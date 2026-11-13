"""Explicit StateGraph nodes, bounded critique loop and interrupt-based HITL."""
import json
import re

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from ..llm import LLM
from ..schemas import INSUFFICIENT, RAGAnswer, SearchResult
from .audit import AuditLog
from .routing import AgentRouter
from .schemas import AgentResult, AgentState, Approval, Critique, I_MAX, ReportAction
from .tools import AgentTools


def after_route(state: AgentState) -> str:
    return {"document": "retrieve", "time": "call_tools", "side_effect": "approve"}.get(
        state["route"], "finalize")


def after_retrieve(state: AgentState) -> str:
    return "generate" if state["retrieved"] and not state["error"] else "finalize"


def after_critique(state: AgentState) -> str:
    critique = state["critique"] or {}
    if critique.get("decision") == "fix" and state["iterations"] < I_MAX:
        return "generate"
    return "finalize"


class AgentNodes:
    def __init__(self, tools: AgentTools, llm: LLM, router: AgentRouter, audit: AuditLog) -> None:
        self.tools, self.llm, self.router, self.audit = tools, llm, router, audit
        self.critic_prompt = (router.prompts_dir / "agent_critique.md").read_text(encoding="utf-8")

    def event(self, state: AgentState, config: RunnableConfig, node: str, **metadata) -> None:
        self.audit.emit(thread_id=config["configurable"]["thread_id"],
                        run_id=state["run_id"], node=node,
                        iterations=state["iterations"], **metadata)

    def route(self, state: AgentState, config: RunnableConfig) -> dict:
        question = state["question"]
        # Resolve explicit short follow-ups using only the previous user question.
        # Context and drafts from a previous turn are never reused as evidence.
        previous = [m.content for m in state["messages"][:-1] if isinstance(m, HumanMessage)]
        if (state.get("route") == "document" and previous
                and re.match(r"^(а\s|и\s|подробнее\b|поясни\b|уточни\b)", question, re.I)):
            question = f"Предыдущий вопрос: {previous[-1]}\nУточнение: {question}"
        route, action, error = self.router.classify(question)
        update = dict(question=question, route=route, retrieved=[], draft="", critique=None,
                      iterations=0, approved=False, pending_action=action,
                      tool_result=None, error=error, final=None)
        self.event({**state, **update}, config, "route", route=route,
                   transition=after_route(update), error=error)
        return update

    def retrieve(self, state: AgentState, config: RunnableConfig) -> dict:
        self.event(state, config, "retrieve", event="tool_call", tool="document_retrieval")
        try:
            chunks = self.tools.document_retrieval(state["question"])
            if any(not isinstance(x["metadata"].get(key), str) or not x["metadata"][key]
                   for x in chunks for key in ("source", "chunk_id")):
                raise ValueError("Missing provenance")
            update = dict(retrieved=chunks, error=None)
        except Exception:
            update = dict(retrieved=[], error="retrieval_error")
        self.event({**state, **update}, config, "retrieve", count=len(update["retrieved"]),
                   transition=after_retrieve({**state, **update}), error=update["error"])
        return update

    def call_tools(self, state: AgentState, config: RunnableConfig) -> dict:
        self.event(state, config, "call_tools", event="tool_call", tool="get_current_time")
        result = self.tools.get_current_time()
        self.event(state, config, "call_tools", transition="finalize")
        return {"tool_result": result}

    def generate(self, state: AgentState, config: RunnableConfig) -> dict:
        iteration = state["iterations"] + 1
        if iteration > I_MAX:
            raise RuntimeError("Generation budget exceeded")
        try:
            chunks = [SearchResult(x["text"], x["metadata"], x["score"]) for x in state["retrieved"]]
            # Reuse the exact Lab 1 grounded + structured prompt construction.
            system, user, _ = self.tools.pipeline._prompt(state["question"], chunks, "zero_shot")
            if state["critique"]:
                user += "\n\nЗамечания предыдущей проверки (исправьте ответ):\n" + json.dumps(
                    state["critique"], ensure_ascii=False)
            draft = self.llm.generate(system, user)
            error = None
        except Exception:
            draft, error = "", "generation_error"
        update = dict(draft=draft, iterations=iteration, error=error)
        self.event({**state, **update}, config, "generate", transition="critique", error=error)
        return update

    @staticmethod
    def validate_draft(state: AgentState) -> RAGAnswer:
        answer = RAGAnswer.model_validate_json(state["draft"], strict=True)
        allowed = {(x["metadata"]["source"], x["metadata"]["chunk_id"]) for x in state["retrieved"]}
        if not answer.answer.strip():
            raise ValueError("empty_answer")
        if any((c.source, c.chunk_id) not in allowed for c in answer.citations):
            raise ValueError("unknown_citation")
        if answer.insufficient_context and answer.citations:
            raise ValueError("insufficient_with_citations")
        if not answer.insufficient_context and not answer.citations:
            raise ValueError("missing_citations")
        return answer

    def critique(self, state: AgentState, config: RunnableConfig) -> dict:
        result: Critique
        try:
            answer = self.validate_draft(state)
        except (ValidationError, ValueError):
            result = Critique(decision="fix", issues=["Исправьте схему JSON и цитаты: только source/chunk_id из CONTEXT; "
                              "достаточный ответ требует цитат, недостаточный — пустого списка."],
                              relevant=False, grounded=False)
        else:
            if answer.insufficient_context or not state["retrieved"]:
                result = Critique(decision="fallback", issues=["Недостаточно контекста."],
                                  relevant=False, grounded=False)
            else:
                # Only cited chunks can substantiate claims. Full retrieval is not
                # allowed to hide unsupported claims behind an unrelated valid ID.
                cited = {(c.source, c.chunk_id) for c in answer.citations}
                context = [x for x in state["retrieved"]
                           if (x["metadata"]["source"], x["metadata"]["chunk_id"]) in cited]
                payload = json.dumps({"QUESTION": state["question"], "DRAFT": answer.model_dump(),
                                      "CONTEXT": context}, ensure_ascii=False)
                try:
                    result = Critique.model_validate_json(
                        self.llm.generate(self.critic_prompt, payload))
                    if result.decision == "accept" and (not result.grounded or not result.relevant or result.issues):
                        result = Critique(decision="fix", issues=result.issues or ["Ответ не обоснован или нерелевантен."],
                                          relevant=result.relevant, grounded=result.grounded)
                    elif result.decision == "fix" and not result.issues:
                        result.issues = ["Перепроверьте релевантность и опору каждого утверждения на CONTEXT."]
                except Exception:
                    result = Critique(decision="fallback", issues=["Проверка обоснованности недоступна."],
                                      grounded=False, relevant=False)
        update = {"critique": result.model_dump()}
        self.event({**state, **update}, config, "critique", decision=result.decision,
                   grounded=result.grounded, relevant=result.relevant, issues_count=len(result.issues),
                   transition=after_critique({**state, **update}))
        return update

    def approve(self, state: AgentState, config: RunnableConfig) -> dict:
        action = ReportAction.model_validate(state["pending_action"])
        # No effect and no broad exception handler before this interrupt: LangGraph
        # restarts this node on resume, returning the supplied value here.
        response = interrupt({"tool": "send_report", "action_id": action.action_id,
                              "arguments": {"body": action.body}, "simulated": True,
                              "prompt": "Подтвердить локальную имитацию отправки отчёта?"})
        try:
            approval = Approval.model_validate(response)
            approved = approval.approved is True and approval.action_id == action.action_id
        except ValidationError:
            approved = False
        self.event(state, config, "approve", event="hitl_decision", approved=approved,
                   action_id=action.action_id, transition="call_side_effect" if approved else "finalize")
        confirmation = "Подтверждаю" if approved else "Отклоняю"
        return {"approved": approved, "messages": [HumanMessage(
            content=f"{confirmation} действие {action.action_id}.", id=state["run_id"] + ":approval")]}

    def call_side_effect(self, state: AgentState, config: RunnableConfig) -> dict:
        if state["approved"] is not True:
            return {"error": "approval_required"}
        action = ReportAction.model_validate(state["pending_action"])
        self.event(state, config, "call_side_effect", event="tool_call", tool="send_report",
                   action_id=action.action_id)
        try:
            update = {"tool_result": self.tools.send_report(action, approved=state["approved"]), "error": None}
        except Exception:
            update = {"tool_result": None, "error": "report_error"}
        self.event(state, config, "call_side_effect", transition="finalize", error=update["error"])
        return update

    def finalize(self, state: AgentState, config: RunnableConfig) -> dict:
        status = "insufficient_context"
        answer = INSUFFICIENT.model_copy(deep=True)
        route, error = state["route"], state["error"]
        if route == "out_of_scope":
            status = "refused"
            answer = self.service_answer("Я помогаю с документацией ALT Linux. Доступны поиск, текущее время и "
                                         "команда «Отправь отчёт: текст» с подтверждением.")
        elif error:
            status = "error"
            answer.answer = "Не удалось безопасно завершить запрос. Проверьте настройки retrieval и LLM."
        elif route == "smalltalk":
            status, answer = "smalltalk", self.service_answer("Здравствуйте! Задайте вопрос по документации ALT Linux.")
        elif route == "time":
            result = state["tool_result"]
            status, answer = "time", self.service_answer(f"{result['iso_time']} ({result['timezone']})")
        elif route == "side_effect":
            if state["approved"] and state["tool_result"]:
                status, answer = "action_completed", self.service_answer("Фиктивная отправка отчёта зарегистрирована локально.")
            else:
                status, answer = "action_rejected", self.service_answer("Отправка отчёта отклонена.")
        elif state["critique"] and state["critique"]["decision"] == "accept":
            status, answer = "answered", self.validate_draft(state)
        elif state["iterations"] == I_MAX and state["critique"]["decision"] == "fix":
            status = "iteration_limit"
        result = AgentResult(**answer.model_dump(), status=status,
                             thread_id=config["configurable"]["thread_id"], run_id=state["run_id"],
                             iterations=state["iterations"], tool_result=state["tool_result"])
        self.event(state, config, "finalize", status=status, transition="END")
        return {"final": result.model_dump(), "messages": [AIMessage(
            content=result.model_dump_json(), id=state["run_id"] + ":final")]}

    @staticmethod
    def service_answer(text: str) -> RAGAnswer:
        return RAGAnswer(answer=text, citations=[], confidence="high", insufficient_context=False)


def build_graph(tools: AgentTools, llm: LLM, router: AgentRouter,
                audit: AuditLog, checkpointer: BaseCheckpointSaver):
    nodes = AgentNodes(tools, llm, router, audit)
    graph = StateGraph(AgentState)
    for name in ("route", "retrieve", "call_tools", "generate", "critique", "approve", "call_side_effect", "finalize"):
        graph.add_node(name, getattr(nodes, name))
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route, ["retrieve", "call_tools", "approve", "finalize"])
    graph.add_conditional_edges("retrieve", after_retrieve, ["generate", "finalize"])
    graph.add_edge("call_tools", "finalize")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges("critique", after_critique, ["generate", "finalize"])
    graph.add_conditional_edges("approve", lambda state: "call_side_effect" if state["approved"] else "finalize",
                                ["call_side_effect", "finalize"])
    graph.add_edge("call_side_effect", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
