"""Deterministic tests execute the real StateGraph and SQLite checkpointer."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from langgraph.types import Command

from src.agent.runtime import open_agent
from src.agent.schemas import AgentSettings, I_MAX, ReportAction
from src.agent.tools import AgentTools
from src.rag import RAGPipeline
from src.schemas import Document, SearchResult

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
DOC = json.dumps({"route": "document"})
ACCEPT = json.dumps({"decision": "accept", "issues": [], "relevant": True, "grounded": True})
FIX = json.dumps({"decision": "fix", "issues": ["Уберите неподтверждённое утверждение."],
                  "relevant": True, "grounded": False}, ensure_ascii=False)
VALID = json.dumps({"answer": "Для обновления списка пакетов используется apt-get update.",
                    "citations": [{"source": "guide.md", "chunk_id": "chunk-1"}],
                    "confidence": "high", "insufficient_context": False}, ensure_ascii=False)


class SequenceLLM:
    def __init__(self, *responses):
        self.responses, self.prompts = list(responses), []

    def generate(self, system, user):
        self.prompts.append((system, user))
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class Store:
    def __init__(self, empty=False, score=.9):
        self.calls = 0
        self.results = [] if empty else [SearchResult(
            "Для обновления списка пакетов используется apt-get update.",
            {"source": "guide.md", "chunk_id": "chunk-1", "page": 7, "custom": "preserved"}, score)]

    def search(self, question, k):
        self.calls += 1
        return self.results[:k]

    def all(self):
        return [Document(x.text, x.metadata) for x in self.results]


@pytest.fixture
def settings(tmp_path):
    return AgentSettings(_env_file=None, prompts_dir=PROMPTS, retrieval_mode="hybrid",
                         agent_checkpoint_db=tmp_path / "checkpoints.sqlite",
                         agent_reports_db=tmp_path / "reports.sqlite",
                         agent_log_path=tmp_path / "graph.jsonl")


def opened(settings, llm, store=None, **kwargs):
    pipeline = RAGPipeline(store or Store(), llm, PROMPTS)
    return open_agent(settings, llm=llm, pipeline_factory=lambda: pipeline, mode="test", **kwargs)


def test_factual_preserves_context_citations_and_real_trace(settings):
    llm = SequenceLLM(DOC, VALID, ACCEPT)
    with opened(settings, llm) as agent:
        outcome = agent.ask("Как обновить пакеты ALT Linux?", "one")
        result = outcome.result
        state = agent.state("one").values
    assert result.status == "answered" and result.iterations == 1
    assert result.citations[0].source == "guide.md"
    assert state["retrieved"][0]["metadata"]["custom"] == "preserved"
    assert "chunk_id=chunk-1" in llm.prompts[1][1]
    assert json.loads(llm.prompts[2][1])["CONTEXT"][0]["metadata"]["page"] == 7
    events = [json.loads(line) for line in settings.agent_log_path.read_text().splitlines()]
    assert [e["node"] for e in events if e["event"] == "node"] == [
        "route", "retrieve", "generate", "critique", "finalize"]
    assert all(e["thread_id"] == "one" and e["run_id"] == result.run_id for e in events)
    assert events[-1]["status"] == "answered"


@pytest.mark.parametrize("empty,score", [(True, .9), (False, .01)])
def test_insufficient_context_without_generation(settings, empty, score):
    llm = SequenceLLM(DOC)
    with opened(settings, llm, Store(empty=empty, score=score)) as agent:
        result = agent.ask("Что сказано в документах ALT Linux?", "empty").result
    assert result.insufficient_context and result.citations == [] and result.iterations == 0
    assert len(llm.prompts) == 1


def test_semantic_critique_repairs_grounding_with_valid_citation(settings):
    unsupported = json.loads(VALID)
    unsupported["answer"] = "apt-get update автоматически перезагружает сервер."
    llm = SequenceLLM(DOC, json.dumps(unsupported), FIX, VALID, ACCEPT)
    with opened(settings, llm) as agent:
        result = agent.ask("Как обновить пакеты ALT Linux?", "repair").result
    assert result.status == "answered" and result.iterations == 2
    assert "Уберите неподтверждённое" in llm.prompts[3][1]
    assert len(llm.prompts) == 5


@pytest.mark.parametrize("bad", [
    "not json",
    json.dumps({**json.loads(VALID), "citations": []}),
    json.dumps({**json.loads(VALID), "citations": [{"source": "invented.md", "chunk_id": "chunk-1"}]}),
    json.dumps({**json.loads(VALID), "citations": [{"source": "guide.md", "chunk_id": "invented"}]}),
    json.dumps({**json.loads(VALID), "insufficient_context": True}),
])
def test_structural_critique_retries_then_limits_to_three(settings, bad):
    llm = SequenceLLM(DOC, bad, bad, bad)
    with opened(settings, llm) as agent:
        result = agent.ask("Как обновить ALT Linux?", "limit").result
    assert result.status == "iteration_limit" and result.iterations == I_MAX
    assert result.insufficient_context and result.citations == []
    assert len(llm.prompts) == 1 + I_MAX  # no semantic judge for invalid structure


def test_semantic_fixes_also_obey_budget(settings):
    llm = SequenceLLM(DOC, VALID, FIX, VALID, FIX, VALID, FIX)
    with opened(settings, llm) as agent:
        result = agent.ask("Как обновить ALT Linux?", "semantic-limit").result
    assert result.status == "iteration_limit" and result.iterations == 3
    assert len(llm.prompts) == 7


@pytest.mark.parametrize("judgment", ["garbage", RuntimeError("secret API key"), json.dumps({
    "decision": "fallback", "issues": ["Нет доказательств"], "relevant": False, "grounded": False})])
def test_failed_or_negative_critic_cannot_accept(settings, judgment):
    with opened(settings, SequenceLLM(DOC, VALID, judgment)) as agent:
        result = agent.ask("Как обновить ALT Linux?", "bad-critic").result
    assert result.status == "insufficient_context" and result.citations == []


def test_inconsistent_accept_is_repaired(settings):
    dishonest_accept = json.dumps({"decision": "accept", "issues": [], "grounded": False, "relevant": True})
    with opened(settings, SequenceLLM(DOC, VALID, dishonest_accept, VALID, ACCEPT)) as agent:
        result = agent.ask("Как обновить ALT Linux?", "inconsistent").result
    assert result.iterations == 2 and result.status == "answered"


def test_model_insufficient_answer_is_canonicalized(settings):
    insufficient = json.dumps({**json.loads(VALID), "answer": "Текст, который нельзя выдавать как факт.",
                               "insufficient_context": True, "citations": []})
    with opened(settings, SequenceLLM(DOC, insufficient)) as agent:
        result = agent.ask("Что сказано в документах?", "model-insufficient").result
    assert result.insufficient_context and "Текст, который" not in result.answer


def test_generation_failures_are_counted_and_private(settings):
    llm = SequenceLLM(DOC, *(RuntimeError("secret-token") for _ in range(3)))
    with opened(settings, llm) as agent:
        result = agent.ask("Как обновить ALT Linux?", "api-errors").result
    assert result.iterations == 3 and result.status == "error"
    assert "secret-token" not in result.model_dump_json()
    assert "secret-token" not in settings.agent_log_path.read_text()


@pytest.mark.parametrize("decision", [True, False])
def test_hitl_persists_across_reopen_and_executes_only_on_yes(settings, decision):
    with opened(settings, SequenceLLM()) as agent:
        outcome = agent.ask("Отправь отчёт: тестовый отчёт ALT Linux", "hitl")
        assert outcome.status == "interrupted" and outcome.result is None
        assert agent.state("hitl").next == ("approve",)
        assert not settings.agent_reports_db.exists()
        with pytest.raises(ValueError):
            agent.ask("Привет", "hitl")
        with pytest.raises(ValueError):
            agent.continue_run("hitl")
    with opened(settings, SequenceLLM()) as reopened:
        result = reopened.resume("hitl", decision).result
        with pytest.raises(ValueError):
            reopened.resume("hitl", True)
    assert result.status == ("action_completed" if decision else "action_rejected")
    if decision:
        with sqlite3.connect(settings.agent_reports_db) as conn:
            assert conn.execute("SELECT count(*) FROM reports").fetchone()[0] == 1
    else:
        assert not settings.agent_reports_db.exists()
    events = [json.loads(x) for x in settings.agent_log_path.read_text().splitlines()]
    assert len([e for e in events if e.get("tool") == "send_report"]) == int(decision)
    assert "тестовый отчёт" not in settings.agent_log_path.read_text()


@pytest.mark.parametrize("payload", [True, "yes", {"approved": "true", "action_id": "wrong"},
                                     {"approved": True, "action_id": "wrong"}])
def test_approval_is_strict_and_bound_to_action(settings, payload):
    with opened(settings, SequenceLLM()) as agent:
        agent.ask("Отправь отчёт: проверка", "strict")
        output = agent.graph.invoke(Command(resume=payload), config=agent.config("strict"))
    assert output["final"]["status"] == "action_rejected"
    assert not settings.agent_reports_db.exists()


def test_empty_resume_mapping_stays_safely_interrupted(settings):
    # In LangGraph 1.2, {} is an empty interrupt-ID -> value mapping.
    with opened(settings, SequenceLLM()) as agent:
        agent.ask("Отправь отчёт: проверка", "empty-resume")
        output = agent.graph.invoke(Command(resume={}), config=agent.config("empty-resume"))
        assert output.get("__interrupt__") and agent.state("empty-resume").next == ("approve",)
    assert not settings.agent_reports_db.exists()


def test_effect_is_idempotent_after_crash_before_checkpoint(settings):
    def crash_after_write(event):
        if event["node"] == "call_side_effect" and event["event"] == "node":
            raise RuntimeError("simulated crash after the committed local effect")

    with opened(settings, SequenceLLM(), observer=crash_after_write) as agent:
        agent.ask("Отправь отчёт: crash test", "crash")
        with pytest.raises(RuntimeError, match="simulated crash"):
            agent.resume("crash", True)
    with opened(settings, SequenceLLM()) as agent:
        assert agent.continue_run("crash").result.status == "action_completed"
    with sqlite3.connect(settings.agent_reports_db) as conn:
        assert conn.execute("SELECT count(*) FROM reports").fetchone()[0] == 1


def test_report_tool_refuses_without_approval_before_creating_file(settings):
    tools = AgentTools(lambda: None, "dense", "Europe/Moscow", settings.agent_reports_db)
    action = ReportAction(action_id="a" * 32, body="test")
    for value in (False, "true", 1, None):
        with pytest.raises(PermissionError):
            tools.send_report(action, approved=value)
    assert not settings.agent_reports_db.exists()


def test_persistent_history_isolation_and_per_turn_reset(settings):
    with opened(settings, SequenceLLM(DOC, VALID, ACCEPT)) as agent:
        agent.ask("Как обновить ALT Linux?", "alice")
        agent.ask("Отправь отчёт: alice", "pending-alice")
        agent.resume("pending-alice", True)
    with opened(settings, SequenceLLM()) as agent:
        assert len(agent.state("alice").values["messages"]) == 2
        assert agent.state("bob").values == {}
        agent.ask("Привет", "alice")
        state = agent.state("alice").values
        assert len(state["messages"]) == 4
        assert state["retrieved"] == [] and state["draft"] == "" and state["critique"] is None
        assert state["iterations"] == 0 and state["tool_result"] is None
        assert state["approved"] is False and state["pending_action"] is None
        agent.ask("Привет", "pending-alice")
        state = agent.state("pending-alice").values
        assert state["approved"] is False and state["pending_action"] is None
        assert state["tool_result"] is None
        agent.ask("Привет", "bob")
        assert len(agent.state("bob").values["messages"]) == 2


def test_follow_up_retrieves_fresh_context(settings):
    llm = SequenceLLM(DOC, VALID, ACCEPT, DOC, VALID, ACCEPT)
    store = Store()
    with opened(settings, llm, store) as agent:
        agent.ask("Как обновить ALT Linux?", "follow-up")
        agent.ask("А подробнее?", "follow-up")
    assert "Предыдущий вопрос:" in llm.prompts[3][1]
    assert "Уточнение: А подробнее?" in llm.prompts[4][1]
    assert store.calls == 4  # retrieval + independent dense gate, twice


@pytest.mark.parametrize("question,response,status", [
    ("Привет!", None, "smalltalk"),
    ("Сколько времени?", None, "time"),
    ("Напиши рецепт пирога", '{"route":"out_of_scope"}', "refused"),
    ("Выполни rm -rf /", '{"route":"side_effect"}', "refused"),
    ("Что-то непонятное", "invalid classifier JSON", "refused"),
    ("Ошибка провайдера", RuntimeError("secret-token"), "refused"),
])
def test_router_and_safe_tools_skip_retrieval(settings, question, response, status):
    def forbidden_pipeline():
        pytest.fail("Service route must not initialize retrieval")
    llm = SequenceLLM(*([] if response is None else [response]))
    with open_agent(settings, llm=llm, pipeline_factory=forbidden_pipeline) as agent:
        result = agent.ask(question, "routing").result
    assert result.status == status and result.iterations == 0
    if status == "time":
        assert result.tool_result["timezone"] == "Europe/Moscow"
        assert datetime.fromisoformat(result.tool_result["iso_time"]).utcoffset().total_seconds() == 10800


def test_retrieval_failure_cannot_leak_exception(settings):
    def unavailable():
        raise RuntimeError("secret-token in unavailable store")
    with open_agent(settings, llm=SequenceLLM(DOC), pipeline_factory=unavailable) as agent:
        result = agent.ask("Как обновить ALT Linux?", "retrieval-error").result
    assert result.status == "error" and result.insufficient_context
    assert "secret-token" not in settings.agent_log_path.read_text()
