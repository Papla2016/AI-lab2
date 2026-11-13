"""Persistent LangGraph lifecycle shared by the CLI, demos and tests."""
import re
from contextlib import contextmanager
from typing import Callable, Iterator, Literal
from uuid import uuid4

from filelock import FileLock, Timeout
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from ..llm import LLM, OpenAICompatibleLLM
from ..main import components
from ..rag import RAGPipeline
from ..reranker import Reranker
from .audit import AuditLog
from .graph import build_graph
from .routing import AgentRouter
from .schemas import AgentResult, AgentSettings
from .tools import AgentTools


class Outcome(BaseModel):
    thread_id: str
    run_id: str
    status: Literal["interrupted", "completed"]
    result: AgentResult | None = None
    interrupt: dict | None = None


class LazyLLM:
    def __init__(self, settings: AgentSettings) -> None:
        self.settings, self._llm = settings, None

    def generate(self, system: str, user: str) -> str:
        if self._llm is None:
            s = self.settings
            self._llm = OpenAICompatibleLLM(s.llm_base_url, s.llm_api_key, s.llm_model)
        return self._llm.generate(system, user)


class GraphAgent:
    def __init__(self, graph, audit: AuditLog) -> None:
        self.graph, self.audit = graph, audit

    @staticmethod
    def config(thread_id: str) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", thread_id):
            raise ValueError("thread_id: 1–64 символа A–Z, a–z, 0–9, _ или -.")
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 32}

    def state(self, thread_id: str):
        return self.graph.get_state(self.config(thread_id))

    def threads(self) -> list[str]:
        """Discover persisted dialogues through the checkpointer's public API."""
        return sorted({item.config["configurable"]["thread_id"]
                       for item in self.graph.checkpointer.list(None)})

    def ask(self, question: str, thread_id: str) -> Outcome:
        if not question.strip() or len(question) > 8000:
            raise ValueError("Вопрос должен содержать от 1 до 8000 символов.")
        if self.state(thread_id).next:
            raise ValueError("Диалог приостановлен. Используйте status и resume/continue.")
        run_id = uuid4().hex
        return self._invoke(thread_id, {"question": question.strip(), "run_id": run_id,
                                      "messages": [HumanMessage(content=question.strip(), id=run_id + ":user")]})

    def resume(self, thread_id: str, approved: bool, action_id: str | None = None) -> Outcome:
        if type(approved) is not bool:
            raise ValueError("Подтверждение должно быть boolean.")
        snapshot = self.state(thread_id)
        interrupts = [item for task in snapshot.tasks for item in task.interrupts]
        if not interrupts:
            raise ValueError("В этом диалоге нет ожидающего подтверждения.")
        expected = interrupts[0].value["action_id"]
        self.audit.emit(thread_id=thread_id, run_id=snapshot.values["run_id"], node="approve",
                        iterations=snapshot.values["iterations"], event="hitl_resume", approved=approved)
        return self._invoke(thread_id, Command(resume={"approved": approved, "action_id": action_id or expected}))

    def continue_run(self, thread_id: str) -> Outcome:
        snapshot = self.state(thread_id)
        if not snapshot.next:
            raise ValueError("В этом диалоге нет незавершённого запуска.")
        if any(task.interrupts for task in snapshot.tasks):
            raise ValueError("Требуется явное решение: resume --decision yes/no.")
        return self._invoke(thread_id, None)

    def _invoke(self, thread_id: str, value) -> Outcome:
        output = self.graph.invoke(value, config=self.config(thread_id))
        if output.get("__interrupt__"):
            request = output["__interrupt__"][0].value
            self.audit.emit(thread_id=thread_id, run_id=output["run_id"], node="approve",
                            iterations=output["iterations"], event="hitl_interrupt", transition="interrupt",
                            action_id=request["action_id"])
            return Outcome(thread_id=thread_id, run_id=output["run_id"], status="interrupted", interrupt=request)
        return Outcome(thread_id=thread_id, run_id=output["run_id"], status="completed",
                       result=AgentResult.model_validate(output["final"]))


class AgentBusyError(RuntimeError):
    """Another process/session currently owns this checkpoint database."""


@contextmanager
def open_agent(settings: AgentSettings | None = None, *, llm: LLM | None = None,
               pipeline_factory: Callable[[], RAGPipeline] | None = None,
               observer: Callable[[dict], None] | None = None,
               mode: str = "live") -> Iterator[GraphAgent]:
    s = settings or AgentSettings()
    backend = llm or LazyLLM(s)

    def default_pipeline() -> RAGPipeline:
        store, _ = components(s)
        return RAGPipeline(store, backend, s.prompts_dir, s.top_n, s.final_k,
                           s.min_dense_score, s.rrf_k, Reranker(s.rerank_model, s.rerank_enabled))

    tools = AgentTools(pipeline_factory or default_pipeline, s.retrieval_mode, s.agent_timezone, s.agent_reports_db)
    audit = AuditLog(s.agent_log_path, observer, mode)
    s.agent_checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    # A separate saver/connection per operation, serialized across CLI processes
    # and Streamlit sessions. Never hold/cache a SQLite connection in the UI.
    lock = FileLock(str(s.agent_checkpoint_db.resolve()) + ".lock")
    try:
        lock.acquire(timeout=0)
    except Timeout:
        raise AgentBusyError("Агент занят. Повторите операцию после завершения запроса.") from None
    try:
        with SqliteSaver.from_conn_string(str(s.agent_checkpoint_db)) as checkpointer:
            yield GraphAgent(build_graph(tools, backend, AgentRouter(backend, s.prompts_dir), audit, checkpointer), audit)
    finally:
        lock.release()
