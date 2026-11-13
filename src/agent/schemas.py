"""JSON-compatible checkpoints and explicit public result contracts."""
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from ..config import ROOT, Settings
from ..schemas import RAGAnswer

Route = Literal["document", "smalltalk", "time", "side_effect", "out_of_scope"]
Status = Literal[
    "answered", "insufficient_context", "iteration_limit", "smalltalk", "time",
    "refused", "action_completed", "action_rejected", "error",
]
I_MAX = 3


class AgentSettings(Settings):
    agent_checkpoint_db: Path = ROOT / "storage/agent_checkpoints.sqlite"
    agent_reports_db: Path = ROOT / "storage/agent_reports.sqlite"
    agent_log_path: Path = ROOT / "logs/graph_run.jsonl"
    agent_timezone: str = "Europe/Moscow"


class Chunk(TypedDict):
    text: str
    metadata: dict[str, Any]
    score: float


class ReportAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    body: str = Field(min_length=1, max_length=4000)


class Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    approved: bool
    action_id: str


class Critique(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["accept", "fix", "fallback"]
    issues: list[str] = Field(max_length=10)
    relevant: bool
    grounded: bool


class AgentResult(RAGAnswer):
    status: Status
    thread_id: str
    run_id: str
    iterations: int = Field(ge=0, le=I_MAX)
    tool_result: dict[str, Any] | None = None


class AgentState(TypedDict, total=False):
    # Only the dialogue accumulates. Every other channel uses last-value updates.
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    run_id: str
    route: Route
    retrieved: list[Chunk]
    draft: str
    critique: dict[str, Any] | None
    critique_history: list[dict[str, Any]]
    iterations: int
    approved: bool
    pending_action: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    error: str | None
    final: dict[str, Any] | None
