"""Narrow deterministic tool commands; a fail-closed document classifier."""
import re
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from ..llm import LLM
from .schemas import ReportAction, Route


class DocumentRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    route: Route


class AgentRouter:
    def __init__(self, llm: LLM, prompts_dir: Path) -> None:
        self.llm = llm
        self.prompts_dir = prompts_dir
        self.prompt = (prompts_dir / "agent_router.md").read_text(encoding="utf-8")

    def classify(self, question: str) -> tuple[Route, dict | None, str | None]:
        normalized = question.strip().lower().rstrip(".!? ")
        if normalized in {"привет", "здравствуйте", "добрый день", "добрый вечер", "hello", "hi"}:
            return "smalltalk", None, None
        if normalized in {"который час", "сколько времени", "текущее время", "what time is it"}:
            return "time", None, None
        report = re.fullmatch(r"(?:отправь|отправьте|отправить)\s+отч[её]т\s*:\s*(.+)",
                              question.strip(), flags=re.IGNORECASE | re.DOTALL)
        if report:
            try:
                action = ReportAction(action_id=uuid4().hex, body=report[1].strip())
                return "side_effect", action.model_dump(), None
            except ValidationError:
                return "out_of_scope", None, "invalid_report"
        if not normalized:
            return "out_of_scope", None, None
        try:
            decision = DocumentRoute.model_validate_json(self.llm.generate(self.prompt, question)).route
            # A model cannot introduce a tool call. Those need the exact commands above.
            if decision != "document":
                return "out_of_scope", None, None
            return "document", None, None
        except Exception:
            # Provider exception bodies and malformed classifier output stay private.
            return "out_of_scope", None, "routing_error"
