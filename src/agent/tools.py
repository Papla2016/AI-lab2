"""Three allowlisted tools; retrieval is exclusively the Lab 1 implementation."""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from ..rag import RAGPipeline
from .schemas import Chunk, ReportAction


class AgentTools:
    def __init__(self, pipeline_factory: Callable[[], RAGPipeline], mode: str,
                 timezone: str, reports_db: Path) -> None:
        self._factory, self.mode = pipeline_factory, mode
        self.zone = ZoneInfo(timezone)
        self.reports_db = reports_db
        self._pipeline: RAGPipeline | None = None

    @property
    def pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            self._pipeline = self._factory()
        return self._pipeline

    def document_retrieval(self, question: str) -> list[Chunk]:
        pipeline = self.pipeline
        _, selected = pipeline.retrieve(question, self.mode)
        # Preserve Lab 1's semantic gate: hybrid RRF / cross-encoder scores
        # cannot be compared to a dense similarity threshold.
        dense_best = pipeline.store.search(question, 1)
        if not selected or not dense_best or dense_best[0].score < pipeline.min_score:
            return []
        return [Chunk(text=x.text, metadata=dict(x.metadata), score=x.score) for x in selected]

    def get_current_time(self) -> dict[str, str]:
        return {"iso_time": datetime.now(self.zone).isoformat(), "timezone": self.zone.key}

    def send_report(self, action: ReportAction, *, approved: bool) -> dict:
        if approved is not True:
            raise PermissionError("Explicit approval is required")
        # The local DB is both the simulated delivery and its durable idempotency
        # ledger: one transaction, one UNIQUE action_id, no network or subprocess.
        self.reports_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.reports_db) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS reports ("
                               "action_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, body TEXT NOT NULL)")
            connection.execute("INSERT OR IGNORE INTO reports VALUES (?, ?, ?)",
                               (action.action_id, datetime.now(self.zone).isoformat(), action.body))
            saved = connection.execute("SELECT body FROM reports WHERE action_id = ?",
                                       (action.action_id,)).fetchone()
            if saved[0] != action.body:
                raise ValueError("Action identifier collision")
        return {"tool": "send_report", "action_id": action.action_id, "simulated": True}
