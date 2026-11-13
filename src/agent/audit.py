"""Content-free JSONL audit emitted by actual graph execution."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class AuditLog:
    def __init__(self, path: Path, observer: Callable[[dict], None] | None = None,
                 mode: str = "live") -> None:
        self.path, self.observer, self.mode = path, observer, mode

    def emit(self, *, thread_id: str, run_id: str, node: str, iterations: int,
             event: str = "node", **metadata) -> None:
        # Callers pass only fixed decisions, counts, codes and opaque identifiers.
        # Never log question, body, documents, prompts or exception/LLM text.
        record = dict(timestamp=datetime.now(timezone.utc).isoformat(), mode=self.mode,
                      thread_id=thread_id, run_id=run_id, node=node,
                      iterations=iterations, event=event, **metadata)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.observer:
            self.observer(record)
