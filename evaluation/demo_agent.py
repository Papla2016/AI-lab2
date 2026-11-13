"""Seven reproducible demo requests; LLM and corpus are EXPLICIT fixtures.

The real StateGraph, checkpoints, interrupt/resume, tool calls and audit execute.
This is a control-flow demonstration, not a real ALT Linux retrieval evaluation.
"""
import argparse
import json
from pathlib import Path
from uuid import uuid4

from src.agent.runtime import open_agent
from src.agent.schemas import AgentSettings
from src.config import ROOT
from src.rag import RAGPipeline
from src.schemas import Document, SearchResult


class FixtureStore:
    def __init__(self, empty=False):
        self.results = [] if empty else [SearchResult(
            "Демонстрационный фрагмент: список пакетов обновляется apt-get update.",
            {"source": "fixture-guide.md", "chunk_id": "fixture-apt", "page": 1}, .9)]

    def search(self, query, k):
        return self.results[:k]

    def all(self):
        return [Document(x.text, x.metadata) for x in self.results]


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, system, user):
        if not self.responses:
            raise RuntimeError("No scripted response: fixture must not call a live LLM")
        return json.dumps(self.responses.pop(0), ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/agent-demo")
    parser.add_argument("--log-path", type=Path, default=ROOT / "logs/graph_run.jsonl")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch = uuid4().hex[:8]
    settings = AgentSettings(_env_file=None, agent_checkpoint_db=args.output_dir / "checkpoints.sqlite",
                             agent_reports_db=args.output_dir / "reports.sqlite", agent_log_path=args.log_path)
    good = dict(answer="Список пакетов обновляется apt-get update.",
                citations=[dict(source="fixture-guide.md", chunk_id="fixture-apt")],
                confidence="high", insufficient_context=False)
    bad = dict(good, answer="apt-get update автоматически перезагружает компьютер.")
    cases = [
        ("smalltalk", "Привет", [], False, None),
        ("out_of_scope", "Напиши рецепт пирога", [dict(route="out_of_scope")], False, None),
        ("time", "Сколько времени?", [], False, None),
        ("critique_fix", "Как обновить список пакетов ALT Linux?", [dict(route="document"), bad,
         dict(decision="fix", issues=["В контексте нет перезагрузки."], relevant=True, grounded=False), good,
         dict(decision="accept", issues=[], relevant=True, grounded=True)], False, None),
        ("insufficient", "Что сказано в документах ALT Linux об отсутствующей функции?", [dict(route="document")], True, None),
        ("report_yes", "Отправь отчёт: демонстрация согласия", [], False, True),
        ("report_no", "Отправь отчёт: демонстрация отказа", [], False, False),
    ]
    results = []
    for name, question, responses, empty, approval in cases:
        llm = ScriptedLLM(responses)
        pipeline = RAGPipeline(FixtureStore(empty), llm, settings.prompts_dir)
        thread = f"demo-{batch}-{name}"
        with open_agent(settings, llm=llm, pipeline_factory=lambda: pipeline, mode="fixture") as agent:
            outcome = agent.ask(question, thread)
        interrupted = outcome.status == "interrupted"
        if interrupted:
            # Reopen the SQLite checkpointer to demonstrate durable HITL.
            with open_agent(settings, llm=llm, pipeline_factory=lambda: pipeline, mode="fixture") as agent:
                outcome = agent.resume(thread, approval)
        row = dict(scenario=name, question=question, mode="fixture", interrupted=interrupted,
                   **outcome.model_dump())
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))
    (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
