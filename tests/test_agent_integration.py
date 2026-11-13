"""Real Chroma / real subprocesses; external models are a separate opt-in test."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from src.agent.runtime import open_agent
from src.agent.schemas import AgentSettings
from src.rag import RAGPipeline
from src.schemas import Document
from src.vector_store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


class FixtureEmbeddings:
    """Predictable vectors, explicitly not a semantic embedding benchmark."""
    def encode_documents(self, texts):
        return [[1., 0., 0.] if "пакет" in x else [0., 1., 0.] for x in texts]

    def encode_query(self, text):
        return [1., 0., 0.]


class ContextLLM:
    def __init__(self):
        self.generated = False

    def generate(self, system, user):
        if "маршрутизатор" in system:
            return '{"route":"document"}'
        if "независимый критик" in system:
            payload = json.loads(user)
            assert payload["CONTEXT"][0]["metadata"]["chunk_id"] == "real-chroma-chunk"
            return '{"decision":"accept","issues":[],"grounded":true,"relevant":true}'
        assert "chunk_id=real-chroma-chunk" in user and "apt-get update" in user
        self.generated = True
        return json.dumps(dict(answer="Список пакетов обновляет apt-get update.",
                               citations=[dict(source="fixture.md", chunk_id="real-chroma-chunk")],
                               confidence="high", insufficient_context=False))


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_graph_uses_real_lab1_chroma_and_reranker(tmp_path, mode):
    s = AgentSettings(_env_file=None, retrieval_mode=mode,
                      agent_checkpoint_db=tmp_path / "checkpoints.sqlite",
                      agent_reports_db=tmp_path / "reports.sqlite", agent_log_path=tmp_path / "graph.jsonl")
    store = VectorStore(tmp_path / "chroma", "agent_integration", FixtureEmbeddings())
    store.replace([Document("Список пакетов обновляет apt-get update.",
                            {"source": "fixture.md", "chunk_id": "real-chroma-chunk", "page": 1}),
                   Document("Справка о другом разделе.",
                            {"source": "other.md", "chunk_id": "other-chunk", "page": 2})])

    class SpyReranker:
        calls = 0

        def rerank(self, query, results, k):
            self.calls += 1
            assert any(x.metadata["chunk_id"] == "real-chroma-chunk" for x in results)
            return results[:k]

    llm, reranker = ContextLLM(), SpyReranker()
    pipeline = RAGPipeline(store, llm, s.prompts_dir, final_k=1, reranker=reranker)
    with open_agent(s, llm=llm, pipeline_factory=lambda: pipeline, mode="fixture") as agent:
        result = agent.ask("Как обновить список пакетов ALT Linux?", "chroma").result
    assert result.status == "answered" and llm.generated and reranker.calls == 1
    assert result.citations[0].chunk_id == "real-chroma-chunk"


def test_cli_separate_processes_preserve_and_resume_hitl(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT),
               AGENT_CHECKPOINT_DB=str(tmp_path / "checkpoint.sqlite"),
               AGENT_REPORTS_DB=str(tmp_path / "reports.sqlite"),
               AGENT_LOG_PATH=str(tmp_path / "graph.jsonl"))

    def command(*args, code=0):
        completed = subprocess.run([sys.executable, "-m", "src.agent.cli", *args],
                                   cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        assert completed.returncode == code, completed.stderr
        return json.loads(completed.stdout) if code == 0 else None

    start = command("ask", "Отправь отчёт: subprocess fixture", "--thread-id", "cli")
    assert start["status"] == "interrupted" and not (tmp_path / "reports.sqlite").exists()
    status = command("status", "--thread-id", "cli")
    assert status["interrupt"]["action_id"] == start["interrupt"]["action_id"]
    assert command("status", "--thread-id", "other")["history_messages"] == 0
    resumed = command("resume", "--thread-id", "cli", "--decision", "yes")
    assert resumed["result"]["status"] == "action_completed"
    command("resume", "--thread-id", "cli", "--decision", "yes", code=2)
    assert command("ask", "Привет", "--thread-id", "cli")["result"]["status"] == "smalltalk"
    with sqlite3.connect(tmp_path / "reports.sqlite") as conn:
        assert conn.execute("SELECT count(*) FROM reports").fetchone()[0] == 1


@pytest.mark.live
@pytest.mark.skipif(os.getenv("RUN_LIVE_AGENT") != "1", reason="Requires local ALT Linux corpus, embeddings and LLM")
def test_live_alt_linux_corpus_and_llm(tmp_path):
    question = os.environ.get("AGENT_LIVE_QUESTION")
    assert question, "Set AGENT_LIVE_QUESTION to a question answered by your indexed ALT Linux corpus"
    s = AgentSettings(agent_checkpoint_db=tmp_path / "live.sqlite",
                      agent_reports_db=tmp_path / "reports.sqlite", agent_log_path=tmp_path / "live.jsonl")
    with open_agent(s) as agent:
        result = agent.ask(question, "live-integration").result
        state = agent.state("live-integration").values
    assert result.status == "answered", result.model_dump_json()
    allowed = {(x["metadata"]["source"], x["metadata"]["chunk_id"]) for x in state["retrieved"]}
    assert result.citations and all((c.source, c.chunk_id) in allowed for c in result.citations)
