"""Real Streamlit + GraphAgent + SQLite, deterministic LLM/store fixtures only."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

from src.agent import streamlit_ui as ui
from src.agent.runtime import AgentBusyError, open_agent
from src.agent.schemas import AgentSettings
from src.agent.tools import AgentTools
from src.agent.web_service import AgentWebService
from src.rag import RAGPipeline
from src.schemas import Document, SearchResult

ROOT = Path(__file__).resolve().parents[1]
DOC = {"route": "document"}
VALID = {"answer": "Используйте apt-get update.", "citations": [{"source": "guide.md", "chunk_id": "chunk-1"}],
         "confidence": "high", "insufficient_context": False}
ACCEPT = {"decision": "accept", "issues": [], "grounded": True, "relevant": True}
FIX = {"decision": "fix", "issues": ["Уберите неподтверждённое утверждение."], "grounded": False, "relevant": True}


class MockLLM:
    def __init__(self):
        self.responses, self.calls = [], []

    def generate(self, system, user):
        self.calls.append((system, user))
        assert self.responses, "Unexpected LLM call"
        response = self.responses.pop(0)
        return json.dumps(response, ensure_ascii=False)


class MockStore:
    empty = False

    def search(self, query, k):
        return [] if self.empty else [SearchResult("Используйте apt-get update.",
                                                  {"source": "guide.md", "chunk_id": "chunk-1"}, .9)]

    def all(self):
        return [Document(x.text, x.metadata) for x in self.search("", 5)]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    s = AgentSettings(_env_file=None, agent_checkpoint_db=tmp_path / "checkpoints.sqlite",
                      agent_reports_db=tmp_path / "reports.sqlite", agent_log_path=tmp_path / "audit.jsonl")
    llm, store = MockLLM(), MockStore()

    @contextmanager
    def opener(settings, observer=None):
        pipeline = RAGPipeline(store, llm, settings.prompts_dir)
        with open_agent(settings, llm=llm, pipeline_factory=lambda: pipeline,
                        observer=observer, mode="test") as agent:
            yield agent

    service = AgentWebService(s, opener=opener)
    monkeypatch.setattr(ui, "make_service", lambda: service)

    def forbidden(*args, **kwargs):
        pytest.fail("UI must not initialize embeddings or bypass GraphAgent via RAGPipeline.ask")

    monkeypatch.setattr(ui, "cached_embeddings", forbidden)
    monkeypatch.setattr(RAGPipeline, "ask", forbidden)
    return service, llm, store


def app(thread="ui"):
    test = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15)
    test.query_params["agent_thread"] = thread
    test.run()
    assert not test.exception
    return test


def button(test, label):
    return next(item for item in test.button if item.label == label)


def rendered(test):
    return "\n".join(str(item.value) for kind in ("text", "caption", "markdown", "info", "warning", "error")
                     for item in getattr(test, kind))


def send(test, text):
    test.chat_input[0].set_value(text).run()
    assert not test.exception


def test_factual_chat_citations_restoration_and_lab1_tabs(harness):
    service, llm, _ = harness
    llm.responses.extend([DOC, VALID, ACCEPT])
    test = app()
    assert [tab.label for tab in test.tabs] == [
        "Документы", "Поиск", "RAG-ответ", "Сравнение", "Антигаллюцинации", "AI-агент (LangGraph)"]
    assert button(test, "Выполнить RAG") and button(test, "Найти чанки")
    send(test, "Как обновить пакеты ALT Linux?")
    assert "guide.md" in rendered(test) and "chunk-1" in rendered(test)
    assert "confidence: высокая" in rendered(test) and "insufficient_context: нет" in rendered(test)
    assert "Генерация 1: accept" in rendered(test)
    assert test.dataframe[-1].value["Узел"].tolist() == [
        "route", "retrieve", "retrieve", "generate", "critique", "finalize"]
    assert "critique" in service.inspect("ui").mermaid
    # A fresh session (including query param) restores SQLite, without an LLM call.
    reopened = app()
    assert len(reopened.chat_message) == 2 and "apt-get update" in rendered(reopened)
    assert len(llm.calls) == 3
    send(reopened, "Привет")
    assert len(reopened.chat_message) == 4 and len(llm.calls) == 3


def test_empty_retrieval_ui(harness):
    _, llm, store = harness
    store.empty = True
    llm.responses.append(DOC)
    test = app()
    send(test, "Что сказано в документах ALT Linux?")
    assert "insufficient_context: да" in rendered(test)
    assert "Финальный статус: insufficient_context" in rendered(test)
    assert len(llm.calls) == 1


def test_semantic_fallback_is_visible(harness):
    _, llm, _ = harness
    llm.responses.extend([DOC, VALID, {"decision": "fallback", "issues": ["Недостаточно доказательств."],
                                       "grounded": False, "relevant": True}])
    test = app()
    send(test, "Как обновить пакеты ALT Linux?")
    assert "Генерация 1: fallback" in rendered(test)
    assert "Недостаточно доказательств." in rendered(test)
    assert "insufficient_context: да" in rendered(test)


@pytest.mark.parametrize("limited", [False, True])
def test_critique_iterations_and_history_are_real(harness, limited):
    service, llm, _ = harness
    responses = [DOC, VALID, FIX, VALID, FIX, VALID, FIX] if limited else [DOC, VALID, FIX, VALID, ACCEPT]
    llm.responses.extend(responses)
    test = app()
    send(test, "Как обновить пакеты ALT Linux?")
    text = rendered(test)
    assert "Генерация 1: fix" in text and FIX["issues"][0] in text
    assert f"Генерация {3 if limited else 2}: {'fix' if limited else 'accept'}" in text
    assert "I_max=3" in text
    if limited:
        assert "iteration_limit" in text and "insufficient_context: да" in text
    trace = test.dataframe[-1].value
    assert len(trace[(trace["Узел"] == "critique") & (trace["Переход"] == "generate")]) == (2 if limited else 1)
    # Switching to a later turn does not erase persisted feedback for the old run.
    old_run = service.inspect("ui").selected_run
    send(test, "Привет")
    next(x for x in test.selectbox if x.label == "История выполнения").select(old_run).run()
    assert "Генерация 1: fix" in rendered(test) and FIX["issues"][0] in rendered(test)
    audit = service.settings.agent_log_path.read_text()
    assert FIX["issues"][0] not in audit and "apt-get update" not in audit


@pytest.mark.parametrize("approved", [True, False])
def test_hitl_reopen_buttons_resume_and_rerun_do_not_repeat_effect(harness, monkeypatch, approved):
    service, llm, _ = harness
    calls = []
    original = AgentTools.send_report

    def spy(self, action, **kwargs):
        calls.append(action.action_id)
        return original(self, action, **kwargs)

    monkeypatch.setattr(AgentTools, "send_report", spy)
    test = app("hitl")
    send(test, "Отправь отчёт: Проверка лабораторной работы")
    pending = service.inspect("hitl").pending
    assert pending and test.chat_input[0].disabled
    assert not service.settings.agent_reports_db.exists() and not calls
    # Normal rerun and a completely new UI session preserve the true interrupt.
    test.run()
    test = app("hitl")
    assert service.inspect("hitl").pending["action_id"] == pending["action_id"]
    assert "Проверка лабораторной работы" in rendered(test)
    assert not service.settings.agent_reports_db.exists()
    button(test, "Подтвердить" if approved else "Отклонить").click().run()
    assert not test.exception
    expected = "action_completed" if approved else "action_rejected"
    assert expected in rendered(test) and not test.chat_input[0].disabled
    for _ in range(2):
        test.run()
    app("hitl")
    assert len(calls) == int(approved) and not llm.calls
    events = service.inspect("hitl").events
    assert any(e["event"] == "hitl_interrupt" for e in events)
    assert any(e["event"] == "hitl_resume" for e in events)
    assert sum(e.get("tool") == "send_report" for e in events) == int(approved)
    if approved:
        with sqlite3.connect(service.settings.agent_reports_db) as conn:
            assert conn.execute("SELECT count(*) FROM reports").fetchone()[0] == 1
    else:
        assert not service.settings.agent_reports_db.exists()


def test_thread_controls_and_isolation(harness):
    service, _, _ = harness
    test = app("alice")
    send(test, "Отправь отчёт: Только Alice")
    button(test, "Создать новый диалог").click().run()
    assert not test.exception and not test.chat_message and not test.chat_input[0].disabled
    next(x for x in test.text_input if x.label == "thread_id").set_value("bob")
    button(test, "Открыть диалог").click().run()
    assert not test.exception and not test.chat_message
    send(test, "Привет")
    assert len(service.inspect("bob").messages) == 2
    assert len(service.inspect("alice").messages) == 1 and service.inspect("alice").pending
    next(x for x in test.selectbox if x.label == "Сохранённые диалоги").select("alice").run()
    button(test, "Открыть выбранный").click().run()
    assert not test.exception and test.chat_input[0].disabled
    assert "Только Alice" in rendered(test) and "Здравствуйте!" not in rendered(test)


def test_stale_hitl_action_does_not_resolve_current_interrupt(harness):
    service, _, _ = harness
    first = service.ask("stale", "Отправь отчёт: первый")
    service.resume("stale", False, first.interrupt["action_id"])
    second = service.ask("stale", "Отправь отчёт: второй")
    with pytest.raises(ValueError, match="устарело"):
        service.resume("stale", True, first.interrupt["action_id"])
    assert service.inspect("stale").pending["action_id"] == second.interrupt["action_id"]
    assert not service.settings.agent_reports_db.exists()
    with pytest.raises(ValueError):
        service.ask("stale", "Привет")


def test_simultaneous_connections_are_rejected_and_closed(harness):
    service, _, _ = harness
    with service.opener(service.settings):
        with pytest.raises(AgentBusyError):
            service.ask("busy", "Привет")
    assert service.ask("busy", "Привет").result.status == "smalltalk"


def test_unfinished_run_requires_explicit_continue_in_ui(harness):
    service, _, _ = harness
    request = service.ask("crash", "Отправь отчёт: восстановление")

    def crash(event):
        if event["node"] == "call_side_effect" and event["event"] == "node":
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError):
        service.resume("crash", True, request.interrupt["action_id"], crash)
    before = service.settings.agent_log_path.read_text()
    test = app("crash")
    assert test.chat_input[0].disabled
    test.run()
    assert service.settings.agent_log_path.read_text() == before
    button(test, "Продолжить незавершённый запуск").click().run()
    assert not test.exception and "action_completed" in rendered(test)
    with sqlite3.connect(service.settings.agent_reports_db) as conn:
        assert conn.execute("SELECT count(*) FROM reports").fetchone()[0] == 1


def test_missing_audit_does_not_invent_trace(harness):
    service, _, _ = harness
    service.ask("missing", "Привет")
    service.settings.agent_log_path.unlink()
    test = app("missing")
    assert "Событий аудита этого запуска нет" in rendered(test)
    assert not service.inspect("missing").events


def test_invalid_thread_input_is_recoverable(harness):
    test = app()
    next(x for x in test.text_input if x.label == "thread_id").set_value("../../bad")
    button(test, "Открыть диалог").click().run()
    assert not test.exception and "thread_id: 1–64" in rendered(test)
    send(test, "Привет")
    assert "smalltalk" in rendered(test)


@pytest.mark.parametrize("exception", [RuntimeError("secret-token"), ValueError("secret-token")])
def test_execution_error_is_safe_and_does_not_retry_on_rerun(harness, monkeypatch, exception):
    service, _, _ = harness
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise exception

    monkeypatch.setattr(service, "ask", fail)
    test = app()
    send(test, "Привет")
    assert test.error and "secret-token" not in rendered(test)
    test.run()
    assert len(calls) == 1 and not test.chat_message
