"""Sixth tab of app.py. All durable state and execution belong to LangGraph."""
from threading import RLock
from uuid import uuid4

from langchain_core.messages import HumanMessage
import streamlit as st

from ..embeddings import SentenceTransformerEmbeddings
from ..error_messages import describe_error
from ..rag import RAGPipeline
from ..reranker import Reranker
from ..vector_store import VectorStore
from .runtime import AgentBusyError, GraphAgent, LazyLLM, open_agent
from .schemas import AgentSettings, I_MAX
from .web_service import AgentWebService, Dialogue, result_message

CONFIDENCE = {"low": "низкая", "medium": "средняя", "high": "высокая"}
STATUSES = {
    "answered": "Ответ принят", "insufficient_context": "Недостаточно контекста",
    "iteration_limit": "Достигнут предел генераций", "smalltalk": "Приветствие",
    "time": "Время получено", "refused": "Запрос вне области агента",
    "action_completed": "Действие выполнено", "action_rejected": "Действие отклонено",
    "error": "Ошибка выполнения",
}


class LockedModel:
    """Only model inference is cached; SQLite, pipelines and LLMs are per call."""
    def __init__(self, model) -> None:
        self.model, self.lock = model, RLock()

    def encode_documents(self, texts):
        with self.lock:
            return self.model.encode_documents(texts)

    def encode_query(self, text):
        with self.lock:
            return self.model.encode_query(text)

    def rerank(self, query, results, k):
        with self.lock:
            return self.model.rerank(query, results, k)


@st.cache_resource(show_spinner="Загрузка embedding-модели…")
def cached_embeddings(name: str):
    return LockedModel(SentenceTransformerEmbeddings(name))


@st.cache_resource(show_spinner="Загрузка reranker…")
def cached_reranker(name: str, enabled: bool):
    return LockedModel(Reranker(name, enabled))


def make_service() -> AgentWebService:
    settings = AgentSettings()

    def opener(s, observer=None):
        backend = LazyLLM(s)

        def pipeline():
            store = VectorStore(s.storage_dir, s.collection_name, cached_embeddings(s.embedding_model))
            return RAGPipeline(store, backend, s.prompts_dir, s.top_n, s.final_k,
                               s.min_dense_score, s.rrf_k,
                               cached_reranker(s.rerank_model, s.rerank_enabled))

        return open_agent(s, llm=backend, pipeline_factory=pipeline, observer=observer)

    return AgentWebService(settings, opener=opener)


def select_thread(thread_id: str) -> None:
    GraphAgent.config(thread_id)
    st.session_state["agent_thread"] = thread_id
    st.query_params["agent_thread"] = thread_id


def thread_controls(service: AgentWebService) -> str:
    if "agent_thread" not in st.session_state:
        try:
            select_thread(st.query_params.get("agent_thread") or uuid4().hex)
        except ValueError:
            st.warning("Некорректный thread_id в ссылке. Создан новый диалог.")
            select_thread(uuid4().hex)
    thread_id = st.session_state["agent_thread"]
    st.caption(f"Текущий thread_id: {thread_id}")
    with st.expander("Диалоги и память"):
        if st.button("Создать новый диалог", key="agent_new"):
            select_thread(uuid4().hex)
            st.rerun()
        with st.form("agent_open_form"):
            candidate = st.text_input("thread_id", value=thread_id, key="agent_open_id_" + thread_id, max_chars=64)
            if st.form_submit_button("Открыть диалог"):
                try:
                    select_thread(candidate.strip())
                except ValueError:
                    st.error("thread_id: 1–64 символа A–Z, a–z, 0–9, _ или -.")
                else:
                    st.rerun()
        threads = service.threads()
        saved = st.selectbox("Сохранённые диалоги", [None, *threads],
                             format_func=lambda value: value or "Выберите диалог", key="agent_saved")
        if st.button("Открыть выбранный", disabled=saved is None, key="agent_choose"):
            select_thread(saved)
            st.rerun()
        st.caption("История и ожидающие подтверждения хранятся в SQLite. Ссылка с agent_thread восстанавливает выбор после обновления страницы.")
    return thread_id


def show_result(result) -> None:
    if result.insufficient_context:
        st.warning(result.answer)
    else:
        st.write(result.answer)
    st.caption(f"Статус: {STATUSES.get(result.status, result.status)} ({result.status}) | "
               f"confidence: {CONFIDENCE.get(result.confidence, result.confidence)} | "
               f"insufficient_context: {'да' if result.insufficient_context else 'нет'}")
    if result.citations:
        st.markdown("**Источники ответа**")
        for citation in result.citations:
            st.text(f"source: {citation.source} · chunk_id: {citation.chunk_id}")


def show_history(view: Dialogue) -> None:
    with st.container(height=420, border=True):
        if not view.messages:
            st.info("Задайте вопрос по корпусу ALT Linux, поздоровайтесь или запросите время.")
        for message in view.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                result = result_message(message)
                if result:
                    with st.chat_message("assistant"):
                        show_result(result)


def execution_panels(view: Dialogue) -> None:
    with st.expander("Выполнение графа", expanded=True):
        state = view.execution
        final = state.get("final") or {}
        status = final.get("status") or ("interrupted" if view.pending and view.selected_run == view.values.get("run_id")
                                        else "unfinished" if view.selected_run else "idle")
        st.write(f"Маршрут: {state.get('route', '—')} · Генераций: {state.get('iterations', 0)}/{I_MAX} · I_max={I_MAX}")
        st.write(f"Финальный статус: {status}")
        if view.trace_truncated:
            st.warning("Показаны последние 500 событий запуска; начало трассы обрезано.")
        if not view.events:
            st.info("Событий аудита этого запуска нет. Путь выполнения не восстанавливается по ответу.")
        # Every row is an observed event. Keep tool calls and pauses distinct
        # from completed node events, including a replayed approve after resume.
        rows = []
        for number, event in enumerate(view.events, 1):
            rows.append({"№": number, "Узел": event["node"], "Событие": event["event"],
                         "Переход": event.get("transition", ""), "Инструмент": event.get("tool", ""),
                         "Critique": event.get("decision", ""), "Генерация": event["iterations"],
                         "Статус": event.get("status", "")})
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        calls = [event["tool"] for event in view.events if event.get("event") == "tool_call"]
        st.caption("Вызванные инструменты: " + (", ".join(calls) or "нет"))
    with st.expander("Проверка ответа (Critique)", expanded=True):
        critiques = view.execution.get("critique_history", [])
        for critique in critiques:
            st.markdown(f"**Генерация {critique['iteration']}: {critique['decision']}**")
            st.caption(f"Замечания: {'есть' if critique['issues'] else 'нет'} · "
                       f"grounded={critique['grounded']} · relevant={critique['relevant']}")
            for issue in critique["issues"]:
                st.text(issue)
        if not critiques:
            st.caption("Критик не выполнялся в этом запуске. Для старых checkpoints история отдельных итераций может отсутствовать.")
        if view.execution.get("final"):
            status = view.execution["final"]["status"]
            st.write("Итог проверки: " + STATUSES.get(status, status))
    with st.expander("Статическая схема LangGraph"):
        st.caption("Схема получена из скомпилированного графа. Пунктир — условные переходы.")
        st.graphviz_chart(view.diagram, use_container_width=True)
        st.download_button("Скачать Mermaid", view.mermaid, "agent-graph.mermaid", key="agent_mermaid")


def perform(command) -> None:
    try:
        with st.status("Агент выполняет запрос…", expanded=True) as status:
            def observer(event):
                status.update(label=f"Агент: {event['node']} · генерация {event['iterations']}/{I_MAX}")
            outcome = command(observer)
            status.update(label="Ожидается подтверждение" if outcome.status == "interrupted" else "Выполнение завершено",
                          state="complete")
    except AgentBusyError:
        st.warning("Агент занят другим запросом. Повторите операцию после его завершения.")
    except ValueError:
        st.error("Операция отклонена: проверьте вопрос (1–8000 символов) и актуальное состояние диалога. Обновите страницу перед повтором.")
    except Exception as exc:
        st.error("Не удалось завершить запрос. Состояние сохранено в SQLite. " + describe_error(exc))
    else:
        st.rerun()


def render_agent_tab() -> None:
    st.subheader("6. AI-агент (LangGraph)")
    st.caption("Настройки агента берутся из .env; параметры эксперимента ЛР1 в боковой панели относятся к первым пяти вкладкам.")
    try:
        service = make_service()
        thread_id = thread_controls(service)
        view = service.inspect(thread_id)
    except AgentBusyError:
        st.warning("Агент занят другим запросом. Обновите страницу после его завершения.")
        return
    except Exception:
        st.error("Не удалось открыть состояние агента. Проверьте настройки AGENT_* и доступ к SQLite.")
        return
    status = "Ожидается подтверждение HITL" if view.pending else (
        "Запуск не завершён" if view.next_nodes else "Готов к запросу")
    st.info("Текущий статус агента: " + status)
    show_history(view)
    if view.pending:
        pending = view.pending
        st.warning("Действие ожидает вашего решения. Новый запрос в этом диалоге заблокирован.")
        st.markdown("**Подтверждение действия (HITL)**")
        st.text(f"Инструмент: {pending['tool']}\naction_id: {pending['action_id']}")
        st.caption("Локальная имитация отправки: запись в SQLite, без сетевой отправки.")
        st.text(pending.get("arguments", {}).get("body", ""))
        approve, reject = st.columns(2)
        with approve:
            if st.button("Подтвердить", type="primary", key="agent_yes_" + pending["action_id"]):
                perform(lambda observer: service.resume(thread_id, True, pending["action_id"], observer))
        with reject:
            if st.button("Отклонить", key="agent_no_" + pending["action_id"]):
                perform(lambda observer: service.resume(thread_id, False, pending["action_id"], observer))
    elif view.next_nodes:
        st.warning("Обнаружен незавершённый запуск. Продолжите его перед новым вопросом.")
        if st.button("Продолжить незавершённый запуск", key="agent_continue"):
            perform(lambda observer: service.continue_run(thread_id, observer))
    question = st.chat_input("Вопрос агенту или «Отправь отчёт: текст»", key="agent_question",
                             disabled=bool(view.next_nodes), max_chars=8000)
    if question:
        perform(lambda observer: service.ask(thread_id, question, observer))
    if view.runs:
        selected = st.selectbox("История выполнения", list(reversed(view.runs)),
                                key="agent_run_" + thread_id + "_" + view.values.get("run_id", ""))
        if selected != view.selected_run:
            try:
                view = service.inspect(thread_id, selected)
            except AgentBusyError:
                st.warning("Агент занят. История выполнения будет доступна после завершения запроса.")
                return
    execution_panels(view)
