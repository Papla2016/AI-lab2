"""Streamlit laboratory interface for the grounded RAG pipeline."""
import csv
from io import StringIO
import json

import streamlit as st

from src.config import ROOT, Settings
from src.error_messages import describe_error
from src.schemas import SearchResult
from src.web_service import RAGLabService
from src.agent.streamlit_ui import render_agent_tab

PROMPTS = {
    "Zero-shot": "zero_shot",
    "Few-shot": "few_shot",
    "Evidence selection + structured": "cot_structured",
}
SEARCH_MODES = {"Dense (embeddings)": "dense", "Hybrid (dense + BM25/RRF)": "hybrid"}
CONFIDENCE = {"low": "низкая", "medium": "средняя", "high": "высокая"}


def service_for(settings: Settings) -> RAGLabService:
    if st.session_state.get("lab_settings") != settings:
        st.session_state["lab_service"] = RAGLabService(settings)
        st.session_state["lab_settings"] = settings
    return st.session_state["lab_service"]


def chunk_dict(item: SearchResult) -> dict:
    return {
        "source": item.metadata.get("source", "?"),
        "page": item.metadata.get("page"),
        "chunk_id": item.metadata.get("chunk_id", "?"),
        "score": round(item.score, 5),
        "text": item.text,
    }


def show_chunks(items: list[SearchResult], heading: str) -> None:
    st.markdown(f"**{heading} ({len(items)})**")
    if not items:
        st.info("Фрагменты не найдены.")
        return
    for i, item in enumerate(items, 1):
        data = chunk_dict(item)
        page = f", страница {data['page']}" if data["page"] else ""
        with st.expander(f"{i}. {data['source']}{page} | {data['chunk_id']} | score={data['score']:.5f}"):
            st.text(data["text"])
    st.caption("Score зависит от этапа: cosine similarity, RRF или оценка reranker. Не сравнивайте эти шкалы напрямую.")


def show_answer(answer, trace: dict) -> None:
    if answer.insufficient_context:
        st.warning(answer.answer)
    else:
        st.markdown(answer.answer)
    st.caption(
        f"Уверенность: {CONFIDENCE.get(answer.confidence, answer.confidence)} | "
        f"Недостаточно контекста: {'да' if answer.insufficient_context else 'нет'} | "
        f"Маршрут: {trace.get('router', 'нет')}"
    )
    st.markdown("**Подтверждённые цитаты**")
    if not answer.citations:
        st.info("Цитат нет. При недостаточном контексте это ожидаемое поведение.")
    else:
        chunks = trace.get("after_rerank", [])
        for cite in answer.citations:
            match = next((x for x in chunks if x.metadata.get("source") == cite.source
                          and x.metadata.get("chunk_id") == cite.chunk_id), None)
            with st.expander(f"{cite.source} | {cite.chunk_id}", expanded=True):
                if match is not None:
                    st.caption(f"Страница: {match.metadata.get('page', 'не указана')}")
                    st.text(match.text)
                else:
                    st.error("Цитируемый чанк отсутствует в трассировке.")
    with st.expander("Структурированный ответ (Pydantic JSON)"):
        st.json(answer.model_dump())
    with st.expander("Диагностика: поиск, контекст, промпт", expanded=True):
        st.write("Решение маршрутизатора:", trace.get("router", "не выполнялся"))
        show_chunks(trace.get("before_rerank", []), "Кандидаты retrieval")
        show_chunks(trace.get("after_rerank", []), "Чанки после reranking (попали в контекст)")
        if "evidence_selection" in trace:
            st.markdown("**Предварительный выбор доказательств (CoT без длинных рассуждений)**")
            st.code(trace["evidence_selection"], language="json")
        st.markdown("**CONTEXT - точный текст, переданный для генерации**")
        st.code(trace.get("context", "Контекст не формировался: маршрутизатор отклонил запрос или релевантных чанков нет."),
                language=None)
        st.markdown("**Итоговый SYSTEM + USER prompt**")
        st.code(trace.get("final_prompt", "Генерация ответа не запускалась."), language=None)


def documents_tab(service: RAGLabService, settings: Settings) -> None:
    st.subheader("1. Корпус и индексация")
    st.write("Загрузите 5–10 файлов по одной теме (PDF, Markdown, TXT). После изменений запустите индексацию вручную.")
    uploaded = st.file_uploader(
        "Документы корпуса", type=["pdf", "md", "txt"], accept_multiple_files=True, key="lab_uploads"
    )
    if st.button("Сохранить выбранные файлы", disabled=not uploaded):
        for entry in uploaded or []:
            try:
                name = service.save_document(entry.name, entry.getvalue())
                st.success(f"Сохранено: {name}")
            except ValueError as exc:
                st.warning(str(exc))
            except OSError:
                st.error("Ошибка записи файла в data/. Проверьте права доступа.")
    names = service.files()
    st.write(f"Файлов в data/: **{len(names)}**")
    if names:
        with st.expander("Список файлов", expanded=True):
            for name in names:
                st.text(name)
    if len(names) < 5:
        st.warning("В задании требуется минимум 5–10 файлов одной тематики; добавьте документы перед сдачей.")
    st.caption(
        f"Чанкинг: рекурсивный, size={settings.chunk_size}, overlap={settings.chunk_overlap}; "
        f"embeddings: {settings.embedding_model}; векторное хранилище: Chroma."
    )
    if st.button("Переиндексировать корпус", type="primary"):
        try:
            with st.spinner("Чтение PDF и построение embeddings…"):
                result = service.index_documents()
            st.session_state["lab_index"] = result
            st.success(
                f"Проиндексировано файлов: {result.files}; частей документов: {result.document_parts}; "
                f"чанков: {result.chunks}."
            )
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Индексация не выполнена: {describe_error(exc)}")
    if st.session_state.get("lab_index"):
        summary = st.session_state["lab_index"]
        st.caption(f"Последняя индексация в этой сессии: {summary.files} файлов, {summary.chunks} чанков.")
    st.info("Индекс не обновляется автоматически после загрузки. Повторная индексация заменяет прежний корпус.")


def search_tab(service: RAGLabService, options: dict) -> None:
    st.subheader("2. Retrieval без генерации")
    st.write("Проверьте top-k, источники, идентификаторы и текст фрагментов до вызова LLM.")
    with st.form("search_form"):
        question = st.text_input("Поисковый запрос", key="lab_search_question")
        submit = st.form_submit_button("Найти чанки")
    if submit:
        try:
            with st.spinner("Выполняется поиск…"):
                before, after = service.search(question, options["mode"], options["top_n"],
                                               options["final_k"], options["rerank"])
            st.session_state["lab_search_results"] = (before, after)
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Поиск не выполнен: {describe_error(exc)}")
    result = st.session_state.get("lab_search_results")
    if result:
        before, after = result
        show_chunks(before, "Кандидаты поиска")
        show_chunks(after, "Чанки после опционального reranking")


def question_tab(service: RAGLabService, options: dict) -> None:
    st.subheader("3. Ответ строго по документам")
    st.write("Ответ должен содержать проверенные source/chunk_id или явный отказ при недостаточном контексте.")
    with st.form("ask_form"):
        question = st.text_area("Вопрос к корпусу", key="lab_ask_question", height=100)
        submit = st.form_submit_button("Выполнить RAG", type="primary")
    if submit:
        try:
            with st.spinner("Маршрутизация → поиск → генерация → Pydantic-валидация…"):
                answer, trace = service.ask(
                    question, options["mode"], options["strategy"], options["top_n"],
                    options["final_k"], options["rerank"], options["routing"],
                )
            st.session_state["lab_last_answer"] = (answer, trace)
        except ValueError as exc:
            st.error(f"Ошибка проверки результата: {exc}")
        except Exception as exc:
            st.error(f"RAG не выполнен: {describe_error(exc)}")
    result = st.session_state.get("lab_last_answer")
    if result:
        show_answer(*result)


def default_questions() -> str:
    path = ROOT / "evaluation" / "questions.json"
    try:
        questions = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join(item["question"] for item in questions[:5])
    except (OSError, ValueError, KeyError, TypeError):
        return ""


def comparison_tab(service: RAGLabService, options: dict) -> None:
    st.subheader("4. Сравнение retrieval и промптов")
    st.write(
        "Одинаковые 5–7 вопросов проверяются с dense и hybrid, а также с zero-shot, few-shot "
        "и CoT + structured output. Каждый результат получается реальным вызовом пайплайна."
    )
    st.caption("Примерные вопросы относятся к документации о пакетах. Замените их вопросами по своему корпусу.")
    questions_raw = st.text_area(
        "Вопросы (один в строке)", value=default_questions(), height=180, key="lab_eval_questions"
    )
    questions = [q.strip() for q in questions_raw.splitlines() if q.strip()]
    total = len(questions) * 6
    st.caption(f"Вопросов: {len(questions)} | комбинаций: {total} | режим маршрутизации: принудительный RAG для честного сравнения")
    if st.button("Запустить матрицу сравнения", type="primary"):
        if not 5 <= len(questions) <= 7:
            st.warning("Для сравнения требуется от 5 до 7 непустых вопросов.")
        else:
            rows: list[dict] = []
            progress = st.progress(0.0)
            for i, question in enumerate(questions):
                for mode in ("dense", "hybrid"):
                    for strategy in ("zero_shot", "few_shot", "cot_structured"):
                        row = {
                            "question_id": i + 1, "question": question,
                            "retrieval": mode, "prompt": strategy,
                            "answer": "", "citations": "", "confidence": "",
                            "insufficient_context": "", "retrieved_chunks": "", "error": "",
                        }
                        try:
                            answer, trace = service.ask(
                                question, mode, strategy, options["top_n"],
                                options["final_k"], options["rerank"], routing=False,
                            )
                            row.update(
                                answer=answer.answer,
                                citations="; ".join(f"{c.source}:{c.chunk_id}" for c in answer.citations),
                                confidence=answer.confidence,
                                insufficient_context=answer.insufficient_context,
                                retrieved_chunks="; ".join(
                                    f"{r.metadata['source']}:{r.metadata['chunk_id']}"
                                    for r in trace.get("after_rerank", [])
                                ),
                            )
                        except Exception as exc:
                            row["error"] = describe_error(exc)
                        rows.append(row)
                        progress.progress(len(rows) / total)
            st.session_state["lab_comparison_rows"] = rows
    rows = st.session_state.get("lab_comparison_rows")
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        export = StringIO()
        writer = csv.DictWriter(export, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            "Скачать результаты CSV для отчёта",
            export.getvalue().encode("utf-8-sig"),
            file_name="rag_comparison.csv", mime="text/csv",
        )
        st.caption("Отказы, цитаты и ошибки оценивайте по реальным ответам. Этот экран не объявляет галлюцинации автоматически.")


def safety_tab(service: RAGLabService, options: dict) -> None:
    st.subheader("5. Демонстрация антигаллюцинаций")
    st.write(
        "Выберите вопрос, ответа на который нет в документах. Сравните экспериментальный "
        "ответ LLM **без контекста** с RAG-ответом и проверенными цитатами."
    )
    question = st.text_input("Вопрос вне корпуса", value="Какова температура поверхности Венеры?",
                             key="lab_safety_question")
    if st.button("Сравнить без RAG и с RAG"):
        try:
            with st.spinner("Выполняется эксперимент…"):
                baseline = service.baseline(question)
                answer, trace = service.ask(
                    question, options["mode"], options["strategy"], options["top_n"],
                    options["final_k"], options["rerank"], routing=True,
                )
            st.session_state["lab_safety_result"] = (baseline, answer, trace)
        except Exception as exc:
            st.error(f"Эксперимент не выполнен: {describe_error(exc)}")
    result = st.session_state.get("lab_safety_result")
    if result:
        baseline, answer, trace = result
        st.markdown("**Без RAG - не подтверждено документами**")
        st.code(baseline, language=None)
        st.markdown("**С RAG - проверенный структурированный ответ**")
        show_answer(answer, trace)
    st.caption(
        "Результат без RAG не обязательно содержит ошибку: для отчёта отдельно оцените, "
        "является ли утверждение выдуманным. Чанки отделены от system prompt маркерами CONTEXT."
    )


def main() -> None:
    st.set_page_config(page_title="ЛР 1–2 | RAG и LangGraph", page_icon="📚", layout="wide")
    st.title("RAG-лаборатория")
    st.caption("ЛР 1: документы → чанки → dense/hybrid retrieval → промпты → структурированный ответ с цитатами")
    st.caption("ЛР 2: LangGraph → инструменты → критик → SQLite-память → подтверждение действий")
    try:
        settings = Settings()
    except Exception:
        st.error("Ошибки конфигурации .env (например, некорректный CHUNK_OVERLAP).")
        return
    service = service_for(settings)
    with st.sidebar:
        st.header("Параметры эксперимента")
        retrieval_label = st.selectbox("Поиск", list(SEARCH_MODES), index=1 if settings.retrieval_mode == "hybrid" else 0)
        prompt_label = st.selectbox("Промпт", list(PROMPTS))
        top_n = int(st.number_input("Top-N кандидатов", min_value=1, max_value=100,
                                    value=min(settings.top_n, 100), step=1))
        final_k = int(st.number_input("Финальных чанков", min_value=1, max_value=top_n,
                                      value=min(settings.final_k, top_n), step=1))
        rerank = st.checkbox("Cross-encoder reranking", value=settings.rerank_enabled)
        routing = st.checkbox("Agentic RAG: LLM-маршрутизация", value=True,
                              help="При отключении принудительно выполняется RAG. В сравнении маршрутизация отключена.")
        st.divider()
        st.caption(f"LLM: {settings.llm_model}")
        if st.button("Проверить LLM API", help="Проверяет модель и JSON-режим без Chroma, embeddings и индекса."):
            try:
                with st.spinner("Проверка LLM API…"):
                    result = service.llm.generate(
                        'Верните строго JSON объект вида {"status":"ok"}.',
                        "Проверка подключения. Верните JSON со status=ok.",
                    )
                decoded = json.loads(result)
                if isinstance(decoded, dict) and decoded.get("status") == "ok":
                    st.success("LLM отвечает и поддерживает используемый JSON-режим.")
                else:
                    st.warning("LLM ответила, но вернула неожиданный JSON.")
                    st.json(decoded)
            except Exception as exc:
                st.error(f"Проверка LLM API: {describe_error(exc)}")
        st.caption(f"Embeddings: {settings.embedding_model}")
        st.caption("Ключ API из .env не показывается.")
    options = {
        "mode": SEARCH_MODES[retrieval_label], "strategy": PROMPTS[prompt_label],
        "top_n": top_n, "final_k": final_k, "rerank": rerank, "routing": routing,
    }
    documents, search, question, comparison, safety, agent = st.tabs([
        "Документы", "Поиск", "RAG-ответ", "Сравнение", "Антигаллюцинации",
        "AI-агент (LangGraph)",
    ])
    with documents:
        documents_tab(service, settings)
    with search:
        search_tab(service, options)
    with question:
        question_tab(service, options)
    with comparison:
        comparison_tab(service, options)
    with safety:
        safety_tab(service, options)
    with agent:
        render_agent_tab()
    st.divider()
    st.caption("Приложение использует тот же RAGPipeline, что и python -m src.main. Негрунтированный чат не подменяет RAG.")


if __name__ == "__main__":
    main()
