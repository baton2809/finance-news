"""
Streamlit Web Interface for Financial Literacy RAG System
Веб-интерфейс для RAG системы финансовой грамотности
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Import RAG components from main
from main import (
    get_embedding_model,
    get_llm_client,
    load_knowledge_base,
    build_faiss_index,
    retrieve_faiss,
    llm_rerank,
    build_context,
    generate_answer,
    evaluate_answer,
    chunk_text,
    COST,
    logger
)

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="RAG Финансовая грамотность",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Custom CSS ----------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .context-box {
        background-color: #f5f5f5;
        border-left: 4px solid #9E9E9E;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .metrics-card {
        background-color: #fff;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #c62828;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #2e7d32;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------- Session State ----------------------
def init_session_state():
    """Initialize session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None


# ---------------------- System Loading ----------------------
@st.cache_resource
def load_system(mode: str = "v2"):
    """Load and cache the RAG system components"""
    try:
        # Load knowledge base
        df = load_knowledge_base("./train_data.csv")

        # Build/Load FAISS index
        index_mode = "v1" if mode == "v1" else "v2"
        index_file = f"faiss_index_{index_mode}_e5small.bin"
        meta_file = f"faiss_meta_{index_mode}_e5small.pkl"

        index, metadata = build_faiss_index(df, index_file, meta_file, mode=index_mode)

        return df, index, metadata, None
    except FileNotFoundError as e:
        return None, None, None, f"Файл не найден: {str(e)}"
    except Exception as e:
        return None, None, None, f"Ошибка загрузки системы: {str(e)}"


def check_api_key() -> Tuple[bool, str]:
    """Check if API key is configured"""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return False, "API ключ не настроен. Добавьте LLM_API_KEY в .env файл."
    if api_key == "sk-your-deepseek-api-key-here":
        return False, "Используется пример API ключа. Замените на реальный ключ DeepSeek."
    return True, "API ключ настроен"


# ---------------------- RAG Query ----------------------
def run_query(question: str, mode: str, df: pd.DataFrame, index, metadata, top_k: int, final_k: int) -> Dict[str, Any]:
    """Execute RAG query and return results"""
    result = {
        "question": question,
        "answer": None,
        "context": None,
        "metrics": None,
        "error": None,
        "time": 0
    }

    start_time = time.time()

    try:
        # Validate input
        if not question or len(question.strip()) < 3:
            result["error"] = "Вопрос слишком короткий. Введите не менее 3 символов."
            return result

        # 1. Retrieve
        hits = retrieve_faiss(question, index, metadata, top_k)

        if not hits:
            result["error"] = "Не найдено релевантных документов в базе знаний."
            return result

        # 2. Rerank (if v3)
        if mode == "v3":
            hits = llm_rerank(question, hits, df, final_k)
        else:
            hits = hits[:final_k]

        # 3. Build context
        context = build_context(hits, df)
        result["context"] = context

        # 4. Generate answer
        answer = generate_answer(question, context)
        result["answer"] = answer

        # 5. Evaluate
        metrics = evaluate_answer(question, answer, context)
        result["metrics"] = metrics

        result["time"] = time.time() - start_time

    except Exception as e:
        result["error"] = f"Ошибка обработки запроса: {str(e)}"
        logger.error(f"Query error: {e}")

    return result


# ---------------------- UI Components ----------------------
def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.markdown("## Настройки")

        # Mode selection
        mode = st.selectbox(
            "Режим RAG",
            options=["v1", "v2", "v3"],
            index=1,
            help="v1: Doc-level, v2: Chunks, v3: Chunks + Reranking"
        )

        mode_descriptions = {
            "v1": "Быстрый поиск по документам",
            "v2": "Поиск по чанкам (рекомендуется)",
            "v3": "Чанки + LLM переранжирование (лучшее качество)"
        }
        st.caption(mode_descriptions[mode])

        st.divider()

        # Advanced settings
        with st.expander("Расширенные настройки"):
            top_k = st.slider("Top-K результатов", 3, 30, 20 if mode == "v3" else 5)
            final_k = st.slider("Финальные фрагменты", 1, 5, 3)
            show_context = st.checkbox("Показывать контекст", value=True)
            show_metrics = st.checkbox("Показывать метрики", value=True)

        st.divider()

        # System status
        st.markdown("## Статус системы")

        api_ok, api_msg = check_api_key()
        if api_ok:
            st.success("API ключ настроен")
        else:
            st.error(api_msg)

        if st.session_state.system_ready:
            st.success("Система загружена")
        else:
            st.warning("Система загружается...")

        st.divider()

        # Help section
        with st.expander("Справка"):
            st.markdown("""
            **Как использовать:**
            1. Введите вопрос по финансовой грамотности
            2. Нажмите "Отправить" или Enter
            3. Получите ответ с метриками качества

            **Примеры вопросов:**
            - Что такое инфляция?
            - Как открыть брокерский счет?
            - Какие виды налогов существуют?
            """)

        return mode, top_k, final_k, show_context, show_metrics


def render_metrics(metrics: Dict[str, Any]):
    """Render metrics in a nice format"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Релевантность",
            f"{metrics.get('llm_relevance', 0)}/5",
            help="Насколько ответ соответствует вопросу (LLM оценка)"
        )

    with col2:
        st.metric(
            "Достоверность",
            f"{metrics.get('llm_faithfulness', 0)}/5",
            help="Основан ли ответ на контексте (LLM оценка)"
        )

    with col3:
        st.metric(
            "BERTScore",
            f"{metrics.get('bert_f1', 0):.2f}",
            help="Семантическое сходство с контекстом (0.0-1.0)"
        )


def render_history():
    """Render query history"""
    if st.session_state.history:
        with st.expander(f"История запросов ({len(st.session_state.history)})"):
            for i, item in enumerate(reversed(st.session_state.history[-10:])):
                st.markdown(f"**{len(st.session_state.history) - i}. {item['question'][:50]}...**")
                st.caption(f"Время: {item['time']:.2f}с | Релевантность: {item.get('relevance', 'N/A')}")
                st.divider()


# ---------------------- Main App ----------------------
def main():
    """Main application entry point"""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">RAG Финансовая грамотность</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Система ответов на вопросы с использованием RAG и LLM</div>', unsafe_allow_html=True)

    # Sidebar
    mode, top_k, final_k, show_context, show_metrics = render_sidebar()

    # Load system
    if not st.session_state.system_ready:
        with st.spinner("Загрузка системы... (первый запуск может занять несколько минут)"):
            df, index, metadata, error = load_system(mode)

            if error:
                st.markdown(f'<div class="error-box">{error}</div>', unsafe_allow_html=True)
                st.stop()

            st.session_state.df = df
            st.session_state.index = index
            st.session_state.metadata = metadata
            st.session_state.system_ready = True
            st.rerun()

    # Check API key
    api_ok, api_msg = check_api_key()
    if not api_ok:
        st.markdown(f'<div class="error-box">{api_msg}</div>', unsafe_allow_html=True)
        st.info("Добавьте ваш DeepSeek API ключ в файл .env и перезапустите приложение.")
        st.stop()

    # Main input area
    st.markdown("### Задайте вопрос")

    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Вопрос",
            placeholder="Например: Что такое инфляция и как она влияет на сбережения?",
            label_visibility="collapsed"
        )
    with col2:
        submit_btn = st.button("Отправить", type="primary", use_container_width=True)

    # Example questions
    st.markdown("**Примеры вопросов:**")
    example_cols = st.columns(3)
    examples = [
        "Что такое инфляция?",
        "Как открыть брокерский счет?",
        "Какие налоги платит ИП?"
    ]

    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                question = example
                submit_btn = True

    st.divider()

    # Process query
    if submit_btn and question:
        with st.spinner("Обрабатываю запрос..."):
            result = run_query(
                question=question,
                mode=mode,
                df=st.session_state.df,
                index=st.session_state.index,
                metadata=st.session_state.metadata,
                top_k=top_k,
                final_k=final_k
            )

        if result["error"]:
            st.markdown(f'<div class="error-box">{result["error"]}</div>', unsafe_allow_html=True)
        else:
            # Success message
            st.markdown(f'<div class="success-box">Ответ получен за {result["time"]:.2f} секунд</div>', unsafe_allow_html=True)

            # Answer
            st.markdown("### Ответ")
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

            # Metrics
            if show_metrics and result["metrics"]:
                st.markdown("### Метрики качества")
                render_metrics(result["metrics"])

            # Context
            if show_context and result["context"]:
                with st.expander("Показать контекст (источники)"):
                    st.markdown(f'<div class="context-box">{result["context"][:2000]}...</div>', unsafe_allow_html=True)

            # Add to history
            st.session_state.history.append({
                "question": question,
                "answer": result["answer"],
                "time": result["time"],
                "relevance": result["metrics"].get("llm_relevance", 0) if result["metrics"] else 0
            })

    # History
    render_history()

    # Footer
    st.divider()
    st.caption("RAG система для финансовой грамотности | Использует multilingual-e5-small + FAISS + DeepSeek")


if __name__ == "__main__":
    main()
