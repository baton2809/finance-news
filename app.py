"""
Streamlit Web Interface for Financial Literacy RAG System
Modern UI Design
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
    page_title="Finance RAG Assistant",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Modern CSS ----------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        font-weight: 400;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }

    /* Chat Container */
    .chat-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    /* Question Input Styling */
    .question-box {
        background: #f8f9fc;
        border: 2px solid #e8eaf6;
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }

    .question-box:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    }

    /* Answer Box */
    .answer-container {
        background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 100%);
        border-left: 4px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem;
        margin: 1.5rem 0;
        animation: fadeIn 0.5s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .answer-label {
        color: #667eea;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
    }

    .answer-text {
        color: #1a1a2e;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    /* Metrics Cards */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f5;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    .metric-card.primary {
        border-top: 3px solid #667eea;
    }

    .metric-card.secondary {
        border-top: 3px solid #a8a8a8;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0.5rem 0;
    }

    .metric-bar {
        height: 6px;
        background: #e8eaf6;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 0.75rem;
    }

    .metric-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    .metric-bar-fill.primary {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    .metric-bar-fill.secondary {
        background: linear-gradient(90deg, #9ca3af 0%, #6b7280 100%);
    }

    /* Context Box */
    .context-container {
        background: #fafbfc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #4b5563;
        line-height: 1.6;
        max-height: 300px;
        overflow-y: auto;
    }

    /* Example Questions */
    .example-btn {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #374151;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: left;
        width: 100%;
    }

    .example-btn:hover {
        border-color: #667eea;
        color: #667eea;
        background: #f5f7ff;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .status-badge.success {
        background: #ecfdf5;
        color: #059669;
    }

    .status-badge.error {
        background: #fef2f2;
        color: #dc2626;
    }

    .status-badge.warning {
        background: #fffbeb;
        color: #d97706;
    }

    .status-badge.info {
        background: #eff6ff;
        color: #2563eb;
    }

    /* Sidebar Styling */
    .sidebar-section {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #f0f0f5;
    }

    .sidebar-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* History Items */
    .history-item {
        background: #f8f9fc;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid #667eea;
    }

    .history-question {
        font-weight: 500;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }

    .history-meta {
        font-size: 0.8rem;
        color: #6b7280;
    }

    /* Loading Animation */
    .loading-pulse {
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #a1a1a1;
    }

    /* Button Override */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
    }

    .section-icon {
        font-size: 1.2rem;
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
    if 'references_dict' not in st.session_state:
        st.session_state.references_dict = {}


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

        # Load reference answers for enhanced RAGAS metrics
        references_dict = load_reference_answers()

        return df, index, metadata, references_dict, None
    except FileNotFoundError as e:
        return None, None, None, None, f"File not found: {str(e)}"
    except Exception as e:
        return None, None, None, None, f"System loading error: {str(e)}"


def load_reference_answers() -> Dict[str, str]:
    """Load reference answers from CSV for enhanced RAGAS metrics"""
    try:
        # Try to load the reference file
        if os.path.exists("./questions_with_references.csv"):
            ref_df = pd.read_csv("./questions_with_references.csv")
            # Create a dictionary mapping questions to reference answers
            ref_dict = dict(zip(ref_df["Вопрос"], ref_df["Референсный ответ"]))
            logger.info(f"✅ Loaded {len(ref_dict)} reference answers for enhanced RAGAS metrics")
            return ref_dict
        else:
            logger.warning("⚠️  No reference file found - using basic RAGAS metrics only")
            return {}
    except Exception as e:
        logger.error(f"Failed to load references: {e}")
        return {}


def check_api_key() -> Tuple[bool, str]:
    """Check if API key is configured"""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return False, "API key not configured. Add LLM_API_KEY to .env file."
    if api_key == "sk-your-deepseek-api-key-here":
        return False, "Using example API key. Replace with your real DeepSeek key."
    return True, "API key configured"


# ---------------------- RAG Query ----------------------
def run_query(question: str, mode: str, df: pd.DataFrame, index, metadata, references_dict: Dict[str, str], top_k: int, final_k: int) -> Dict[str, Any]:
    """Execute RAG query and return results"""
    result = {
        "question": question,
        "answer": None,
        "context": None,
        "metrics": None,
        "error": None,
        "time": 0,
        "has_reference": False
    }

    start_time = time.time()

    try:
        # Validate input
        if not question or len(question.strip()) < 3:
            result["error"] = "Question too short. Enter at least 3 characters."
            return result

        # 1. Retrieve
        hits = retrieve_faiss(question, index, metadata, top_k)

        if not hits:
            result["error"] = "No relevant documents found in knowledge base."
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

        # 5. Find reference answer if available
        reference = references_dict.get(question.strip())
        if reference:
            result["has_reference"] = True
            logger.info(f"✅ Found reference answer - will use full RAGAS metrics")
        else:
            logger.info(f"⚠️  No reference answer - using basic RAGAS metrics")

        # 6. Evaluate with reference if available
        metrics = evaluate_answer(question, answer, context, reference=reference)
        result["metrics"] = metrics

        result["time"] = time.time() - start_time

    except Exception as e:
        result["error"] = f"Query processing error: {str(e)}"
        logger.error(f"Query error: {e}")

    return result


# ---------------------- UI Components ----------------------
def render_hero():
    """Render hero header"""
    st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">Финансовый RAG Ассистент</h1>
            <p class="hero-subtitle">ИИ-ответы на ваши вопросы о финансовой грамотности</p>
            <span class="hero-badge">разработано командой A3R</span>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-title">Настройки</div>
            </div>
        """, unsafe_allow_html=True)

        # Mode selection
        mode = st.selectbox(
            "Режим RAG",
            options=["v1", "v2", "v3"],
            index=1,
            help="v1: Уровень документа, v2: Фрагменты, v3: Фрагменты + переранжирование"
        )

        mode_info = {
            "v1": ("Быстрый", "Поиск на уровне документа"),
            "v2": ("Сбалансированный", "Поиск по фрагментам (рекомендуется)"),
            "v3": ("Качественный", "Фрагменты + LLM переранжирование")
        }

        badge_type = "info" if mode == "v1" else "success" if mode == "v2" else "warning"
        st.markdown(f"""
            <div class="status-badge {badge_type}">
                <strong>{mode_info[mode][0]}</strong> - {mode_info[mode][1]}
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Advanced settings
        with st.expander("Расширенные настройки", expanded=False):
            top_k = st.slider("Top-K результатов", 3, 30, 20 if mode == "v3" else 5)
            final_k = st.slider("Финальные фрагменты", 1, 5, 3)
            show_context = st.checkbox("Показать контекст", value=True)
            show_metrics = st.checkbox("Показать метрики", value=True)

        st.markdown("---")

        # System status
        st.markdown("""
            <div class="sidebar-title">Статус системы</div>
        """, unsafe_allow_html=True)

        api_ok, api_msg = check_api_key()
        if api_ok:
            st.markdown('<div class="status-badge success">API подключен</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-badge error">Ошибка API</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.system_ready:
            st.markdown('<div class="status-badge success">Система готова</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge warning">Загрузка...</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Help section
        with st.expander("Помощь", expanded=False):
            st.markdown("""
            **Как использовать:**
            1. Введите ваш финансовый вопрос
            2. Нажмите "Спросить" или Enter
            3. Получите ответ от ИИ с метриками качества

            **Примеры тем:**
            - Инфляция и сбережения
            - Основы инвестиций
            - Налоговые вопросы
            - Банковское дело и кредиты
            """)

        return mode, top_k, final_k, show_context, show_metrics


def render_metrics(metrics: Dict[str, Any]):
    """Render metrics in modern card format"""
    st.markdown('<div class="section-header"><span class="section-icon">📊</span><span class="section-title">Метрики качества</span></div>', unsafe_allow_html=True)

    # Check for RAGAS errors
    ragas_error = metrics.get('ragas_error')
    if ragas_error:
        st.markdown(f"""
            <div class="status-badge error" style="margin-bottom: 1rem; width: 100%;">
                ⚠️ RAGAS Evaluation Error: {ragas_error}
            </div>
        """, unsafe_allow_html=True)
        st.info("RAGAS metrics показывают 0% из-за ошибки. Проверьте логи контейнера для деталей. LLM-оценки работают корректно.")

    # Check if we have context metrics (indicates references are available)
    has_context_metrics = 'ragas_context_precision' in metrics and 'ragas_context_recall' in metrics

    # RAGas metrics (Primary)
    if has_context_metrics:
        st.markdown("**Основные метрики RAGAS** (полная оценка с референсным ответом)")
    else:
        st.markdown("**Основные метрики RAGAS** (базовые метрики)")
        st.info("💡 Вы видите только Faithfulness и Answer Relevancy. Для получения всех 4 RAGAS метрик (+ Context Precision & Recall) выберите вопрос с референсным ответом выше.")

    if has_context_metrics:
        # Display 4 metrics in 2x2 grid
        cols = st.columns(2)
        ragas_metrics = [
            ("Достоверность (Faithfulness)", metrics.get('ragas_faithfulness', 0), "primary", "Соответствует ли ответ контексту?"),
            ("Релевантность (Answer Relevancy)", metrics.get('ragas_answer_relevancy', 0), "primary", "Насколько ответ релевантен вопросу?"),
            ("Точность контекста (Context Precision)", metrics.get('ragas_context_precision', 0), "primary", "Точен ли извлеченный контекст?"),
            ("Полнота контекста (Context Recall)", metrics.get('ragas_context_recall', 0), "primary", "Вся ли нужная информация извлечена?"),
        ]

        for i, (label, value, card_type, tooltip) in enumerate(ragas_metrics):
            with cols[i % 2]:
                pct = value * 100
                st.markdown(f"""
                    <div class="metric-card {card_type}" title="{tooltip}">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{pct:.1f}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill {card_type}" style="width: {pct}%"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        # Display 2 metrics in 1 row
        cols = st.columns(2)
        ragas_metrics = [
            ("Достоверность (Faithfulness)", metrics.get('ragas_faithfulness', 0), "primary", "Соответствует ли ответ контексту?"),
            ("Релевантность (Answer Relevancy)", metrics.get('ragas_answer_relevancy', 0), "primary", "Насколько ответ релевантен вопросу?"),
        ]

        for i, (label, value, card_type, tooltip) in enumerate(ragas_metrics):
            with cols[i]:
                pct = value * 100
                st.markdown(f"""
                    <div class="metric-card {card_type}" title="{tooltip}">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{pct:.1f}%</div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill {card_type}" style="width: {pct}%"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # LLM-as-Judge metrics (Secondary)
    st.markdown("**Дополнительные метрики (LLM-as-Judge)**")
    cols2 = st.columns(2)

    llm_metrics = [
        ("Релевантность", metrics.get('llm_relevance', 0), 5, "secondary", "LLM оценка релевантности (1-5)"),
        ("Достоверность", metrics.get('llm_faithfulness', 0), 5, "secondary", "LLM оценка достоверности (1-5)"),
    ]

    for i, (label, value, max_val, card_type, tooltip) in enumerate(llm_metrics):
        with cols2[i]:
            pct = (value / max_val) * 100 if max_val > 0 else 0
            st.markdown(f"""
                <div class="metric-card {card_type}" title="{tooltip}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}/{max_val}</div>
                    <div class="metric-bar">
                        <div class="metric-bar-fill {card_type}" style="width: {pct}%"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)


def render_history():
    """Render query history"""
    if st.session_state.history:
        with st.expander(f"История запросов ({len(st.session_state.history)} элементов)", expanded=False):
            for i, item in enumerate(reversed(st.session_state.history[-10:])):
                idx = len(st.session_state.history) - i
                st.markdown(f"""
                    <div class="history-item">
                        <div class="history-question">{idx}. {item['question'][:60]}{'...' if len(item['question']) > 60 else ''}</div>
                        <div class="history-meta">Время: {item['time']:.2f}с | Оценка: {item.get('relevance', 'Н/Д')}</div>
                    </div>
                """, unsafe_allow_html=True)


# ---------------------- Main App ----------------------
def main():
    """Main application entry point"""
    init_session_state()

    # Sidebar
    mode, top_k, final_k, show_context, show_metrics = render_sidebar()

    # Hero Header
    render_hero()

    # Load system
    if not st.session_state.system_ready:
        with st.spinner("Загрузка ИИ системы... (первый запуск может занять несколько минут)"):
            df, index, metadata, references_dict, error = load_system(mode)

            if error:
                st.markdown(f'<div class="status-badge error" style="padding: 1rem; width: 100%;">{error}</div>', unsafe_allow_html=True)
                st.stop()

            st.session_state.df = df
            st.session_state.index = index
            st.session_state.metadata = metadata
            st.session_state.references_dict = references_dict
            st.session_state.system_ready = True
            st.rerun()

    # Check API key
    api_ok, api_msg = check_api_key()
    if not api_ok:
        st.markdown(f'<div class="status-badge error" style="padding: 1rem; width: 100%;">{api_msg}</div>', unsafe_allow_html=True)
        st.info("Добавьте свой DeepSeek API ключ в файл .env и перезапустите приложение.")
        st.stop()

    # Main input area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="section-icon">💬</span><span class="section-title">Задайте вопрос</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Question",
            placeholder="Например: Что такое инфляция и как она влияет на мои сбережения?",
            label_visibility="collapsed"
        )
    with col2:
        submit_btn = st.button("Спросить", type="primary", use_container_width=True)

    # Example questions - ONLY show questions with reference answers
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Примеры вопросов с референсными ответами (все 4 RAGAS метрики):**")

    # Show indicator if we have references loaded
    if st.session_state.references_dict:
        st.markdown(f'<div class="status-badge success" style="margin-bottom: 0.5rem;">✨ Загружено {len(st.session_state.references_dict)} вопросов с референсами</div>', unsafe_allow_html=True)

        # Get first 4 questions from references_dict
        examples = list(st.session_state.references_dict.keys())[:4]
    else:
        st.warning("⚠️ Референсные ответы не загружены. Только базовые метрики будут доступны.")
        examples = []

    if examples:
        # Display questions in expandable sections so full text is visible
        for i, example in enumerate(examples):
            # Create a container for each question
            col1, col2 = st.columns([5, 1])
            with col1:
                # Show full question text (no truncation)
                st.markdown(f'<div style="background: #f8f9fc; padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid #667eea;">'
                           f'<strong>🌟 Вопрос {i+1}:</strong> {example}</div>', unsafe_allow_html=True)
            with col2:
                if st.button("Выбрать", key=f"example_{i}", use_container_width=True):
                    question = example
                    submit_btn = True

    st.markdown('</div>', unsafe_allow_html=True)

    # Process query
    if submit_btn and question:
        # Create a placeholder for loading message that can be cleared
        loading_placeholder = st.empty()
        loading_placeholder.markdown('<div class="loading-pulse" style="text-align: center; padding: 2rem;">Обработка вашего вопроса...</div>', unsafe_allow_html=True)

        result = run_query(
            question=question,
            mode=mode,
            df=st.session_state.df,
            index=st.session_state.index,
            metadata=st.session_state.metadata,
            references_dict=st.session_state.references_dict,
            top_k=top_k,
            final_k=final_k
        )

        # Clear the loading message after query completes
        loading_placeholder.empty()

        if result["error"]:
            st.markdown(f'<div class="status-badge error" style="padding: 1rem; width: 100%;">{result["error"]}</div>', unsafe_allow_html=True)
        else:
            # Success badge with reference indicator
            ref_badge = ""
            if result.get("has_reference"):
                ref_badge = ' <span style="background: #fbbf24; color: #78350f; padding: 0.25rem 0.5rem; border-radius: 6px; font-size: 0.75rem; margin-left: 0.5rem;">✨ С референсом - все 4 RAGAS метрики</span>'
            st.markdown(f'<div class="status-badge success" style="margin-bottom: 1rem;">Ответ сгенерирован за {result["time"]:.2f} секунд{ref_badge}</div>', unsafe_allow_html=True)

            # Show the question before the answer
            st.markdown(f"""
                <div style="background: #f0f4ff; border-left: 4px solid #4f46e5; border-radius: 0 12px 12px 0; padding: 1rem 1.5rem; margin-bottom: 1rem;">
                    <div style="color: #4f46e5; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">Ваш вопрос</div>
                    <div style="color: #1a1a2e; font-size: 1.05rem;">{question}</div>
                </div>
            """, unsafe_allow_html=True)

            # Answer - convert markdown bold to HTML bold for proper rendering
            answer_html = result["answer"].replace("**", "<strong>", 1)
            while "**" in answer_html:
                answer_html = answer_html.replace("**", "</strong>", 1)
                if "**" in answer_html:
                    answer_html = answer_html.replace("**", "<strong>", 1)

            st.markdown(f"""
                <div class="answer-container">
                    <div class="answer-label">Ответ ИИ</div>
                    <div class="answer-text">{answer_html}</div>
                </div>
            """, unsafe_allow_html=True)

            # Metrics
            if show_metrics and result["metrics"]:
                render_metrics(result["metrics"])

            # Context
            if show_context and result["context"]:
                with st.expander("Показать источники", expanded=False):
                    st.markdown(f'<div class="context-container">{result["context"][:2000]}{"..." if len(result["context"]) > 2000 else ""}</div>', unsafe_allow_html=True)

            # Add to history
            st.session_state.history.append({
                "question": question,
                "answer": result["answer"],
                "time": result["time"],
                "relevance": result["metrics"].get("ragas_answer_relevancy", 0) if result["metrics"] else 0
            })

    # History
    render_history()

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #6b7280; font-size: 0.85rem;">
            Финансовый RAG Ассистент | multilingual-e5-small + FAISS + DeepSeek + RAGas
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
