import streamlit as st
import pandas as pd
import json
from datetime import datetime
import sys
import os

# Добавляем путь к существующим модулям
sys.path.append('.')

# Настройки страницы
st.set_page_config(
    page_title="Финансовый RAG Чат",
    page_icon="💰",
    layout="wide"
)

# Инициализация состояния сессии
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_mode' not in st.session_state:
    st.session_state.rag_mode = "v3"
if 'show_details' not in st.session_state:
    st.session_state.show_details = False

# Боковая панель с настройками
with st.sidebar:
    st.title("⚙️ Настройки")
    
    # Выбор режима работы
    st.session_state.rag_mode = st.selectbox(
        "Режим RAG:",
        options=["v1", "v2", "v3"],
        index=2,
        help="v1: по документам, v2: по фрагментам, v3: v2 + переранжирование"
    )
    
    # Параметры поиска
    st.subheader("🔍 Параметры поиска")
    top_k = st.slider("Кандидатов для поиска:", 1, 20, 
                     value=20 if st.session_state.rag_mode == "v3" else 5)
    final_k = st.slider("Финальных фрагментов:", 1, 10, value=3)
    
    # Детали
    st.session_state.show_details = st.checkbox("Показать детали поиска", value=False)
    
    # Кнопка очистки истории
    if st.button("🗑️ Очистить историю", type="secondary", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # Статистика
    st.divider()
    st.markdown(f"**Режим:** {st.session_state.rag_mode.upper()}")
    st.markdown(f"**Сообщений:** {len(st.session_state.chat_history)//2}")

# Главная область - чат
st.title("💬 Финансовый RAG Чат")
st.markdown("Задавайте вопросы по финансовой грамотности")

# Контейнер для истории чата
chat_container = st.container()

# Показываем историю чата
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                
                # Показываем детали если есть и включено
                if st.session_state.show_details and "details" in message:
                    with st.expander("🔍 Детали поиска"):
                        details = message["details"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Релевантность", f"{details['metrics']['relevance']}/5")
                        with col2:
                            st.metric("Точность", f"{details['metrics']['faithfulness']}/5")
                        
                        st.markdown("**Найденные фрагменты:**")
                        for i, (doc_info, score) in enumerate(details["hits"], 1):
                            st.markdown(f"**{i}. Док {doc_info[0]}** (сходство: {score:.3f})")

# Поле ввода внизу
with st.container():
    st.markdown("---")
    
    # Два варианта ввода: через форму или chat_input
    question = st.chat_input("Введите ваш вопрос...")
    
    if question:
        # Добавляем вопрос в историю
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat()
        })
        
        # Сразу показываем вопрос
        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)
            
            # Генерируем ответ
            with st.chat_message("assistant"):
                with st.spinner("🔍 Ищу информацию..."):
                    try:
                        # Импортируем модули из существующей системы
                        from main import (
                            load_knowledge_base,
                            build_faiss_index,
                            retrieve_faiss,
                            llm_rerank,
                            build_context,
                            generate_answer,
                            evaluate_answer
                        )
                        
                        # Загружаем данные
                        df = load_knowledge_base("./train_data.csv")
                        
                        # Загружаем индекс FAISS
                        index_mode = "v1" if st.session_state.rag_mode == "v1" else "v2"
                        index_file = f"faiss_index_{index_mode}.bin"
                        meta_file = f"faiss_meta_{index_mode}.pkl"
                        
                        index, metadata = build_faiss_index(
                            df, index_file, meta_file, mode=index_mode
                        )
                        
                        # Поиск релевантных фрагментов
                        hits = retrieve_faiss(question, index, metadata, top_k)
                        
                        # Переранжирование если v3
                        if st.session_state.rag_mode == "v3":
                            hits = llm_rerank(question, hits, df, final_k)
                        else:
                            hits = hits[:final_k]
                        
                        # Генерация ответа
                        context = build_context(hits, df)
                        answer = generate_answer(question, context)
                        
                        # Оценка ответа
                        metrics = evaluate_answer(question, answer, context)
                        
                        # Показываем ответ
                        st.markdown(answer)
                        
                        # Сохраняем в историю с деталями
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "timestamp": datetime.now().isoformat(),
                            "details": {
                                "hits": hits,
                                "metrics": metrics,
                                "mode": st.session_state.rag_mode
                            }
                        })
                        
                    except ImportError as e:
                        error_msg = f"Ошибка импорта: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"Ошибка системы: {error_msg}",
                            "timestamp": datetime.now().isoformat(),
                            "error": True
                        })
                        
                    except Exception as e:
                        error_msg = str(e)
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": f"Произошла ошибка: {error_msg}",
                            "timestamp": datetime.now().isoformat(),
                            "error": True
                        })