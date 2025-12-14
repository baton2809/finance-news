#utils.py
"""
Утилиты для тестирования и отладки RAG решения
"""

import pandas as pd
import numpy as np
from main import (
    load_knowledge_base, 
    build_doc_cache,
    build_chunk_cache,
    retrieve_v1,
    retrieve_v2,
    answer_with_rag_v1,
    answer_with_rag_v2,
    get_embedding
)
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns


def test_single_question(question: str, top_k: int = 5, mode: str = "v2"):
    """
    Тестирование одного вопроса с выводом релевантных документов.
    
    Args:
        question: Вопрос для тестирования
        top_k: Количество документов для показа
        mode: Режим работы (v1 или v2/v3)
    """
    print(f"\n{'='*80}")
    print(f"ВОПРОС: {question}")
    print(f"{'='*80}\n")
    
    # Загружаем базу знаний
    print("Загрузка базы знаний...")
    knowledge_base = load_knowledge_base('./train_data.csv')
    
    if mode == "v1":
        # Загружаем эмбеддинги документов
        print("Загрузка эмбеддингов документов...")
        doc_embs = build_doc_cache(knowledge_base)
        
        # Ищем релевантные документы
        print(f"\nПоиск топ-{top_k} релевантных документов...\n")
        relevant_docs = retrieve_v1(question, knowledge_base, doc_embs, top_k)
        
        # Выводим найденные документы
        for i, (idx, score) in enumerate(relevant_docs, 1):
            doc = knowledge_base.iloc[idx]
            print(f"{'─'*80}")
            print(f"Документ #{i} (ID: {idx}, Релевантность: {score:.4f})")
            print(f"{'─'*80}")
            print(f"Аннотация: {doc['annotation']}\n")
            print(f"Теги: {doc.get('tags', 'N/A')}\n")
            print(f"Текст (первые 500 символов):\n{doc['text'][:500]}...\n")
        
        # Генерируем ответ
        print(f"\n{'='*80}")
        print("ГЕНЕРАЦИЯ ОТВЕТА...")
        print(f"{'='*80}\n")
        
        answer = answer_with_rag_v1(question, knowledge_base, doc_embs, top_k=3)
        
    else:  # v2/v3
        # Загружаем эмбеддинги чанков
        print("Загрузка эмбеддингов чанков...")
        chunk_cache = build_chunk_cache(knowledge_base)
        
        # Ищем релевантные чанки
        print(f"\nПоиск топ-{top_k} релевантных чанков...\n")
        relevant_chunks = retrieve_v2(question, chunk_cache, top_k)
        
        # Выводим найденные чанки
        for i, ((doc_idx, chunk_idx), score) in enumerate(relevant_chunks, 1):
            doc = knowledge_base.iloc[doc_idx]
            print(f"{'─'*80}")
            print(f"Чанк #{i} (Doc: {doc_idx}, Chunk: {chunk_idx}, Релевантность: {score:.4f})")
            print(f"{'─'*80}")
            print(f"Аннотация: {doc['annotation']}\n")
            print(f"Теги: {doc.get('tags', 'N/A')}\n")
            # Reconstruct chunk
            from main import chunk_text
            base = f"{doc['annotation']} {doc['text']}"
            chunks = chunk_text(base)
            chunk_content = chunks[chunk_idx] if chunk_idx < len(chunks) else "N/A"
            print(f"Чанк (первые 500 символов):\n{chunk_content[:500]}...\n")
        
        # Генерируем ответ
        print(f"\n{'='*80}")
        print("ГЕНЕРАЦИЯ ОТВЕТА...")
        print(f"{'='*80}\n")
        
        answer = answer_with_rag_v2(question, knowledge_base, chunk_cache, top_k=5, k_final=3)
    
    print(f"ОТВЕТ:\n{answer}\n")
    print(f"{'='*80}\n")


def analyze_embeddings_distribution(sample_size: int = 100, mode: str = "doc"):
    """
    Анализ распределения эмбеддингов в базе знаний.
    
    Args:
        sample_size: Количество документов для анализа
        mode: "doc" для документов или "chunk" для чанков
    """
    print("Загрузка данных...")
    knowledge_base = load_knowledge_base('./train_data.csv')
    
    if mode == "doc":
        embeddings = build_doc_cache(knowledge_base)
    else:
        cache = build_chunk_cache(knowledge_base)
        embeddings = cache["embeddings"]
    
    # Берем выборку
    sample_indices = np.random.choice(len(embeddings), min(sample_size, len(embeddings)), replace=False)
    sample_embeddings = embeddings[sample_indices]
    
    # Вычисляем матрицу схожести
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(sample_embeddings)
    
    # Визуализация
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0, cbar_kws={"shrink": 0.8})
    plt.title(f'Матрица косинусной схожести ({mode}) (выборка {sample_size})')
    plt.xlabel(f'Индекс {mode}')
    plt.ylabel(f'Индекс {mode}')
    plt.tight_layout()
    plt.savefig(f'similarity_matrix_{mode}.png', dpi=150)
    print(f"Матрица схожести сохранена в similarity_matrix_{mode}.png")
    
    # Статистика
    np.fill_diagonal(similarity_matrix, 0)  # Исключаем диагональ
    flat_similarities = similarity_matrix.flatten()
    flat_similarities = flat_similarities[flat_similarities != 0]
    
    print(f"\nСтатистика схожести ({mode}):")
    print(f"Средняя схожесть: {np.mean(flat_similarities):.4f}")
    print(f"Медианная схожесть: {np.median(flat_similarities):.4f}")
    print(f"Стд. отклонение: {np.std(flat_similarities):.4f}")
    print(f"Минимум: {np.min(flat_similarities):.4f}")
    print(f"Максимум: {np.max(flat_similarities):.4f}")


def benchmark_speed():
    """
    Тест производительности системы.
    """
    import time
    
    print("Загрузка данных...")
    knowledge_base = load_knowledge_base('./train_data.csv')
    
    print("\n1. Тест создания эмбеддинга (локальная модель)...")
    test_text = "Тестовый текст для измерения скорости создания эмбеддинга"
    start = time.time()
    embedding = get_embedding(test_text)
    embedding_time = time.time() - start
    print(f"Время создания эмбеддинга: {embedding_time:.3f} сек")
    
    print("\n2. Тест поиска релевантных документов...")
    doc_embs = build_doc_cache(knowledge_base)
    test_question = "Как получить кредит?"
    
    start = time.time()
    relevant_docs = retrieve_v1(test_question, knowledge_base, doc_embs, top_k=5)
    search_time = time.time() - start
    print(f"Время поиска: {search_time:.3f} сек")
    
    print("\n3. Тест полного цикла RAG...")
    start = time.time()
    answer = answer_with_rag_v1(test_question, knowledge_base, doc_embs, top_k=3)
    full_time = time.time() - start
    print(f"Время полного цикла: {full_time:.3f} сек")
    
    print(f"\n{'='*80}")
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print(f"{'='*80}")
    print(f"Эмбеддинг (локальный): {embedding_time:.3f} сек")
    print(f"Поиск: {search_time:.3f} сек")
    print(f"Генерация (LLM): {full_time - search_time - embedding_time:.3f} сек")
    print(f"Общее время: {full_time:.3f} сек")


if __name__ == "__main__":
    # Пример использования
    print("Утилиты для тестирования RAG решения")
    print("="*80)
    print("\nДоступные функции:")
    print("1. test_single_question() - тест одного вопроса")
