#utils.py
"""
Утилиты для тестирования и отладки RAG решения
"""

import pandas as pd
import numpy as np
from main import (
    load_knowledge_base, 
    create_embeddings_cache, 
    retrieve_relevant_documents,
    answer_with_rag,
    get_embedding
)
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns


def test_single_question(question: str, top_k: int = 5):
    """
    Тестирование одного вопроса с выводом релевантных документов.
    
    Args:
        question: Вопрос для тестирования
        top_k: Количество документов для показа
    """
    print(f"\n{'='*80}")
    print(f"ВОПРОС: {question}")
    print(f"{'='*80}\n")
    
    # Загружаем базу знаний
    print("Загрузка базы знаний...")
    knowledge_base = load_knowledge_base('./train_data.csv')
    
    # Загружаем эмбеддинги
    print("Загрузка эмбеддингов...")
    embeddings = create_embeddings_cache(knowledge_base)
    
    # Ищем релевантные документы
    print(f"\nПоиск топ-{top_k} релевантных документов...\n")
    relevant_docs = retrieve_relevant_documents(question, knowledge_base, embeddings, top_k)
    
    # Выводим найденные документы
    for i, (idx, score) in enumerate(relevant_docs, 1):
        doc = knowledge_base.iloc[idx]
        print(f"{'─'*80}")
        print(f"Документ #{i} (ID: {doc['id']}, Релевантность: {score:.4f})")
        print(f"{'─'*80}")
        print(f"Аннотация: {doc['annotation']}\n")
        print(f"Теги: {doc['tags']}\n")
        print(f"Текст (первые 500 символов):\n{doc['text'][:500]}...\n")
    
    # Генерируем ответ
    print(f"\n{'='*80}")
    print("ГЕНЕРАЦИЯ ОТВЕТА...")
    print(f"{'='*80}\n")
    
    answer = answer_with_rag(question, knowledge_base, embeddings, top_k=3)
    
    print(f"ОТВЕТ:\n{answer}\n")
    print(f"{'='*80}\n")


def analyze_embeddings_distribution(sample_size: int = 100):
    """
    Анализ распределения эмбеддингов в базе знаний.
    
    Args:
        sample_size: Количество документов для анализа
    """
    print("Загрузка данных...")
    knowledge_base = load_knowledge_base('./train_data.csv')
    embeddings = create_embeddings_cache(knowledge_base)
    
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
    plt.title(f'Матрица косинусной схожести документов (выборка {sample_size})')
    plt.xlabel('Индекс документа')
    plt.ylabel('Индекс документа')
    plt.tight_layout()
    plt.savefig('similarity_matrix.png', dpi=150)
    print("Матрица схожести сохранена в similarity_matrix.png")
    
    # Статистика
    np.fill_diagonal(similarity_matrix, 0)  # Исключаем диагональ
    flat_similarities = similarity_matrix.flatten()
    flat_similarities = flat_similarities[flat_similarities != 0]
    
    print(f"\nСтатистика схожести документов:")
    print(f"Средняя схожесть: {np.mean(flat_similarities):.4f}")
    print(f"Медианная схожесть: {np.median(flat_similarities):.4f}")
    print(f"Стд. отклонение: {np.std(flat_similarities):.4f}")
    print(f"Минимум: {np.min(flat_similarities):.4f}")
    print(f"Максимум: {np.max(flat_similarities):.4f}")


def evaluate_retrieval_quality(test_questions: List[str], expected_keywords: List[List[str]], top_k: int = 5):
    """
    Оценка качества поиска релевантных документов.
    
    Args:
        test_questions: Список тестовых вопросов
        expected_keywords: Список списков ключевых слов, которые должны быть в релевантных документах
        top_k: Количество документов для проверки
    """
    print("Загрузка данных...")
    knowledge_base = load_knowledge_base('./train_data.csv')
    embeddings = create_embeddings_cache(knowledge_base)
    
    results = []
    
    for i, (question, keywords) in enumerate(zip(test_questions, expected_keywords), 1):
        print(f"\nВопрос {i}: {question}")
        print(f"Ожидаемые ключевые слова: {', '.join(keywords)}")
        
        relevant_docs = retrieve_relevant_documents(question, knowledge_base, embeddings, top_k)
        
        # Проверяем наличие ключевых слов в найденных документах
        found_keywords = set()
        for idx, score in relevant_docs:
            doc = knowledge_base.iloc[idx]
            doc_text = f"{doc['annotation']} {doc['text']}".lower()
            
            for keyword in keywords:
                if keyword.lower() in doc_text:
                    found_keywords.add(keyword)
        
        precision = len(found_keywords) / len(keywords) if keywords else 0
        results.append({
            'question': question,
            'expected': len(keywords),
            'found': len(found_keywords),
            'precision': precision,
            'top_score': relevant_docs[0][1] if relevant_docs else 0
        })
        
        print(f"Найдено ключевых слов: {len(found_keywords)}/{len(keywords)}")
        print(f"Precision: {precision:.2%}")
        print(f"Топ score: {relevant_docs[0][1]:.4f}")
    
    # Общая статистика
    df_results = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print("ОБЩАЯ СТАТИСТИКА:")
    print(f"{'='*80}")
    print(f"Средний Precision: {df_results['precision'].mean():.2%}")
    print(f"Средний топ score: {df_results['top_score'].mean():.4f}")
    
    return df_results


def compare_models(question: str, models: List[str]):
    """
    Сравнение ответов разных моделей на один вопрос.
    
    Args:
        question: Вопрос для тестирования
        models: Список моделей для сравнения
    """
    print(f"\n{'='*80}")
    print(f"ВОПРОС: {question}")
    print(f"{'='*80}\n")
    
    knowledge_base = load_knowledge_base('./train_data.csv')
    embeddings = create_embeddings_cache(knowledge_base)
    
    # Получаем контекст один раз
    relevant_docs = retrieve_relevant_documents(question, knowledge_base, embeddings, top_k=3)
    from main import create_context_from_documents, answer_generation
    context = create_context_from_documents(relevant_docs, knowledge_base)
    
    for model in models:
        print(f"\n{'─'*80}")
        print(f"МОДЕЛЬ: {model}")
        print(f"{'─'*80}\n")
        
        # Временно меняем модель в answer_generation
        # (в реальности нужно передавать model как параметр)
        answer = answer_generation(question, context)
        
        print(f"ОТВЕТ:\n{answer}\n")


def benchmark_speed():
    """
    Тест производительности системы.
    """
    import time
    
    print("Загрузка данных...")
    knowledge_base = load_knowledge_base('./train_data.csv')
    
    print("\n1. Тест создания эмбеддинга...")
    test_text = "Тестовый текст для измерения скорости создания эмбеддинга"
    start = time.time()
    embedding = get_embedding(test_text)
    embedding_time = time.time() - start
    print(f"Время создания эмбеддинга: {embedding_time:.3f} сек")
    
    print("\n2. Тест поиска релевантных документов...")
    embeddings = create_embeddings_cache(knowledge_base)
    test_question = "Как получить кредит?"
    
    start = time.time()
    relevant_docs = retrieve_relevant_documents(test_question, knowledge_base, embeddings, top_k=5)
    search_time = time.time() - start
    print(f"Время поиска: {search_time:.3f} сек")
    
    print("\n3. Тест полного цикла RAG...")
    start = time.time()
    answer = answer_with_rag(test_question, knowledge_base, embeddings, top_k=3)
    full_time = time.time() - start
    print(f"Время полного цикла: {full_time:.3f} сек")
    
    print(f"\n{'='*80}")
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print(f"{'='*80}")
    print(f"Эмбеддинг: {embedding_time:.3f} сек")
    print(f"Поиск: {search_time:.3f} сек")
    print(f"Генерация (LLM): {full_time - search_time - embedding_time:.3f} сек")
    print(f"Общее время: {full_time:.3f} сек")


if __name__ == "__main__":
    # Пример использования
    print("Утилиты для тестирования RAG решения")
    print("="*80)
    print("\nДоступные функции:")
    print("1. test_single_question() - тест одного вопроса")
    print("2. analyze_embeddings_distribution() - анализ эмбеддингов")
    print("3. evaluate_retrieval_quality() - оценка качества поиска")
    print("4. compare_models() - сравнение разных моделей")
    print("5. benchmark_speed() - тест производительности")
    
    # Запускаем простой тест
    print("\n\nЗапуск примера: тест одного вопроса")
    test_single_question("Как получить налоговый вычет?", top_k=3)
