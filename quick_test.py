#!/usr/bin/env python3
"""
Быстрый тест RAG решения на одном вопросе
Использование: python quick_test.py "Ваш вопрос"
"""

import sys
from main import load_knowledge_base, create_embeddings_cache, answer_with_rag


def quick_test(question: str = None):
    """
    Быстрый тест на одном вопросе.
    
    Args:
        question: Вопрос для тестирования. Если None, используется вопрос по умолчанию.
    """
    if question is None:
        question = "Как получить налоговый вычет при покупке квартиры?"
    
    print("="*80)
    print("БЫСТРЫЙ ТЕСТ RAG РЕШЕНИЯ")
    print("="*80)
    print(f"\nВопрос: {question}\n")
    
    # Загрузка данных
    print("📚 Загрузка базы знаний...")
    knowledge_base = load_knowledge_base('./train_data.csv')
    print(f"   Загружено {len(knowledge_base)} документов")
    
    # Загрузка/создание эмбеддингов
    print("\n🔢 Загрузка эмбеддингов...")
    embeddings = create_embeddings_cache(knowledge_base)
    print(f"   Размерность эмбеддингов: {embeddings.shape}")
    
    # Генерация ответа
    print("\n🤖 Генерация ответа с использованием RAG...")
    print("-"*80)
    answer = answer_with_rag(question, knowledge_base, embeddings, top_k=3)
    
    print("\n📝 ОТВЕТ:")
    print("="*80)
    print(answer)
    print("="*80)
    
    print("\n✅ Тест завершен успешно!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Берем вопрос из аргументов командной строки
        question = " ".join(sys.argv[1:])
        quick_test(question)
    else:
        # Используем вопрос по умолчанию
        print("Использование: python quick_test.py \"Ваш вопрос\"")
        print("Запуск с вопросом по умолчанию...\n")
        quick_test()
