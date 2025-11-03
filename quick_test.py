#!/usr/bin/env python3
"""
Быстрый тест RAG решения на одном вопросе
Использование:
    python quick_test.py "Ваш вопрос"
    python quick_test.py --mode v3 "Как получить налоговый вычет?"
"""

import sys
import argparse
from main import (
    load_knowledge_base,
    build_doc_cache,
    build_chunk_cache,
    answer_with_rag_v1,
    answer_with_rag_v2,
    answer_with_rag_v3,
)

def quick_test(question: str, mode: str = "v2"):
    print("=" * 80)
    print("БЫСТРЫЙ ТЕСТ RAG РЕШЕНИЯ")
    print("=" * 80)
    print(f"\nВопрос: {question}\n")

    print("📚 Загрузка базы знаний...")
    kb = load_knowledge_base("./train_data.csv")

    if mode == "v1":
        print("\n🔢 Загрузка эмбеддингов (док-уровень)...")
        doc_embs = build_doc_cache(kb)
        print(f"   Размерность эмбеддингов: {doc_embs.shape}")
        print("\n🤖 Генерация ответа (v1)...")
        print("-" * 80)
        answer = answer_with_rag_v1(question, kb, doc_embs, top_k=3)
    elif mode == "v2":
        print("\n🔢 Загрузка эмбеддингов (чанки)...")
        cache = build_chunk_cache(kb)
        print(f"   Чанков: {cache['embeddings'].shape[0]}")
        print("\n🤖 Генерация ответа (v2)...")
        print("-" * 80)
        answer = answer_with_rag_v2(question, kb, cache, top_k=5, k_final=3)
    else:  # v3
        print("\n🔢 Загрузка эмбеддингов (чанки)...")
        cache = build_chunk_cache(kb)
        print(f"   Чанков: {cache['embeddings'].shape[0]}")
        print("\n🤖 Генерация ответа (v3, с re-ranking)...")
        print("-" * 80)
        answer = answer_with_rag_v3(question, kb, cache, top_k=8, k_final=3)

    print("\n📝 ОТВЕТ:")
    print("=" * 80)
    print(answer)
    print("=" * 80)
    print("\n✅ Тест завершен успешно!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["v1", "v2", "v3"], default="v2")
    parser.add_argument("question", nargs="*", help="Вопрос в кавычках")
    args = parser.parse_args()

    q = " ".join(args.question) if args.question else "Как получить налоговый вычет при покупке квартиры?"
    quick_test(q, mode=args.mode)
