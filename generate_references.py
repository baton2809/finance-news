"""
Generate Ground Truth References for RAG Evaluation

This script uses the existing RAG system to generate high-quality reference answers
for each question in questions.csv. These references will be used for context_precision
and context_recall metrics in RAGas.

Strategy:
1. Use RAG v3 (best quality) to generate answers
2. Use top-k=10 and final-k=5 for comprehensive context
3. Save references along with relevant document IDs
"""

import os
import logging
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Import RAG components
from main import (
    load_knowledge_base,
    build_faiss_index,
    retrieve_faiss,
    llm_rerank,
    build_context,
    generate_answer,
    logger
)

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_ground_truth_references(output_file="questions_with_references.csv", mode="v3", limit=None):
    """
    Generate ground truth references for all questions using RAG system.

    Args:
        output_file: Path to save questions with references
        mode: RAG mode to use (v3 recommended for best quality)
        limit: Limit number of questions (None for all)
    """
    logger.info("=" * 80)
    logger.info("GENERATING GROUND TRUTH REFERENCES")
    logger.info("=" * 80)

    # Load knowledge base
    logger.info("Loading knowledge base...")
    df = load_knowledge_base("./train_data.csv")
    logger.info(f"Loaded {len(df)} documents")

    # Build FAISS index (use v2 for chunk-level retrieval)
    logger.info("Building FAISS index...")
    index_mode = "v2"
    index_file = f"faiss_index_{index_mode}_e5small.bin"
    meta_file = f"faiss_meta_{index_mode}_e5small.pkl"
    index, metadata = build_faiss_index(df, index_file, meta_file, mode=index_mode)
    logger.info(f"FAISS index ready with {index.ntotal} vectors")

    # Load questions
    logger.info("Loading questions...")
    questions_df = pd.read_csv("./questions.csv")
    logger.info(f"Loaded {len(questions_df)} questions")

    if limit:
        questions_df = questions_df.head(limit)
        logger.info(f"Limited to first {limit} questions")

    # Configuration for best quality
    top_k = 10  # Retrieve more candidates
    final_k = 5  # Keep top 5 after reranking

    # Generate references
    logger.info(f"Generating references using RAG mode: {mode}")
    logger.info(f"Config: top_k={top_k}, final_k={final_k}")

    references = []
    contexts_list = []
    relevant_doc_ids = []

    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Generating references"):
        question_id = row['ID вопроса']
        question = row['Вопрос']

        try:
            # 1. Retrieve contexts
            hits = retrieve_faiss(question, index, metadata, top_k)

            # 2. Rerank (if v3)
            if mode == "v3":
                hits = llm_rerank(question, hits, df, final_k)
            else:
                hits = hits[:final_k]

            # 3. Extract relevant document IDs
            doc_ids = []
            for (doc_idx, chunk_idx), score in hits:
                doc_id = df.iloc[doc_idx]['id']
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)

            # 4. Build context
            context = build_context(hits, df)

            # 5. Generate reference answer
            reference_answer = generate_answer(question, context)

            # Store results
            references.append(reference_answer)
            contexts_list.append(context[:500])  # Store first 500 chars for review
            relevant_doc_ids.append(",".join(doc_ids))

            logger.debug(f"Q{question_id}: Generated reference ({len(reference_answer)} chars)")

        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            references.append("")
            contexts_list.append("")
            relevant_doc_ids.append("")

    # Add references to dataframe
    questions_df['Референсный ответ'] = references
    questions_df['Релевантные документы'] = relevant_doc_ids
    questions_df['Контекст (превью)'] = contexts_list

    # Save
    questions_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info("=" * 80)
    logger.info(f"✅ Successfully generated {len(references)} references")
    logger.info(f"💾 Saved to: {output_file}")
    logger.info("=" * 80)

    # Show sample
    logger.info("\n📋 Sample references:")
    for i in range(min(3, len(questions_df))):
        row = questions_df.iloc[i]
        logger.info(f"\nQ{row['ID вопроса']}: {row['Вопрос'][:80]}...")
        logger.info(f"Reference: {row['Референсный ответ'][:150]}...")
        logger.info(f"Docs: {row['Релевантные документы']}")

    return questions_df

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate ground truth references for RAG evaluation")
    parser.add_argument("--output", default="questions_with_references.csv", help="Output file path")
    parser.add_argument("--mode", choices=["v1", "v2", "v3"], default="v3",
                       help="RAG mode (v3=best quality, recommended)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of questions (for testing)")

    args = parser.parse_args()

    # Check API key
    if not os.getenv("LLM_API_KEY"):
        logger.error("❌ LLM_API_KEY not found in environment!")
        logger.error("Please set your DeepSeek API key in .env file")
        return

    # Generate references
    generate_ground_truth_references(
        output_file=args.output,
        mode=args.mode,
        limit=args.limit
    )

    logger.info("\n✨ Done! You can now use context_precision metric in your evaluation.")

if __name__ == "__main__":
    main()
