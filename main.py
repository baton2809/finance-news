import os
import re
import time
import pickle
import logging
import argparse
import json
from typing import List, Tuple, Dict, Any
from collections import Counter

import numpy as np
import pandas as pd
import faiss  # <<< NEW: Vector DB
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Additional metrics libraries
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------- Env --------------------------
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")

# ---------------------- Cost tracker -----------------
COST = {
    "embedding_calls": 0,
    "llm_calls": 0,
    "embedding_cost": 0.0,
    "llm_cost": 0.0,
    "embedding_tokens": 0,
    "llm_prompt_tokens": 0,
}
PRICE_PER_1K_TOK_LLM = 0.0001 

# ---------------------- Global embedding model -------
EMBEDDING_MODEL = None
EMBEDDING_DIM = 384  # Dimension for multilingual-e5-small (lighter, faster)

def get_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        logger.info("Loading local embedding model: intfloat/multilingual-e5-small")
        EMBEDDING_MODEL = SentenceTransformer('intfloat/multilingual-e5-small')
    return EMBEDDING_MODEL

# ---------------------- Clients ----------------------
def get_llm_client() -> OpenAI:
    """Returns DeepSeek API client (OpenAI-compatible)"""
    return OpenAI(
        base_url="https://api.deepseek.com",
        api_key=LLM_API_KEY
    )

# ---------------------- Utils ------------------------
def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)

def chunk_text(text: str, chunk_size_chars: int = 1000, overlap_chars: int = 200) -> List[str]:
    chunks = []
    i = 0
    n = len(text)
    step = max(1, chunk_size_chars - overlap_chars)
    while i < n:
        chunks.append(text[i:i + chunk_size_chars])
        i += step
    return chunks

# ---------------------- Data loading -----------------
def load_knowledge_base(path: str = "./train_data.csv") -> pd.DataFrame:
    logger.info(f"Loading knowledge base from {path}")
    df = pd.read_csv(path, sep=",", encoding="utf-8")
    return df

# ---------------------- Embeddings -------------------
def get_embedding(text: str) -> np.ndarray:
    model = get_embedding_model()
    # encode returns numpy array by default now
    embedding = model.encode([text], show_progress_bar=False)[0]
    
    t = approx_tokens(text)
    COST["embedding_calls"] += 1
    COST["embedding_tokens"] += t
    
    return embedding

def build_faiss_index(df: pd.DataFrame, index_file: str = "faiss_index.bin", meta_file: str = "faiss_meta.pkl", mode: str = "v2") -> Tuple[faiss.Index, List[Any]]:
    """
    Builds a FAISS index (Vector Database) from the dataframe.
    mode="v1": documents are items.
    mode="v2": chunks are items.
    """
    if os.path.exists(index_file) and os.path.exists(meta_file):
        logger.info(f"Loading FAISS index from {index_file}")
        index = faiss.read_index(index_file)
        with open(meta_file, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    logger.info(f"Building FAISS index (Mode: {mode})...")
    
    texts_to_encode = []
    metadata = [] # Stores (doc_id, chunk_id) or just doc_id

    for doc_idx, row in df.iterrows():
        base_text = f"{row['annotation']} {row['text']}"
        
        if mode == "v1":
            # Document level
            texts_to_encode.append(base_text[:2000]) # Limit length for doc level
            metadata.append((doc_idx, -1))
        else:
            # Chunk level
            chunks = chunk_text(base_text)
            for chunk_idx, chunk in enumerate(chunks):
                texts_to_encode.append(chunk)
                metadata.append((doc_idx, chunk_idx))

    # Batch encode
    model = get_embedding_model()
    logger.info("Encoding texts...")
    embeddings = model.encode(texts_to_encode, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    # Cost tracking
    for t in texts_to_encode:
        COST["embedding_tokens"] += approx_tokens(t)
    COST["embedding_calls"] += len(texts_to_encode)

    # Build FAISS Index
    # We use IndexFlatIP (Inner Product) which is equivalent to Cosine Similarity 
    # IF vectors are normalized. SentenceTransformers outputs normalized vectors by default? 
    # Usually yes, but let's normalize just in case to be safe for cosine.
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    
    logger.info(f"Index built with {index.ntotal} vectors.")
    
    # Save
    faiss.write_index(index, index_file)
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)
        
    return index, metadata

# ---------------------- Retrieval (FAISS) ------------
def retrieve_faiss(question: str, index: faiss.Index, metadata: List[Any], top_k: int = 5) -> List[Tuple[Any, float]]:
    # 1. Embed question
    q_emb = get_embedding(question).reshape(1, -1)
    faiss.normalize_L2(q_emb) # Normalize for cosine similarity
    
    # 2. Search
    D, I = index.search(q_emb, top_k)
    
    # 3. Map back to metadata
    results = []
    for i, idx in enumerate(I[0]):
        if idx != -1: # FAISS returns -1 if not enough neighbors
            meta = metadata[idx]
            score = float(D[0][i])
            results.append((meta, score))
            
    return results

# ---------------------- Re-ranking (LLM) -------------
def llm_rerank(question: str, candidates: List[Tuple[Any, float]], df: pd.DataFrame, k_final: int = 3) -> List[Tuple[Any, float]]:
    # Re-use logic from previous, but adapted for generic metadata
    previews = []
    for (doc_idx, chunk_idx), _ in candidates:
        row = df.iloc[doc_idx]
        base = f"{row['annotation']} {row['text']}"
        if chunk_idx == -1: # Doc level
            preview = base[:500]
        else:
            chunks = chunk_text(base)
            preview = chunks[chunk_idx] if chunk_idx < len(chunks) else ""
        
        previews.append(preview.replace("\n", " ")[:400])

    prompt = (
        "Ты — ассистент по поиску. Дан вопрос и фрагменты.\n"
        "Выбери ТРИ самых полезных фрагмента (индексы 0-N).\n"
        f"Вопрос: {question}\n\n"
        + "\n".join([f"[{i}] {p}" for i, p in enumerate(previews)]) +
        "\n\nВерни ТОЛЬКО цифры, например: 0, 2"
    )

    client = get_llm_client()
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=20
        )
        COST["llm_calls"] += 1
        text = resp.choices[0].message.content
        idxs = [int(x) for x in re.findall(r"\d+", text)]
        picked = [candidates[i] for i in idxs if i < len(candidates)]
        return picked[:k_final] if picked else candidates[:k_final]
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        return candidates[:k_final]

# ---------------------- Context & Answer -------------
def build_context(hits: List[Tuple[Any, float]], df: pd.DataFrame) -> str:
    parts = []
    for (doc_idx, chunk_idx), score in hits:
        row = df.iloc[doc_idx]
        base = f"{row['annotation']} {row['text']}"
        
        if chunk_idx == -1:
            content = base
        else:
            chunks = chunk_text(base)
            content = chunks[chunk_idx] if chunk_idx < len(chunks) else ""

        parts.append(f"Фрагмент (score: {score:.2f}):\n{content}\n---")
    return "\n".join(parts)

def generate_answer(question: str, context: str) -> str:
    client = get_llm_client()
    prompt = (
        "Ответь на вопрос, используя контекст.\n"
        "Если информации нет, скажи 'В контексте нет информации'.\n\n"
        f"КОНТЕКСТ:\n{context}\n\n"
        f"ВОПРОС: {question}"
    )
    
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    COST["llm_calls"] += 1
    return resp.choices[0].message.content

# ---------------------- Evaluation Metrics ----------------

# Initialize ROUGE scorer (cached for efficiency)
# NOTE: use_stemmer=True helps match different word forms in Russian
ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def extract_relevant_context(answer: str, context: str, max_sentences: int = 5) -> str:
    """
    Extract the most relevant sentences from context based on word overlap with answer.
    This helps ROUGE focus on the most relevant parts of a long context.
    """
    # Split context into sentences (simple split by punctuation)
    sentences = re.split(r'[.!?]\s+', context)

    if len(sentences) <= max_sentences:
        return context

    # Tokenize answer for comparison
    answer_words = set(answer.lower().split())

    # Score each sentence by word overlap with answer
    sentence_scores = []
    for sent in sentences:
        sent_words = set(sent.lower().split())
        overlap = len(answer_words & sent_words)
        sentence_scores.append((sent, overlap))

    # Get top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    # Restore original order
    top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[0]))

    return '. '.join([sent for sent, _ in top_sentences])

def calculate_rouge_scores(answer: str, context: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores between answer and context.

    Improvements:
    1. Uses stemming to match word variations (e.g., цен, цены, ценам)
    2. Normalizes text (lowercase, strip whitespace)
    3. Extracts most relevant context sentences for fairer comparison
    4. Returns both F1 and recall scores
    """
    # Normalize texts
    answer_norm = answer.strip().lower()
    context_norm = context.strip().lower()

    # Extract most relevant parts of context (helps with long contexts)
    # This gives more meaningful scores when context is much longer than answer
    relevant_context = extract_relevant_context(answer_norm, context_norm, max_sentences=10)

    # Debug logging
    logger.debug(f"Answer length: {len(answer_norm)} chars")
    logger.debug(f"Context length: {len(context_norm)} chars")
    logger.debug(f"Relevant context length: {len(relevant_context)} chars")
    logger.debug(f"Answer preview: {answer_norm[:100]}...")
    logger.debug(f"Relevant context preview: {relevant_context[:200]}...")

    # Calculate ROUGE scores
    scores = ROUGE_SCORER.score(relevant_context, answer_norm)

    logger.debug(f"ROUGE-L F1: {scores['rougeL'].fmeasure:.3f}, Recall: {scores['rougeL'].recall:.3f}")

    return {
        "rouge1_f": scores['rouge1'].fmeasure,
        "rouge1_r": scores['rouge1'].recall,  # How much of answer is in context
        "rouge2_f": scores['rouge2'].fmeasure,
        "rougeL_f": scores['rougeL'].fmeasure,
        "rougeL_r": scores['rougeL'].recall,  # Longest common subsequence recall
    }

def calculate_bert_score(answer: str, context: str) -> Dict[str, float]:
    """Calculate BERTScore for semantic similarity."""
    try:
        P, R, F1 = bert_score_fn(
            [answer], [context],
            lang="ru",  # Russian language
            verbose=False,
            rescale_with_baseline=True
        )
        return {
            "bert_precision": float(P[0]),
            "bert_recall": float(R[0]),
            "bert_f1": float(F1[0]),
        }
    except Exception as e:
        logger.warning(f"BERTScore calculation failed: {e}")
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}

def calculate_answer_quality(question: str, answer: str, context: str) -> Dict[str, float]:
    """Calculate lexical and structural quality metrics."""
    # Answer length metrics
    answer_words = len(answer.split())
    context_words = len(context.split())

    # Context coverage (how much of context keywords appear in answer)
    context_tokens = set(context.lower().split())
    answer_tokens = set(answer.lower().split())
    coverage = len(context_tokens & answer_tokens) / max(len(context_tokens), 1)

    # Question keyword overlap
    question_tokens = set(question.lower().split())
    question_coverage = len(question_tokens & answer_tokens) / max(len(question_tokens), 1)

    # TF-IDF similarity between answer and context
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([context, answer])
        tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        tfidf_sim = 0.0

    return {
        "answer_length": answer_words,
        "context_coverage": coverage,
        "question_coverage": question_coverage,
        "tfidf_similarity": tfidf_sim,
    }

def evaluate_answer_llm(question: str, answer: str, context: str) -> Dict[str, int]:
    """
    LLM-as-a-Judge: Asks the LLM to rate the answer on scale 1-5.
    Returns: {'llm_relevance': int, 'llm_faithfulness': int}
    """
    client = get_llm_client()

    prompt = f"""
    You are a judge. Evaluate the answer based on the context and question.

    Question: {question}
    Context: {context}
    Answer: {answer}

    1. Relevance: Does the answer directly address the question? (1-5)
    2. Faithfulness: Is the answer fully supported by the context? (1-5)

    Output JSON only: {{"relevance": <int>, "faithfulness": <int>}}
    """

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        COST["llm_calls"] += 1
        content = resp.choices[0].message.content
        json_str = re.search(r'\{.*\}', content, re.DOTALL).group(0)
        result = json.loads(json_str)
        return {
            "llm_relevance": result.get("relevance", 0),
            "llm_faithfulness": result.get("faithfulness", 0)
        }
    except:
        return {"llm_relevance": 0, "llm_faithfulness": 0}

def evaluate_answer(question: str, answer: str, context: str) -> Dict[str, float]:
    """
    Comprehensive evaluation combining multiple metrics:
    - LLM-as-Judge (relevance, faithfulness)
    - ROUGE scores (lexical overlap)
    - BERTScore (semantic similarity)
    - Answer quality metrics (coverage, length, TF-IDF)
    """
    metrics = {}

    # 1. LLM-as-Judge (subjective but comprehensive)
    llm_metrics = evaluate_answer_llm(question, answer, context)
    metrics.update(llm_metrics)

    # 2. ROUGE scores (lexical overlap with context)
    rouge_metrics = calculate_rouge_scores(answer, context)
    metrics.update(rouge_metrics)

    # 3. BERTScore (semantic similarity)
    bert_metrics = calculate_bert_score(answer, context)
    metrics.update(bert_metrics)

    # 4. Answer quality metrics
    quality_metrics = calculate_answer_quality(question, answer, context)
    metrics.update(quality_metrics)

    # Legacy compatibility: keep 'relevance' and 'faithfulness' for backward compat
    metrics["relevance"] = metrics["llm_relevance"]
    metrics["faithfulness"] = metrics["llm_faithfulness"]

    return metrics

# ---------------------- Pipelines --------------------
def run_rag(mode: str, question: str, df: pd.DataFrame, index: faiss.Index, metadata: List[Any], top_k: int, final_k: int) -> Tuple[str, Dict]:
    # 1. Retrieve
    hits = retrieve_faiss(question, index, metadata, top_k)
    
    # 2. Rerank (if v3)
    if mode == "v3":
        hits = llm_rerank(question, hits, df, final_k)
    else:
        hits = hits[:final_k]
        
    # 3. Context
    ctx = build_context(hits, df)
    
    # 4. Generate
    ans = generate_answer(question, ctx)
    
    # 5. Evaluate (Metric)
    metrics = evaluate_answer(question, ans, ctx)
    
    return ans, metrics

# ---------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG System for Financial Literacy Q&A")
    parser.add_argument("--mode", choices=["v1", "v2", "v3"], default="v2",
                        help="RAG mode: v1=doc-level, v2=chunk, v3=chunk+rerank")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions to process")
    parser.add_argument("--skip-bert", action="store_true",
                        help="Skip BERTScore calculation (faster)")
    args = parser.parse_args()

    # Config based on mode
    top_k = 20 if args.mode == "v3" else 5
    final_k = 3

    # Load Data
    df = load_knowledge_base("./train_data.csv")

    # Build FAISS (Vector DB) - Note: need to rebuild with new embedding model
    index_mode = "v1" if args.mode == "v1" else "v2"
    index_file = f"faiss_index_{index_mode}_e5small.bin"  # e5-small model
    meta_file = f"faiss_meta_{index_mode}_e5small.pkl"

    index, metadata = build_faiss_index(df, index_file, meta_file, mode=index_mode)

    # Load Questions
    questions_df = pd.read_csv("./questions.csv")
    questions = questions_df["Вопрос"].tolist()
    if args.limit:
        questions = questions[:args.limit]

    answers = []
    all_metrics = []

    # Initialize accumulators for all metrics
    metric_keys = [
        "llm_relevance", "llm_faithfulness",
        "rouge1_f", "rouge2_f", "rougeL_f",
        "bert_precision", "bert_recall", "bert_f1",
        "context_coverage", "question_coverage", "tfidf_similarity"
    ]
    total_metrics = {k: 0.0 for k in metric_keys}

    for q in tqdm(questions, desc="RAG Pipeline"):
        ans, metrics = run_rag(args.mode, q, df, index, metadata, top_k, final_k)
        answers.append(ans)
        all_metrics.append(metrics)

        # Accumulate metrics
        for k in metric_keys:
            total_metrics[k] += metrics.get(k, 0.0)

    # Save Results
    questions_df = questions_df.iloc[:len(answers)]
    questions_df["Ответы"] = answers
    questions_df.to_csv("submission.csv", index=False)

    # Save detailed metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("metrics_detailed.csv", index=False)

    # Report Metrics
    n = len(answers)
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("")
    logger.info("📊 LLM-as-Judge Metrics (1-5 scale):")
    logger.info(f"   Relevance:    {total_metrics['llm_relevance']/n:.2f}")
    logger.info(f"   Faithfulness: {total_metrics['llm_faithfulness']/n:.2f}")
    logger.info("")
    logger.info("📝 ROUGE Scores (lexical overlap):")
    logger.info(f"   ROUGE-1 F1: {total_metrics['rouge1_f']/n:.4f}")
    logger.info(f"   ROUGE-2 F1: {total_metrics['rouge2_f']/n:.4f}")
    logger.info(f"   ROUGE-L F1: {total_metrics['rougeL_f']/n:.4f}")
    logger.info("")
    logger.info("🔤 BERTScore (semantic similarity):")
    logger.info(f"   Precision: {total_metrics['bert_precision']/n:.4f}")
    logger.info(f"   Recall:    {total_metrics['bert_recall']/n:.4f}")
    logger.info(f"   F1:        {total_metrics['bert_f1']/n:.4f}")
    logger.info("")
    logger.info("📈 Answer Quality Metrics:")
    logger.info(f"   Context Coverage:  {total_metrics['context_coverage']/n:.4f}")
    logger.info(f"   Question Coverage: {total_metrics['question_coverage']/n:.4f}")
    logger.info(f"   TF-IDF Similarity: {total_metrics['tfidf_similarity']/n:.4f}")
    logger.info("=" * 60)
    logger.info(f"Detailed metrics saved to: metrics_detailed.csv")

if __name__ == "__main__":
    main()
