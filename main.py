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

# RAGas metrics (primary)
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

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
        try:
            # Try to load from local cache first (offline mode)
            EMBEDDING_MODEL = SentenceTransformer('intfloat/multilingual-e5-small', local_files_only=True)
            logger.info("Loaded model from local cache")
        except Exception as e:
            logger.warning(f"Could not load from cache, attempting download: {e}")
            try:
                # If cache doesn't exist, download the model
                EMBEDDING_MODEL = SentenceTransformer('intfloat/multilingual-e5-small')
                logger.info("Model downloaded successfully")
            except Exception as download_error:
                logger.error(f"Failed to download model: {download_error}")
                raise Exception(
                    "Cannot load embedding model. Either:\n"
                    "1. Check your internet connection and DNS settings\n"
                    "2. Or download the model manually and place it in the cache directory\n"
                    f"Original error: {download_error}"
                )
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
        "Если информации нет, скажи 'В контексте нет информации'.\n"
        "ВАЖНО: Всегда выделяй **жирным** ключевые термины, цифры и важные понятия в ответе.\n\n"
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

def evaluate_with_ragas(question: str, answer: str, context: str, reference: str = None) -> Dict[str, float]:
    """
    Evaluate answer using RAGas metrics (primary metrics).

    RAGas provides:
    - faithfulness: Is the answer faithful to the context?
    - answer_relevancy: Is the answer relevant to the question?
    - context_precision: How precise is the retrieved context? (requires reference)
    - context_recall: How much relevant info was retrieved? (requires reference)

    Args:
        question: The user question
        answer: The generated answer
        context: The retrieved context
        reference: Ground truth answer (optional, enables context_precision & context_recall)
    """
    try:
        # Split context into individual context chunks (RAGas works better with separated contexts)
        # The context string contains multiple fragments separated by "---"
        context_parts = [part.strip() for part in context.split("---") if part.strip()]

        # Clean up context parts - remove score lines and keep only the actual content
        cleaned_contexts = []
        for part in context_parts:
            # Remove "Фрагмент (score: X.XX):" lines
            lines = part.split("\n")
            content_lines = [line for line in lines if not line.startswith("Фрагмент (score:")]
            cleaned_context = "\n".join(content_lines).strip()
            if cleaned_context:
                cleaned_contexts.append(cleaned_context)

        # Use cleaned contexts or fallback to original
        if not cleaned_contexts:
            cleaned_contexts = [context]

        # Prepare data in RAGas format
        data = {
            "user_input": [question],
            "response": [answer],
            "retrieved_contexts": [cleaned_contexts],  # List of context chunks
        }

        # Add reference if available
        if reference:
            data["reference"] = [reference]

        dataset = Dataset.from_dict(data)

        # Configure LLM: DeepSeek via OpenAI-compatible API
        # Note: DeepSeek only supports n=1 (no parallel completions)
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("LLM_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0.0,
            timeout=60,  # Increased timeout for complex evaluations
            max_retries=3,  # More retries for reliability
        )
        ragas_llm = LangchainLLMWrapper(llm)

        # Configure embeddings: local HuggingFace model
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

        # Select metrics based on whether we have reference
        # Configure answer_relevancy with strictness=1 to avoid n>1 API calls (DeepSeek doesn't support n>1)
        answer_relevancy_configured = AnswerRelevancy(strictness=1)

        if reference:
            # Full evaluation with reference
            metrics_to_use = [faithfulness, answer_relevancy_configured, context_precision, context_recall]
            logger.info("Using all RAGas metrics (with reference)")
        else:
            # Basic evaluation without reference
            metrics_to_use = [faithfulness, answer_relevancy_configured]
            logger.info("Using basic RAGas metrics (no reference)")

        # Run RAGas evaluation with DeepSeek LLM and local embeddings
        logger.info("Starting RAGas evaluation...")
        result = evaluate(
            dataset,
            metrics=metrics_to_use,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        # Build results dictionary
        # Handle both scalar and list returns from RAGAS (different versions return different formats)
        def extract_metric(value):
            """Extract float from RAGAS metric (handles both list and scalar formats)"""
            if isinstance(value, list):
                return float(value[0]) if len(value) > 0 else 0.0
            return float(value)

        results = {
            "ragas_faithfulness": extract_metric(result["faithfulness"]),
            "ragas_answer_relevancy": extract_metric(result["answer_relevancy"]),
            "ragas_error": None,  # No error
        }

        # Add context metrics if available
        if reference:
            results["ragas_context_precision"] = extract_metric(result["context_precision"])
            results["ragas_context_recall"] = extract_metric(result["context_recall"])

        logger.info(f"✅ RAGas evaluation successful: faithfulness={results['ragas_faithfulness']:.3f}, relevancy={results['ragas_answer_relevancy']:.3f}")
        return results

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"❌ RAGas evaluation failed: {error_msg}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Return error info instead of just zeros
        base_results = {
            "ragas_faithfulness": 0.0,
            "ragas_answer_relevancy": 0.0,
            "ragas_error": error_msg,  # Store error message
        }
        if reference:
            base_results["ragas_context_precision"] = 0.0
            base_results["ragas_context_recall"] = 0.0
        return base_results

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

def evaluate_answer(question: str, answer: str, context: str, reference: str = None) -> Dict[str, float]:
    """
    Comprehensive evaluation combining:
    - RAGas metrics (PRIMARY): faithfulness, answer_relevancy, [context_precision, context_recall if reference available]
    - LLM-as-Judge (SECONDARY): relevance, faithfulness (1-5 scale)

    Args:
        question: The user question
        answer: The generated answer
        context: The retrieved context
        reference: Ground truth answer (optional, enables additional metrics)
    """
    metrics = {}

    # 1. RAGas metrics (PRIMARY)
    ragas_metrics = evaluate_with_ragas(question, answer, context, reference)
    metrics.update(ragas_metrics)

    # 2. LLM-as-Judge (SECONDARY)
    llm_metrics = evaluate_answer_llm(question, answer, context)
    metrics.update(llm_metrics)

    return metrics

# ---------------------- Pipelines --------------------
def run_rag(mode: str, question: str, df: pd.DataFrame, index: faiss.Index, metadata: List[Any], top_k: int, final_k: int, reference: str = None) -> Tuple[str, Dict]:
    """
    Run RAG pipeline with evaluation.

    Args:
        mode: RAG mode (v1, v2, v3)
        question: User question
        df: Knowledge base dataframe
        index: FAISS index
        metadata: FAISS metadata
        top_k: Number of candidates to retrieve
        final_k: Number of final contexts to use
        reference: Ground truth answer (optional, for enhanced metrics)
    """
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

    # 5. Evaluate (Metric) - pass reference if available
    metrics = evaluate_answer(question, ans, ctx, reference)

    return ans, metrics

# ---------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG System for Financial Literacy Q&A")
    parser.add_argument("--mode", choices=["v1", "v2", "v3"], default="v2",
                        help="RAG mode: v1=doc-level, v2=chunk, v3=chunk+rerank")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions to process")
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

    # Load Questions (try with references first, fallback to basic questions)
    references_file = "./questions_with_references.csv"
    use_references = False

    if os.path.exists(references_file):
        logger.info(f"Loading questions with references from {references_file}")
        questions_df = pd.read_csv(references_file)
        use_references = True
        logger.info("✅ References available - will use context_precision and context_recall")
    else:
        logger.info("Loading questions from ./questions.csv (no references)")
        questions_df = pd.read_csv("./questions.csv")
        logger.info("⚠️  No references available - using basic metrics only")

    questions = questions_df["Вопрос"].tolist()
    references = questions_df["Референсный ответ"].tolist() if use_references else [None] * len(questions)

    if args.limit:
        questions = questions[:args.limit]
        references = references[:args.limit]

    answers = []
    all_metrics = []

    # Initialize accumulators for all metrics
    metric_keys = [
        # RAGas metrics (PRIMARY)
        "ragas_faithfulness", "ragas_answer_relevancy",
        # LLM-as-Judge (SECONDARY)
        "llm_relevance", "llm_faithfulness",
    ]

    # Add context metrics if using references
    if use_references:
        metric_keys.extend(["ragas_context_precision", "ragas_context_recall"])

    total_metrics = {k: 0.0 for k in metric_keys}

    for q, ref in tqdm(zip(questions, references), total=len(questions), desc="RAG Pipeline"):
        ans, metrics = run_rag(args.mode, q, df, index, metadata, top_k, final_k, reference=ref)
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
    logger.info("PRIMARY METRICS - RAGas (0-1 scale):")
    logger.info(f"   Faithfulness:       {total_metrics['ragas_faithfulness']/n:.4f}")
    logger.info(f"   Answer Relevancy:   {total_metrics['ragas_answer_relevancy']/n:.4f}")

    # Show context metrics if available
    if use_references:
        logger.info(f"   Context Precision:  {total_metrics['ragas_context_precision']/n:.4f}")
        logger.info(f"   Context Recall:     {total_metrics['ragas_context_recall']/n:.4f}")

    logger.info("")
    logger.info("SECONDARY METRICS - LLM-as-Judge (1-5 scale):")
    logger.info(f"   Relevance:    {total_metrics['llm_relevance']/n:.2f}")
    logger.info(f"   Faithfulness: {total_metrics['llm_faithfulness']/n:.2f}")
    logger.info("=" * 60)
    logger.info(f"Detailed metrics saved to: metrics_detailed.csv")

if __name__ == "__main__":
    main()
