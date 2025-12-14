import os
import re
import time
import pickle
import logging
import argparse
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import faiss  # <<< NEW: Vector DB
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
EMBEDDING_DIM = 384  # Dimension for MiniLM-L12-v2

def get_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        logger.info("Loading local embedding model: paraphrase-multilingual-MiniLM-L12-v2")
        EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return EMBEDDING_MODEL

# ---------------------- Clients ----------------------
def get_llm_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.perplexity.ai",
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
            model="sonar",
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
        model="sonar",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    COST["llm_calls"] += 1
    return resp.choices[0].message.content

# ---------------------- LLM-as-a-Judge ----------------
def evaluate_answer(question: str, answer: str, context: str) -> Dict[str, int]:
    """
    Asks the LLM to rate the answer on scale 1-5.
    Returns: {'relevance': int, 'faithfulness': int}
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
            model="sonar",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = resp.choices[0].message.content
        # Extract JSON
        json_str = re.search(r'\{.*\}', content, re.DOTALL).group(0)
        return json.loads(json_str)
    except:
        return {"relevance": 0, "faithfulness": 0}

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["v1", "v2", "v3"], default="v2")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Config based on mode
    top_k = 20 if args.mode == "v3" else 5
    final_k = 3

    # Load Data
    df = load_knowledge_base("./train_data.csv")
    
    # Build FAISS (Vector DB)
    index_mode = "v1" if args.mode == "v1" else "v2"
    index_file = f"faiss_index_{index_mode}.bin"
    meta_file = f"faiss_meta_{index_mode}.pkl"
    
    index, metadata = build_faiss_index(df, index_file, meta_file, mode=index_mode)

    # Load Questions
    questions_df = pd.read_csv("./questions.csv")
    questions = questions_df["Вопрос"].tolist()
    if args.limit: questions = questions[:args.limit]

    answers = []
    total_metrics = {"relevance": 0, "faithfulness": 0}
    
    for q in tqdm(questions, desc="RAG Pipeline"):
        ans, metrics = run_rag(args.mode, q, df, index, metadata, top_k, final_k)
        answers.append(ans)
        
        # Accumulate metrics
        total_metrics["relevance"] += metrics["relevance"]
        total_metrics["faithfulness"] += metrics["faithfulness"]

    # Save Results
    questions_df = questions_df.iloc[:len(answers)]
    questions_df["Ответы"] = answers
    questions_df.to_csv("submission.csv", index=False)
    
    # Report Metrics
    n = len(answers)
    logger.info("="*40)
    logger.info(f"RESULTS (Avg Score 1-5)")
    logger.info(f"Relevance: {total_metrics['relevance']/n:.2f}")
    logger.info(f"Faithfulness: {total_metrics['faithfulness']/n:.2f}")
    logger.info("="*40)

if __name__ == "__main__":
    main()
