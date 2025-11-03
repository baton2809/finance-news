import os
import re
import time
import pickle
import logging
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------- Env --------------------------
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

# ---------------------- Cost tracker -----------------
COST = {
    "embedding_calls": 0,
    "llm_calls": 0,
    "embedding_cost": 0.0,     # rough estimate
    "llm_cost": 0.0,           # rough estimate
    "embedding_tokens": 0,     # <<< NEW: token counters
    "llm_prompt_tokens": 0,
}
# price assumptions (adjust if you know exact tariffs)
PRICE_PER_1K_TOK_EMB = 0.00002   # text-embedding-3-small (approx)
PRICE_PER_1K_TOK_LLM = 0.0004    # llama-3-70b-instruct (approx)

# ---------------------- Clients ----------------------
def get_llm_client() -> OpenAI:
    return OpenAI(base_url="https://ai-for-finance-hack.up.railway.app/", api_key=LLM_API_KEY)

def get_emb_client() -> OpenAI:
    return OpenAI(base_url="https://ai-for-finance-hack.up.railway.app/", api_key=EMBEDDER_API_KEY)

# ---------------------- Utils ------------------------
def approx_tokens(s: str) -> int:
    # ~4 chars per token (very rough)
    return max(1, len(s) // 4)

def chunk_text(text: str, chunk_size_chars: int = 1800, overlap_chars: int = 200) -> List[str]:
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
    logger.info(f"Загрузка базы знаний из {path}")
    df = pd.read_csv(path, sep=",", encoding="utf-8")
    logger.info(f"Загружено {len(df)} документов")
    return df

# ---------------------- Embeddings -------------------
def get_embedding(text: str) -> List[float]:
    client = get_emb_client()
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    # cost est.
    t = approx_tokens(text)
    COST["embedding_calls"] += 1
    COST["embedding_tokens"] += t
    COST["embedding_cost"] += PRICE_PER_1K_TOK_EMB * (t / 1000.0)
    return resp.data[0].embedding

def build_doc_cache(df: pd.DataFrame, cache_file: str = "embeddings_cache_doc.pkl") -> np.ndarray:
    if os.path.exists(cache_file):
        logger.info(f"Загрузка эмбеддингов (документы) из кэша: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    logger.info("Создание эмбеддингов на уровне ДОКУМЕНТОВ (v1)...")
    embs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Эмбеддинги док-уровня"):
        combined = f"{row['annotation']} {row['text'][:1000]}"
        embs.append(get_embedding(combined))
    arr = np.array(embs)
    with open(cache_file, "wb") as f:
        pickle.dump(arr, f)
    return arr

def build_chunk_cache(df: pd.DataFrame, cache_file: str = "embeddings_cache_chunks.pkl") -> Dict[str, Any]:
    if os.path.exists(cache_file):
        logger.info(f"Загрузка эмбеддингов (чанки) из кэша: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    logger.info("Создание эмбеддингов на уровне ЧАНКОВ (v2/v3)...")
    all_embs = []
    index = []  # list of (doc_idx, chunk_idx)
    for doc_idx, row in tqdm(df.iterrows(), total=len(df), desc="Эмбеддинги чанков"):
        combined = f"{row['annotation']} {row['text']}"
        chunks = chunk_text(combined, chunk_size_chars=1800, overlap_chars=200)
        for ci, ch in enumerate(chunks):
            all_embs.append(get_embedding(ch))
            index.append((int(doc_idx), int(ci)))
    arr = np.array(all_embs)
    data = {"embeddings": arr, "index": index}
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    return data

# ---------------------- Retrieval --------------------
def retrieve_v1(question: str, df: pd.DataFrame, doc_embs: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    q = np.array(get_embedding(question)).reshape(1, -1)
    sims = cosine_similarity(q, doc_embs)[0]
    top = np.argsort(sims)[-top_k:][::-1]
    return [(int(i), float(sims[i])) for i in top]

def retrieve_v2(question: str, chunk_cache: Dict[str, Any], top_k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
    embs = chunk_cache["embeddings"]
    idx = chunk_cache["index"]
    q = np.array(get_embedding(question)).reshape(1, -1)
    sims = cosine_similarity(q, embs)[0]
    top = np.argsort(sims)[-top_k:][::-1]
    return [((idx[int(i)][0], idx[int(i)][1]), float(sims[i])) for i in top]

# ---------------------- Re-ranking (LLM) -------------
def llm_rerank(question: str, candidates: List[Tuple[Tuple[int, int], float]], df: pd.DataFrame,
               k_final: int = 3) -> List[Tuple[Tuple[int, int], float]]:
    """
    candidates: [ ((doc_idx, chunk_idx), cosine_score) ]
    Return: top k_final candidates re-ordered by LLM.
    Fallback: cosine order if parsing fails.
    """
    # Build short previews for LLM
    previews = []
    for (doc_idx, chunk_idx), _ in candidates:
        row = df.iloc[doc_idx]
        base = f"{row['annotation']} {row['text']}"
        chunks = chunk_text(base, chunk_size_chars=1800, overlap_chars=200)
        preview = chunks[chunk_idx][:400].replace("\n", " ")
        previews.append(preview)

    prompt = (
        "Ты — ассистент по извлечению фактов. "
        "Дан вопрос и список коротких фрагментов. "
        "Выбери ТРИ наиболее релевантных фрагмента (0-based индексы) и верни только их индексы через запятую.\n\n"
        f"Вопрос: {question}\n\n"
        "Фрагменты:\n" +
        "\n".join([f"[{i}] {p}" for i, p in enumerate(previews)]) +
        "\n\nОтвет: "
    )

    client = get_llm_client()
    resp = client.chat.completions.create(
        model="openrouter/meta-llama/llama-3-70b-instruct",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=0.1,
        max_tokens=16,
    )
    t = approx_tokens(prompt)
    COST["llm_calls"] += 1
    COST["llm_prompt_tokens"] += t
    COST["llm_cost"] += PRICE_PER_1K_TOK_LLM * (t / 1000.0)

    text = resp.choices[0].message.content
    idxs = [int(x) for x in re.findall(r"\d+", text)]
    if not idxs:
        return candidates[:k_final]
    idxs = [i for i in idxs if 0 <= i < len(candidates)]
    if not idxs:
        return candidates[:k_final]

    picked = [candidates[i] for i in idxs[:k_final]]
    if len(picked) < k_final:
        for c in candidates:
            if c not in picked:
                picked.append(c)
            if len(picked) == k_final:
                break
    return picked[:k_final]

# ---------------------- Context & Answer -------------
def build_context_v1(doc_indices: List[Tuple[int, float]], df: pd.DataFrame, max_tokens: int = 3000) -> str:
    parts, tokens = [], 0
    for doc_idx, score in doc_indices:
        row = df.iloc[doc_idx]
        block = (
            f"Документ {doc_idx} (релевантность: {score:.3f}):\n"
            f"Аннотация: {row['annotation']}\n"
            f"Содержание: {row['text']}\n"
            + "-" * 80 + "\n\n"
        )
        t = approx_tokens(block)
        if tokens + t > max_tokens: break
        parts.append(block); tokens += t
    return "".join(parts)

def build_context_v2_v3(chunk_indices: List[Tuple[Tuple[int, int], float]], df: pd.DataFrame,
                        max_tokens: int = 3000) -> str:
    parts, tokens = [], 0
    for (doc_idx, chunk_idx), score in chunk_indices:
        row = df.iloc[doc_idx]
        base = f"{row['annotation']} {row['text']}"
        chunks = chunk_text(base, chunk_size_chars=1800, overlap_chars=200)
        chunk_text_str = chunks[chunk_idx]
        block = (
            f"Документ {doc_idx}, фрагмент {chunk_idx} (релевантность: {score:.3f}):\n"
            f"Аннотация: {row['annotation']}\n"
            f"Фрагмент: {chunk_text_str}\n"
            + "-" * 80 + "\n\n"
        )
        t = approx_tokens(block)
        if tokens + t > max_tokens: break
        parts.append(block); tokens += t
    return "".join(parts)

def generate_answer(question: str, context: str) -> str:
    client = get_llm_client()
    if context:
        prompt = (
            "Ты - эксперт по финансовой грамотности. Используй контекст для ответа на вопрос.\n\n"
            "Правила:\n"
            "1) Отвечай кратко и точно.\n"
            "2) Не выдумывай фактов вне контекста; если данных не хватает — укажи, чего не хватает.\n"
            "3) В конце дай короткий 'Вывод: ...'.\n\n"
            f"КОНТЕКСТ:\n{context}\n\n"
            f"ВОПРОС: {question}\n"
        )
    else:
        prompt = f"Ответь на вопрос: {question}"

    resp = client.chat.completions.create(
        model="openrouter/meta-llama/llama-3-70b-instruct",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=0.2,
        max_tokens=800,
    )
    t = approx_tokens(prompt)
    COST["llm_calls"] += 1
    COST["llm_prompt_tokens"] += t
    COST["llm_cost"] += PRICE_PER_1K_TOK_LLM * (t / 1000.0)
    return resp.choices[0].message.content

# ---------------------- Pipelines --------------------
def answer_with_rag_v1(question: str, df: pd.DataFrame, doc_embs: np.ndarray, top_k: int = 3) -> str:
    hits = retrieve_v1(question, df, doc_embs, top_k=top_k)
    ctx = build_context_v1(hits, df)
    ans = generate_answer(question, ctx)
    sources = "; ".join([f"{i}" for i, _ in hits])
    return f"{ans}\n\nИсточники: {sources}"

def answer_with_rag_v2(question: str, df: pd.DataFrame, chunk_cache: Dict[str, Any], top_k: int = 5, k_final: int = 3) -> str:
    hits = retrieve_v2(question, chunk_cache, top_k=top_k)
    top = hits[:k_final]
    ctx = build_context_v2_v3(top, df)
    ans = generate_answer(question, ctx)
    sources = "; ".join([f"{d}-{c}" for (d, c), _ in top])
    return f"{ans}\n\nИсточники: {sources}"

def answer_with_rag_v3(question: str, df: pd.DataFrame, chunk_cache: Dict[str, Any], top_k: int = 8, k_final: int = 3) -> str:
    hits = retrieve_v2(question, chunk_cache, top_k=top_k)
    top = llm_rerank(question, hits, df, k_final=k_final)
    ctx = build_context_v2_v3(top, df)
    ans = generate_answer(question, ctx)
    sources = "; ".join([f"{d}-{c}" for (d, c), _ in top])
    return f"{ans}\n\nИсточники: {sources}"

# ---------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["v1", "v2", "v3"], default="v2",
                        help="v1: doc-level; v2: chunking; v3: chunking + LLM rerank")
    parser.add_argument("--top_k", type=int, default=None, help="retrieval pool")
    parser.add_argument("--final_k", type=int, default=None, help="context chunks")
    parser.add_argument("--limit", type=int, default=None, help="limit # of questions for dev")
    args = parser.parse_args()

    logger.info(f"Режим: {args.mode}")

    # Fixed params per mode (we also respect explicit flags)
    if args.mode == "v1":
        retrieval_top_k = args.top_k or 3
        final_k = None
    elif args.mode == "v2":
        retrieval_top_k = args.top_k or 5
        final_k = args.final_k or 3
    else:  # v3
        retrieval_top_k = args.top_k or 8
        final_k = args.final_k or 3

    df = load_knowledge_base("./train_data.csv")

    # Build/load cache(s)
    if args.mode == "v1":
        doc_embs = build_doc_cache(df, cache_file="embeddings_cache_doc.pkl")
    else:
        chunk_cache = build_chunk_cache(df, cache_file="embeddings_cache_chunks.pkl")

    # Load questions
    questions = pd.read_csv("./questions.csv", sep=",")
    if "Вопрос" in questions.columns:
        qcol = "Вопрос"
    elif "question" in questions.columns:
        qcol = "question"
    else:
        raise KeyError(f"Expected 'Вопрос' or 'question' column, got: {list(questions.columns)}")

    questions_list = questions[qcol].tolist()
    if args.limit:
        questions_list = questions_list[: args.limit]

    answers = []
    t0 = time.time()  # <<< NEW: timer start
    for i, q in enumerate(tqdm(questions_list, desc="Генерация ответов")):
        logger.info(f"[{i+1}/{len(questions_list)}] Вопрос: {q[:80]}...")
        if args.mode == "v1":
            ans = answer_with_rag_v1(q, df, doc_embs, top_k=retrieval_top_k)
        elif args.mode == "v2":
            ans = answer_with_rag_v2(q, df, chunk_cache, top_k=retrieval_top_k, k_final=final_k)
        else:  # v3
            ans = answer_with_rag_v3(q, df, chunk_cache, top_k=retrieval_top_k, k_final=final_k)
        answers.append(ans)
    elapsed = time.time() - t0  # <<< NEW: timer end

    # Save answers (pad if partial)
    if len(answers) < len(questions):
        logger.warning(f"⚠️ Only {len(answers)} of {len(questions)} questions answered. Filling missing with empty strings.")
        answers += [""] * (len(questions) - len(answers))

    questions["Ответы на вопрос"] = answers
    questions.to_csv("submission.csv", index=False, sep=",")

    # ---- Performance summary & table row ----
    total_cost = COST["embedding_cost"] + COST["llm_cost"]
    n_proc = len(answers) if len(answers) <= len(questions) else len(questions)
    sec_per_q = elapsed / max(1, len(questions_list))

    # Log pretty summary
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Questions processed: {len(questions_list)}")
    logger.info(f"Elapsed: {elapsed:.2f} sec | {sec_per_q:.2f} sec/question")
    logger.info(f"Embedding calls: {COST['embedding_calls']}, tokens≈{COST['embedding_tokens']}, cost ≈ ${COST['embedding_cost']:.4f}")
    logger.info(f"LLM calls:       {COST['llm_calls']}, prompt_tokens≈{COST['llm_prompt_tokens']}, cost ≈ ${COST['llm_cost']:.4f}")
    logger.info(f"TOTAL cost ≈ ${total_cost:.4f}")
    logger.info(f"Retrieval top_k: {retrieval_top_k} | Final_k: {final_k}")
    logger.info("=" * 60)
    logger.info("Готово! submission.csv создан.")

    # Append a row to perf_metrics.csv (create if missing)
    perf_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "questions_processed": len(questions_list),
        "elapsed_sec": round(elapsed, 2),
        "sec_per_question": round(sec_per_q, 2),
        "retrieval_top_k": retrieval_top_k,
        "final_k": final_k if final_k is not None else "",
        "embedding_calls": COST["embedding_calls"],
        "embedding_tokens": COST["embedding_tokens"],
        "embedding_cost_usd": round(COST["embedding_cost"], 6),
        "llm_calls": COST["llm_calls"],
        "llm_prompt_tokens": COST["llm_prompt_tokens"],
        "llm_cost_usd": round(COST["llm_cost"], 6),
        "total_cost_usd": round(total_cost, 6),
    }
    perf_path = "perf_metrics.csv"
    if os.path.exists(perf_path):
        pd.DataFrame([perf_row]).to_csv(perf_path, mode="a", header=False, index=False)
    else:
        pd.DataFrame([perf_row]).to_csv(perf_path, index=False)

if __name__ == "__main__":
    main()
