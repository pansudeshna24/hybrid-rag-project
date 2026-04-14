import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_data():
    index_path = os.path.join(BASE_DIR, "data", "index.faiss")
    chunk_path = os.path.join(BASE_DIR, "data", "chunks.json")

    index = faiss.read_index(index_path)

    with open(chunk_path, "r", encoding="utf-8") as f:
        texts = json.load(f)

    return index, texts


# 🔥 NEW: RRF FUNCTION
def rrf_fusion(dense_ids, sparse_ids, k=60):
    scores = {}

    for rank, idx in enumerate(dense_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

    for rank, idx in enumerate(sparse_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked


def retrieve(query, bm25):
    index, texts = load_data()

    # 🔹 Dense Retrieval (FAISS)
    q_emb = model.encode(query)
    q_emb = np.array([q_emb]).astype("float32")

    # 🔥 CHANGE HERE (k=10 instead of 6)
    D, I = index.search(q_emb, k=10)
    dense_ids = [i for i in I[0] if i < len(texts)]

    # 🔹 Sparse Retrieval (BM25)
    scores = bm25.get_scores(query.split())

    # 🔥 CHANGE HERE (top 10 instead of 6)
    sparse_ids = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:10]

    # 🔥 RRF Fusion
    fused = rrf_fusion(dense_ids, sparse_ids)

    # 🔥 Convert to passages + RRF scores
    results = []
    rrf_scores = []

    for idx, score in fused[:8]:  # keep top 8 final
        results.append(texts[idx])
        rrf_scores.append(score)

    return results, rrf_scores