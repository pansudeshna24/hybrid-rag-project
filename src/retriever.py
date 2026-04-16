import faiss
import numpy as np
import json
import os
from src.model import get_model

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 🔥 GLOBAL CACHE (VERY IMPORTANT)
index = None
texts = None


def load_data_once():
    global index, texts

    if index is None:
        index_path = os.path.join(BASE_DIR, "data", "index.faiss")
        chunk_path = os.path.join(BASE_DIR, "data", "chunks.json")

        print("✅ Loading FAISS + JSON (only once)...")

        index = faiss.read_index(index_path)

        with open(chunk_path, "r", encoding="utf-8") as f:
            texts = json.load(f)

    return index, texts


def rrf_fusion(dense_ids, sparse_ids, k=60):
    scores = {}

    for rank, idx in enumerate(dense_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

    for rank, idx in enumerate(sparse_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked


def retrieve(query, bm25):
    # 🔥 USE CACHED DATA
    index, texts = load_data_once()

    model = get_model()

    q_emb = model.encode(query)
    q_emb = np.array([q_emb]).astype("float32")

    # 🔥 VERY SMALL k (IMPORTANT FOR RENDER)
    D, I = index.search(q_emb, k=3)
    dense_ids = [i for i in I[0] if i < len(texts)]

    scores = bm25.get_scores(query.split())

    sparse_ids = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:3]

    fused = rrf_fusion(dense_ids, sparse_ids)

    results = []
    rrf_scores = []

    # 🔥 FINAL REDUCED OUTPUT
    for idx, score in fused[:3]:
        results.append(texts[idx])
        rrf_scores.append(score)

    return results, rrf_scores