import faiss
import numpy as np
import json
import os
from src.model import get_model

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_data():
    index_path = os.path.join(BASE_DIR, "data", "index.faiss")
    chunk_path = os.path.join(BASE_DIR, "data", "chunks.json")

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
    index, texts = load_data()

    model = get_model()

    q_emb = model.encode(query)
    q_emb = np.array([q_emb]).astype("float32")

    # 🔥 REDUCED k
    D, I = index.search(q_emb, k=5)
    dense_ids = [i for i in I[0] if i < len(texts)]

    scores = bm25.get_scores(query.split())

    # 🔥 REDUCED
    sparse_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

    fused = rrf_fusion(dense_ids, sparse_ids)

    results = []
    rrf_scores = []

    # 🔥 REDUCED final
    for idx, score in fused[:5]:
        results.append(texts[idx])
        rrf_scores.append(score)

    return results, rrf_scores