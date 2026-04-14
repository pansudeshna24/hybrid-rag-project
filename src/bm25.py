import json
from rank_bm25 import BM25Okapi

def load_bm25():
    with open("data/chunks.json") as f:
        data = json.load(f)

    corpus = [d["text"].split() for d in data]
    bm25 = BM25Okapi(corpus)

    return bm25, data