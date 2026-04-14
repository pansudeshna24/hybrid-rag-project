import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings():
    with open("data/chunks.json") as f:
        data = json.load(f)

    embeddings = []

    for item in data:
        emb = model.encode(item["text"])
        embeddings.append(emb)

    np.save("data/embeddings.npy", embeddings)

    print("✅ Embeddings created")

if __name__ == "__main__":
    create_embeddings()