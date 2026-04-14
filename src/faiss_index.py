import faiss
import numpy as np

def create_index():
    embeddings = np.load("data/embeddings.npy")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, "data/index.faiss")

    print("✅ FAISS index created")

if __name__ == "__main__":
    create_index()