import fitz
import os
import json

DATASET_PATH = "dataset"

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

def process():
    all_chunks = []

    for file in os.listdir(DATASET_PATH):
        if file.endswith(".pdf"):
            path = os.path.join(DATASET_PATH, file)
            text = extract_text(path)
            chunks = chunk_text(text)

            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": file
                })

    os.makedirs("data", exist_ok=True)

    with open("data/chunks.json", "w") as f:
        json.dump(all_chunks, f)

    print("✅ Preprocessing Done")

if __name__ == "__main__":
    process()