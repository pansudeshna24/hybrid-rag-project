from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def strict_gate(query, passages, threshold=0.55):
    if len(passages) == 0:
        return False

    texts = [p["text"] for p in passages]

    query_emb = model.encode(query, convert_to_tensor=True)
    passage_emb = model.encode(texts, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, passage_emb)[0]

    max_score = max(scores).item()

    print("🔍 Gate score:", max_score)

    # 🔥 STRICT threshold
    if max_score < threshold:
        return False

    return True