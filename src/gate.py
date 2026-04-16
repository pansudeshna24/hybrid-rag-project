from sentence_transformers import util
from src.model import get_model

def strict_gate(query, passages, threshold=0.5):
    if len(passages) == 0:
        return False

    model = get_model()

    texts = [p["text"] for p in passages]

    query_emb = model.encode(query, convert_to_tensor=True)
    passage_emb = model.encode(texts, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, passage_emb)[0]

    max_score = max(scores).item()

    print("🔍 Gate score:", max_score)

    return max_score >= threshold