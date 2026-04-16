from sentence_transformers import util
from src.model import get_model

def rerank(query, passages):
    model = get_model()

    texts = [p["text"] for p in passages]

    query_emb = model.encode(query, convert_to_tensor=True)
    passage_emb = model.encode(texts, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, passage_emb)[0]

    ranked = sorted(
        zip(passages, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 🔥 REDUCED
    return [(r[0], float(r[1])) for r in ranked[:3]]