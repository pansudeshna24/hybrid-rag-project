import re
from sentence_transformers import util
from src.model import get_model


def clean_text(text):
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'(http\S+|www\S+|\S+@\S+)', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def generate(query, context):

    model = get_model()

    sentences = context.split(". ")

    # 🔥 LIMIT SENTENCES (VERY IMPORTANT)
    sentences = sentences[:10]

    sentences = [clean_text(s) for s in sentences if len(s.split()) > 6]
    sentences = list(dict.fromkeys(sentences))

    if len(sentences) == 0:
        return "❌ No useful content found."

    query_emb = model.encode(query, convert_to_tensor=True)
    sent_emb = model.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, sent_emb)[0]

    ranked = sorted(
        [(sentences[i], scores[i].item()) for i in range(len(sentences))],
        key=lambda x: x[1],
        reverse=True
    )

    final_points = []
    seen = set()

    for sent, score in ranked:

        s = sent.lower()

        if any(x in s for x in [
            "call", "email", "visit", "scan", "click", "login", "contact"
        ]):
            continue

        short = " ".join(sent.split()[:15]).lower()

        if short in seen:
            continue

        seen.add(short)

        clean = " ".join(sent.split()[:15])
        final_points.append(clean)

        # 🔥 REDUCED
        if len(final_points) == 2:
            break

    if len(final_points) == 0:
        final_points = [s[0] for s in ranked[:2]]

    answer = "### Answer:\n\n"

    for point in final_points:
        answer += f"• {point.strip()}.\n\n"

    return answer