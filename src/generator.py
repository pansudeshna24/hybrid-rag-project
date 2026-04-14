import re
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    # remove numbers like 15 16 10
    text = re.sub(r'\b\d+\b', '', text)

    # remove emails/phones/urls
    text = re.sub(r'(http\S+|www\S+|\S+@\S+)', '', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def generate(query, context):

    sentences = context.split(". ")
    sentences = [clean_text(s) for s in sentences if len(s.split()) > 6]

    # remove duplicates
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

        # ❌ Remove noise
        if any(x in s for x in [
            "call", "email", "visit", "scan", "click", "login",
            "for any queries", "contact"
        ]):
            continue

        short = " ".join(sent.split()[:15]).lower()

        if short in seen:
            continue

        seen.add(short)

        # keep only meaningful length
        clean = " ".join(sent.split()[:18])

        final_points.append(clean)

        if len(final_points) == 4:
            break

    # fallback
    if len(final_points) == 0:
        final_points = [s[0] for s in ranked[:3]]

    # format nicely
    answer = "### Startup Benefits:\n\n"

    for point in final_points:
        answer += f"• {point.strip()}.\n\n"

    return answer