from src.bm25 import load_bm25
from src.retriever import retrieve
from src.reranker import rerank
from src.gate import strict_gate
from src.generator import generate


def run(query):

    # 🔥 Handle multiple questions
    if "?" in query:
        parts = query.split("?")
        query = parts[0].strip()

    query_lower = query.lower()

    # 🔥 Intent Filter
    invalid_keywords = [
        "ceo", "founder", "owner", "who is",
        "name of", "person", "leader", "head"
    ]

    if any(word in query_lower for word in invalid_keywords):
        return {
            "answer": "❌ This type of information (person/CEO) is not available.",
            "chunks": [],
            "scores": [],
            "rrf_scores": [],
            "precision": 0,
            "hallucination": "HIGH"
        }

    bm25, _ = load_bm25()

    # 🔹 Step 1: Retrieval + RRF
    results, rrf_scores = retrieve(query, bm25)

    # 🔹 Step 2: Rerank
    ranked = rerank(query, results)

    passages = [r[0] for r in ranked]
    rerank_scores = [r[1] for r in ranked]

    if len(passages) == 0:
        return {
            "answer": "❌ No relevant documents found.",
            "chunks": [],
            "scores": [],
            "rrf_scores": [],
            "precision": 0,
            "hallucination": "HIGH"
        }

    max_score = max(rerank_scores)

    # 🔹 Step 3: Gate
    if not strict_gate(query, passages):

        if max_score < 0.5:
            return {
                "answer": "❌ Out of scope.",
                "chunks": [],
                "scores": [],
                "rrf_scores": [],
                "precision": 0,
                "hallucination": "HIGH"
            }

        context = "\n".join([p["text"] for p in passages])
        answer = generate(query, context)

        return {
            "answer": answer,
            "chunks": passages,
            "scores": rerank_scores,
            "rrf_scores": rrf_scores,
            "precision": round(max_score, 3),
            "hallucination": "MEDIUM"
        }

    # 🔹 Step 4: Generate
    context = "\n".join([p["text"] for p in passages])
    answer = generate(query, context)

    precision = sum(rerank_scores) / len(rerank_scores)

    hallucination = (
        "LOW" if precision > 0.6
        else "MEDIUM" if precision > 0.4
        else "HIGH"
    )

    return {
        "answer": answer,
        "chunks": passages,
        "scores": rerank_scores,
        "rrf_scores": rrf_scores,
        "precision": round(precision, 3),
        "hallucination": hallucination
    }