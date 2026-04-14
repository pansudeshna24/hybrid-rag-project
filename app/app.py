import streamlit as st
import requests

# 🔹 API URL (local)
API_URL = "http://127.0.0.1:8000/ask"

st.title("🔍 Hybrid RAG with Evaluation")

query = st.text_input("Ask a question")

if st.button("Search"):

    if not query.strip():
        st.warning("⚠️ Please enter a question")
    else:
        try:
            # 🔥 Call FastAPI backend
            response = requests.post(
                API_URL,
                json={"query": query}
            )

            # Convert response to JSON
            result = response.json()

            # ✅ Answer
            st.subheader("📌 Answer")
            st.write(result.get("answer", "No answer found"))

            # ✅ Metrics
            st.subheader("📊 Evaluation")
            st.write(f"Precision Score: {result.get('precision', 0)}")
            st.write(f"Hallucination Risk: {result.get('hallucination', 'N/A')}")

            # ✅ Chunks
            st.subheader("📄 Retrieved Chunks")

            chunks = result.get("chunks", [])
            scores = result.get("scores", [])
            rrf_scores = result.get("rrf_scores", [])

            for i, (chunk, score, rrf) in enumerate(
                zip(chunks, scores, rrf_scores)
            ):
                st.markdown(f"""
                **Rank {i+1}**
                - 🔷 RRF Score: {round(rrf,4)}
                - 🔶 Rerank Score: {round(score,4)}
                """)
                st.write(chunk["text"][:500])
                st.write(f"Source: {chunk['source']}")
                st.markdown("---")

        except Exception as e:
            st.error(f"❌ Error connecting to backend: {e}")