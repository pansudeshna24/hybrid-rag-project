from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline import run

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Hybrid RAG API is running 🚀"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    result = run(request.query)
    return result