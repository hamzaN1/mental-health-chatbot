from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import hf_hub_download
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download FAISS index from HF Hub at startup
os.makedirs("faiss_index", exist_ok=True)
hf_hub_download(repo_id="hamzaN1/mental-health-faiss", filename="index.faiss",
                repo_type="dataset", local_dir="faiss_index")
hf_hub_download(repo_id="hamzaN1/mental-health-faiss", filename="index.pkl",
                repo_type="dataset", local_dir="faiss_index")

# Load vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "Mental Health Chatbot API is running!"}

@app.post("/ask")
def ask(q: Question):
    results = vectorstore.similarity_search(q.question, k=1)
    context = results[0].page_content
    if "Answer:" in context:
        answer = context.split("Answer:")[1].strip()
    else:
        answer = context
    return {"question": q.question, "answer": answer}
