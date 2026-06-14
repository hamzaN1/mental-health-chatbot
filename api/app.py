from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import hf_hub_download, InferenceClient
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("faiss_index", exist_ok=True)
hf_hub_download(repo_id="hamzaN1/mental-health-faiss", filename="index.faiss",
                repo_type="dataset", local_dir="faiss_index")
hf_hub_download(repo_id="hamzaN1/mental-health-faiss", filename="index.pkl",
                repo_type="dataset", local_dir="faiss_index")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3")

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
        raw_answer = context.split("Answer:")[1].strip()
    else:
        raw_answer = context

    prompt = f"""You are a compassionate mental health assistant. 
Using ONLY the information below, give a concise, warm, and direct answer in 2-3 sentences maximum.
Do not add anything not in the information below.

Information: {raw_answer}

Question: {q.question}
Answer:"""

    response = client.text_generation(prompt, max_new_tokens=150, temperature=0.5)
    return {"question": q.question, "answer": response.strip()}
