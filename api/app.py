from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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