# 🧠 MindEase — Mental Health Chatbot

> A fine-tuned, RAG-enhanced AI chatbot for mental health Q&A. Built as a CCP assignment for Parallel and Distributed Computing.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue?style=flat-square)](https://hamzaN1.github.io/mental-health-chatbot)
[![API](https://img.shields.io/badge/API-HuggingFace%20Spaces-yellow?style=flat-square)](https://hamzaN1-mental-health-chatbot.hf.space)
[![Model](https://img.shields.io/badge/Model-HuggingFace%20Hub-orange?style=flat-square)](https://huggingface.co/hamzaN1/mental-health-chatbot)

---

## 📌 Overview

MindEase is an AI-powered chatbot trained to answer mental health questions accurately and compassionately. It uses a fine-tuned `google/flan-t5-base` model combined with Retrieval-Augmented Generation (RAG) to ground every response in a curated dataset of mental health FAQs.

The system retrieves the most semantically relevant Q&A from the dataset using FAISS vector search, then uses Mistral-7B to summarize and deliver a concise, human-friendly answer.

---

## 🗂️ Project Structure

```
mental-health-chatbot/
├── data/
│   └── Mental_Health_FAQ.csv        # Dataset (390+ Q&A pairs + greetings)
├── notebooks/
│   └── finetuning.ipynb             # Fine-tuning notebook (Google Colab)
├── rag/
│   ├── rag_pipeline.ipynb           # RAG pipeline notebook
│   ├── index.faiss                  # FAISS vector index
│   └── index.pkl                    # FAISS metadata
├── api/
│   ├── app.py                       # FastAPI backend
│   └── requirements.txt             # Python dependencies
├── index.html                       # Frontend landing page
├── Dockerfile                       # HuggingFace Spaces deployment
└── README.md
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Base Model | `google/flan-t5-base` |
| Fine-tuning | HuggingFace `transformers` + `trl` on Google Colab (T4 GPU) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| RAG Orchestration | LangChain |
| Backend | FastAPI + Uvicorn |
| Deployment | HuggingFace Spaces (Docker) |
| Frontend | HTML / CSS / JavaScript |
| Hosting | GitHub Pages |

---

## 🚀 How It Works

1. **User asks a question** via the frontend chat interface
2. **FAISS vector search** finds the most semantically similar Q&A from the dataset
3. **Mistral-7B** receives the retrieved context and generates a concise, focused answer
4. **Answer is returned** to the user in 2-3 sentences

---

## 📊 Dataset

- **Source:** [Mental Health FAQ — Kaggle](https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot)
- **Augmented with:** 250 additional mental health Q&A pairs + 40 conversational/greeting exchanges (synthetic data)
- **Total:** ~390 Q&A pairs

---

## 🔧 Fine-Tuning Details

- **Model:** `google/flan-t5-base`
- **Epochs:** 10
- **Batch size:** 8
- **Training loss:** 11.2 → 3.1 (across 10 epochs)
- **Hardware:** Google Colab T4 GPU (free tier)
- **Training time:** ~14 minutes

---

## 🌐 Deployment

| Service | URL |
|---|---|
| Frontend | https://hamzaN1.github.io/mental-health-chatbot |
| API | https://hamzaN1-mental-health-chatbot.hf.space |
| API Docs | https://hamzaN1-mental-health-chatbot.hf.space/docs |
| Model | https://huggingface.co/hamzaN1/mental-health-chatbot |
| FAISS Dataset | https://huggingface.co/datasets/hamzaN1/mental-health-faiss |

---

## ⚙️ Running Locally

```bash
# Clone the repo
git clone https://github.com/hamzaN1/mental-health-chatbot.git
cd mental-health-chatbot/api

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app:app --reload
```

API will be available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## 📁 API Reference

### `GET /`
Returns API status.

### `POST /ask`
```json
// Request
{ "question": "What is anxiety disorder?" }

// Response
{ "question": "What is anxiety disorder?", "answer": "..." }
```

---

## ⚠️ Disclaimer

MindEase is an academic project and is **not a substitute for professional mental health care**. If you or someone you know is in crisis, please contact a licensed mental health professional or a crisis helpline.

---

## 👤 Author

**Hamza Naveed Siddiqui**
Iqra University — BS Computer Science
[LinkedIn](https://linkedin.com/in/hamzanaveedsiddiqui) | [GitHub](https://github.com/hamzaN1)
