# 📄 RAG-Based Document Q&A System

## 🚀 Overview
This project is a Generative AI application that enables users to ask questions based on uploaded documents. It uses a Retrieval-Augmented Generation (RAG) pipeline to fetch relevant context and generate accurate answers.
## 🧠 Features
- Semantic search using embeddings
- Context-aware answers using LLM
- Efficient document retrieval

## 🛠️ Tech Stack
- Python
- LangChain
- OpenAI API
- FAISS
- HuggingFace Embeddings

## ⚙️ How It Works
1. Load document  
2. Convert into embeddings  
3. Store in vector database  
4. Retrieve relevant data  
5. Generate answer using LLM  

## ▶️ How to Run
```bash
pip install -r requirements.txt
python app.py
