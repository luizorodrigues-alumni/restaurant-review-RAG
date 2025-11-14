# 🍕 Restaurant Reviews RAG — LangChain + Ollama + Chroma

This project is a **lightweight Retrieval-Augmented Generation (RAG) system** built with **LangChain**, **Ollama**, and **ChromaDB**.  
It allows users to ask questions about a pizza restaurant, and the model answers **based solely on real customer reviews** stored in a CSV file.

The system embeds the reviews, stores them in a local vector database, retrieves the most relevant ones, and generates an answer using an LLM running locally via **Ollama**—no API keys from OpenAI or Google needed.

---

## 🚀 Features

- **Local LLM inference** using Ollama (`llama3:8b`)
- **Local embeddings** using `mxbai-embed-large`
- **Vector store** powered by ChromaDB
- **RAG pipeline** fully built with LangChain
- Answers forced to be **in Brazilian Portuguese**
- If the model does not know the answer, it responds honestly
- Simple CLI for asking questions

---

## 🧠 How It Works

1. Load restaurant reviews from a CSV file  
2. Embed each review using **OllamaEmbeddings**
3. Store embeddings in a **Chroma** vector database
4. Retrieve the top-k most similar reviews based on the user question
5. Build a structured LangChain prompt
6. Generate answers using **Llama 3 (8B)** running locally

This creates a faithful RAG setup where the LLM answers **only** using the retrieved content.

## 📦 Requirements

- Python 3.10+
- Ollama installed locally  
  https://ollama.com/download
- Models used (any one of your preference will do) :
  ```bash
  ollama pull llama3:8b
  ollama pull mxbai-embed-large
  ```
- Python dependencies:
  ```bash
   pip install langchain langchain-core langchain-ollama langchain-chroma pandas python-dotenv
  ```
## ⚙️ Environment Variables

- Create a .env file in the project root:
  ```ini
  BASE_URL=http://localhost:11434
  ```
## ▶️ Running the Project

- Run the script:
  ```bash
  python main.py
  ```
- Then ask your question:
  ```bash
   Is the restaurant good for families?
  ```


