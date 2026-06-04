# MediRAG: AI-Powered Clinical Mental Health Assistant

MediRAG is a robust, privacy-focused conversational AI agent designed to provide mental health support by grounding its responses in verified clinical psychology workbooks and guidelines. Using an advanced Retrieval-Augmented Generation (RAG) pipeline and stateful memory, MediRAG tracks longitudinal symptom data and offers evidence-based coping strategies.

## Features

- **Stateful AI Agent**: Engineered with **LangGraph**, the agent maintains context across sessions and securely tracks emotional states (mood, anxiety, sleep) using **Neon PostgreSQL** for persistent memory.
- **Advanced Hybrid RAG Pipeline**: Built on **ChromaDB** to index clinical psychology PDFs parsed via **PyMuPDF**. It utilizes a hybrid retrieval approach combining dense vector embeddings (`HuggingFace MiniLM`) and sparse keyword matching (`BM25Okapi`), followed by **Cross-Encoder** semantic re-ranking for extreme precision.
- **Automated Clinical Data Extraction**: Uses **LangChain** tool calling and **Pydantic** to dynamically parse unstructured conversational input into structured clinical metrics (severity, duration, appetite, etc.), logging the data securely.
- **LLM-as-a-Judge Evaluation**: Features a custom benchmarking suite (inspired by Ragas) to evaluate pipeline performance automatically. It achieves mathematically verified, robust metrics of **0.88 Faithfulness** and **0.86 Answer Relevancy** across complex synthetic queries.
- **Full-Stack Implementation**: A modern, responsive **React (Vite)** frontend communicating with an asynchronous **FastAPI** backend, load-tested with **Locust** for concurrency.
- **Privacy First**: User authentication is secured with **bcrypt**, and vector database namespaces are strictly segregated to ensure isolation of patient sessions and clinical data.

## Tech Stack

- **Backend**: Python, FastAPI, LangChain, LangGraph, Pydantic
- **Frontend**: React, Vite, Vanilla CSS
- **Database**: Neon (Serverless PostgreSQL), ChromaDB (Vector Store)
- **AI/ML**: OpenRouter (LLM Routing), HuggingFace Sentence-Transformers, Cross-Encoders, BM25
- **Testing**: Locust (Load Testing), Custom LLM Benchmarking Suite

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js & npm
- An OpenRouter API Key
- A Neon PostgreSQL Database DSN

### 1. Backend Setup

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/ranjansanskar/MediRAG.git
cd MediRAG

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (ensure you have the required ML libraries)
pip install -r requirements.txt
```

Set up your `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_api_key_here
NEON_DSN=postgresql://user:password@ep-xxxx.region.aws.neon.tech/dbname?sslmode=require
```

Start the FastAPI backend server:
```bash
uvicorn backend_server:app --reload
```

### 2. Frontend Setup

In a new terminal window, start the Vite development server:
```bash
cd frontend  # or run directly in root if Vite is configured there
npm install
npm run dev
```

Navigate to `http://localhost:5173/` in your browser to interact with the MediRAG assistant!

## Automated Benchmarking

To run the custom LLM-as-a-judge evaluation suite (ensure the backend is stopped as ChromaDB requires exclusive file access):

```bash
PYTHONUNBUFFERED=1 python3 benchmarks/benchmark_llm_judge.py
```
*Note: The script contains deliberate 15-second API delays to bypass free-tier rate limits, taking approximately 6-8 minutes to complete a 20-query evaluation.*
