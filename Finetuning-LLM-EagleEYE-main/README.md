# ⚓ Maritime RAG Pipeline

## 🚀 Project Overview
This project implements a **Retrieval-Augmented Generation (RAG) pipeline** for maritime logistics data.  
It allows users to ask **natural language questions** (e.g., *“How many tanker vessels arrived at Klaipėda on 2021-01-10?”*) and get answers **grounded in structured CSV port traffic data**.

### 🔑 Features
- **Data ingestion & preprocessing** – cleans and chunks static maritime dataset (`final_data.csv`).
- **Vector database (FAISS)** – stores embeddings for efficient similarity search.
- **RAG pipeline** – combines retrieval + LLM for natural language answers.
- **Models used:**
  - `sentence-transformers/all-MiniLM-L6-v2` → embeddings  
  - `TinyLlama-1.1B-Chat` (local CPU) → question answering  
  - (Optional) GPT-4 or other API models can be plugged in.
- **Why RAG (not fine-tuning)?**  
  - Data is static but large → retrieval is enough.  
  - Fine-tuning (QLoRA) is scaffolded but optional, needs GPU.

---

## 🛠️ Setup Instructions (Windows 10/11)

### 1. Navigate to project folder
```powershell
cd C:\Users\ADMIN\OneDrive\Desktop\maritime_rag_pipeline

2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1


⚠️ If activation is blocked, run once:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

3. Install dependencies
pip install -r requirements.txt
pip install -U langchain-huggingface langchain-community

📊 Build the Vector Database (FAISS)

Convert your CSV (final_data.csv) into searchable embeddings:

python .\scripts\chunk_data.py --input ".\data\final_data.csv" --output .\vector_db --sample_rows 50000 --chunk_size 1000 --chunk_overlap 100 --embedding_model sentence-transformers/all-MiniLM-L6-v2


This will:

Load first 50k rows of your dataset

Split into chunks of 1000 rows (with overlap 100)

Create embeddings using MiniLM

Save FAISS index → .\vector_db\index.faiss + metadata

💬 Ask Questions (RAG QA)

Run natural-language queries:

python .\scripts\rag_qa.py --index .\vector_db --query "How many tanker vessels arrived at Klaipėda on 2021-01-10?" --top_k 2


Example output:

🔹 Loading FAISS index...
🔹 Loading model TinyLlama/TinyLlama-1.1B-Chat-v1.0...
🔹 Running query...

✅ Answer:
On 2021-01-10, 2 tanker vessels arrived at Klaipėda.
Sources: retrieved chunks from FAISS


Other examples:

python .\scripts\rag_qa.py --index .\vector_db --query "Which vessels stayed in Klaipėda longer than 48 hours?" --top_k 3
python .\scripts\rag_qa.py --index .\vector_db --query "What ports had the most tanker arrivals in January 2021?" --top_k 5

📂 Project Structure
maritime_rag_pipeline/
│
├── data/
│   └── final_data.csv         # your dataset
│
├── vector_db/                 # generated FAISS index & metadata
│
├── scripts/
│   ├── chunk_data.py          # build FAISS vector DB
│   ├── rag_qa.py              # retrieval + LLM QA
│   ├── evaluate_retrieval.py  # check recall@k
│   └── fine_tune_qlora.py     # optional QLoRA fine-tuning
│
├── requirements.txt           # dependencies
└── README.md                  # this guide

⚡ Notes for Recruiters

Tech stack: Python, Pandas, LangChain, FAISS, HuggingFace, TinyLlama.

What it does: Lets maritime analysts query vessel data in plain English.

Why it matters:

Detect port congestion, tanker arrivals, dwell times.

Real-world RAG application on structured data.

Scalability: Runs locally, can scale to cloud APIs with GPUs.

Extensibility: Can integrate CO₂ estimates, congestion thresholds, and trend analysis.
