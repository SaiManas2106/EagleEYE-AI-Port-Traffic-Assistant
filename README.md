# Maritime RAG Pipeline — EagleEYE

A **Retrieval-Augmented Generation (RAG)** pipeline for maritime port traffic analysis. Ask natural-language questions about vessel arrivals, dwell times, and port activity — and get answers grounded in structured maritime CSV data.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Build the Vector Database](#1-build-the-vector-database)
  - [2. Run RAG Question Answering](#2-run-rag-question-answering)
  - [3. Generate Synthetic Training Data](#3-generate-synthetic-training-data)
  - [4. Fine-Tune with QLoRA (Optional)](#4-fine-tune-with-qlora-optional)
  - [5. Evaluate Retrieval Performance](#5-evaluate-retrieval-performance)
  - [6. Evaluate LLM with Fine-Tuned Adapter](#6-evaluate-llm-with-fine-tuned-adapter)
- [Why RAG Instead of Pure Fine-Tuning?](#why-rag-instead-of-pure-fine-tuning)
- [Evaluation](#evaluation)
- [Scalability & Extensibility](#scalability--extensibility)
- [Requirements](#requirements)
- [License](#license)

---

## Overview

This project implements an end-to-end RAG pipeline on top of Baltic Sea port traffic data. Users can query the system in plain English, such as:

- *"How many tanker vessels arrived at Klaipėda on 2021-01-10?"*
- *"Which vessels stayed in Ventspils longer than 48 hours?"*
- *"What ports had the most tanker arrivals in January 2021?"*

The system retrieves the most relevant data chunks from a FAISS vector store and passes them as context to a local language model (TinyLlama) to generate a grounded answer.

---

## Features

- **CSV Data Ingestion** — Loads and preprocesses the maritime vessel traffic dataset (`final_data.csv`), converting rows into searchable plain-text representations.
- **FAISS Vector Database** — Embeds data chunks using `sentence-transformers/all-MiniLM-L6-v2` and stores them for efficient similarity search.
- **RAG Question Answering** — Retrieves the top-K relevant chunks and passes them with the question to a local LLM for answer generation.
- **Synthetic Instruction Generation** — Automatically generates instruction–response pairs from the dataset for supervised fine-tuning.
- **QLoRA Fine-Tuning** — Optional parameter-efficient fine-tuning using 4-bit quantization and LoRA adapters (requires GPU).
- **Retrieval Evaluation** — Measures Recall@K for the FAISS retrieval layer against a labeled eval set.
- **LLM Evaluation** — Tests the fine-tuned model with retrieved context and computes exact-match accuracy.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Data Processing | Pandas |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (via `langchain-community`) |
| LLM (local / CPU) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| LLM (fine-tuning base) | `meta-llama/Llama-3-8b-instruct` (or any HuggingFace causal LM) |
| Fine-Tuning | QLoRA via PEFT + `bitsandbytes` |
| Training Framework | HuggingFace Transformers + Trainer API |
| Evaluation | Custom recall and exact-match metrics |

---

## Project Structure

```
Finetuning-LLM-EagleEYE-main/
│
├── final_merged_dataset_cleaned.csv       # Full raw/merged dataset
│
└── maritime_rag_pipeline/
    │
    ├── data/
    │   └── final_data.csv                 # Cleaned dataset used for indexing
    │
    ├── vector_db/                         # Full FAISS index (50k rows)
    │   ├── index.faiss
    │   └── index.pkl
    │
    ├── vector_db_small/                   # Smaller FAISS index for quick testing
    │   ├── index.faiss
    │   └── index.pkl
    │
    ├── scripts/
    │   ├── chunk_data.py                  # Build FAISS vector database from CSV
    │   ├── rag_qa.py                      # Run RAG question-answering
    │   ├── generate_instructions.py       # Generate synthetic instruction-response pairs
    │   ├── fine_tune_qlora.py             # QLoRA fine-tuning (GPU required)
    │   ├── evaluate_retrieval.py          # Evaluate FAISS retrieval (Recall@K)
    │   ├── evaluate_llm.py               # Evaluate fine-tuned LLM on eval set
    │   └── test_rag.py                    # Quick smoke test for RAG pipeline
    │
    ├── training_data/
    │   └── eval_questions.json            # Labeled evaluation questions
    │
    └── requirements.txt                   # Python dependencies
```

---

## Dataset

The dataset (`final_data.csv`) contains maritime vessel port traffic records for Baltic Sea ports in 2021. Each row represents a vessel port call with the following key columns:

| Column | Description |
|---|---|
| `portID` | Unique port identifier |
| `portName` | Name of the port (e.g., VENTSPILS, KLAIPĖDA) |
| `portLocode` | UN/LOCODE of the port |
| `portArrival` | Timestamp of vessel arrival |
| `portDeparture` | Timestamp of vessel departure |
| `vesselMMSI` | Maritime Mobile Service Identity number |
| `vesselIMO` | IMO vessel number |
| `vesselName` | Name of the vessel |
| `vesselType` | Type of vessel (e.g., TANKER, CARGO SHIP) |
| `ais_VesselType` | AIS-reported vessel type |
| `ais_Flag` | Flag state of the vessel |
| `ais_Length` / `ais_Width` | Vessel dimensions |
| `ais_Draught` | Vessel draught in meters |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- (For fine-tuning) NVIDIA GPU with CUDA support and sufficient VRAM (16 GB+ recommended)

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/your-username/Finetuning-LLM-EagleEYE.git
cd Finetuning-LLM-EagleEYE/maritime_rag_pipeline
```

**2. Create and activate a virtual environment**

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **Windows note:** If activation is blocked by execution policy, run this once:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

**3. Install dependencies**

```bash
pip install -r requirements.txt
pip install -U langchain-huggingface langchain-community
```

---

## Usage

### 1. Build the Vector Database

Convert the CSV dataset into a searchable FAISS vector store:

```bash
python scripts/chunk_data.py \
  --input data/final_data.csv \
  --output vector_db \
  --sample_rows 50000 \
  --chunk_size 800 \
  --chunk_overlap 150 \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to the cleaned CSV file |
| `--output` | `vector_db` | Directory to save the FAISS index |
| `--sample_rows` | `0` (all) | Number of rows to index; `0` = all rows |
| `--chunk_size` | `800` | Characters per chunk |
| `--chunk_overlap` | `150` | Overlap characters between adjacent chunks |
| `--embedding_model` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformer model |

This will create `vector_db/index.faiss` and `vector_db/index.pkl`.

---

### 2. Run RAG Question Answering

Ask natural-language questions about the maritime data:

```bash
python scripts/rag_qa.py \
  --index vector_db \
  --query "How many tanker vessels arrived at Klaipėda on 2021-01-10?" \
  --top_k 3
```

**More examples:**

```bash
python scripts/rag_qa.py --index vector_db \
  --query "Which vessels stayed in Klaipėda longer than 48 hours?" --top_k 3

python scripts/rag_qa.py --index vector_db \
  --query "What ports had the most tanker arrivals in January 2021?" --top_k 5

python scripts/rag_qa.py --index vector_db \
  --query "List all cargo ships that arrived at Ventspils in March 2021." --top_k 5
```

**Example output:**

```
Loading FAISS index...
Retrieving relevant chunks...
Loading local model (TinyLlama)...
Running query...

Final Answer:
On 2021-01-10, 2 tanker vessels arrived at Klaipėda.

Sources:
[1] Port=KLAIPĖDA | Arrival=2021-01-10T08:42:00 | Departure=2021-01-11T... | VesselType=TANKER...
[2] Port=KLAIPĖDA | Arrival=2021-01-10T14:15:00 | Departure=2021-01-12T... | VesselType=TANKER...
```

> The pipeline runs entirely on CPU using TinyLlama. No GPU or API key is required for inference.

---

### 3. Generate Synthetic Training Data

Auto-generate instruction–response pairs from the dataset for supervised fine-tuning:

```bash
python scripts/generate_instructions.py \
  --input data/final_data.csv \
  --output training_data/instructions.jsonl \
  --num_samples 20000
```

This produces a JSONL file where each line is a JSON object with `instruction`, `input`, and `output` keys — covering questions about vessel arrivals, dwell times, and vessel type distributions.

---

### 4. Fine-Tune with QLoRA (Optional)

> **Requires:** NVIDIA GPU with 16 GB+ VRAM and CUDA toolkit installed.

Fine-tune a base LLM on the generated instruction pairs using 4-bit quantization (QLoRA):

```bash
python scripts/fine_tune_qlora.py \
  --dataset training_data/instructions.jsonl \
  --model meta-llama/Llama-3-8b-instruct \
  --output_dir training_data/qlora-adapter \
  --epochs 2 \
  --batch_size 2 \
  --learning_rate 2e-4
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Path to JSONL instruction file |
| `--model` | *(required)* | Base HuggingFace model identifier |
| `--output_dir` | *(required)* | Directory to save the trained LoRA adapter |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `2` | Per-device training batch size |
| `--learning_rate` | `2e-4` | Learning rate for LoRA parameters |
| `--max_length` | `1024` | Maximum tokenized sequence length |

**LoRA configuration used:**
- Rank `r = 8`, Alpha `= 16`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Dropout: `0.05`
- Task type: `CAUSAL_LM`

---

### 5. Evaluate Retrieval Performance

Measure how well the FAISS retriever finds relevant chunks (Recall@K):

```bash
python scripts/evaluate_retrieval.py \
  --index_dir vector_db \
  --eval_set training_data/eval_questions.json \
  --top_k 5
```

The eval set (`eval_questions.json`) is a JSON array where each item has:

```json
{
  "query": "How many tanker vessels arrived at Ventspils on 2021-03-15?",
  "port": "VENTSPILS",
  "date": "2021-03-15",
  "vessel_type": "TANKER"
}
```

The script reports per-query recall and an average Recall@K across all queries.

---

### 6. Evaluate LLM with Fine-Tuned Adapter

Test the full pipeline (retrieval + fine-tuned LLM) and compute exact-match accuracy:

```bash
python scripts/evaluate_llm.py \
  --index_dir vector_db \
  --adapter_dir training_data/qlora-adapter \
  --eval_set training_data/eval_questions.json \
  --model meta-llama/Llama-3-8b-instruct \
  --top_k 5
```

---

## Why RAG Instead of Pure Fine-Tuning?

| Consideration | RAG | Pure Fine-Tuning |
|---|---|---|
| Dataset type | Static, structured CSV | Same |
| Hardware | CPU sufficient for inference | GPU required |
| Up-to-date answers | Retrieval always reflects current data | Requires re-training on data changes |
| Hallucination risk | Low (grounded in retrieved context) | Higher without retrieval |
| Implementation effort | Moderate | High |

The dataset is large and static, making retrieval a natural fit. Fine-tuning is scaffolded in this repo as an optional enhancement — it can further improve answer quality once GPU resources are available.

---

## Evaluation

The project includes two evaluation layers:

**Retrieval layer (`evaluate_retrieval.py`):**  
Checks whether retrieved chunks contain the expected port name, date, and vessel type. Reports Recall@K per query and an overall average.

**LLM layer (`evaluate_llm.py`):**  
Runs end-to-end inference for each eval question using the fine-tuned adapter and computes exact-match accuracy against ground-truth answers (where provided in the eval set).

---

## Scalability & Extensibility

- **Cloud deployment:** Swap TinyLlama for GPT-4 or any API-based model by updating the LLM initialization in `rag_qa.py`.
- **Larger datasets:** Increase `--sample_rows` or remove the limit entirely to index the full dataset.
- **Additional analytics:** Extend `generate_instructions.py` with templates for CO₂ emission estimates, congestion threshold detection, or trend analysis across time periods.
- **Richer embeddings:** Replace `all-MiniLM-L6-v2` with a domain-adapted maritime embedding model for improved retrieval quality.
- **API serving:** Wrap `rag_qa.py` logic in a FastAPI endpoint for production deployment.

---

## Requirements

```
pandas>=1.5
langchain>=0.3.0
langchain-community>=0.3.0
langchain-huggingface
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
Jinja2>=3.1
Faker>=19.6
peft>=0.5
transformers>=4.36
datasets>=2.12
trl>=0.7.7
```

For GPU-based fine-tuning, additionally install:

```bash
pip install bitsandbytes accelerate
```

---

## License

This project is released for educational and research purposes. Dataset usage is subject to the terms of the original maritime data provider.
