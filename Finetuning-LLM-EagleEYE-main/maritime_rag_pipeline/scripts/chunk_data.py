#!/usr/bin/env python
"""
chunk_data.py
-----------------

This script reads a cleaned maritime dataset (CSV), converts each row into a plain‑text
representation, splits the combined text into overlapping chunks, embeds the chunks
using a sentence‑transformer model and stores them in a FAISS vector database.

Usage:
    python chunk_data.py --input data/final_data.csv --output vector_db \
        --chunk_size 800 --chunk_overlap 150 --sample_rows 0 \
        --embedding_model sentence-transformers/all-MiniLM-L6-v2

Arguments:
    --input            Path to the cleaned CSV file (required).
    --output           Directory where the FAISS index will be stored (default: vector_db).
    --chunk_size       Maximum characters per chunk (default: 800).
    --chunk_overlap    Characters of overlap between chunks (default: 150).
    --sample_rows      Number of rows to sample from the CSV for quick experimentation.  Set to 0 to use all rows.
    --embedding_model  Name of the HuggingFace sentence‑transformer model to use (default: all-MiniLM-L6-v2).

The script uses LangChain's ``RecursiveCharacterTextSplitter`` to break the text into
chunks and ``langchain_community.vectorstores.FAISS`` to build the vector store.  The
index and document store are saved into the output directory.
"""

import argparse
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk maritime data and build FAISS index")
    parser.add_argument("--input", required=True, help="Path to cleaned CSV file")
    parser.add_argument("--output", default="vector_db", help="Directory to save FAISS index")
    parser.add_argument("--chunk_size", type=int, default=800, help="Number of characters per chunk")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Overlap between chunks")
    parser.add_argument("--sample_rows", type=int, default=0, help="Use only the first N rows (0 for all)")
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the HuggingFace sentence-transformer model",
    )
    return parser.parse_args()


def row_to_text(row: pd.Series) -> str:
    """Convert a single DataFrame row into a structured text representation.

    The output uses simple key=value pairs separated by pipes.  Only a few columns
    are used by default; you can extend this function to include more context
    (e.g. dwell time, cargo type) if available.
    """
    port = row.get("portName", "")
    arr = row.get("portArrival", "")
    dep = row.get("portDeparture", "")
    vtype = row.get("ais_VesselType", "")
    return f"Port={port} | Arrival={arr} | Departure={dep} | VesselType={vtype}"


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_dir = args.output
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    sample_rows = args.sample_rows
    embedding_model = args.embedding_model

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading CSV from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

    # Optionally sample a subset for quick testing
    if sample_rows and sample_rows > 0:
        df = df.head(sample_rows)
        print(f"Using only the first {len(df)} rows for indexing")

    # Convert each row to a plain‑text string
    texts = [row_to_text(r) for _, r in df.iterrows()]
    if not texts:
        raise ValueError("No text extracted from the CSV. Check your column names.")

    # Join all texts with newline separators and split into chunks
    full_text = "\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(full_text)
    print(f"Generated {len(chunks):,} chunks")

    # Wrap each chunk in a Document with minimal metadata
    documents = [Document(page_content=c, metadata={"source": os.path.basename(input_path)}) for c in chunks]

    # Create embeddings and FAISS index
    print(f"Loading embedding model {embedding_model}...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    print("Encoding and indexing chunks...")
    db = FAISS.from_documents(documents, embeddings)

    # Save index to disk
    print(f"Saving FAISS index to {output_dir}...")
    db.save_local(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()