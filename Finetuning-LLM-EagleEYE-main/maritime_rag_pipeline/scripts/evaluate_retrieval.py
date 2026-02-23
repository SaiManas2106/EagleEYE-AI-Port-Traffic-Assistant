#!/usr/bin/env python
"""
evaluate_retrieval.py
----------------------

This script evaluates the performance of the FAISS retrieval layer using a set
of predefined queries.  For each query, it computes recall based on known
ground‑truth metadata extracted from the query itself (port and date).  The
evaluation set should be a JSON array of objects with at least the fields
``query``, ``port`` and ``date`` (YYYY‑MM‑DD).  A ``vessel_type`` field can
optionally be included to restrict the ground truth to a specific type.

The script reports the percentage of relevant documents retrieved in the top
``k`` results for each query and prints a summary at the end.

Usage:
    python evaluate_retrieval.py --index_dir vector_db --eval_set training_data/eval_questions.json --top_k 5

Arguments:
    --index_dir   Directory containing the FAISS index created by ``chunk_data.py``.
    --eval_set    Path to a JSON file with evaluation queries and metadata.
    --top_k       Number of top documents to retrieve for each query (default: 5).

"""

import argparse
import json
import os
from typing import List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FAISS retrieval performance")
    parser.add_argument("--index_dir", required=True, help="Directory with FAISS index files")
    parser.add_argument("--eval_set", required=True, help="JSON file containing evaluation queries")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top documents to retrieve")
    return parser.parse_args()


def load_queries(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main() -> None:
    args = parse_args()
    queries = load_queries(args.eval_set)
    if not queries:
        print("No queries found in eval set")
        return

    # Load vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(args.index_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": args.top_k})

    total_queries = len(queries)
    total_recall = 0.0
    for q in queries:
        query_text = q["query"]
        port = q.get("port")
        date = q.get("date")
        vessel_type = q.get("vessel_type")  # optional

        results = retriever.get_relevant_documents(query_text)

        # Compute recall: count docs that mention both port and date (and type, if provided)
        relevant_count = 0
        for doc in results:
            text = doc.page_content
            if port and port not in text:
                continue
            if date and date not in text:
                continue
            if vessel_type and vessel_type not in text:
                continue
            relevant_count += 1

        recall = relevant_count / args.top_k
        total_recall += recall
        print(f"Query: {query_text}")
        print(f"Recall: {recall:.2f} ({relevant_count}/{args.top_k})")
        print("---")

    avg_recall = total_recall / total_queries
    print(f"Average recall@{args.top_k}: {avg_recall:.2f}")


if __name__ == "__main__":
    main()