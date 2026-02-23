#!/usr/bin/env python
"""
evaluate_llm.py
----------------

This script evaluates a fine‑tuned language model in combination with a
retrieval layer.  For each question in an evaluation set it retrieves the
most relevant context from the FAISS index, prepends it to the question and
generates an answer using the model.  If the evaluation set provides a
ground‑truth ``expected`` field, the script computes a simple accuracy
metric (exact match).

Usage:
    python evaluate_llm.py --index_dir vector_db --adapter_dir training_data/qlora-adapter \
        --eval_set training_data/eval_questions.json --model meta-llama/Llama-3-8b-instruct \
        --top_k 5

Arguments:
    --index_dir    Directory with the FAISS index (built via chunk_data.py).
    --adapter_dir  Directory containing the fine‑tuned LoRA adapter and tokenizer.
    --eval_set     JSON file with evaluation questions; each item should have
                   ``query`` and optionally ``expected``.
    --model        Base model identifier (same as used during fine‑tuning).
    --top_k        Number of top documents to retrieve per query (default: 5).
    --max_length   Max tokens to generate for each answer (default: 256).

"""

import argparse
import json
import os
from typing import Dict, List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine‑tuned LLM with retrieval")
    parser.add_argument("--index_dir", required=True, help="Directory of the FAISS index")
    parser.add_argument("--adapter_dir", required=True, help="Directory with LoRA adapter and tokenizer")
    parser.add_argument("--eval_set", required=True, help="JSON evaluation file")
    parser.add_argument("--model", required=True, help="Base model identifier")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve per query")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum tokens to generate for each answer")
    return parser.parse_args()


def load_eval_set(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(question: str, context_docs: List[str]) -> str:
    context = "\n".join(context_docs)
    prompt = (
        "### Context\n"
        f"{context}\n\n"
        "### Question\n"
        f"{question}\n\n"
        "### Answer\n"
    )
    return prompt


def main() -> None:
    args = parse_args()
    eval_set = load_eval_set(args.eval_set)
    if not eval_set:
        print("Evaluation set is empty")
        return

    # Load vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(args.index_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": args.top_k})

    # Load base model and apply LoRA adapter
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_4bit=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

    total = 0
    correct = 0
    for item in eval_set:
        query = item.get("query")
        expected = item.get("expected")
        docs = retriever.get_relevant_documents(query)
        context_docs = [doc.page_content for doc in docs]
        prompt = build_prompt(query, context_docs)
        output = gen(prompt, max_new_tokens=args.max_length, do_sample=False)[0]["generated_text"].strip()
        print(f"Q: {query}\nA: {output}\n---")
        if expected is not None:
            total += 1
            if output.strip() == expected.strip():
                correct += 1

    if total > 0:
        accuracy = correct / total
        print(f"Exact match accuracy on {total} questions: {accuracy:.2%}")


if __name__ == "__main__":
    main()