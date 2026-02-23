import argparse
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to FAISS index directory")
    parser.add_argument("--query", required=True, help="Query to test")
    args = parser.parse_args()

    # Load embedding model
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index
    db = FAISS.load_local(args.index, emb, allow_dangerous_deserialization=True)

    # Run retrieval
    docs = db.similarity_search(args.query, k=5)

    print(f"\n✅ Query: {args.query}\n")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.page_content[:400]}...\n")

if __name__ == "__main__":
    main()
