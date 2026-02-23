from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True, help="Path to FAISS index folder")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    print("🔹 Loading FAISS index...")
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(args.index, emb, allow_dangerous_deserialization=True)

    print("🔹 Retrieving relevant chunks...")
    docs = db.similarity_search(args.query, k=args.top_k)

    context = "\n".join([d.page_content for d in docs])

    print("🔹 Loading local model (TinyLlama)...")
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=-1,  # CPU
        max_new_tokens=256,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    print("🔹 Running query...")
    prompt = f"Context:\n{context}\n\nQuestion: {args.query}\nAnswer:"
    answer = llm(prompt)

    print("\n✅ Final Answer:")
    print(answer)

    print("\n📖 Sources:")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.page_content[:200]}...")

if __name__ == "__main__":
    main()
