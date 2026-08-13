import sys
import time
import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "BAAI/bge-m3"
device = "cuda" if torch.cuda.is_available() else "cpu"
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": device},
    encode_kwargs=encode_kwargs,
)

vector_store = Chroma(
    persist_directory="../../embeddings_output/chroma_db_bgem3",
    embedding_function=embeddings,
)

def search(query, k=5):
    start = time.perf_counter()
    query_vector = vector_store._embedding_function.embed_query(query)
    embed_time = time.perf_counter() - start
    print(f"[Vector creation time: {embed_time:.4f} s]")

    start = time.perf_counter()
    results = vector_store.similarity_search_by_vector_with_relevance_scores(
        query_vector, k=k
    )
    search_time = time.perf_counter() - start
    print(f"[Vector search time: {search_time:.4f} s]")
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    else:
        query = input("Enter your question: ").strip()
        k = int(input("Number of results (default 5): ") or 5)

    print(f"\nQuery: {query}\n")
    for i, (doc, score) in enumerate(search(query, k), start=1):
        print(f"--- Result {i} (relevance score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Images: {doc.metadata.get('images', 'none')}")
        print(doc.page_content)
        print()
