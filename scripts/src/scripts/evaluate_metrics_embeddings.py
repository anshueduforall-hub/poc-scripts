import ast
import numpy as np
import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import pickle
import torch

# ===================================================================
# PHASE 2: EVALUATION METRICS ENGINE
# ===================================================================


def calculate_hit_rate(retrieved_docs, ground_truth_contexts):
    """Hit Rate @ K: Returns 1 if any ground truth chunk is in retrieved docs, else 0."""
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    for gt in ground_truth_contexts:
        if any(gt in text or text in gt for text in retrieved_texts):
            return 1.0
    return 0.0


def calculate_mrr(retrieved_docs, ground_truth_contexts):
    """Mean Reciprocal Rank (MRR @ K): Reciprocal rank of the FIRST relevant chunk found."""
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    for rank, text in enumerate(retrieved_texts, start=1):
        for gt in ground_truth_contexts:
            if gt in text or text in gt:
                return 1.0 / rank
    return 0.0


def calculate_ndcg(retrieved_docs, ground_truth_contexts, k=5):
    """nDCG @ K: Measures ranking quality by penalizing relevant items ranked lower."""
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    relevance = []

    for text in retrieved_texts[:k]:
        is_relevant = any(
            gt in text or text in gt for gt in ground_truth_contexts
        )
        relevance.append(1 if is_relevant else 0)

    if not any(relevance):
        return 0.0

    # Discounted Cumulative Gain (DCG)
    dcg = sum(
        rel / np.log2(idx + 2) for idx, rel in enumerate(relevance[:k])
    )

    # Ideal DCG (IDCG) - best possible sorting of ground truths
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum(
        rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance[:k])
    )

    return dcg / idcg if idcg > 0 else 0.0


# ===================================================================
# PHASE 3: RETRIEVAL PIPELINES SETUP
# ===================================================================

# Load all chunks (Document objects with metadata) from pickle
with open("../../embeddings_output/final_docs_list.pkl", "rb") as f:
    final_docs = pickle.load(f)
print(f"Successfully loaded {len(final_docs)} documents")


print("Setting up retrieval pipelines...")

# 1. Pipeline A: Vanilla Dense Retriever
device = "cuda" if torch.cuda.is_available() else "cpu"
vector_store = Chroma(
    persist_directory="../../embeddings_output/chroma_db_bgem3",
    embedding_function=HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    ),
)
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 2. Pipeline B: Hybrid Retriever (Dense BGE-M3 + Sparse BM25)
# Re-extract all original documents from vector DB to build BM25 index
#all_docs = vector_store.get()["documents"]
#from langchain_core.documents import Document

#documents = [Document(page_content=text) for text in all_docs]

bm25_retriever = BM25Retriever.from_documents(final_docs)
bm25_retriever.k = 5


def hybrid_retrieve(query, top_k=5, alpha=0.5):
    """Combines Dense and BM25 results using Reciprocal Rank Fusion (RRF)."""
    dense_results = dense_retriever.invoke(query)
    bm25_results = bm25_retriever.invoke(query)

    # Simple RRF scoring
    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc.page_content
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + alpha * (
            1.0 / (60 + rank)
        )

    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.page_content
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - alpha) * (
            1.0 / (60 + rank)
        )

    sorted_docs = sorted(
        rrf_scores.items(), key=lambda x: x[1], reverse=True
    )
    return [doc_map[doc_id] for doc_id, _ in sorted_docs[:top_k]]


# 3. Pipeline C: Re-Ranker Setup
print("Loading Cross-Encoder Re-ranker (BAAI/bge-reranker-v2-m3)...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)


def reranked_retrieve(query, initial_k=20, final_k=5):
    """Retrieves candidates using Hybrid search, then re-ranks using CrossEncoder."""
    # Retrieve top candidates using hybrid search
    candidate_docs = hybrid_retrieve(query, top_k=initial_k)

    if not candidate_docs:
        return []

    # Prepare pairs for cross-encoder scoring
    pairs = [[query, doc.page_content] for doc in candidate_docs]
    scores = reranker.predict(pairs)

    # Sort documents based on re-ranker scores
    scored_docs = sorted(
        zip(candidate_docs, scores), key=lambda x: x[1], reverse=True
    )
    return [doc for doc, _ in scored_docs[:final_k]]


# ===================================================================
# EXECUTE BENCHMARK ON GROUND TRUTH DATASET
# ===================================================================

# Load generated ground truth dataset from Phase 1
df_gt = pd.read_csv("../../generated/ollama_rag_evaluation_ground_truth_100.csv")

# Parse string representations of lists if necessary
if isinstance(df_gt["reference_contexts"].iloc[0], str):
    df_gt["reference_contexts"] = df_gt["reference_contexts"].apply(
        ast.literal_eval
    )

results_summary = {
    "Vanilla Dense": {"Hit_Rate": [], "MRR": [], "nDCG": []},
    "Hybrid (Dense+BM25)": {"Hit_Rate": [], "MRR": [], "nDCG": []},
    "Hybrid + Re-Ranker": {"Hit_Rate": [], "MRR": [], "nDCG": []},
}

print("\nRunning Retrieval Evaluation Benchmark...")

for _, row in tqdm(df_gt.iterrows(), total=len(df_gt)):
    query = row["user_input"]
    gt_contexts = row["reference_contexts"]

    # 1. Run Vanilla Dense
    vanilla_docs = dense_retriever.invoke(query)
    results_summary["Vanilla Dense"]["Hit_Rate"].append(
        calculate_hit_rate(vanilla_docs, gt_contexts)
    )
    results_summary["Vanilla Dense"]["MRR"].append(
        calculate_mrr(vanilla_docs, gt_contexts)
    )
    results_summary["Vanilla Dense"]["nDCG"].append(
        calculate_ndcg(vanilla_docs, gt_contexts)
    )

    # 2. Run Hybrid
    hybrid_docs = hybrid_retrieve(query, top_k=5)
    results_summary["Hybrid (Dense+BM25)"]["Hit_Rate"].append(
        calculate_hit_rate(hybrid_docs, gt_contexts)
    )
    results_summary["Hybrid (Dense+BM25)"]["MRR"].append(
        calculate_mrr(hybrid_docs, gt_contexts)
    )
    results_summary["Hybrid (Dense+BM25)"]["nDCG"].append(
        calculate_ndcg(hybrid_docs, gt_contexts)
    )

    # 3. Run Re-Ranker
    reranked_docs = reranked_retrieve(query, initial_k=20, final_k=5)
    results_summary["Hybrid + Re-Ranker"]["Hit_Rate"].append(
        calculate_hit_rate(reranked_docs, gt_contexts)
    )
    results_summary["Hybrid + Re-Ranker"]["MRR"].append(
        calculate_mrr(reranked_docs, gt_contexts)
    )
    results_summary["Hybrid + Re-Ranker"]["nDCG"].append(
        calculate_ndcg(reranked_docs, gt_contexts)
    )

# ===================================================================
# DISPLAY PERFORMANCE BENCHMARK COMPARISON
# ===================================================================

evaluation_metrics = []

for pipeline, metrics in results_summary.items():
    evaluation_metrics.append({
        "Retrieval Strategy": pipeline,
        "Hit Rate @ 5": f"{np.mean(metrics['Hit_Rate']):.4f}",
        "MRR @ 5": f"{np.mean(metrics['MRR']):.4f}",
        "nDCG @ 5": f"{np.mean(metrics['nDCG']):.4f}",
    })

df_metrics = pd.DataFrame(evaluation_metrics)
print("\n" + "=" * 50)
print("FINAL RETRIEVAL PERFORMANCE EVALUATION")
print("=" * 50)
print(df_metrics.to_string(index=False))