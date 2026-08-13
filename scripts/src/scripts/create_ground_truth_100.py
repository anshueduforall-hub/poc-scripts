import sys
import types
import pickle
import torch
import pandas as pd
from openai import AsyncOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Workaround for ragas 0.4.3 incompatibility with langchain-community >= 0.4.2
_shim = types.ModuleType("langchain_community.chat_models.vertexai")
_shim.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _shim

import langchain_community.llms as _llms
if not hasattr(_llms, "VertexAI"):
    _llms.VertexAI = type("VertexAI", (), {})

from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig

# Load all chunks
with open("../../embeddings_output/final_docs_list.pkl", "rb") as f:
    final_docs = pickle.load(f)
print(f"Successfully loaded {len(final_docs)} documents.")

# -------------------------------------------------------------------
# 1. Initialize the Generator LLM using Ollama (ASYNC)
# -------------------------------------------------------------------
ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=120.0,
    max_retries=2
)

# Low temperature: Ragas needs strict JSON, not creativity here.
generator_llm = llm_factory(
    "qwen2.5:7b",
    client=ollama_client,
    temperature=0.1
)

# -------------------------------------------------------------------
# 2. Initialize the Embedding Model
# -------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
generator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# -------------------------------------------------------------------
# 3. Initialize the TestsetGenerator
# -------------------------------------------------------------------
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings
)

print("Starting synthetic dataset generation using Ollama...")
print("Note: 100 samples on local qwen2.5:7b will take several hours.")

# -------------------------------------------------------------------
# 4. Generate the dataset with Concurrency Control
# -------------------------------------------------------------------
# Throttle Ragas so it doesn't overwhelm local Ollama
run_config = RunConfig(
    max_workers=1,  # Forces sequential processing (Ollama's default capability)
    timeout=180     # Prevents infinite hanging if a chunk is too large
)

# Target ~100 examples. Default query distribution (single-hop / multi-hop
# abstract / multi-hop specific) is kept for diversity; ceil-rounding on the
# 3-way split may push the final count slightly above 100, and multi-hop
# under-production could pull it slightly below.
testset_size = 100

dataset = generator.generate_with_langchain_docs(
    documents=final_docs,
    testset_size=testset_size,
    run_config=run_config
)

# -------------------------------------------------------------------
# 5. Convert to Pandas DataFrame and Save
# -------------------------------------------------------------------
df_ground_truth = dataset.to_pandas()

print(df_ground_truth.head())
df_ground_truth.to_csv("../../generated/ollama_rag_evaluation_ground_truth_100.csv", index=False)
print(f"\nSynthetic dataset successfully saved with {len(df_ground_truth)} examples to 'ollama_rag_evaluation_ground_truth_100.csv'")
