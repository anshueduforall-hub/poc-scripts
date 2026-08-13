import sys
import types
import pickle
import torch
import pandas as pd
from openai import AsyncOpenAI  # CHANGED: Must use the Async client
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
from ragas.run_config import RunConfig  # ADDED: To control concurrency

# Load all chunks
with open("../../embeddings_output/final_docs_list.pkl", "rb") as f:
    final_docs = pickle.load(f)
print(f"Successfully loaded {len(final_docs)} documents.")

# -------------------------------------------------------------------
# 1. Initialize the Generator LLM using Ollama (ASYNC)
# -------------------------------------------------------------------
# CHANGED: Use AsyncOpenAI with timeouts so it crashes gracefully instead of hanging
ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="ollama",
    timeout=120.0,
    max_retries=2
)

# CHANGED: Lower temperature to 0.1. Ragas needs strict JSON, not creativity here.
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

# -------------------------------------------------------------------
# 4. Generate the dataset with Concurrency Control
# -------------------------------------------------------------------
# ADDED: Throttle Ragas so it doesn't overwhelm local Ollama
run_config = RunConfig(
    max_workers=1,  # Forces sequential processing (Ollama's default capability)
    timeout=180     # Prevents infinite hanging if a chunk is too large
)

testset_size = 5 

# PRO-TIP: If your `final_docs` contains thousands of chunks, generating the initial 
# Knowledge Graph will take hours locally. Slice it `final_docs[:50]` for your first test!
dataset = generator.generate_with_langchain_docs(
    documents=final_docs[:50], # Sliced to 50 chunks for testing speed
    testset_size=testset_size,
    run_config=run_config
)

# -------------------------------------------------------------------
# 5. Convert to Pandas DataFrame and Save
# -------------------------------------------------------------------
df_ground_truth = dataset.to_pandas()

print(df_ground_truth.head())
df_ground_truth.to_csv("ollama_rag_evaluation_ground_truth.csv", index=False)
print("\nSynthetic dataset successfully saved to 'ollama_rag_evaluation_ground_truth.csv'")