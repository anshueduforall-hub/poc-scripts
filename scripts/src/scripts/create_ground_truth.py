import sys
import types
import pickle
import torch
import pandas as pd
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Workaround for ragas 0.4.3 incompatibility with langchain-community >= 0.4.2:
# ragas/llms/base.py unconditionally imports ChatVertexAI from a module that
# was removed from langchain-community. Stub the missing module before
# importing ragas so the import no longer crashes.
_shim = types.ModuleType("langchain_community.chat_models.vertexai")
_shim.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _shim

import langchain_community.llms as _llms

if not hasattr(_llms, "VertexAI"):
    _llms.VertexAI = type("VertexAI", (), {})

from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

# Load all chunks
# Load the saved final_docs
with open("../../embeddings_output/final_docs_list.pkl", "rb") as f:
    final_docs = pickle.load(f)
print(f"Successfully loaded {len(final_docs)} documents.")

# -------------------------------------------------------------------
# 1. Initialize the Generator LLM using Ollama
# -------------------------------------------------------------------
# Point the OpenAI-compatible client at Ollama's /v1 endpoint
# Increase timeout if your local GPU takes time to process large contexts
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# Instructor-based structured output for Ragas compatibility (enforces JSON)
generator_llm = llm_factory("qwen2.5:7b", client=ollama_client, temperature=0.7)

# -------------------------------------------------------------------
# 2. Initialize the Embedding Model
# -------------------------------------------------------------------
# Reusing the BAAI/bge-m3 model you set up earlier
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
# Wrap the LangChain embedding model for Ragas compatibility
generator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# -------------------------------------------------------------------
# 3. Initialize the TestsetGenerator
# -------------------------------------------------------------------
generator = TestsetGenerator(
    llm=generator_llm, 
    embedding_model=generator_embeddings
)

print("Starting synthetic dataset generation using Ollama...")
print("Note: Local LLM generation may take several minutes depending on hardware.")

# -------------------------------------------------------------------
# 4. Generate the dataset using your existing 'final_docs'
# -------------------------------------------------------------------
testset_size = 5  # Start small (e.g., 5) to test local inference speed and JSON stability
dataset = generator.generate_with_langchain_docs(
    documents=final_docs, 
    testset_size=testset_size
)

# -------------------------------------------------------------------
# 5. Convert to Pandas DataFrame and Save
# -------------------------------------------------------------------
df_ground_truth = dataset.to_pandas()

# Display the first few rows of the generated evaluation dataset
print(df_ground_truth.head())

# Save to CSV so you can use it for your retrieval experiments
df_ground_truth.to_csv("ollama_rag_evaluation_ground_truth.csv", index=False)
print("\nSynthetic dataset successfully saved to 'ollama_rag_evaluation_ground_truth.csv'")