import re
import pickle
import torch
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# -------------------------------------------------------------------
# 1. Load Markdown Content
# -------------------------------------------------------------------
with open("../../generated_107_notusellm/local_parsed_document.md", "r", encoding="utf-8") as f:
    markdown_document = f.read()

# -------------------------------------------------------------------
# 2. Split by Markdown Headings
# -------------------------------------------------------------------
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = markdown_splitter.split_text(markdown_document)

# -------------------------------------------------------------------
# 3. Detect Images and Prepare Metadata
# -------------------------------------------------------------------
image_pattern = re.compile(r"!\[.*?\]\((.*?)\)")

docs_with_image_metadata = []
for doc in md_header_splits:
    image_matches = image_pattern.findall(doc.page_content)

    updated_metadata = doc.metadata.copy()
    updated_metadata["images"] = image_matches if image_matches else []
    updated_metadata["has_images"] = len(image_matches) > 0

    docs_with_image_metadata.append(
        Document(page_content=doc.page_content, metadata=updated_metadata)
    )

# -------------------------------------------------------------------
# 4. Chunk Large Sections
# -------------------------------------------------------------------
# Note: bge-m3 supports context lengths up to 8192 tokens,
# so you can safely use larger chunk sizes if needed!
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)
final_docs = text_splitter.split_documents(docs_with_image_metadata)

# Save final_docs to a file so other scripts can use it
with open("../../embeddings_output/final_docs_list.pkl", "wb") as f:
    pickle.dump(final_docs, f)
print("Saved final_docs to 'final_docs_list.pkl'")

# Refine image metadata per specific split chunk
for doc in final_docs:
    found_images = image_pattern.findall(doc.page_content)
    # Chroma metadata requires scalar types or lists of primitive types
    doc.metadata["images"] = (
        ", ".join(found_images) if found_images else "none"
    )
    doc.metadata["has_images"] = len(found_images) > 0
    doc.metadata["source"] = "local_parsed_document.md"

# -------------------------------------------------------------------
# 5. Initialize BAAI/bge-m3 Model
# -------------------------------------------------------------------
# Configure model execution parameters
model_name = "BAAI/bge-m3"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_kwargs = {"device": device}
encode_kwargs = {
    "normalize_embeddings": True
}  # Recommended for cosine similarity

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# -------------------------------------------------------------------
# 6. Store Embeddings in ChromaDB
# -------------------------------------------------------------------
vector_store = Chroma.from_documents(
    documents=final_docs,
    embedding=embeddings,
    persist_directory="../../embeddings_output/chroma_db_bgem3",
)

print(
    f"Successfully embedded and stored {len(final_docs)} chunks using BAAI/bge-m3!"
)