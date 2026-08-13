import pickle

pickle_path = "../../embeddings_output/final_docs_list.pkl"
output_path = "../../embeddings_output/chunks_export.md"

with open(pickle_path, "rb") as f:
    final_docs = pickle.load(f)

with open(output_path, "w", encoding="utf-8") as f:
    for i, doc in enumerate(final_docs, start=1):
        f.write(f"## Chunk {i}\n\n")
        f.write(f"- **Source:** {doc.metadata.get('source', 'unknown')}\n")
        f.write(f"- **Images:** {doc.metadata.get('images', 'none')}\n")
        f.write(f"- **Has images:** {doc.metadata.get('has_images', False)}\n\n")
        f.write("```\n")
        f.write(doc.page_content)
        f.write("\n```\n\n")

print(f"Exported {len(final_docs)} chunks to '{output_path}'")
