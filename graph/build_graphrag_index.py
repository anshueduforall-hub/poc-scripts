#!/usr/bin/env python3
"""End-to-end Neo4j-native GraphRAG index builder (separate from the Chroma pipeline).

Pipeline:
  1. Chunk `processed_notes_with_descriptions.md` using the exact splitter config
     used by `graph/langchain_final.ipynb` (MarkdownHeaderTextSplitter h1-h3 +
     RecursiveCharacterTextSplitter 1000/100) -> 99 chunks (verified 1:1 against
     the extraction pickle).
  2. Embed chunk text with BAAI/bge-m3 (cosine-normalized, 1024-d).
  3. Load into Neo4j: `(:Chunk {chunk_id, text, header, images, text_embedding})`
     nodes, `(:Entity)-[:MENTIONED_IN]->(:Chunk)` relationships built from the
     entity-extraction pickle, and a VECTOR index over `Chunk.text_embedding`.

The entity-extraction pickle is only used for the entity<->chunk mapping; the
chunk text itself is re-derived deterministically from the markdown source.
"""

import argparse
import hashlib
import pickle
import re

from neo4j import GraphDatabase

MD_PATH = "/home/kanshu/poc-scripts/scripts/generated_107_notusellm/processed_notes_with_descriptions.md"
PICKLE_PATH = "/home/kanshu/poc-scripts/graph/extracted_graph_docs.pkl"
EMBED_MODEL = "BAAI/bge-m3"
VECTOR_DIM = 1024

IMAGE_PATTERN = re.compile(r"!\[.*?\]\((.*?)\)")


def chunk_markdown(md_path):
    from langchain_core.documents import Document
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    raw_text = open(md_path, encoding="utf-8").read()

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
        strip_headers=False,
    )
    md_header_splits = markdown_splitter.split_text(raw_text)

    docs_with_image_metadata = []
    for doc in md_header_splits:
        image_matches = IMAGE_PATTERN.findall(doc.page_content)
        docs_with_image_metadata.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "images": image_matches,
                    "has_images": len(image_matches) > 0,
                    **doc.metadata,
                },
            )
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    return text_splitter.split_documents(docs_with_image_metadata)


def chunk_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="neo4jadmin")
    parser.add_argument("--md", default=MD_PATH)
    parser.add_argument("--pickle", default=PICKLE_PATH)
    args = parser.parse_args()

    print("[1/4] Chunking markdown...")
    chunks = chunk_markdown(args.md)
    print(f"      {len(chunks)} chunks from {args.md}")

    print("[2/4] Cross-checking against extraction pickle...")
    with open(args.pickle, "rb") as f:
        graph_docs = pickle.load(f)
    pickle_texts = [g.source.page_content for g in graph_docs]
    from langchain_core.documents import Document

    rechunked = {c.page_content for c in chunks}
    match = sum(1 for t in pickle_texts if t in rechunked)
    if match == len(pickle_texts) == len(chunks):
        print("      OK: chunk texts match pickle 1:1")
    else:
        print(f"      WARN: only {match}/{len(pickle_texts)} pickle sources match "
              f"({len(chunks)} chunks). Using pickle texts as canonical source.")
        seen = set()
        chunks = []
        for text in pickle_texts:
            if text not in seen:
                seen.add(text)
                chunks.append(Document(page_content=text, metadata={}))

    print("[3/4] Embedding chunks with bge-m3...")
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    print(f"      embedded {len(vectors)} vectors x {len(vectors[0])} dims")

    print("[4/4] Loading into Neo4j...")
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as s:
            s.run(
                "CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.text_embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`: $dim, "
                "`vector.similarity_function`: 'cosine'}}",
                dim=VECTOR_DIM,
            ).consume()

            chunk_ids = []
            for doc, vec in zip(chunks, vectors):
                cid = chunk_id(doc.page_content)
                chunk_ids.append(cid)
                header = doc.metadata.get("Header 1") or doc.metadata.get("Header 2") \
                    or doc.metadata.get("Header 3") or ""
                images = doc.metadata.get("images", [])
                if isinstance(images, (list, tuple)):
                    images = ", ".join(images)
                s.run(
                    "MERGE (c:Chunk {chunk_id: $cid}) "
                    "SET c.text = $text, c.header = $header, "
                    "c.images = $images, c.text_embedding = $vec",
                    cid=cid, text=doc.page_content, header=header,
                    images=images, vec=vec,
                ).consume()
            print(f"      upserted {len(chunk_ids)} Chunk nodes")

            chunk_id_set = set(chunk_ids)
            linked = 0
            for g in graph_docs:
                cid = chunk_id(g.source.page_content)
                if cid not in chunk_id_set:
                    continue
                for n in g.nodes:
                    res = s.run(
                        "MATCH (e {id: $id}) "
                        "MATCH (c:Chunk {chunk_id: $cid}) "
                        "MERGE (e)-[:MENTIONED_IN]->(c)",
                        id=n.id, cid=cid,
                    ).consume()
                    linked += res.counters.relationships_created
            print(f"      created {linked} MENTIONED_IN relationships")

            chunks_n = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
            links_n = s.run(
                "MATCH ()-[r:MENTIONED_IN]->() RETURN count(r) AS n"
            ).single()["n"]
            idx = s.run(
                "SHOW INDEXES YIELD name, type "
                "WHERE name = 'chunk_embeddings' RETURN name, type"
            ).single()
            print(f"      DB state: {chunks_n} chunks, {links_n} MENTIONED_IN "
                  f"links, index: {dict(idx) if idx else None}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
