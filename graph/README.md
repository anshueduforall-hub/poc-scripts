# Graph Utilities

Export and import tools for the Neo4j knowledge graph built by `chunk_embed_graph_pipeline.ipynb`.

## What Gets Exported

The export captures the entire connected component rooted at the KG entity labels
(Topic, Subtopic, Concept, Definition, Formula, Theorem, Example), including:

- **KG entity nodes** with all properties (name, id, type, etc.)
- **Chunk nodes** with properties (text, header, images, chunk_id, previous_chunk_id, next_chunk_id, text_embedding)
- **All relationships** between them (COVERS, EXPLAINS, ILLUSTRATES, USES_FORMULA, PART_OF, PREREQUISITE_FOR, MENTIONED_IN)

Unrelated subgraphs (e.g. demo datasets) are excluded.

## Files

| File | Description |
|------|-------------|
| `export_neo4j.py` | Exports the knowledge graph from Neo4j as a portable `.cypher` file |
| `import_cypher.py` | Imports a `.cypher` file into a Neo4j instance |
| `exported_nodes_and_relationships.cypher` | The exported Cypher file (2.4 MB) |

## Export

```bash
scripts/.venv/bin/python graph/export_neo4j.py --output graph/exported_nodes_and_relationships.cypher
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--uri` | `bolt://localhost:7687` | Neo4j bolt URI |
| `--user` | `neo4j` | Neo4j username |
| `--password` | `neo4jadmin` | Neo4j password |
| `--output` | `ontology-rnj.cypher` | Output file path |

## Import

```bash
scripts/.venv/bin/python graph/import_cypher.py --input graph/exported_nodes_and_relationships.cypher --cleanup
```

This loads all nodes and relationships from the exported `.cypher` file into the target Neo4j instance.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--uri` | `bolt://localhost:7687` | Neo4j bolt URI |
| `--user` | `neo4j` | Neo4j username |
| `--password` | `neo4jadmin` | Neo4j password |
| `--input` | `ontology-rnj.cypher` | Path to the `.cypher` file to import |
| `--cleanup` | off | Remove temporary `UNIQUE IMPORT LABEL` / `UNIQUE IMPORT ID` properties after import |

### What the import does

1. Parses the `.cypher` file into transaction blocks (`:begin`/`:commit` delimited)
2. Creates the vector index and unique constraint (schema block)
3. Waits for indexes to be ready (`CALL db.awaitIndexes(300)`)
4. Imports all node and relationship data in batched UNWIND blocks
5. Optionally cleans up the temporary `UNIQUE IMPORT LABEL` / `UNIQUE IMPORT ID` properties

### Importing into a fresh database

```bash
# Start with a clean database, then:
scripts/.venv/bin/python graph/import_cypher.py \
    --input graph/exported_nodes_and_relationships.cypher \
    --cleanup
```

The `--cleanup` flag is recommended to remove APOC's temporary import scaffolding after the data is loaded.
