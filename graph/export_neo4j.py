#!/usr/bin/env python3
"""Export the knowledge-graph connected component from Neo4j as a portable .cypher file.

The export is scoped to the knowledge graph (Topic/Subtopic/Concept/Definition/
Formula/Theorem/Example) plus everything connected to it: the neighbouring
Process/Constant/Variable nodes, the `Chunk` nodes (with `text_embedding` for
RAG) and their `MENTIONED_IN` relationships. Unrelated subgraphs such as the
movies demo dataset are excluded.

Output is readable CREATE statements that can be imported on any Neo4j Community
instance with:

    cypher-shell -u neo4j -p <password> -a bolt://<host>:7687 < output.cypher

Uses APOC streaming (apoc.export.cypher.query), so no apoc.conf or server
restart is required.
"""

import argparse

from neo4j import GraphDatabase

KG_LABELS = {
    "Topic",
    "Subtopic",
    "Concept",
    "Definition",
    "Formula",
    "Theorem",
    "Example",
}


def connected_component(driver):
    """Return (nodes, relationships) of the connected component(s) rooted at the KG labels."""
    with driver.session() as s:
        nodes = [rec["n"] for rec in s.run("MATCH (n) RETURN n")]
        rels = [rec["r"] for rec in s.run("MATCH ()-[r]->() RETURN r")]

    adjacency = {}
    for r in rels:
        a, b = r.start_node.element_id, r.end_node.element_id
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    seeds = {n.element_id for n in nodes if any(l in KG_LABELS for l in n.labels)}
    reached = set(seeds)
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        for nxt in adjacency.get(cur, ()):
            if nxt not in reached:
                reached.add(nxt)
                stack.append(nxt)

    comp_nodes = [n for n in nodes if n.element_id in reached]
    comp_rels = [
        r
        for r in rels
        if r.start_node.element_id in reached and r.end_node.element_id in reached
    ]
    return comp_nodes, comp_rels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="neo4jadmin")
    parser.add_argument("--output", default="ontology-rnj.cypher")
    args = parser.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        nodes, rels = connected_component(driver)

        label_counts = {}
        for n in nodes:
            for l in n.labels:
                label_counts[l] = label_counts.get(l, 0) + 1
        print(f"Component nodes: {len(nodes)}  ({label_counts})")
        print(f"Component relationships: {len(rels)}")

        labels = set()
        for n in nodes:
            labels.update(n.labels)
        pred_n = " OR ".join(f"n:{l}" for l in sorted(labels))
        pred_m = " OR ".join(f"m:{l}" for l in sorted(labels))
        stmt = (
            f"MATCH (n) WHERE {pred_n} "
            f"OPTIONAL MATCH (n)-[r]-(m) WHERE {pred_m} "
            f"RETURN n, r, m"
        )
        escaped = stmt.replace("\\", "\\\\").replace('"', '\\"')
        call = (
            f'CALL apoc.export.cypher.query("{escaped}", null, '
            "{stream: true, useNativeTypes: true})"
        )

        with driver.session() as s:
            res = s.run(call)
            statements = []
            for rec in res:
                for key in ("schemaStatements", "cypherStatements"):
                    val = rec.get(key)
                    if val:
                        statements.append(val)

        with open(args.output, "w") as f:
            f.write("\n".join(statements))
            f.write("\n")

        print(f"Wrote {len(statements)} statement block(s) to {args.output}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
