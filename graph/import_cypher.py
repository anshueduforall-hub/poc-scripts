#!/usr/bin/env python3
"""Import a .cypher file into Neo4j.

Handles APOC-style exports with :begin/:commit transaction delimiters
and the UNIQUE IMPORT LABEL / UNIQUE IMPORT ID cleanup pattern.

Usage:
    python graph/import_cypher.py --input graph/ontology-rnj.cypher
    python graph/import_cypher.py --input graph/ontology-rnj.cypher --cleanup
"""

import argparse

from neo4j import GraphDatabase


def parse_cypher(content):
    """Parse a .cypher file into a list of executable statement strings.

    Handles :begin/:commit delimited transaction blocks and standalone
    statements (like CALL db.awaitIndexes(...)).
    """
    blocks = []
    current = []
    in_block = False

    for line in content.splitlines():
        stripped = line.strip()
        if stripped == ":begin":
            in_block = True
            current = []
        elif stripped == ":commit":
            in_block = False
            if current:
                blocks.append("\n".join(current))
        elif in_block:
            current.append(line)
        elif stripped and not stripped.startswith(":"):
            blocks.append(stripped)

    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="neo4jadmin")
    parser.add_argument("--input", default="ontology-rnj.cypher")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove UNIQUE IMPORT LABEL and UNIQUE IMPORT ID after import",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        content = f.read()

    blocks = parse_cypher(content)
    print(f"Parsed {len(blocks)} statement block(s) from {args.input}")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as s:
            for i, block in enumerate(blocks):
                preview = block.splitlines()[0][:100]
                print(f"  [{i + 1}/{len(blocks)}] {preview}...")
                s.run(block)

            if args.cleanup:
                s.run(
                    "MATCH (n:`UNIQUE IMPORT LABEL`) "
                    "REMOVE n:`UNIQUE IMPORT LABEL` "
                    "REMOVE n.`UNIQUE IMPORT ID`"
                )
                print("  Cleaned up UNIQUE IMPORT LABEL / UNIQUE IMPORT ID")

        print("Import complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
