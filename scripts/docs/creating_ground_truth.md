# Test Dataset / Ground Truth Generation Plan

## Goal
Generate ~100 synthetic evaluation examples (RAG ground-truth testset) from the
parsed physics textbook (`local_parsed_document.md`) using Ragas 0.4.3 + local
Ollama (`qwen2.5:7b`) and `BAAI/bge-m3` embeddings.

## Current state
- Chunks: 69 chunks saved to `embeddings_output/final_docs_list.pkl`
  (Chroma vector store at `embeddings_output/chroma_db_bgem3`).
- `create_ground_truth1.py` currently runs with `testset_size = 5` and only
  `final_docs[:50]` -> produced 7 examples in
  `ollama_rag_evaluation_ground_truth.csv`.

## Why only 7 examples?
- The number of examples is set by `testset_size`, NOT chunk count.
- Ragas splits `testset_size` across 3 query types
  (`default_query_distribution`): single-hop, multi-hop abstract, multi-hop
  specific, each with prob 1/3.
- `calculate_split_values` uses `ceil(n * prob)` -> 5 -> 2+2+2, and per-node
  sampling overshoots, yielding 7.
- 50 chunks is enough for 5 samples; it only matters as an upper bound.

## Changes to `create_ground_truth1.py`
1. `testset_size = 5` -> `testset_size = 100`
2. `final_docs[:50]` -> `final_docs` (use all 69 chunks so multi-hop cluster
   constraints don't starve the 100-sample target)
3. Save to new file `ollama_rag_evaluation_ground_truth_100.csv`
   (do not clobber the 7-example file)
4. Keep default query distribution (equal single-hop / multi-hop split)
5. Keep `RunConfig(max_workers=1, timeout=180)` for Ollama stability

## Expected outcome
- ~100-105 examples (ceil-rounding on the 3-way split can slightly exceed 100;
  may land slightly below if multi-hop entity-overlap clusters run out, but
  69 chunks should keep it close).

## Runtime expectations
- SummaryExtractor alone took ~59 min on 66 nodes last run (one-time cost).
- 100 samples on local `qwen2.5:7b` sequential -> several hours total.
- Run in background / screen; monitor the progress bar.

## Pre-run checks
- Kill any lingering hung `python3.11 create_ground_truth.py` process
  (non-async client that hung at CustomNodeFilter) to free GPU/RAM.
- Confirm Ollama is up: `curl http://localhost:11434/v1/models`.

## Files
- Script: `src/scripts/create_ground_truth1.py`
- Input chunks: `embeddings_output/final_docs_list.pkl`
- Output: `src/scripts/ollama_rag_evaluation_ground_truth_100.csv`
- Columns: `user_input`, `reference_contexts`, `reference`, `persona_name`,
  `query_style`, `query_length`, `synthesizer_name`

## Optional follow-ups
- If multi-hop under-produces, bias distribution toward single-hop
  (e.g. 0.7/0.15/0.15) to guarantee ~100.
- Verify retrieval with `vector_search.py` using the generated `user_input`s.
