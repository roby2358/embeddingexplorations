# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

EvolvAttention is the single project in this `embeddingexplorations` repo: a FastAPI server that scores and evolves text strings against an embedding-space target. It computes a normalized **cosine barycenter** (mean of normalized sentence embeddings) of a set of target strings, then uses cosine similarity to that barycenter as a fitness function for a character-level genetic algorithm.

## Commands

Dependencies are managed with `uv`. Run everything through `uv run` (this resolves the `pythonpath = ["src"]` setting so imports like `from evolvattention...` work).

```bash
uv sync                                           # install all deps (incl. test/lint tools)

uv run web                                        # run server (localhost:8042); alias for the line below
uv run python -m src.evolvattention.server        # run server (localhost:8042)
uv run uvicorn src.evolvattention.server:app --port 8042   # alt: run with reload, etc.

uv run pytest                                     # all tests (tests/ + tests_slow/)
uv run pytest -m fast                             # only fast tests (no model loading)
uv run pytest -m slow                             # only slow tests (load all-MiniLM-L6-v2)
uv run pytest tests/test_server.py                # single file
uv run pytest tests/evolvattention/test_evolution.py::TestEvolution::test_x   # single test

uv run black .                                    # format (line-length 88)
uv run isort .                                    # sort imports (black profile)
uv run mypy .                                     # type check (disallow_untyped_defs=True)
```

**Fast vs. slow tests**: `tests/` are fast — they use `TestClient` and mock `VecBookIndex` so no model loads. `tests_slow/` actually load the sentence-transformers model and download `all-MiniLM-L6-v2` on first run (network + slow). When adding tests, mark model-loading tests `@pytest.mark.slow` and keep server-logic tests in `tests/` with a mocked index.

## Architecture

Three layers, wired together lazily at request time:

- **`server.py`** — FastAPI app. All routes under `/api/v1`. Holds a module-level `ServerState` (barycenter, evolution session, `VecBookIndex`, `EvolutionaryAlgorithm`). **The server is intentionally stateful and single-user**: `/barycenter` stores a target in memory that subsequent calls (`/cosine-similarities`, evolution, attention) depend on — calling those before setting a barycenter returns a 400. Also serves the static UI from `resources/public/` (index.html/css/js).
- **`vecx/vecbook_index.py`** — `VecBookIndex`, the embedding layer. Wraps `SentenceTransformer("all-MiniLM-L6-v2")` and FAISS. Owns barycenter math (`set_target_strings`, `cosine_barycenter_similarity`, `compare_against_barycenter`) and similarity search (`IndexFlatIP` over L2-normalized vectors = cosine). This is the **only** place embeddings/FAISS are touched.
- **`evolve/evolution.py`** — `EvolutionaryAlgorithm` plus `Individual`/`Population` dataclasses. Pure GA logic (tournament selection, elitism, mutation/crossover). It does **not** embed anything itself — it calls back into the injected `VecBookIndex` to evaluate fitness (`_evaluate_fitness` → cosine similarity to the stored barycenter). **Genome is pluggable via a codec** (`genome_mode`): `CharCodec` evolves a sequence of characters; `TokenCodec` evolves a sequence of GPT/BPE token ids (`tiktoken`) so output is real word-pieces rather than character soup. An `Individual` carries `genome` (list of units) plus the decoded `.text`; all operators work on the unit list and the codec's `to_text` produces the string. Crossover/mutation are codec-agnostic — to add a new representation, add a codec (`random_unit`/`random_genome`/`to_text`), not new operators. Token mode reaches roughly 2× the fitness ceiling of char mode on the same target. `set_genome_mode()` switches the codec; the `/evolution/initialize` endpoint exposes it. `tiktoken` fetches its BPE ranks from the network on first use (then caches).
- **`textrec/text_records.py`** — discovers `.txt` files recursively and parses them into records split on `---` separators, tracking file path / byte offset / index.

**Lazy initialization is deliberate.** `server.py` never imports `VecBookIndex` or `EvolutionaryAlgorithm` at module level — `get_vecbook_index()` and `get_evolutionary_algorithm()` construct them on first use. This keeps imports (and the fast test suite) free of the heavy sentence-transformers/FAISS dependency and avoids loading the model until an endpoint actually needs it. Preserve this pattern: don't add top-level imports of the embedding/GA modules into `server.py`.

**Data flow for a typical session**: `POST /barycenter` (set target → `VecBookIndex.set_target_strings`) → `POST /evolution/initialize` (build population, evaluate fitness vs. barycenter) → `POST /evolution/step` (run N generations) / `GET /evolution/status`. The barycenter set in step 1 is the shared dependency for everything after.

## Source of truth

`src/evolvattention/SPEC.md` is the authoritative API contract (request/response shapes, MUST requirements, error formats `{"error": ..., "status": "error"}`). When changing endpoints, update SPEC.md alongside the code. `src/evolvattention/README.md` documents GA defaults (population 50, elite 5, tournament 3, crossover 0.8, mutation 0.1).

## Notes

- `requires-python = ">=3.8"`; mypy/black target py38.
- The `data/` directory is created on demand and is where `VecBookIndex` looks for text records.
- `bak/`, `.idea/`, `.venv/` are gitignored. `epics.txt`, `ideas.txt`, `evolvattention.txt` are planning notes, not code.
