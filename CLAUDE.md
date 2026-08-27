# CLAUDE.md

Guidance for Claude Code (and anyone else) working in this repo.

## Project

RAG system over the Vachanamrut (273 discourses). Pipeline: PDF/text ingestion →
normalization/parsing → parent-child chunking → embedding (BGE) → Chroma index →
(retrieval + generation, in progress). See `PROJECT_PLAN.md` for current status
and what's next — check it before starting new work, and update it as milestones
land.

## Code style

Match the conventions in `src/rag/` (the most current module) over older modules
like `src/text_normalization/` where they conflict.

- **Single quotes everywhere** — strings, dict keys, imports. Only use double
  quotes in comments, docstrings, and when the string itself needs to contain 
  an apostrophe/single quote and escaping would hurt readability 
  (e.g. `f"{parent['id']}__c{i:03d}"`).
- **Docstrings use triple single-quotes** (`'''...'''`), not `"""..."""`.
  One-line docstrings are a single line inside triple quotes; longer ones use
  a summary line, blank line, then detail.
- **Type hints** on function signatures, using builtin generics (`list[dict]`,
  `str | None`) — no `typing.List`/`typing.Optional`. `from __future__ import
  annotations` at the top of modules that need forward references.
- **Small, pure, testable functions.** Prefer a handful of single-purpose
  helpers (`_select_overlap_text`, `_make_chunk`, `_build_units`) composed by
  one public entrypoint, over one large function.
- **Config lives in `src/rag/config.py`.** No magic numbers for chunk sizes,
  model names, batch sizes, or retrieval k-values scattered in other files —
  add a constant there and import it. Env-overridable constants use
  `os.getenv('NAME', default)`.
- **Lazy imports for heavy/optional dependencies** (`sentence_transformers`,
  `chromadb`) inside the function that needs them, not at module top-level —
  keeps `--skip-embed`-style paths usable without those packages installed.
- **Comments are rare and explain 'why,' not 'what.'** A non-obvious invariant
  or workaround gets one line; nothing else.
- **Tests live under `tests/`, mirroring `src/`** (`tests/rag/test_chunk.py`
  next to `src/rag/chunk.py`). Mock heavy externals (real models, Chroma
  network calls) in unit tests; gate real end-to-end tests behind
  `@pytest.mark.live` (see `pytest.ini` — live tests are excluded by default).
- Every new module gets a test file before moving to the next module — write
  and run pytest for it before proceeding to other work.

## Running things

- Rebuild the full index: `python -m scripts.build_index` (flags:
  `--skip-chunk`, `--skip-embed`, `--rebuild`).
- Run tests: `pytest` (live/model-downloading tests are skipped by default;
  run them explicitly with `pytest -m live`).

## Git workflow

The user runs all git commands themselves, from their own terminal. Never run
`git commit`, `git push`, `git rebase`, `git merge`, or branch/PR operations —
provide the exact commands and commit messages in chat and let the user execute
them. Read-only inspection (`git status`, `git log`, `git diff`, `git fetch`) is
fine to run directly to gather context.

Standard flow for every change, even solo: create a feature branch off `main`,
commit there, push, open a PR into `main`, let GitHub Copilot's review run,
address feedback with more commits on the branch, then squash-merge and delete
the branch. Never commit directly to `main`. See `PROJECT_PLAN.md` for the full
workflow description.

## Things to avoid

- Don't add sparse/hybrid retrieval, reranking, or other roadmap extensions
  speculatively — `PROJECT_PLAN.md` lists what's intentionally deferred and why.
- Don't introduce a new config pattern (YAML, separate settings module, etc.)
  — everything tunable belongs in `src/rag/config.py`.
