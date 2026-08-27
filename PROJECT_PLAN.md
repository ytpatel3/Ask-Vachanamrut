# Ask-Vachanamrut — Project Plan

Living plan for the rest of the project. Update this file as milestones land or
priorities shift; it is the source of truth for "what's next," not the original
roadmap doc.

## Workflow

Solo project, but using standard branch/PR practices rather than committing
directly to `main`:

1. Branch off `main` for each piece of work: `git checkout -b feat/<short-name>`
   (or `fix/`, `chore/` as appropriate).
2. Commit on the branch, push with `-u origin <branch>`.
3. Open a PR into `main` on GitHub. Let GitHub Copilot's automated review run.
4. Address review feedback with additional commits on the branch.
5. Squash-merge the PR once satisfied, then delete the branch.
6. Pull `main` locally before starting the next branch.

Claude provides commands and commit/PR messages; the user runs all git and
GitHub operations from their own terminal. See `CLAUDE.md`.

## Done

- **PDF/text ingestion** (`src/pdf_ingestion/`, `src/text_normalization/`) — raw
  Vachanamrut text cleaned, 273 discourses parsed into `data/clean/vachanamrut.jsonl`
  with header/title/text/id fields.
- **Chunking** (`src/rag/chunk.py`) — parent discourses split into ~400-token child
  chunks with paragraph/sentence-aware overlap, capped so overlap can't balloon on
  oversized trailing units. Deterministic, unit-tested.
- **Embedding + indexing** (`src/rag/embed.py`, `scripts/build_index.py`) — child
  chunks embedded with `BAAI/bge-large-en-v1.5` (asymmetric query/passage handling)
  and upserted into a persistent Chroma collection. CLI chains chunk → embed → index.

## Next: retrieval + generation loop

The core RAG loop — the part that actually answers a question — hasn't started.
This is the priority.

1. **`src/rag/retrieve.py`**
   - `retrieve(query: str, k: int = config.K_INITIAL) -> list[dict]`: embed the
     query with `embed_query`, run `collection.query(...)`, return chunk dicts
     with scores.
   - Parent-expansion: use `config.EXPAND_NEIGHBORS` / `config.MAX_EXPANDED_TOKENS`
     to pull in sibling chunks from the same parent discourse when useful, capped
     by token budget. (Config already declares these; nothing consumes them yet.)
   - Decide now whether reranking (`config.RERANKER_MODEL`, already declared) is
     in scope for v1 or a later pass — don't leave it half-wired.

2. **`src/rag/generate.py`**
   - Prompt template: system prompt (concise, cite chunk/discourse ids, admit
     uncertainty) + user message with assembled context + question.
   - Context assembly: concatenate top-k (or expanded) chunks under a token
     budget, with source markers (e.g. `[Gadhada I-1]`).
   - Call `RAG_MODEL` (`claude-sonnet-4-6`) via the `anthropic` SDK (already a
     dependency). Return answer text + the chunk ids actually cited.

3. **`src/rag/pipeline.py`** (or fold into `generate.py` if it stays small)
   - `answer(query: str) -> dict` wiring retrieve → assemble → generate →
     return `{answer, sources}`.
   - This is the one function the UI and eval harness both call.

## Then: evaluation

- Build a small ground-truth Q&A set (start with ~20-30 pairs, grow later) with
  expected supporting chunk/discourse ids. Store as JSONL under `data/eval/`.
- `src/rag/eval.py`: recall@k and MRR against the test set; simple citation-
  precision check (did the generated answer cite an expected chunk).
- Run once before touching the UI so retrieval quality is known and any chunking/
  embedding regressions are caught early.

## Then: UI

- Minimal Streamlit app (`app.py` or `src/rag/app.py`): query box, answer display
  with inline citations, a "show retrieved chunks" debug panel.
- Keep it thin — it should only call `pipeline.answer()`, no logic duplicated
  here.

## Then: polish / deployment

- Flesh out `README.md`: setup instructions, how to rebuild the index
  (`python -m scripts.build_index`), how to run the app, env vars needed
  (`ANTHROPIC_API_KEY`, optional `EMBEDDING_MODEL`/`RERANKER_MODEL`/`RAG_MODEL`
  overrides).
- Dockerfile for the Streamlit app, if still targeting zero-cost deployment
  (Streamlit Community Cloud / local).
- Optional: `.env.example` documenting required env vars.

## Explicit scope decisions (deviations from the original roadmap, kept intentionally)

- **Dense-only retrieval**, no TF-IDF/BM25 hybrid fusion. Revisit only if eval
  shows a recall gap that hybrid would plausibly close — don't add sparse
  retrieval speculatively.
- **Chroma instead of raw FAISS.** Chosen for persistence and simpler API;
  revisit only if index size/performance becomes a real constraint.
- **Anthropic (Claude) instead of OpenAI** for generation, matching the
  `anthropic` dependency already in `requirements.txt`.

## Future (post-v1, not currently scoped)

- Additional scriptures beyond the Vachanamrut.
- JSON → TOON migration for storage format, if it materializes as a real format.
- Conversational RAG (multi-turn, history-conditioned retrieval).
- Cross-encoder reranking, if eval shows it's needed.
