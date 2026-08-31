'''Query-time retrieval: embed a question, fetch the nearest chunks from Chroma,
and expand each hit with neighboring sibling chunks for more context.

`config.K_INITIAL` is reserved for a future reranking stage and is intentionally
unused here -- v1 has no reranker, so there's no benefit to fetching a broader
candidate pool than we return. Retrieval queries Chroma directly for `k` results.
'''

from __future__ import annotations

import functools
import json
from pathlib import Path

from src.rag import config
from src.rag.chunk import count_tokens
from src.rag.embed import embed_query, get_collection, load_model


@functools.lru_cache(maxsize=None)
def _load_chunks_index(chunks_path: Path) -> dict[str, dict[int, dict]]:
    '''Group chunk records by parent_id, then by chunk_index, for neighbor lookup.

    Result is cached keyed on *chunks_path* so the JSONL file is parsed at most
    once per distinct path for the lifetime of the process.
    '''
    index: dict[str, dict[int, dict]] = {}
    with chunks_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            index.setdefault(chunk['parent_id'], {})[chunk['chunk_index']] = chunk
    return index


@functools.lru_cache(maxsize=None)
def _load_discourse_lookup(discourses_path: Path) -> dict[str, str]:
    '''Map parent_id -> full discourse text, for expanding chunks back into their source.

    Result is cached keyed on *discourses_path* so the JSONL file is parsed at
    most once per distinct path for the lifetime of the process.
    '''
    lookup: dict[str, str] = {}
    with discourses_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            discourse = json.loads(line)
            lookup[discourse['id']] = discourse['text']
    return lookup


def _expand_chunk(
    *,
    parent_id: str,
    chunk_index: int,
    text: str,
    n_tokens: int,
    chunks_index: dict[str, dict[int, dict]],
    discourse_lookup: dict[str, str],
    neighbors: int,
    max_tokens: int,
) -> tuple[str, int, bool]:
    '''Try to expand `text` with up to `neighbors` sibling chunks on each side
    (by chunk_index), slicing a clean span out of the parent discourse via char
    offsets so the shared overlap between adjacent chunks isn't duplicated.

    Falls back to (text, n_tokens, False) when there's nothing to expand into,
    the parent text isn't available, or the expanded span would exceed
    `max_tokens`.
    '''
    siblings = chunks_index.get(parent_id, {})
    window = [i for i in range(chunk_index - neighbors, chunk_index + neighbors + 1) if i in siblings]
    if len(window) <= 1:
        return text, n_tokens, False

    parent_text = discourse_lookup.get(parent_id)
    if parent_text is None:
        return text, n_tokens, False

    first, last = siblings[window[0]], siblings[window[-1]]
    expanded_text = parent_text[first['char_start']:last['char_end']]
    expanded_tokens = count_tokens(expanded_text)
    if expanded_tokens > max_tokens:
        return text, n_tokens, False

    return expanded_text, expanded_tokens, True


def retrieve(
    query: str,
    *,
    k: int = config.K_FINAL,
    neighbors: int = config.EXPAND_NEIGHBORS,
    max_expanded_tokens: int = config.MAX_EXPANDED_TOKENS,
    chunks_path: Path = config.CHUNKS_PATH,
    discourses_path: Path = config.DISCOURSES_PATH,
    persist_dir: Path = config.CHROMA_DIR,
    collection_name: str = config.COLLECTION_NAME,
) -> list[dict]:
    '''Embed `query`, fetch the top `k` chunks from Chroma, and expand each into
    its surrounding context. Returns results sorted by descending score
    (Chroma's native nearest-neighbor order).
    '''
    model = load_model()
    collection = get_collection(persist_dir, collection_name)
    qvec = embed_query(model, query)

    res = collection.query(
        query_embeddings=[qvec.tolist()],
        n_results=k,
        include=['documents', 'metadatas', 'distances'],
    )

    if not res['ids'][0]:
        return []

    chunks_index = _load_chunks_index(chunks_path)
    discourse_lookup = _load_discourse_lookup(discourses_path)

    results = []
    for chunk_id, doc, meta, dist in zip(
        res['ids'][0], res['documents'][0], res['metadatas'][0], res['distances'][0]
    ):
        text, n_tokens, expanded = _expand_chunk(
            parent_id=meta['parent_id'],
            chunk_index=meta['chunk_index'],
            text=doc,
            n_tokens=meta['n_tokens'],
            chunks_index=chunks_index,
            discourse_lookup=discourse_lookup,
            neighbors=neighbors,
            max_tokens=max_expanded_tokens,
        )
        results.append({
            'chunk_id': chunk_id,
            'parent_id': meta['parent_id'],
            'header': meta['header'],
            'title': meta['title'],
            'score': 1 - dist,
            'text': text,
            'n_tokens': n_tokens,
            'expanded': expanded,
        })

    return results
