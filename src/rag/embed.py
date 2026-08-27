'''Embed child chunks with BAAI/bge-large-en-v1.5 and upsert into a Chroma collection.

BGE is asymmetric: passages are embedded as-is, queries are prepended with the
BGE_QUERY_INSTRUCTION. We provide vectors directly to Chroma (no implicit embedder).
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.rag import config


_MODEL = None


def load_model(name: str = config.EMBEDDING_MODEL, device: str | None = None):
    '''Lazy-load and cache the sentence-transformers model.'''
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(name, device=device)
    return _MODEL


def embed_passages(model, texts: Sequence[str], batch_size: int = config.EMBED_BATCH_SIZE) -> np.ndarray:
    '''Embed passages (no query instruction). Returns L2-normalized (N, D) array.'''
    return np.asarray(model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ))


def embed_query(model, query: str) -> np.ndarray:
    '''Embed a query, prepending the BGE instruction prefix.'''
    prefixed = f'{config.BGE_QUERY_INSTRUCTION}{query}'
    vec = model.encode(
        [prefixed],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vec[0])


def get_collection(persist_dir: Path, name: str):
    '''Open or create the Chroma collection. Cosine distance, no implicit embedder.'''
    import chromadb
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=name,
        embedding_function=None,
        metadata={'hnsw:space': 'cosine'},
    )


def _iter_chunks(path: Path) -> Iterable[dict]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _chunk_to_metadata(chunk: dict) -> dict:
    '''Chroma metadata fields must be primitives.'''
    return {
        'parent_id': chunk['parent_id'],
        'header': chunk['header'],
        'title': chunk['title'],
        'chunk_index': int(chunk['chunk_index']),
        'n_tokens': int(chunk['n_tokens']),
    }


def _delete_stale(collection, current_ids: set[str]) -> int:
    '''Remove ids in the collection that are no longer present in the chunks file.'''
    existing = collection.get(include=[])  # ids only
    existing_ids = set(existing.get('ids', []))
    stale = list(existing_ids - current_ids)
    if stale:
        collection.delete(ids=stale)
    return len(stale)


def build_index(
    chunks_path: Path,
    persist_dir: Path,
    collection_name: str,
    *,
    rebuild: bool = False,
    batch_size: int = config.EMBED_BATCH_SIZE,
    progress: bool = True,
) -> int:
    '''Embed every chunk in `chunks_path` and upsert into the collection.

    Returns the count of vectors in the collection after indexing.
    `rebuild=True` deletes the collection first.
    '''
    import chromadb
    from tqdm import tqdm

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    if rebuild:
        try:
            client.delete_collection(collection_name)
        except (ValueError, chromadb.errors.NotFoundError if hasattr(chromadb, 'errors') else Exception):
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
        metadata={'hnsw:space': 'cosine'},
    )

    chunks = list(_iter_chunks(chunks_path))
    if not chunks:
        return collection.count()

    model = load_model()

    ids = [c['chunk_id'] for c in chunks]
    documents = [c['text'] for c in chunks]
    metadatas = [_chunk_to_metadata(c) for c in chunks]

    iterator = range(0, len(chunks), batch_size)
    if progress:
        iterator = tqdm(iterator, desc='embedding', unit='batch')

    for start in iterator:
        end = min(start + batch_size, len(chunks))
        batch_texts = documents[start:end]
        batch_emb = embed_passages(model, batch_texts, batch_size=batch_size)
        collection.upsert(
            ids=ids[start:end],
            embeddings=batch_emb.tolist(),
            documents=batch_texts,
            metadatas=metadatas[start:end],
        )

    _delete_stale(collection, set(ids))
    return collection.count()
