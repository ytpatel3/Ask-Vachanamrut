'''Unit tests for src.rag.embed.

Tests use mocks for SentenceTransformer and Chroma where reasonable so the core
logic (query-instruction prefix, batching, idempotent upsert, stale-id deletion,
metadata schema) can be verified without downloading the 1.3GB BGE model.
A separate live integration test (marked) exercises the real model + Chroma.
'''

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.rag import config
from src.rag import embed as em


# ---- pure logic tests -------------------------------------------------------

def test_chunk_to_metadata_keeps_only_primitives():
    chunk = {
        'chunk_id': 'X__c000',
        'parent_id': 'X',
        'header': 'X-1',
        'title': 'Some title',
        'chunk_index': 0,
        'text': 'lorem ipsum',
        'char_start': 0,
        'char_end': 50,
        'n_tokens': 12,
    }
    md = em._chunk_to_metadata(chunk)
    assert set(md.keys()) == {'parent_id', 'header', 'title', 'chunk_index', 'n_tokens'}
    for v in md.values():
        assert isinstance(v, (str, int, float, bool))


def test_iter_chunks_yields_each_record(tmp_path: Path):
    p = tmp_path / 'chunks.jsonl'
    rows = [{'chunk_id': f'A__c{i:03d}', 'text': f'row {i}'} for i in range(3)]
    p.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    out = list(em._iter_chunks(p))
    assert out == rows


def test_iter_chunks_skips_blank_lines(tmp_path: Path):
    p = tmp_path / 'chunks.jsonl'
    p.write_text('\n{"a": 1}\n\n  \n{"b": 2}\n')
    out = list(em._iter_chunks(p))
    assert out == [{'a': 1}, {'b': 2}]


# ---- query-instruction prefix (core BGE asymmetry) -------------------------

def test_embed_query_prepends_bge_instruction():
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 4), dtype=np.float32)
    em.embed_query(fake_model, 'what is bhakti?')
    args, kwargs = fake_model.encode.call_args
    inputs = args[0]
    assert len(inputs) == 1
    assert inputs[0].startswith(config.BGE_QUERY_INSTRUCTION)
    assert inputs[0].endswith('what is bhakti?')
    # Must request normalization for cosine similarity to behave correctly.
    assert kwargs.get('normalize_embeddings') is True


def test_embed_passages_does_not_prepend_instruction():
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((2, 4), dtype=np.float32)
    em.embed_passages(fake_model, ['passage one', 'passage two'])
    args, kwargs = fake_model.encode.call_args
    inputs = args[0]
    assert inputs == ['passage one', 'passage two']
    assert kwargs.get('normalize_embeddings') is True


# ---- stale-id deletion -----------------------------------------------------

def test_delete_stale_removes_only_missing_ids():
    collection = MagicMock()
    collection.get.return_value = {'ids': ['A__c000', 'A__c001', 'A__c002', 'A__c003']}
    n = em._delete_stale(collection, current_ids={'A__c000', 'A__c001'})
    assert n == 2
    args, kwargs = collection.delete.call_args
    deleted = set(kwargs.get('ids') or args[0])
    assert deleted == {'A__c002', 'A__c003'}


def test_delete_stale_noop_when_all_present():
    collection = MagicMock()
    collection.get.return_value = {'ids': ['A__c000']}
    n = em._delete_stale(collection, current_ids={'A__c000'})
    assert n == 0
    collection.delete.assert_not_called()


# ---- build_index orchestration (no model, no real chroma) -----------------

def test_build_index_orchestrates_correctly(tmp_path: Path):
    chunks_path = tmp_path / 'chunks.jsonl'
    chunks = [
        {'chunk_id': f'P__c{i:03d}', 'parent_id': 'P', 'header': 'P-1', 'title': 't',
         'chunk_index': i, 'text': f'doc {i}', 'char_start': 0, 'char_end': 1, 'n_tokens': 5}
        for i in range(7)
    ]
    chunks_path.write_text('\n'.join(json.dumps(c) for c in chunks) + '\n')

    fake_collection = MagicMock()
    fake_collection.get.return_value = {'ids': []}
    fake_collection.count.return_value = 7
    fake_client = MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    fake_model = MagicMock()
    fake_model.encode.side_effect = lambda texts, **kw: np.zeros((len(texts), 8), dtype=np.float32)

    with patch.object(em, 'load_model', return_value=fake_model):
        with patch('chromadb.PersistentClient', return_value=fake_client):
            count = em.build_index(
                chunks_path=chunks_path,
                persist_dir=tmp_path / 'chroma',
                collection_name='test_v1',
                batch_size=3,
                progress=False,
            )

    assert count == 7
    # Verify upsert was called multiple times (batched: 3 + 3 + 1 = 3 calls).
    assert fake_collection.upsert.call_count == 3
    # Collect all upserted ids.
    all_ids = []
    for call in fake_collection.upsert.call_args_list:
        all_ids.extend(call.kwargs['ids'])
    assert all_ids == [c['chunk_id'] for c in chunks]
    # Each upsert call must include embeddings, documents, metadatas of matching length.
    for call in fake_collection.upsert.call_args_list:
        n = len(call.kwargs['ids'])
        assert len(call.kwargs['embeddings']) == n
        assert len(call.kwargs['documents']) == n
        assert len(call.kwargs['metadatas']) == n


def test_build_index_rebuild_drops_existing_collection(tmp_path: Path):
    chunks_path = tmp_path / 'chunks.jsonl'
    chunks_path.write_text(json.dumps({
        'chunk_id': 'P__c000', 'parent_id': 'P', 'header': 'P-1', 'title': 't',
        'chunk_index': 0, 'text': 'd', 'char_start': 0, 'char_end': 1, 'n_tokens': 1,
    }) + '\n')

    fake_collection = MagicMock()
    fake_collection.get.return_value = {'ids': []}
    fake_collection.count.return_value = 1
    fake_client = MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 4), dtype=np.float32)

    with patch.object(em, 'load_model', return_value=fake_model):
        with patch('chromadb.PersistentClient', return_value=fake_client):
            em.build_index(
                chunks_path=chunks_path,
                persist_dir=tmp_path / 'chroma',
                collection_name='test_v1',
                rebuild=True,
                progress=False,
            )

    fake_client.delete_collection.assert_called_once_with('test_v1')


def test_build_index_handles_empty_chunks_file(tmp_path: Path):
    chunks_path = tmp_path / 'chunks.jsonl'
    chunks_path.write_text('')

    fake_collection = MagicMock()
    fake_collection.count.return_value = 0
    fake_client = MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    with patch('chromadb.PersistentClient', return_value=fake_client):
        count = em.build_index(
            chunks_path=chunks_path,
            persist_dir=tmp_path / 'chroma',
            collection_name='test_v1',
            progress=False,
        )

    assert count == 0
    fake_collection.upsert.assert_not_called()


# ---- live integration (marked; runs only when explicitly opted in) --------

@pytest.mark.live
def test_live_embed_smoke(tmp_path: Path):
    '''End-to-end with the real model on a few synthetic chunks. Slow.'''
    chunks_path = tmp_path / 'chunks.jsonl'
    rows = [
        {'chunk_id': f'X__c{i:03d}', 'parent_id': 'X', 'header': 'X-1', 'title': 't',
         'chunk_index': i, 'text': txt, 'char_start': 0, 'char_end': len(txt), 'n_tokens': 0}
        for i, txt in enumerate([
            'The devotee should always remember God.',
            'Persistence in spiritual endeavour leads to liberation.',
            'Maya is anything that obstructs meditation on God.',
        ])
    ]
    chunks_path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')

    persist = tmp_path / 'chroma'
    n = em.build_index(
        chunks_path=chunks_path,
        persist_dir=persist,
        collection_name='live_smoke_v1',
        progress=False,
    )
    assert n == 3
    # Query with the BGE prefix and confirm the most-relevant doc is returned.
    model = em.load_model()
    qvec = em.embed_query(model, 'how do I remember God?')
    coll = em.get_collection(persist, 'live_smoke_v1')
    res = coll.query(query_embeddings=[qvec.tolist()], n_results=3)
    top_id = res['ids'][0][0]
    assert top_id == 'X__c000'  # remembrance-of-God doc should rank first


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
