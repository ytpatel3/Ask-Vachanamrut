'''Unit tests for src.rag.retrieve.'''

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

from src.rag import chunk as chunk_mod
from src.rag import retrieve as rt


# ---- helpers ----------------------------------------------------------------

def _index_by_parent(chunks: list[dict]) -> dict[str, dict[int, dict]]:
    index: dict[str, dict[int, dict]] = {}
    for c in chunks:
        index.setdefault(c['parent_id'], {})[c['chunk_index']] = c
    return index


# ---- _load_chunks_index / _load_discourse_lookup ----------------------------

def test_load_chunks_index_groups_by_parent_and_index(tmp_path: Path):
    rows = [
        {'chunk_id': 'A__c000', 'parent_id': 'A', 'chunk_index': 0, 'text': 'a0'},
        {'chunk_id': 'A__c001', 'parent_id': 'A', 'chunk_index': 1, 'text': 'a1'},
        {'chunk_id': 'B__c000', 'parent_id': 'B', 'chunk_index': 0, 'text': 'b0'},
    ]
    p = tmp_path / 'chunks.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')

    index = rt._load_chunks_index(p)
    assert set(index.keys()) == {'A', 'B'}
    assert index['A'][0]['text'] == 'a0'
    assert index['A'][1]['text'] == 'a1'
    assert index['B'][0]['text'] == 'b0'


def test_load_discourse_lookup_maps_id_to_text(tmp_path: Path):
    rows = [{'id': 'A', 'text': 'discourse a'}, {'id': 'B', 'text': 'discourse b'}]
    p = tmp_path / 'discourses.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')

    lookup = rt._load_discourse_lookup(p)
    assert lookup == {'A': 'discourse a', 'B': 'discourse b'}


# ---- _expand_chunk: built on real chunk_discourse() output -----------------
# Using real chunks (rather than hand-crafted char offsets) exercises the
# actual overlap/offset semantics from chunk.py.

def _make_multi_chunk_parent() -> tuple[list[dict], str]:
    paragraphs = [f'Paragraph {i} talks about topic {i} at some length here.' for i in range(5)]
    text = '\n\n'.join(paragraphs)
    discourse = {'id': 'P', 'header': 'P-1', 'title': 't', 'text': text}
    chunks = chunk_mod.chunk_discourse(
        discourse, target=10, max_tokens=14, overlap=2, overlap_max=4, min_tokens=1,
    )
    assert len(chunks) >= 5, 'expected one chunk per paragraph for this test to be meaningful'
    return chunks, text


def test_expand_chunk_expands_middle_chunk_using_char_offsets():
    chunks, parent_text = _make_multi_chunk_parent()
    index = _index_by_parent(chunks)
    lookup = {'P': parent_text}
    mid = chunks[2]

    text, n_tokens, expanded = rt._expand_chunk(
        parent_id='P', chunk_index=2, text=mid['text'], n_tokens=mid['n_tokens'],
        chunks_index=index, discourse_lookup=lookup, neighbors=1, max_tokens=1000,
    )

    assert expanded is True
    expected = parent_text[chunks[1]['char_start']:chunks[3]['char_end']]
    assert text == expected
    assert 'topic 1' in text and 'topic 2' in text and 'topic 3' in text
    assert 'topic 0' not in text and 'topic 4' not in text
    assert n_tokens == chunk_mod.count_tokens(expected)


def test_expand_chunk_at_first_index_only_expands_right():
    chunks, parent_text = _make_multi_chunk_parent()
    index = _index_by_parent(chunks)
    lookup = {'P': parent_text}
    first = chunks[0]

    text, n_tokens, expanded = rt._expand_chunk(
        parent_id='P', chunk_index=0, text=first['text'], n_tokens=first['n_tokens'],
        chunks_index=index, discourse_lookup=lookup, neighbors=1, max_tokens=1000,
    )

    assert expanded is True
    expected = parent_text[chunks[0]['char_start']:chunks[1]['char_end']]
    assert text == expected


def test_expand_chunk_at_last_index_only_expands_left():
    chunks, parent_text = _make_multi_chunk_parent()
    index = _index_by_parent(chunks)
    lookup = {'P': parent_text}
    last = chunks[-1]
    last_idx = last['chunk_index']

    text, n_tokens, expanded = rt._expand_chunk(
        parent_id='P', chunk_index=last_idx, text=last['text'], n_tokens=last['n_tokens'],
        chunks_index=index, discourse_lookup=lookup, neighbors=1, max_tokens=1000,
    )

    assert expanded is True
    expected = parent_text[chunks[-2]['char_start']:chunks[-1]['char_end']]
    assert text == expected


def test_expand_chunk_returns_unexpanded_for_single_chunk_parent():
    discourse = {'id': 'S', 'header': 'S-1', 'title': 't', 'text': 'A single short discourse.'}
    chunks = chunk_mod.chunk_discourse(discourse)
    assert len(chunks) == 1
    index = _index_by_parent(chunks)
    lookup = {'S': discourse['text']}

    text, n_tokens, expanded = rt._expand_chunk(
        parent_id='S', chunk_index=0, text=chunks[0]['text'], n_tokens=chunks[0]['n_tokens'],
        chunks_index=index, discourse_lookup=lookup, neighbors=1, max_tokens=1000,
    )

    assert expanded is False
    assert text == chunks[0]['text']
    assert n_tokens == chunks[0]['n_tokens']


def test_expand_chunk_falls_back_when_over_token_budget():
    chunks, parent_text = _make_multi_chunk_parent()
    index = _index_by_parent(chunks)
    lookup = {'P': parent_text}
    mid = chunks[2]

    text, n_tokens, expanded = rt._expand_chunk(
        parent_id='P', chunk_index=2, text=mid['text'], n_tokens=mid['n_tokens'],
        chunks_index=index, discourse_lookup=lookup, neighbors=1, max_tokens=1,
    )

    assert expanded is False
    assert text == mid['text']
    assert n_tokens == mid['n_tokens']


def test_expand_chunk_falls_back_when_parent_text_missing():
    chunks, _ = _make_multi_chunk_parent()
    index = _index_by_parent(chunks)
    mid = chunks[2]

    text, n_tokens, expanded = rt._expand_chunk(
        parent_id='P', chunk_index=2, text=mid['text'], n_tokens=mid['n_tokens'],
        chunks_index=index, discourse_lookup={}, neighbors=1, max_tokens=1000,
    )

    assert expanded is False
    assert text == mid['text']


# ---- retrieve(): mocked orchestration ---------------------------------------

def test_retrieve_orchestrates_correctly(tmp_path: Path):
    chunk = {
        'chunk_id': 'A__c000', 'parent_id': 'A', 'chunk_index': 0, 'header': 'A-1',
        'title': 't', 'text': 'hello world', 'char_start': 0, 'char_end': 11, 'n_tokens': 2,
    }
    chunks_path = tmp_path / 'chunks.jsonl'
    chunks_path.write_text(json.dumps(chunk) + '\n')
    discourses_path = tmp_path / 'discourses.jsonl'
    discourses_path.write_text(json.dumps({'id': 'A', 'text': 'hello world'}) + '\n')

    fake_model = MagicMock()
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        'ids': [['A__c000']],
        'documents': [['hello world']],
        'metadatas': [[{'parent_id': 'A', 'chunk_index': 0, 'header': 'A-1', 'title': 't', 'n_tokens': 2}]],
        'distances': [[0.2]],
    }

    with patch.object(rt, 'load_model', return_value=fake_model), \
         patch.object(rt, 'get_collection', return_value=fake_collection) as mock_get_coll, \
         patch.object(rt, 'embed_query', return_value=np.zeros(4)) as mock_embed_query:
        results = rt.retrieve(
            'hello?', k=5, chunks_path=chunks_path, discourses_path=discourses_path,
        )

    mock_embed_query.assert_called_once_with(fake_model, 'hello?')
    mock_get_coll.assert_called_once()
    _, query_kwargs = fake_collection.query.call_args
    assert query_kwargs['n_results'] == 5

    assert len(results) == 1
    r = results[0]
    assert r['chunk_id'] == 'A__c000'
    assert r['header'] == 'A-1'
    assert r['score'] == pytest.approx(0.8)  # 1 - distance
    assert r['expanded'] is False  # single-chunk parent, nothing to expand into


def test_retrieve_returns_empty_list_when_no_hits(tmp_path: Path):
    chunks_path = tmp_path / 'chunks.jsonl'
    chunks_path.write_text('')
    discourses_path = tmp_path / 'discourses.jsonl'
    discourses_path.write_text('')

    fake_collection = MagicMock()
    fake_collection.query.return_value = {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

    with patch.object(rt, 'load_model', return_value=MagicMock()), \
         patch.object(rt, 'get_collection', return_value=fake_collection), \
         patch.object(rt, 'embed_query', return_value=np.zeros(4)):
        results = rt.retrieve('anything', chunks_path=chunks_path, discourses_path=discourses_path)

    assert results == []


# ---- live integration (marked; runs only when explicitly opted in) --------

@pytest.mark.live
def test_live_retrieve_returns_relevant_results():
    '''End-to-end against the real built index. Slow; requires the index to exist.'''
    results = rt.retrieve('What is the nature of maya?', k=3)
    assert len(results) == 3
    for r in results:
        assert -1.0 <= r['score'] <= 1.0
        assert r['text']
        assert r['n_tokens'] > 0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
