'''Unit tests for src.rag.chunk.'''

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.rag import chunk as cm


# ---- helpers ----------------------------------------------------------------

def _make_discourse(text: str, *, _id: str = 'Test_1', header: str = 'Test 1', title: str = 'Title') -> dict:
    return {'id': _id, 'header': header, 'title': title, 'text': text}


def _para(n_words: int, seed: str = 'word') -> str:
    return ' '.join(f'{seed}{i}' for i in range(n_words)) + '.'


# ---- tokens & splitting -----------------------------------------------------

def test_count_tokens_grows_with_text():
    assert cm.count_tokens('hello') < cm.count_tokens('hello world how are you today')


def test_split_paragraphs_offsets_roundtrip():
    text = 'First paragraph.\n\nSecond para.\n\n\nThird block here.'
    paras = cm.split_paragraphs(text)
    assert [p[0] for p in paras] == ['First paragraph.', 'Second para.', 'Third block here.']
    # Offsets must locate the paragraph in the original text.
    for body, start, end in paras:
        assert text[start:end] == body


def test_split_paragraphs_skips_empty_blocks():
    text = '\n\n\n\nFirst.\n\n\n\n   \n\nSecond.\n\n'
    paras = cm.split_paragraphs(text)
    assert [p[0] for p in paras] == ['First.', 'Second.']


def test_split_sentences_basic():
    s = 'He said this. She replied that! Did they agree? Yes, they did.'
    out = cm.split_sentences(s)
    assert len(out) == 4
    assert out[0] == 'He said this.'
    assert out[-1] == 'Yes, they did.'


# ---- chunk_discourse: small / medium / large -------------------------------

def test_tiny_discourse_emits_single_chunk():
    text = 'A short discourse.\n\nWith two paragraphs.'
    chunks = cm.chunk_discourse(_make_discourse(text))
    assert len(chunks) == 1
    c = chunks[0]
    assert c['chunk_id'] == 'Test_1__c000'
    assert c['parent_id'] == 'Test_1'
    assert c['chunk_index'] == 0
    assert c['header'] == 'Test 1'
    assert c['title'] == 'Title'
    assert 'A short discourse.' in c['text']
    assert 'With two paragraphs.' in c['text']
    assert c['n_tokens'] == cm.count_tokens(c['text'])


def test_medium_discourse_emits_multiple_chunks_with_overlap():
    paragraphs = [_para(120, f's{k}') for k in range(8)]  # ~120 tokens each → ~960 total
    text = '\n\n'.join(paragraphs)
    chunks = cm.chunk_discourse(_make_discourse(text), target=400, max_tokens=480, overlap=50, min_tokens=80)
    assert len(chunks) >= 2
    # Sequential indices, deterministic ids.
    for i, c in enumerate(chunks):
        assert c['chunk_index'] == i
        assert c['chunk_id'] == f'Test_1__c{i:03d}'
    # Every chunk respects max_tokens within ~25% (overlap can push slightly above target but under max).
    for c in chunks:
        assert c['n_tokens'] <= 480 + 100, f'chunk {c["chunk_id"]} too big: {c["n_tokens"]}'
    # Overlap: every non-first chunk shares at least one paragraph with the previous one.
    for i in range(1, len(chunks)):
        prev_paras = [p.strip() for p in chunks[i - 1]['text'].split('\n\n') if p.strip()]
        curr_paras = [p.strip() for p in chunks[i]['text'].split('\n\n') if p.strip()]
        shared = set(prev_paras) & set(curr_paras)
        assert shared, f'no paragraph overlap between chunk {i - 1} and {i}'


def test_paragraph_packing_respects_target():
    # Real-prose paragraphs (~30-40 tokens each via tiktoken on natural English).
    paragraph_template = (
        'The devotee should always remember God with deep love and reverence. '
        'Through such constant remembrance, the mind becomes still and the soul '
        'is drawn ever closer to the divine presence within.'
    )
    paragraphs = [f'{paragraph_template} (Number {k}.)' for k in range(40)]
    text = '\n\n'.join(paragraphs)
    chunks = cm.chunk_discourse(_make_discourse(text), target=400, max_tokens=480, overlap=50, min_tokens=80)
    # Sanity: at least 2 chunks (real packing happened) and each chunk fits the budget.
    assert len(chunks) >= 2
    for c in chunks:
        assert c['n_tokens'] <= 480 + 100, f'chunk {c["chunk_id"]} exceeds budget: {c["n_tokens"]}'
    # Every paragraph must appear at least once across all chunks (no content lost).
    joined = '\n\n'.join(c['text'] for c in chunks)
    for orig in paragraphs:
        assert orig in joined


# ---- oversize-paragraph fallback -------------------------------------------

def test_oversize_paragraph_triggers_sentence_fallback():
    # Build one paragraph of many sentences that blows past max_tokens.
    sentences = [f'This is sentence {k} with some filler words.' for k in range(150)]
    huge = ' '.join(sentences)
    text = f'Intro paragraph.\n\n{huge}\n\nClosing paragraph.'
    chunks = cm.chunk_discourse(_make_discourse(text), target=400, max_tokens=480, overlap=50, min_tokens=80)
    # Must produce multiple chunks (huge paragraph alone > max).
    assert len(chunks) >= 2
    for c in chunks:
        assert c['n_tokens'] <= 480 + 100
    # Concatenation of all chunk text contains the huge paragraph's content.
    joined = ' '.join(c['text'] for c in chunks)
    assert 'sentence 0 ' in joined
    assert 'sentence 149' in joined


# ---- determinism ------------------------------------------------------------

def test_chunking_is_deterministic():
    paragraphs = [_para(80, f's{k}') for k in range(20)]
    text = '\n\n'.join(paragraphs)
    a = cm.chunk_discourse(_make_discourse(text))
    b = cm.chunk_discourse(_make_discourse(text))
    assert [c['chunk_id'] for c in a] == [c['chunk_id'] for c in b]
    assert [c['text'] for c in a] == [c['text'] for c in b]
    assert [c['n_tokens'] for c in a] == [c['n_tokens'] for c in b]


# ---- chunk_jsonl integration -----------------------------------------------

def test_chunk_jsonl_roundtrip(tmp_path: Path):
    in_path = tmp_path / 'in.jsonl'
    out_path = tmp_path / 'out.jsonl'
    discourses = [
        _make_discourse('Para A.\n\nPara B.', _id='A', header='A', title='ta'),
        _make_discourse('\n\n'.join(_para(100, f'b{k}') for k in range(8)), _id='B', header='B', title='tb'),
    ]
    in_path.write_text('\n'.join(json.dumps(d) for d in discourses) + '\n', encoding='utf-8')

    n = cm.chunk_jsonl(in_path, out_path)
    rows = [json.loads(line) for line in out_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert n == len(rows)
    assert n >= 2
    # Every chunk references an actual parent.
    parent_ids = {r['parent_id'] for r in rows}
    assert parent_ids == {'A', 'B'}
    # Required schema fields.
    required = {'chunk_id', 'parent_id', 'header', 'title', 'chunk_index', 'text', 'char_start', 'char_end', 'n_tokens'}
    for r in rows:
        assert required.issubset(r.keys())


# ---- char offsets are within parent text -----------------------------------

def test_char_offsets_index_into_parent_text():
    paragraphs = [_para(120, f's{k}') for k in range(6)]
    text = '\n\n'.join(paragraphs)
    d = _make_discourse(text)
    chunks = cm.chunk_discourse(d, target=400, overlap=50)
    for c in chunks:
        assert 0 <= c['char_start'] < c['char_end'] <= len(text)
        # The character at char_start must be the start of *some* paragraph in the parent.
        assert text[c['char_start']:c['char_start'] + 4] in text


# ---- empty edge case -------------------------------------------------------

def test_empty_discourse_produces_no_chunks():
    chunks = cm.chunk_discourse(_make_discourse(''))
    assert chunks == []


def test_single_short_paragraph_below_min_tokens():
    text = 'Hi there.'
    chunks = cm.chunk_discourse(_make_discourse(text))
    assert len(chunks) == 1
    assert chunks[0]['text'] == 'Hi there.'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
