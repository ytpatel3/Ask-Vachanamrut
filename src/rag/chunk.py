'''Parent-child chunking for the Vachanamrut.

Each parent discourse from `vachanamrut.jsonl` is split into smaller child chunks
sized for embedding (~400 tokens, paragraph-boundary overlap of ~50 tokens).
Output is `vachanamrut_chunks.jsonl`, one JSON per line.
'''

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from src.rag import config

# Paragraph break: one blank line (allows internal whitespace).
_PARA_BREAK = re.compile(r'\n\s*\n')

# Sentence boundary: period/?/! followed by whitespace then a capital or quote.
_SENT_BOUNDARY = re.compile(r'(?<=[.?!])\s+(?=["\'A-Z])')

_ENCODING = None


def _get_encoding():
    global _ENCODING
    if _ENCODING is None:
        import tiktoken
        _ENCODING = tiktoken.get_encoding('cl100k_base')
    return _ENCODING


def count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))


def split_paragraphs(text: str) -> list[tuple[str, int, int]]:
    '''Split on blank-line boundaries; return (paragraph, start, end) with offsets in `text`.'''
    paragraphs: list[tuple[str, int, int]] = []
    cursor = 0
    for match in _PARA_BREAK.finditer(text):
        block = text[cursor:match.start()]
        stripped = block.strip()
        if stripped:
            offset = cursor + (len(block) - len(block.lstrip()))
            paragraphs.append((stripped, offset, offset + len(stripped)))
        cursor = match.end()
    if cursor < len(text):
        block = text[cursor:]
        stripped = block.strip()
        if stripped:
            offset = cursor + (len(block) - len(block.lstrip()))
            paragraphs.append((stripped, offset, offset + len(stripped)))
    return paragraphs


def split_sentences(paragraph: str) -> list[str]:
    '''Best-effort sentence splitter (regex; works for English prose).'''
    parts = _SENT_BOUNDARY.split(paragraph)
    return [p.strip() for p in parts if p.strip()]


def _sentence_units(para_text: str, para_start: int) -> list[dict]:
    '''Sentence-split an oversize paragraph; emit unit dicts with parent-text offsets.'''
    units: list[dict] = []
    cursor = 0
    for sent in split_sentences(para_text):
        idx = para_text.find(sent, cursor)
        if idx == -1:
            idx = cursor
        units.append({
            'text': sent,
            'start': para_start + idx,
            'end': para_start + idx + len(sent),
            'tokens': count_tokens(sent),
        })
        cursor = idx + len(sent)
    return units


def _build_units(text: str, *, max_tokens: int) -> list[dict]:
    '''Convert discourse text into atomic units (paragraphs, with sentence fallback).'''
    units: list[dict] = []
    for para_text, p_start, p_end in split_paragraphs(text):
        para_tokens = count_tokens(para_text)
        if para_tokens > max_tokens:
            units.extend(_sentence_units(para_text, p_start))
        else:
            units.append({
                'text': para_text,
                'start': p_start,
                'end': p_end,
                'tokens': para_tokens,
            })
    return units


def _select_overlap_text(
    prev_primary: list[int],
    units: list[dict],
    min_overlap: int,
    max_overlap: int,
) -> str:
    '''Build an overlap prefix from the tail of `prev_primary`.

    Whole-unit overlap is preferred; if the trailing unit alone exceeds `max_overlap`,
    fall back to taking its trailing sentences.
    '''
    parts: list[str] = []
    tokens = 0
    for j in reversed(prev_primary):
        if tokens >= min_overlap:
            break
        u_text = units[j]['text']
        u_tok = units[j]['tokens']
        if tokens + u_tok <= max_overlap:
            parts.insert(0, u_text)
            tokens += u_tok
            continue
        # Trailing unit is too big to include whole. Take its tail sentences,
        # but only sentences that fit under max_overlap. If none fit, skip overlap.
        sentences = split_sentences(u_text)
        tail: list[str] = []
        tail_tok = 0
        for s in reversed(sentences):
            s_tok = count_tokens(s)
            if tokens + tail_tok + s_tok > max_overlap:
                break
            tail.insert(0, s)
            tail_tok += s_tok
            if tokens + tail_tok >= min_overlap:
                break
        if tail:
            parts.insert(0, ' '.join(tail))
            tokens += tail_tok
        break  # don't keep walking back past a huge unit
    return '\n\n'.join(parts)


def _make_chunk(
    *,
    parent: dict,
    chunk_index: int,
    overlap_text: str,
    primary_indices: list[int],
    units: list[dict],
) -> dict:
    primary_text = '\n\n'.join(units[i]['text'] for i in primary_indices)
    text = f'{overlap_text}\n\n{primary_text}' if overlap_text else primary_text
    primary_units = [units[i] for i in primary_indices]
    return {
        'chunk_id': f"{parent['id']}__c{chunk_index:03d}",
        'parent_id': parent['id'],
        'header': parent['header'],
        'title': parent['title'],
        'chunk_index': chunk_index,
        'text': text,
        'char_start': primary_units[0]['start'],
        'char_end': primary_units[-1]['end'],
        'n_tokens': count_tokens(text),
    }


def chunk_discourse(
    discourse: dict,
    *,
    target: int = config.CHUNK_TARGET_TOKENS,
    max_tokens: int = config.CHUNK_MAX_TOKENS,
    overlap: int = config.CHUNK_OVERLAP_TOKENS,
    overlap_max: int = config.CHUNK_OVERLAP_MAX_TOKENS,
    min_tokens: int = config.CHUNK_MIN_TOKENS,
) -> list[dict]:
    '''Split one discourse into child chunks. Deterministic.'''
    units = _build_units(discourse['text'], max_tokens=max_tokens)
    if not units:
        return []

    total = sum(u['tokens'] for u in units)
    if total <= target or total < min_tokens:
        return [_make_chunk(
            parent=discourse,
            chunk_index=0,
            overlap_text='',
            primary_indices=list(range(len(units))),
            units=units,
        )]

    chunks: list[dict] = []
    primary: list[int] = []
    primary_tokens = 0
    chunk_idx = 0
    last_primary: list[int] = []

    def flush():
        nonlocal chunk_idx, last_primary, primary, primary_tokens
        ov_text = _select_overlap_text(last_primary, units, overlap, overlap_max) if last_primary else ''
        chunks.append(_make_chunk(
            parent=discourse,
            chunk_index=chunk_idx,
            overlap_text=ov_text,
            primary_indices=primary,
            units=units,
        ))
        chunk_idx += 1
        last_primary = primary
        primary = []
        primary_tokens = 0

    i = 0
    while i < len(units):
        u = units[i]
        # Always allow at least one unit; otherwise pack while under target.
        if not primary or primary_tokens + u['tokens'] <= target:
            primary.append(i)
            primary_tokens += u['tokens']
            i += 1
        else:
            flush()
            # Re-process unit `i` against the fresh chunk on next iteration.

    if primary:
        flush()

    return chunks


def iter_chunks(discourses: Iterable[dict], **kwargs) -> Iterable[dict]:
    for d in discourses:
        yield from chunk_discourse(d, **kwargs)


def chunk_jsonl(in_path: Path, out_path: Path, **kwargs) -> int:
    '''Read discourses from `in_path`, write chunks to `out_path`. Returns chunk count.'''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with in_path.open('r', encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            discourse = json.loads(line)
            for chunk in chunk_discourse(discourse, **kwargs):
                fout.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                n += 1
    return n
