'''Unit tests for src.rag.eval.'''

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.rag import eval as ev


def _make_source(chunk_id: str, parent_id: str) -> dict:
    return {
        'chunk_id': chunk_id, 'parent_id': parent_id, 'header': 'H', 'title': 't',
        'score': 0.9, 'text': 'text', 'n_tokens': 5, 'expanded': False,
    }


# ---- load_qa_set -------------------------------------------------------------

def test_load_qa_set_reads_jsonl(tmp_path: Path):
    rows = [
        {'id': 'eval_001', 'question': 'q1', 'expected_parent_ids': ['A']},
        {'id': 'eval_002', 'question': 'q2', 'expected_parent_ids': ['B', 'C']},
    ]
    p = tmp_path / 'qa.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')

    loaded = ev.load_qa_set(p)
    assert loaded == rows


def test_load_qa_set_skips_blank_lines(tmp_path: Path):
    p = tmp_path / 'qa.jsonl'
    p.write_text('\n{"id": "eval_001", "question": "q", "expected_parent_ids": ["A"]}\n\n')
    loaded = ev.load_qa_set(p)
    assert len(loaded) == 1


# ---- recall_at_k --------------------------------------------------------------

def test_recall_at_k_true_when_expected_parent_present():
    assert ev.recall_at_k(['X', 'A', 'Y'], ['A']) is True


def test_recall_at_k_false_when_expected_parent_absent():
    assert ev.recall_at_k(['X', 'Y', 'Z'], ['A']) is False


def test_recall_at_k_true_with_multiple_expected_parents():
    assert ev.recall_at_k(['X', 'Y'], ['A', 'Y']) is True


def test_recall_at_k_false_for_empty_retrieved():
    assert ev.recall_at_k([], ['A']) is False


# ---- reciprocal_rank ----------------------------------------------------------

def test_reciprocal_rank_first_position():
    assert ev.reciprocal_rank(['A', 'X', 'Y'], ['A']) == 1.0


def test_reciprocal_rank_third_position():
    assert ev.reciprocal_rank(['X', 'Y', 'A'], ['A']) == pytest.approx(1 / 3)


def test_reciprocal_rank_zero_when_not_found():
    assert ev.reciprocal_rank(['X', 'Y'], ['A']) == 0.0


def test_reciprocal_rank_matches_first_of_multiple_expected():
    assert ev.reciprocal_rank(['X', 'B', 'A'], ['A', 'B']) == pytest.approx(1 / 2)


# ---- citation_precision --------------------------------------------------------

def test_citation_precision_true_when_cited_chunk_belongs_to_expected_parent():
    sources = [_make_source('P__c000', 'P'), _make_source('Q__c000', 'Q')]
    assert ev.citation_precision(['P__c000'], sources, ['P']) is True


def test_citation_precision_false_when_cited_chunk_belongs_to_other_parent():
    sources = [_make_source('Q__c000', 'Q')]
    assert ev.citation_precision(['Q__c000'], sources, ['P']) is False


def test_citation_precision_false_when_nothing_cited():
    sources = [_make_source('P__c000', 'P')]
    assert ev.citation_precision([], sources, ['P']) is False


def test_citation_precision_ignores_unknown_chunk_ids():
    sources = [_make_source('P__c000', 'P')]
    assert ev.citation_precision(['not-a-real-chunk'], sources, ['P']) is False


# ---- evaluate(): mocked orchestration ------------------------------------------

def test_evaluate_aggregates_recall_mrr_and_citation_precision():
    qa_set = [
        {'id': 'eval_001', 'question': 'q1', 'expected_parent_ids': ['A']},
        {'id': 'eval_002', 'question': 'q2', 'expected_parent_ids': ['Z']},
    ]
    sources_q1 = [_make_source('A__c000', 'A')]
    sources_q2 = [_make_source('B__c000', 'B')]

    def fake_retrieve(question, **kwargs):
        return sources_q1 if question == 'q1' else sources_q2

    def fake_generate(question, sources, **kwargs):
        if question == 'q1':
            return {'answer': 'ans', 'cited': ['A__c000']}
        return {'answer': 'ans', 'cited': []}

    with patch.object(ev, 'retrieve', side_effect=fake_retrieve), \
         patch.object(ev, 'generate', side_effect=fake_generate):
        report = ev.evaluate(qa_set)

    assert report['n'] == 2
    assert report['recall_at_k'] == 0.5  # only q1 hit
    assert report['mrr'] == 0.5          # q1: 1/1, q2: 0 -> avg 0.5
    assert report['citation_precision'] == 0.5  # only q1 cited correctly
    assert len(report['per_question']) == 2
    assert report['per_question'][0]['recall_at_k'] is True
    assert report['per_question'][1]['recall_at_k'] is False


def test_evaluate_skips_generation_when_use_generation_false():
    qa_set = [{'id': 'eval_001', 'question': 'q1', 'expected_parent_ids': ['A']}]
    sources = [_make_source('A__c000', 'A')]

    with patch.object(ev, 'retrieve', return_value=sources) as mock_retrieve, \
         patch.object(ev, 'generate') as mock_generate:
        report = ev.evaluate(qa_set, use_generation=False)

    mock_generate.assert_not_called()
    mock_retrieve.assert_called_once()
    assert report['citation_precision'] is None
    assert report['per_question'][0]['citation_precision'] is None
    assert report['recall_at_k'] == 1.0


# ---- live integration (marked; runs only when explicitly opted in) --------

@pytest.mark.live
def test_live_evaluate_against_real_qa_set():
    if not config_eval_qa_exists():
        pytest.skip('data/eval/qa_set.jsonl does not exist yet')
    if not os.getenv('ANTHROPIC_API_KEY'):
        pytest.skip('ANTHROPIC_API_KEY not set')

    from src.rag import config
    qa_set = ev.load_qa_set(config.EVAL_QA_PATH)
    report = ev.evaluate(qa_set)
    assert report['n'] == len(qa_set)
    assert 0.0 <= report['recall_at_k'] <= 1.0
    assert 0.0 <= report['mrr'] <= 1.0


def config_eval_qa_exists() -> bool:
    from src.rag import config
    return config.EVAL_QA_PATH.exists()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
