'''Unit tests for src.rag.pipeline.'''

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.rag import pipeline as pl


def _fake_source(chunk_id: str) -> dict:
    return {
        'chunk_id': chunk_id, 'parent_id': 'P', 'header': 'P-1', 'title': 't',
        'score': 0.9, 'text': 'some passage text', 'n_tokens': 5, 'expanded': False,
    }


def test_answer_wires_retrieve_into_generate():
    sources = [_fake_source('P__c000')]

    with patch.object(pl, 'retrieve', return_value=sources) as mock_retrieve, \
         patch.object(pl, 'generate', return_value={'answer': 'the answer', 'cited': ['P__c000']}) as mock_generate:
        result = pl.answer('what is bhakti?')

    mock_retrieve.assert_called_once()
    retrieve_args, retrieve_kwargs = mock_retrieve.call_args
    assert retrieve_args[0] == 'what is bhakti?'

    mock_generate.assert_called_once()
    generate_args, generate_kwargs = mock_generate.call_args
    assert generate_args[0] == 'what is bhakti?'
    assert generate_args[1] == sources

    assert result == {'answer': 'the answer', 'cited': ['P__c000'], 'sources': sources}


def test_answer_passes_through_retrieval_and_generation_params():
    with patch.object(pl, 'retrieve', return_value=[]) as mock_retrieve, \
         patch.object(pl, 'generate', return_value={'answer': 'no sources', 'cited': []}):
        pl.answer(
            'q', k=2, neighbors=0, max_expanded_tokens=500,
            model='claude-sonnet-4-6', temperature=0.7, max_tokens=200,
        )

    _, retrieve_kwargs = mock_retrieve.call_args
    assert retrieve_kwargs['k'] == 2
    assert retrieve_kwargs['neighbors'] == 0
    assert retrieve_kwargs['max_expanded_tokens'] == 500


def test_answer_returns_empty_sources_list_when_nothing_retrieved():
    with patch.object(pl, 'retrieve', return_value=[]), \
         patch.object(pl, 'generate', return_value={'answer': 'nothing found', 'cited': []}):
        result = pl.answer('obscure question')

    assert result['sources'] == []
    assert result['cited'] == []


# ---- live integration (marked; runs only when explicitly opted in) --------

@pytest.mark.live
def test_live_answer_end_to_end():
    if not os.getenv('GEMINI_API_KEY'):
        pytest.skip('GEMINI_API_KEY not set')

    result = pl.answer('What is the nature of maya?')
    assert result['answer']
    assert isinstance(result['cited'], list)
    assert isinstance(result['sources'], list)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
