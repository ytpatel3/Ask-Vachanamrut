'''Unit tests for src.rag.generate.'''

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.rag import generate as gen


def _make_source(chunk_id: str, text: str, header: str = 'Gadhada I-1', title: str = 'A Title') -> dict:
    return {
        'chunk_id': chunk_id, 'parent_id': 'Gadhada_I_1', 'header': header, 'title': title,
        'score': 0.9, 'text': text, 'n_tokens': 10, 'expanded': False,
    }


# ---- _format_context / _build_user_message ----------------------------------

def test_format_context_labels_each_block_with_id_and_header():
    sources = [_make_source('X__c000', 'first passage text')]
    ctx = gen._format_context(sources)
    assert '[X__c000]' in ctx
    assert 'Gadhada I-1' in ctx
    assert 'A Title' in ctx
    assert 'first passage text' in ctx


def test_format_context_omits_dash_when_title_is_empty():
    sources = [_make_source('X__c000', 'text', title='')]
    ctx = gen._format_context(sources)
    assert 'Gadhada I-1 --' not in ctx


def test_build_user_message_includes_question_and_context():
    sources = [_make_source('X__c000', 'relevant text')]
    msg = gen._build_user_message('What is bhakti?', sources)
    assert 'What is bhakti?' in msg
    assert 'relevant text' in msg


# ---- _extract_cited_ids ------------------------------------------------------

def test_extract_cited_ids_finds_valid_citations():
    sources = [_make_source('A__c000', 't'), _make_source('B__c001', 't')]
    answer = 'According to [A__c000], devotion matters. See also [B__c001].'
    cited = gen._extract_cited_ids(answer, sources)
    assert cited == ['A__c000', 'B__c001']


def test_extract_cited_ids_ignores_unknown_bracketed_text():
    sources = [_make_source('A__c000', 't')]
    answer = 'This references [A__c000] and also [not-a-real-id] and [footnote 1].'
    cited = gen._extract_cited_ids(answer, sources)
    assert cited == ['A__c000']


def test_extract_cited_ids_dedupes_and_preserves_first_order():
    sources = [_make_source('A__c000', 't'), _make_source('B__c001', 't')]
    answer = '[B__c001] then [A__c000] then [B__c001] again.'
    cited = gen._extract_cited_ids(answer, sources)
    assert cited == ['B__c001', 'A__c000']


def test_extract_cited_ids_returns_empty_when_no_citations():
    sources = [_make_source('A__c000', 't')]
    assert gen._extract_cited_ids('No citations here.', sources) == []


# ---- generate(): mocked API boundary ----------------------------------------

def test_generate_returns_fallback_when_no_sources():
    result = gen.generate('any question', [])
    assert result['cited'] == []
    assert 'answer' in result


def test_generate_calls_gemini_and_extracts_citations():
    sources = [_make_source('A__c000', 'devotion is central')]

    with patch.object(gen, '_call_gemini', return_value='Per [A__c000], devotion is key.') as mock_call:
        result = gen.generate('What matters most?', sources, model='gemini-2.5-flash')

    mock_call.assert_called_once()
    args, kwargs = mock_call.call_args
    assert 'What matters most?' in args[0]
    assert kwargs['model'] == 'gemini-2.5-flash'
    assert result['answer'] == 'Per [A__c000], devotion is key.'
    assert result['cited'] == ['A__c000']


def test_generate_passes_through_temperature_and_max_tokens():
    sources = [_make_source('A__c000', 'text')]
    with patch.object(gen, '_call_gemini', return_value='ok') as mock_call:
        gen.generate('q', sources, temperature=0.5, max_tokens=42)

    _, kwargs = mock_call.call_args
    assert kwargs['temperature'] == 0.5
    assert kwargs['max_tokens'] == 42


# ---- _call_gemini: verify Gemini client wiring (mocked SDK) -----------------

def _patch_genai_sdk(fake_client: MagicMock) -> dict:
    '''Build the sys.modules patch dict for `from google import genai` /
    `from google.genai import types`, wired to `fake_client`.
    '''
    fake_genai_module = MagicMock()
    fake_genai_module.Client.return_value = fake_client
    fake_types_module = MagicMock()
    fake_genai_module.types = fake_types_module

    fake_google_module = MagicMock()
    fake_google_module.genai = fake_genai_module

    return {
        'google': fake_google_module,
        'google.genai': fake_genai_module,
        'google.genai.types': fake_types_module,
    }


def test_call_gemini_sends_system_instruction_and_contents():
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = MagicMock(text='an answer')

    with patch.dict(sys.modules, _patch_genai_sdk(fake_client)):
        result = gen._call_gemini(
            'my user message', model='gemini-2.5-flash', temperature=0.2, max_tokens=100,
        )

    assert result == 'an answer'
    _, generate_kwargs = fake_client.models.generate_content.call_args
    assert generate_kwargs['model'] == 'gemini-2.5-flash'
    assert generate_kwargs['contents'] == 'my user message'
    assert 'config' in generate_kwargs


def test_call_gemini_passes_temperature_and_max_output_tokens_to_config():
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = MagicMock(text='ok')
    patch_dict = _patch_genai_sdk(fake_client)
    fake_types_module = patch_dict['google.genai.types']

    with patch.dict(sys.modules, patch_dict):
        gen._call_gemini('msg', model='gemini-2.5-flash', temperature=0.2, max_tokens=100)

    _, config_kwargs = fake_types_module.GenerateContentConfig.call_args
    assert config_kwargs['system_instruction'] == gen.SYSTEM_PROMPT
    assert config_kwargs['temperature'] == 0.2
    assert config_kwargs['max_output_tokens'] == 100


# ---- live integration (marked; runs only when explicitly opted in) --------

@pytest.mark.live
def test_live_generate_answers_from_real_index():
    '''End-to-end: retrieve real chunks, then call the real Gemini API.'''
    if not os.getenv('GEMINI_API_KEY'):
        pytest.skip('GEMINI_API_KEY not set')

    from src.rag.retrieve import retrieve

    sources = retrieve('What is the nature of maya?', k=3)
    result = gen.generate('What is the nature of maya?', sources)
    assert result['answer']
    assert isinstance(result['cited'], list)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
