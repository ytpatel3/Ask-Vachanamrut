'''Smoke tests for app.py using Streamlit's AppTest harness.

Only exercises that the script runs without raising and renders the expected
elements -- `pipeline.answer()` itself is fully covered by
`tests/rag/test_pipeline.py`, so it's mocked here rather than re-tested.
`src.rag.usage` is also mocked throughout so tests never read/write the real
`data/usage/request_log.json`.
'''

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from streamlit.testing.v1 import AppTest

from src.rag import config

APP_PATH = str(PROJECT_ROOT / 'app.py')


def _fake_answer(query: str, **kwargs) -> dict:
    return {
        'answer': 'Per [A__c000], the answer is 42.',
        'cited': ['A__c000'],
        'sources': [{
            'chunk_id': 'A__c000', 'parent_id': 'A', 'header': 'Gadhada I-1', 'title': 'T',
            'score': 0.9, 'text': 'source text', 'n_tokens': 5, 'expanded': False,
        }],
    }


@contextmanager
def _patched_usage():
    '''patch.multiple() only returns auto-created (DEFAULT) mocks from its
    context manager, so build ours explicitly and yield them for assertions.
    '''
    mocks = {
        'requests_remaining_today': MagicMock(return_value=499),
        'requests_used_today': MagicMock(return_value=1),
        'record_request': MagicMock(),
    }
    with patch.multiple('src.rag.usage', **mocks):
        yield mocks


def test_app_loads_without_error():
    with _patched_usage():
        at = AppTest.from_file(APP_PATH).run()
    assert not at.exception


def test_app_renders_answer_and_sources_after_query_submission():
    with _patched_usage():
        at = AppTest.from_file(APP_PATH).run()

        with patch('src.rag.pipeline.answer', side_effect=_fake_answer):
            at.text_input[0].input('What is maya?')
            at.button[0].click().run()

    assert not at.exception
    rendered = '\n'.join(md.value for md in at.markdown)
    assert 'the answer is 42' in rendered
    assert 'citation-badge' in rendered
    assert 'Gadhada I-1' in rendered


def test_app_shows_friendly_error_when_pipeline_raises():
    with _patched_usage():
        at = AppTest.from_file(APP_PATH).run()

        with patch('src.rag.pipeline.answer', side_effect=RuntimeError('no API key')):
            at.text_input[0].input('What is maya?')
            at.button[0].click().run()

    assert not at.exception
    assert any('no API key' in e.value for e in at.error)


def test_app_records_usage_on_successful_answer():
    with _patched_usage() as mocks:
        at = AppTest.from_file(APP_PATH).run()

        with patch('src.rag.pipeline.answer', side_effect=_fake_answer):
            at.text_input[0].input('What is maya?')
            at.button[0].click().run()

    mocks['record_request'].assert_called_once_with(config.AVAILABLE_MODELS[0])


def test_app_does_not_record_usage_when_pipeline_raises():
    with _patched_usage() as mocks:
        at = AppTest.from_file(APP_PATH).run()

        with patch('src.rag.pipeline.answer', side_effect=RuntimeError('no API key')):
            at.text_input[0].input('What is maya?')
            at.button[0].click().run()

    mocks['record_request'].assert_not_called()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
