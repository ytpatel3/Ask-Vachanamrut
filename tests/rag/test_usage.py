'''Unit tests for src.rag.usage.'''

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.rag import usage


def test_load_log_returns_empty_dict_when_file_missing(tmp_path: Path):
    assert usage._load_log(tmp_path / 'missing.json') == {}


def test_load_log_returns_empty_dict_for_corrupt_json(tmp_path: Path):
    path = tmp_path / 'log.json'
    path.write_text('not json{{{')
    assert usage._load_log(path) == {}


def test_record_request_creates_log_file(tmp_path: Path):
    path = tmp_path / 'nested' / 'log.json'
    with patch.object(usage, '_today', return_value='2026-09-01'):
        usage.record_request('gemini-3.5-flash-lite', path=path)
    assert path.exists()


def test_record_request_increments_same_day_same_model(tmp_path: Path):
    path = tmp_path / 'log.json'
    with patch.object(usage, '_today', return_value='2026-09-01'):
        usage.record_request('gemini-3.5-flash-lite', path=path)
        usage.record_request('gemini-3.5-flash-lite', path=path)

    assert usage.requests_used_today('gemini-3.5-flash-lite', path=path) == 2


def test_record_request_tracks_models_independently(tmp_path: Path):
    path = tmp_path / 'log.json'
    with patch.object(usage, '_today', return_value='2026-09-01'):
        usage.record_request('gemini-3.5-flash-lite', path=path)
        usage.record_request('other-model', path=path)

    assert usage.requests_used_today('gemini-3.5-flash-lite', path=path) == 1
    assert usage.requests_used_today('other-model', path=path) == 1


def test_requests_used_today_zero_when_no_log(tmp_path: Path):
    assert usage.requests_used_today('gemini-3.5-flash-lite', path=tmp_path / 'missing.json') == 0


def test_requests_used_today_zero_for_a_past_day(tmp_path: Path):
    path = tmp_path / 'log.json'
    with patch.object(usage, '_today', return_value='2026-08-31'):
        usage.record_request('gemini-3.5-flash-lite', path=path)

    with patch.object(usage, '_today', return_value='2026-09-01'):
        assert usage.requests_used_today('gemini-3.5-flash-lite', path=path) == 0


def test_requests_remaining_today_subtracts_used_from_limit(tmp_path: Path):
    path = tmp_path / 'log.json'
    with patch.object(usage, '_today', return_value='2026-09-01'):
        usage.record_request('gemini-3.5-flash-lite', path=path)
        remaining = usage.requests_remaining_today('gemini-3.5-flash-lite', path=path)

    assert remaining == usage.config.MODEL_DAILY_LIMITS['gemini-3.5-flash-lite'] - 1


def test_requests_remaining_today_clamped_at_zero(tmp_path: Path):
    path = tmp_path / 'log.json'
    with patch.object(usage, 'requests_used_today', return_value=9999):
        assert usage.requests_remaining_today('gemini-3.5-flash-lite', path=path) == 0


def test_requests_remaining_today_none_for_unconfigured_model(tmp_path: Path):
    assert usage.requests_remaining_today('some-other-model', path=tmp_path / 'log.json') is None


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
