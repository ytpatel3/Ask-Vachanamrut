'''Local, best-effort tracking of how many generation requests the Streamlit
app has made today, per model -- powers the sidebar's "requests left today"
estimate.

This is NOT an authoritative count from Google -- the Gemini free tier
doesn't expose remaining-quota via the API, only via the AI Studio dashboard
UI. It only counts requests `app.py` itself made, persisted to a small JSON
file so the count survives Streamlit reruns and process restarts.
'''

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.rag import config


def _today() -> str:
    return date.today().isoformat()


def _load_log(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_log(path: Path, log: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log), encoding='utf-8')


def record_request(model: str, *, path: Path = config.USAGE_LOG_PATH) -> None:
    '''Increment today's request counter for `model`.'''
    log = _load_log(path)
    today_counts = log.setdefault(_today(), {})
    today_counts[model] = today_counts.get(model, 0) + 1
    _save_log(path, log)


def requests_used_today(model: str, *, path: Path = config.USAGE_LOG_PATH) -> int:
    '''Requests recorded for `model` so far today.'''
    log = _load_log(path)
    return log.get(_today(), {}).get(model, 0)


def requests_remaining_today(model: str, *, path: Path = config.USAGE_LOG_PATH) -> int | None:
    '''Estimated requests left today for `model`, or None if it has no
    configured daily limit in `config.MODEL_DAILY_LIMITS`.
    '''
    limit = config.MODEL_DAILY_LIMITS.get(model)
    if limit is None:
        return None
    return max(limit - requests_used_today(model, path=path), 0)
