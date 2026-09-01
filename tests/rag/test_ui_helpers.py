'''Unit tests for src.rag.ui_helpers.'''

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.rag import config, ui_helpers as ui


def _make_source(chunk_id: str, header: str = 'Gadhada I-1', title: str = 'A Title') -> dict:
    return {'chunk_id': chunk_id, 'header': header, 'title': title}


# ---- format_answer_html -------------------------------------------------------

def test_format_answer_html_wraps_cited_id_in_badge():
    html = ui.format_answer_html('Per [A__c000], devotion is key.', ['A__c000'])
    assert '<span class="citation-badge">[A__c000]</span>' in html


def test_format_answer_html_leaves_uncited_bracket_unwrapped():
    html = ui.format_answer_html('See [A__c000] and [B__c001].', ['A__c000'])
    assert '<span class="citation-badge">[A__c000]</span>' in html
    assert '<span class="citation-badge">[B__c001]</span>' not in html
    assert '[B__c001]' in html


def test_format_answer_html_escapes_html_in_answer_text():
    html = ui.format_answer_html('<script>alert(1)</script> [A__c000]', ['A__c000'])
    assert '<script>' not in html
    assert '&lt;script&gt;' in html


def test_format_answer_html_converts_newlines_to_br():
    html = ui.format_answer_html('line one\nline two', [])
    assert '<br>' in html


# ---- build_source_label --------------------------------------------------------

def test_build_source_label_includes_header_and_title():
    label = ui.build_source_label(_make_source('X__c000'))
    assert label == 'Gadhada I-1 -- A Title'


def test_build_source_label_omits_dash_when_title_empty():
    label = ui.build_source_label(_make_source('X__c000', title=''))
    assert label == 'Gadhada I-1'


# ---- build_quote_preview -------------------------------------------------------

def test_build_quote_preview_returns_short_text_unchanged():
    assert ui.build_quote_preview('A short quote.') == 'A short quote.'


def test_build_quote_preview_cuts_at_sentence_boundary():
    text = 'First sentence here. Second sentence here. ' + ('padding word ' * 60)
    preview = ui.build_quote_preview(text, max_chars=50)
    assert preview == 'First sentence here. Second sentence here.'


def test_build_quote_preview_falls_back_to_word_boundary_with_ellipsis():
    text = 'onereallylongwordthatjustkeepsgoingwithoutanysentencepunctuationatallwhatsoever ' * 5
    preview = ui.build_quote_preview(text, max_chars=50)
    assert preview.endswith('...')
    assert len(preview) <= 54


def test_build_quote_preview_strips_surrounding_whitespace():
    assert ui.build_quote_preview('  padded text  ') == 'padded text'


# ---- load_anirdesh_vachno_map / anirdesh_url -----------------------------------

def _write_fake_discourses(path: Path, count: int = 274) -> None:
    with path.open('w', encoding='utf-8') as f:
        for i in range(1, count + 1):
            f.write(json.dumps({'id': f'P{i}', 'text': ''}) + '\n')


def test_load_anirdesh_vachno_map_unchanged_before_the_shift(tmp_path: Path):
    path = tmp_path / 'discourses.jsonl'
    _write_fake_discourses(path)
    mapping = ui.load_anirdesh_vachno_map(path)
    assert mapping['P1'] == 1
    assert mapping['P262'] == 262


def test_load_anirdesh_vachno_map_shifts_amdavad4_through_jetalpur5_by_one(tmp_path: Path):
    path = tmp_path / 'discourses.jsonl'
    _write_fake_discourses(path)
    mapping = ui.load_anirdesh_vachno_map(path)
    assert mapping['P263'] == 264
    assert mapping['P273'] == 274


def test_load_anirdesh_vachno_map_places_bhugol_khagol_at_263(tmp_path: Path):
    path = tmp_path / 'discourses.jsonl'
    _write_fake_discourses(path)
    mapping = ui.load_anirdesh_vachno_map(path)
    assert mapping['P274'] == 263


def test_load_anirdesh_vachno_map_matches_real_corpus_landmarks():
    '''Sanity check against the real corpus (not just synthetic data) for the
    three positions the shift logic actually depends on.
    '''
    mapping = ui.load_anirdesh_vachno_map(config.DISCOURSES_PATH)
    assert mapping['Gadhada_III_39'] == 262
    assert mapping['Amdavad_4'] == 264
    assert mapping['Jetalpur_5'] == 274
    assert mapping['Bhugol_Khagol'] == 263


def test_anirdesh_url_formats_query_string():
    assert ui.anirdesh_url(1) == 'https://www.anirdesh.com/vachanamrut/index.php?format=en&vachno=1'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
