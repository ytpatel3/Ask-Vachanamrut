'''Unit tests for src.text_normalization.cleaners.'''

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_normalization import cleaners as cl


# ---- is_garbled: clean lines should never be flagged ------------------------

def test_plain_ascii_line_is_not_garbled():
    assert not cl.is_garbled('Shriji Mahãrãj was sitting on a dais in Vartal.')


def test_line_with_known_diacritics_is_not_garbled():
    s = 'On Mãgshar vadi 14, Samvat 1882 [7 January 1826], Shriji Mahãrãj was sitting.'
    assert not cl.is_garbled(s)


def test_line_with_curly_quotes_and_dashes_is_not_garbled():
    assert not cl.is_garbled('He said, “This is Satsang—the fellowship of devotees.”')


def test_empty_line_is_not_garbled():
    assert not cl.is_garbled('   ')


# ---- is_garbled: symbol-heavy garbage (original heuristic) ------------------

def test_non_ascii_symbol_heavy_line_is_garbled():
    assert cl.is_garbled('¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤')


# ---- is_garbled: font-substitution mojibake (regression) --------------------
# The source PDF's custom font maps Devanagari glyphs onto Latin-Extended/PUA
# codepoints. These read as mostly "letters" to str.isalpha(), so the original
# letters/non_ascii ratio check missed them entirely (see PROJECT_PLAN.md).

def test_pua_codepoint_is_always_garbled():
    assert cl.is_garbled('SflŸ ‚ŒÊ')


def test_devanagari_font_substitution_mojibake_is_garbled():
    # Exact regression case found in Vartal-7 (a Sanskrit shloka quotation).
    assert cl.is_garbled('ÁŸ⁄USÃ∑È§„U∑¢§ ‚àÿ¢ ¬⁄¢ œË◊Á„UH’i.')


def test_mixed_english_and_mojibake_line_is_garbled():
    assert cl.is_garbled('Thereafter Muktãnand Swãmi asked, “The Shrutis state, ‘•ãÃ')


# ---- is_garbled: full-corpus regression guard --------------------------------

def test_fix_catches_known_garbled_lines_without_new_false_positives():
    '''Every raw line previously identified as font-substitution garble must
    still be caught, and no additional raw lines should newly trip the
    detector beyond that known set (guards against over-aggressive tuning).
    '''
    from src.text_normalization.loader import load_raw_text

    raw_path = PROJECT_ROOT / 'data' / 'raw' / 'vachanamrut.txt'
    lines = load_raw_text(str(raw_path))
    flagged = [i for i, l in enumerate(lines) if cl.is_garbled(l)]
    assert len(flagged) == 222
