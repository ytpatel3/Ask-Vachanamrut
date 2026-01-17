"""Low-level cleaning: diacritics, footers, garbled text from original Sanskrit characters, paragraph merging"""

import re
import unicodedata

DIACRITIC_MAP = {
    'ã': 'a', 'Ã': 'A',
    'â': 'a', 'Â': 'A',
    'á': 'a', 'Á': 'A',
    'à': 'a', 'À': 'A',
    'é': 'e', 'É': 'E',
    'ê': 'e', 'Ê': 'E',
    'í': 'i', 'Í': 'I',
    'ó': 'o', 'Ó': 'O',
    'õ': 'o', 'Õ': 'O',
    'ú': 'u', 'Ú': 'U',
    'ñ': 'n', 'Ñ': 'N',
    '’': "'", '“': '"', '”': '"',
    '—': '-', '–': '-'
}

DIGIT_LINE_RE = re.compile(r'^\s*\d+\s*$')
FOOTER_BAR_RE = re.compile(r'\|\|.*?\|\|')
FOOTER_VACHAN_RE = re.compile(r'^The\s+Vachan[aã]mrut\s+\d+\s*$', re.I)
SECTION_WORD_RE = re.compile(r'^(SECTION|Section)\b', re.I)

def normalize_diacritics(s: str) -> str:
    """Normalize common diacritics (using map above)"""
    for k, v in DIACRITIC_MAP.items():
        s = s.replace(k, v)
    return unicodedata.normalize('NFKC', s)

def is_digit_only(line: str) -> bool:
    """Return True if line contains only digits"""
    return bool(DIGIT_LINE_RE.match(line.strip()))

def is_footer(line: str) -> bool:
    """Return True if line matches known footer patterns"""
    s = line.strip()
    return bool(FOOTER_BAR_RE.search(s), FOOTER_VACHAN_RE.search(s))

def is_section_label(line: str) -> bool:
    """Return True if line is section label"""
    return bool(SECTION_WORD_RE.match(normalize_diacritics(line.strip())))

def is_garbled(line: str) -> bool:
    """Return True if line appears corrupted/garbled"""
    s = line.strip()
    if not s: return False

    total = len(s)
    letters = sum(ch.isalpha() for ch in s)
    non_ascii = sum(ord(ch) > 127 for ch in s)

    return (letters / total < 0.25) and (non_ascii / total > 0.25)

def collapse_lines(lines: list[str]) -> str:
    """Merge line-wrapped text into paragraphs and preserve paragraph breaks"""
    buf = []
    paragraphs = []

    for line in lines:
        s = line.strip()
        if not s:
            if buf: 
                paragraphs.append(' '.join(buf).strip())
                buf = []
            continue
        buf.append(s)

    if buf:
        paragraphs.append(' '.join(buf).strip())
    
    final = '\n\n'.join(paragraphs)
    final = re.sub(r'[ \t]+', ' ', final)
    final = re.sub(r'\n{3,}', '\n\n', final)

    return final.strip()