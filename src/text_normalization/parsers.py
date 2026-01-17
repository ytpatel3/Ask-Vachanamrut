'''
Parse normalized raw text into structured Vachanamrut sections.

High-level cleaning:
- Detects valid Vachanamrut headers using strict patterns
- Groups subsequent lines into section blocks
- Extracts titles
- Produces JSON-ready section records
'''

import re
from .cleaners import (
    normalize_diacritics, is_digit_only, is_footer,
    is_section_label, is_garbled, collapse_lines
)

SECTION_NAMES = ['gadhada-i', 'gadhada-ii', 'gadhada-iii', 'sarangpur', 'kariyani', 'loya', 'panchala', 'vartal', 'amdavad', 'jetalpur', 'ashlali']
HEADER_RE = re.compile(
    r'^('
    r'Gadhada\s+I|Gadhada\s+II|Gadhada\s+III|'
    r'Sarangpur|Kariyani|Loya|Panchala|Vartal|'
    r'Amdavad|Jetalpur|Ashlali'
    r')-(\d{1,2})$',
    re.IGNORECASE
)
NO_TITLE_SECTIONS = {'ashlali', 'jetalpur'}
NO_TITLE_AMDAVAD_NUMBERS = {4, 5, 6, 7, 8}

def is_section_header(line: str) -> bool:
    """Return true if line is a valid header"""
    s = normalize_diacritics(line.strip())
    return bool(HEADER_RE.match(s))

def parse_header(header: str) -> tuple[str, int]:
    """Parse header into (section_name, section_number)."""
    m = HEADER_RE.match(header)
    section = m.group(1).lower().replace(' ', '')
    number = int(m.group(2))
    return section, number

def should_use_header_as_title(header: str) -> bool:
    """Return True if the header itself should be used as the title. (Ashlali, Jetalpur, Amdavad 4-8)"""
    section, number = parse_header(header)

    if section in NO_TITLE_SECTIONS:
        return True

    if section == 'amdavad' and number in NO_TITLE_AMDAVAD_NUMBERS:
        return True

    return False

def extract_title(raw_lines: list[str], start_index: int) -> str:
    '''Extract the title following a header.'''
    for offset in range(1, 6):
        idx = start_index + offset
        if idx >= len(raw_lines):
            return ''

        cand = raw_lines[idx].strip()
        if not cand:
            continue
        if is_digit_only(cand) or is_footer(cand) or is_section_label(cand):
            continue
        if is_section_header(cand):
            return ''
        if not is_garbled(cand):
            return normalize_diacritics(cand)

    return ''

def parse_sections(raw_lines: list[str]) -> list[dict]:
    """Group raw lines into section blocks based on headers."""
    sections = []
    current = None

    for i, raw in enumerate(raw_lines):
        line = raw.strip()

        if is_digit_only(line) or is_footer(line) or is_section_label(line):
            continue

        if is_section_header(line):
            if current:
                current['end_line'] = i
                sections.append(current)

            header = normalize_diacritics(line)

            if should_use_header_as_title(header):
                title = header
            else:
                title = extract_title(raw_lines, i)

            current = {
                'header': header,
                'title': title,
                'lines': [],
                'start_line': i
            }
            continue

        if current and not is_garbled(line) and not is_footer(line):
            current['lines'].append(raw)

    if current:
        current['end_line'] = len(raw_lines)
        sections.append(current)

    return sections

def finalize_sections(sections: list[dict]) -> list[dict]:
    """Convert section blocks into JSON-ready records."""
    output = []

    for sec in sections:
        text = collapse_lines(sec['lines'])
        header_clean = sec['header'].replace('–', '-')
        section_id = re.sub(r'[^A-Za-z0-9]+', '_', header_clean).strip('_')

        output.append({
            'id': section_id,
            'header': sec['header'],
            'title': sec['title'],
            'text': text,
            'start_line': sec['start_line'],
            'end_line': sec['end_line']
        })

    return output
