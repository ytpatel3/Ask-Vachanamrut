'''
End-to-end orchestration of text normalization and section extraction.

Input is always read from data/raw/vachanamrut.txt relative to project root.
'''

from pathlib import Path
from .loader import load_raw_text
from .parsers import parse_sections, finalize_sections
from .writer import write_jsonl


def run_pipeline(out_path: str) -> int:
    '''Run the full pipeline and return the number of sections created.'''
    project_root = Path(__file__).resolve().parents[2]
    in_path = project_root / 'data' / 'raw' / 'vachanamrut.txt'

    raw_lines = load_raw_text(str(in_path))
    sections = parse_sections(raw_lines)
    final = finalize_sections(sections)
    write_jsonl(final, out_path)

    return len(final)