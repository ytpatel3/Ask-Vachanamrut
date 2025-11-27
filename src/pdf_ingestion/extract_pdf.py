"""
author: Yash Patel
file: extract_pdf.py

Extracts text from /data/Vachanamrut.pdf (pages 33..698 inclusive)
and writes a single cleaned raw text file with explicit page-break markers.

Outputs:
  data/extracted_raw.txt
"""

import os
import sys
import argparse
import unicodedata
import re
from typing import List
from tqdm import tqdm
import pdfplumber

DEFAULT_INPUT = 'data/Vachanamrut.pdf'
OUT_DIR = 'data/raw'
OUT_FILE = 'extracted_raw.txt'
PAGE_START = 33
PAGE_END = 698

def normalize_unicode(s: str) -> str:
    """NFKC normalization to fix combined characters, ligatures etc."""
    s = unicodedata.normalize('NFKC', s)
    # Common ligature fixes
    s = s.replace('\u2044', '/').replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    return s

def simple_clean_text(s: str) -> str:
    # replace \r with \n
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    # collapse repeated spaces but keep paragraph breaks
    s = re.sub(r'[ \t]+', ' ', s)
    # collapse many blank lines to two
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s

def extract_pages(pdf_path: str, page_start: int, page_end: int) -> List[str]:
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        ps = max(1, page_start)
        pe = min(page_end, len(pdf.pages))
        for p in tqdm(range(ps-1, pe), desc='Extracting pages'):
            page = pdf.pages[p]
            try:
                text = page.extract_test(x_tolerance=1.5, y_tolerance=1.5) or ''
            except Exception:
                text = page.extract_text() or ''
            text = normalize_unicode(text)
            text = simple_clean_text(text)
            pages_text.append(text)

    return pages_text

def write_output(pages_text: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, OUT_FILE)
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, page_text in enumerate(pages_text, start=PAGE_START):
            f.write(f'\n\n===PAGE {i}===\n\n')
            f.write(page_text.strip())
            f.write('\n')
    print(f'Wrote extracted raw text to {out_path}')

def main():
    parser = argparse.ArgumentParser(description='Extract pages from PDF to raw text')
    parser.add_argument('--pdf', default=DEFAULT_INPUT, help='Path to input PDF')
    parser.add_argument('--start', type=int, default=PAGE_START, help='Start page (1-indexed)')
    parser.add_argument('--end', type=int, default=PAGE_END, help='End page (1-indexed, inclusive)')
    parser.add_argument('--outdir', default=OUT_DIR, help='Output directory')
    args = parser.parse_args()
    pages_text = extract_pages(args.pdf, args.start, args.end)
    write_output(pages_text, args.outdir)