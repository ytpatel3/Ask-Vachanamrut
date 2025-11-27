"""
split_sections

Reads data/raw/extracted_raw.txt and splits into logical 'sections' (e.g., "Gadhada I-1", "Sarangpur-2", etc.). 
Saves JSON Lines to data/clean/sections.jsonl with entries:

{
  "id": "Gadhada_I_1",
  "title": "Gadhada I-1",
  "raw_text": "...",
  "page_start": 33,
  "page_end": 40
}
"""

import os
import re
import json
import argparse
from typing import List, Dict
from tqdm import tqdm

IN_FILE = 'data/raw/extracted_raw.txt'
OUT_DIR = 'data/clean'
OUT_FILE = 'sections.jsonl'

