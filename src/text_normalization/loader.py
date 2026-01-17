"""Load raw text from disk"""

import os

def load_raw_text(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()
    