'''
Write normalized section records to disk.
'''

import json
import os

def write_jsonl(records: list[dict], path: str) -> None:
    '''Write a list of records to a JSONL file.'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
