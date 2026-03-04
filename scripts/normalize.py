'''
Entrypoint: runs the text_normalization pipeline and writes cleaned Vachanamruts to disk as JSONL.
'''

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_normalization.pipeline import run_pipeline


def main() -> None:
    '''Run normalization and report results.'''
    out_path = PROJECT_ROOT / 'data' / 'clean' / 'vachanamrut.jsonl'

    count = run_pipeline(str(out_path))
    print(f'Wrote {count} Vachanamrut sections to {out_path}')


if __name__ == '__main__':
    main()
