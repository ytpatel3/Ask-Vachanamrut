'''
Entrypoint: runs the text_normalization pipeline and writes cleaned Vachanamruts to disk as JSONL.
'''

from pathlib import Path
from text_normalization.pipeline import run_pipeline


def main() -> None:
    '''Run normalization and report results.'''
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / 'data' / 'clean' / 'vachanamrut.jsonl'

    count = run_pipeline(str(out_path))
    print(f'Wrote {count} Vachanamrut sections to {out_path}')


if __name__ == '__main__':
    main()
