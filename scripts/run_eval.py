'''Run the retrieval/generation eval harness and print a report.

    python -m scripts.run_eval                  # full pass (retrieval + generation)
    python -m scripts.run_eval --no-generation   # retrieval only, no Claude API calls
'''

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click

from src.rag import config
from src.rag import eval as eval_mod


@click.command()
@click.option('--no-generation', is_flag=True, help='Skip generation; report retrieval metrics only.')
def main(no_generation: bool) -> None:
    qa_set = eval_mod.load_qa_set(config.EVAL_QA_PATH)
    report = eval_mod.evaluate(qa_set, use_generation=not no_generation)

    print(f"n = {report['n']}")
    print(f"recall@{config.K_FINAL} = {report['recall_at_k']:.2f}")
    print(f"MRR = {report['mrr']:.2f}")
    if report['citation_precision'] is not None:
        print(f"citation precision = {report['citation_precision']:.2f}")

    print()
    for q in report['per_question']:
        marks = f"recall={'Y' if q['recall_at_k'] else 'N'} rr={q['reciprocal_rank']:.2f}"
        if q['citation_precision'] is not None:
            marks += f" cited={'Y' if q['citation_precision'] else 'N'}"
        print(f"[{q['id']}] {marks} -- {q['question']}")


if __name__ == '__main__':
    main()
