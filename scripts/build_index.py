'''Build the RAG index: chunk discourses, embed child chunks, upsert into Chroma.

Phase 1 supports `--skip-embed` (chunk only). Embedding wires in next phase.

    python -m scripts.build_index                 # chunk + embed + index (upsert)
    python -m scripts.build_index --skip-embed    # chunk only
    python -m scripts.build_index --skip-chunk    # embed/index from existing chunks file
    python -m scripts.build_index --rebuild       # delete collection first
'''

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click

from src.rag import chunk as chunk_mod
from src.rag import config


@click.command()
@click.option('--skip-chunk', is_flag=True, help='Skip chunking; reuse existing chunks file.')
@click.option('--skip-embed', is_flag=True, help='Skip embedding/indexing (chunk only).')
@click.option('--rebuild', is_flag=True, help='Drop the Chroma collection before upsert.')
def main(skip_chunk: bool, skip_embed: bool, rebuild: bool) -> None:
    if not skip_chunk:
        n = chunk_mod.chunk_jsonl(config.DISCOURSES_PATH, config.CHUNKS_PATH)
        print(f'Wrote {n} chunks to {config.CHUNKS_PATH}')

    if skip_embed:
        return

    # Lazy import so --skip-embed runs without sentence-transformers installed.
    from src.rag import embed as embed_mod
    count = embed_mod.build_index(
        chunks_path=config.CHUNKS_PATH,
        persist_dir=config.CHROMA_DIR,
        collection_name=config.COLLECTION_NAME,
        rebuild=rebuild,
    )
    print(f"Indexed {count} vectors into collection '{config.COLLECTION_NAME}' at {config.CHROMA_DIR}")


if __name__ == '__main__':
    main()
