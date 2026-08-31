'''End-to-end RAG pipeline: wires retrieve() -> generate() into one entrypoint.

This is the one function the UI and eval harness both call.
'''

from __future__ import annotations

from src.rag import config
from src.rag.generate import generate
from src.rag.retrieve import retrieve


def answer(
    query: str,
    *,
    k: int = config.K_FINAL,
    neighbors: int = config.EXPAND_NEIGHBORS,
    max_expanded_tokens: int = config.MAX_EXPANDED_TOKENS,
    model: str = config.RAG_MODEL,
    temperature: float = config.GENERATION_TEMPERATURE,
    max_tokens: int = config.GENERATION_MAX_TOKENS,
) -> dict:
    '''Answer `query` end-to-end: retrieve supporting chunks, then generate a
    cited answer from them. Returns `{'answer': str, 'cited': list[str],
    'sources': list[dict]}`, where `sources` is retrieve()'s full result list
    (for a UI's "show retrieved chunks" panel or an eval harness).
    '''
    sources = retrieve(query, k=k, neighbors=neighbors, max_expanded_tokens=max_expanded_tokens)
    result = generate(query, sources, model=model, temperature=temperature, max_tokens=max_tokens)

    return {'answer': result['answer'], 'cited': result['cited'], 'sources': sources}
