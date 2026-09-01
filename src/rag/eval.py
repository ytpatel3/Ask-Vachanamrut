'''Evaluation harness: recall@k, MRR, and citation precision against a
ground-truth Q&A set.

Ground truth is anchored at the discourse (parent_id) level, not chunk_id --
chunk boundaries shift whenever chunking logic or the corpus changes (this has
already happened twice in this project), so a chunk-level ground truth would
go stale on every re-chunk. A hit only requires the retrieved/cited chunk to
belong to one of the expected parent discourses.
'''

from __future__ import annotations

import json
from pathlib import Path

from src.rag import config
from src.rag.generate import generate
from src.rag.retrieve import retrieve


def load_qa_set(path: Path = config.EVAL_QA_PATH) -> list[dict]:
    '''Load ground-truth Q&A records from a JSONL file.'''
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def recall_at_k(retrieved_parent_ids: list[str], expected_parent_ids: list[str]) -> bool:
    '''True if any retrieved chunk belongs to one of the expected parent discourses.'''
    return any(pid in expected_parent_ids for pid in retrieved_parent_ids)


def reciprocal_rank(retrieved_parent_ids: list[str], expected_parent_ids: list[str]) -> float:
    '''1 / (rank of the first hit), or 0.0 if no expected parent was retrieved.'''
    for i, pid in enumerate(retrieved_parent_ids, start=1):
        if pid in expected_parent_ids:
            return 1.0 / i
    return 0.0


def citation_precision(
    cited_chunk_ids: list[str], sources: list[dict], expected_parent_ids: list[str],
) -> bool:
    '''True if any chunk the model actually cited belongs to an expected parent.

    `sources` is the retrieve() result the model was given, used to map cited
    chunk_ids back to their parent_id.
    '''
    parent_by_chunk = {s['chunk_id']: s['parent_id'] for s in sources}
    return any(parent_by_chunk.get(cid) in expected_parent_ids for cid in cited_chunk_ids)


def _average(values: list[bool | float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(
    qa_set: list[dict],
    *,
    k: int = config.K_FINAL,
    neighbors: int = config.EXPAND_NEIGHBORS,
    max_expanded_tokens: int = config.MAX_EXPANDED_TOKENS,
    use_generation: bool = True,
    model: str = config.RAG_MODEL,
    temperature: float = config.GENERATION_TEMPERATURE,
    max_tokens: int = config.GENERATION_MAX_TOKENS,
) -> dict:
    '''Run retrieval (and optionally generation) for every question in `qa_set`
    and report aggregate recall@k, MRR, and citation precision.

    `use_generation=False` skips the Claude API call entirely for a fast,
    free retrieval-only pass (citation_precision is reported as None in that
    case) -- useful for iterating on chunking/embedding without spending on
    generation every run.
    '''
    per_question = []
    for item in qa_set:
        expected = item['expected_parent_ids']
        sources = retrieve(item['question'], k=k, neighbors=neighbors, max_expanded_tokens=max_expanded_tokens)
        retrieved_parent_ids = [s['parent_id'] for s in sources]

        hit = recall_at_k(retrieved_parent_ids, expected)
        rr = reciprocal_rank(retrieved_parent_ids, expected)

        cited_hit = None
        if use_generation:
            result = generate(item['question'], sources, model=model, temperature=temperature, max_tokens=max_tokens)
            cited_hit = citation_precision(result['cited'], sources, expected)

        per_question.append({
            'id': item['id'],
            'question': item['question'],
            'recall_at_k': hit,
            'reciprocal_rank': rr,
            'citation_precision': cited_hit,
            'retrieved_parent_ids': retrieved_parent_ids,
        })

    cited_values = [p['citation_precision'] for p in per_question if p['citation_precision'] is not None]

    return {
        'n': len(per_question),
        'recall_at_k': _average([p['recall_at_k'] for p in per_question]),
        'mrr': _average([p['reciprocal_rank'] for p in per_question]),
        'citation_precision': _average(cited_values) if use_generation else None,
        'per_question': per_question,
    }
