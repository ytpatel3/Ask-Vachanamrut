'''Generation: turn retrieved chunks + a question into a grounded, cited answer.

Calls the Gemini API with `config.RAG_MODEL`. Requires `GEMINI_API_KEY` in the
environment (or a `.env` file, loaded via `python-dotenv`).
'''

from __future__ import annotations

import re

from src.rag import config

SYSTEM_PROMPT = (
    'You are a knowledgeable assistant answering questions about the Vachanamrut, '
    'a Hindu scripture of the BAPS Swaminarayan tradition, using only the passages '
    "provided below. Answer solely from these passages -- don't use outside "
    'knowledge. Cite every passage you rely on inline using its bracketed id, '
    'exactly as given (e.g. "[Gadhada I-1__c003]"). If the passages do not contain '
    'enough information to answer the question, say so plainly instead of '
    'guessing.'
)

_CITATION_RE = re.compile(r'\[([^\[\]]+)\]')


def _format_context(sources: list[dict]) -> str:
    '''Render retrieved chunks into labeled context blocks for the prompt.'''
    blocks = []
    for s in sources:
        label = f"{s['header']} -- {s['title']}" if s['title'] else s['header']
        blocks.append(f"[{s['chunk_id']}] ({label})\n{s['text']}")
    return '\n\n'.join(blocks)


def _build_user_message(query: str, sources: list[dict]) -> str:
    context = _format_context(sources)
    return f'Passages:\n\n{context}\n\nQuestion: {query}'


def _extract_cited_ids(answer: str, sources: list[dict]) -> list[str]:
    '''Return the chunk_ids from `sources` that the model actually cited, in
    first-mentioned order, de-duplicated.
    '''
    valid_ids = {s['chunk_id'] for s in sources}
    cited = []
    seen = set()
    for match in _CITATION_RE.findall(answer):
        if match in valid_ids and match not in seen:
            cited.append(match)
            seen.add(match)
    return cited


def _call_gemini(user_message: str, *, model: str, temperature: float, max_tokens: int) -> str:
    '''Call the Gemini API and return the response text. Lazy import keeps this
    module importable without the `google-genai` package installed.

    Some Gemini models silently ignore temperature/top_p/top_k rather than
    erroring (unlike Claude, which hard-rejects them on certain models) -- it's
    safe to always pass it here.
    '''
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text


def generate(
    query: str,
    sources: list[dict],
    *,
    model: str = config.RAG_MODEL,
    temperature: float = config.GENERATION_TEMPERATURE,
    max_tokens: int = config.GENERATION_MAX_TOKENS,
) -> dict:
    '''Generate an answer to `query` grounded in `sources` (as returned by
    `retrieve.retrieve()`). Returns `{'answer': str, 'cited': list[str]}`, where
    `cited` is the subset of source chunk_ids the model actually referenced.
    '''
    if not sources:
        return {
            'answer': "I don't have any passages to answer this question from.",
            'cited': [],
        }

    user_message = _build_user_message(query, sources)
    answer = _call_gemini(user_message, model=model, temperature=temperature, max_tokens=max_tokens)
    cited = _extract_cited_ids(answer, sources)

    return {'answer': answer, 'cited': cited}
