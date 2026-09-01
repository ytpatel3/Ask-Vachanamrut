'''Central configuration for the RAG pipeline. Plain constants, env-overridable.'''

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / '.env')
DATA_DIR = PROJECT_ROOT / 'data'

DISCOURSES_PATH = DATA_DIR / 'clean' / 'vachanamrut.jsonl'
CHUNKS_PATH = DATA_DIR / 'clean' / 'vachanamrut_chunks.jsonl'
CHROMA_DIR = DATA_DIR / 'chroma'
COLLECTION_NAME = 'vachanamrut_chunks_v1'
EVAL_QA_PATH = DATA_DIR / 'eval' / 'qa_set.jsonl'

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5')
RERANKER_MODEL = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-large')
# Gemini free tier: request-capped (500/day), not token-capped, so it's the
# closest to "unlimited" among free options -- see PROJECT_PLAN.md.
RAG_MODEL = os.getenv('RAG_MODEL', 'gemini-2.5-flash')

# BGE asymmetric: queries get this prefix, passages do not.
BGE_QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: '

CHUNK_TARGET_TOKENS = 400
CHUNK_MAX_TOKENS = 480
CHUNK_OVERLAP_TOKENS = 50
CHUNK_OVERLAP_MAX_TOKENS = 100
CHUNK_MIN_TOKENS = 80

K_INITIAL = 30
K_FINAL = 5
EXPAND_NEIGHBORS = 1
MAX_EXPANDED_TOKENS = 1500

EMBED_BATCH_SIZE = 32
RERANK_BATCH_SIZE = 16

GENERATION_TEMPERATURE = 0.2
GENERATION_MAX_TOKENS = 1024
