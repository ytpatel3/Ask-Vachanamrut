Text Normalization Pipeline
===========================

Purpose
-------
This package converts a raw plaintext dump of the Vachanamrut into
structured, clean JSON records suitable for NLP, embeddings, and RAG.

Responsibilities
----------------
- Remove page numbers, footers, and section labels
- Detect valid Vachanamrut headers using strict patterns
- Group text into individual Vachanamrut sections
- Normalize diacritics and spacing
- Output one JSON object per Vachanamrut

Supported Sections
------------------
- Gadhada I
- Gadhada II
- Gadhada III
- Sarangpur
- Kariyani
- Loya
- Panchala
- Vartal
- Amdavad
- Jetalpur
- Ashlali

Output Format
-------------
Each line of the output JSONL file:

{
  "id": "Gadhada_I_1",
  "section": "Gadhada I-1",
  "title": "Title if present",
  "text": "Cleaned full text",
  "start_line": 123,
  "end_line": 256
}

Pipeline Components
----------------
loader.py: Reads raw text from disk

cleaners.py: Low-level normalization - text cleanup utilities

parsers.py: High-level normalization - header detection and section grouping

writer.py: JSONL output writer

pipeline.py: Orchestrates the full process