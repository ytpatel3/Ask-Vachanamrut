# Ask Vachanamrut

The Vachanamrut is the foundational scripture of the Swaminarayan tradition:
273 spiritual discourses delivered by Bhagwan Swaminarayan between 1819 and
1829, filled with philosophical and practical answers to questions about
devotion, the self, and the nature of God.

**Ask Vachanamrut** is a retrieval-augmented generation (RAG) system that
answers questions about the Vachanamrut grounded in its actual text --
every answer cites the specific discourse(s) it's drawn from, so you can
verify it and read further.

**Try it live:** [ask-vachanamrut.streamlit.app](https://ask-vachanamrut.streamlit.app) <!-- TODO: replace with your deployed URL -->

## How it works, briefly

Retrieval and generation are separate, inspectable steps -- this isn't a
single black-box model call:

1. Your question is embedded and matched against a vector index of ~1,200
   passages chunked from all 273 discourses (plus the Bhugol-Khagol
   appendix).
2. The best-matching passages are expanded with surrounding context and
   handed to an LLM, instructed to answer *only* from what's given and to
   cite every passage it relies on.
3. The app shows the answer with inline citations, plus the actual
   retrieved passages and a link to read the full discourse.

## Using the app

- Type a question in the box and hit **Ask** -- it doesn't need to be
  narrowly about Vachanamrut terminology; broad questions ("How do I find
  peace amid uncertainty?") work too.
- Citations in the answer (e.g. `[Gadhada I-1]`) are clickable references
  to the passages listed under **Show retrieved passages**, each linking to
  the full discourse.
- The sidebar lets you pick a model, see an estimated request budget for
  the day, and tune advanced retrieval settings (how many passages are
  retrieved, how much surrounding context each gets, generation
  temperature).

## Project structure

```
app.py                     Streamlit UI entrypoint
src/rag/                   the RAG system itself (chunking, embedding,
                            retrieval, generation, evaluation)
src/text_normalization/    raw text -> structured discourse JSON
src/pdf_ingestion/         one-time PDF -> raw text extraction
scripts/                   CLI entrypoints (build the index, run eval, etc.)
data/                      raw/clean text, the vector index, the eval set
tests/                     mirrors src/, one test file per module
```

## Running it locally

### Requirements

- Python 3.12
- A free [Gemini API key](https://aistudio.google.com/apikey) (Google AI
  Studio -- no billing required for the model this project uses)

### Setup

```bash
git clone https://github.com/ytpatel3/Ask-Vachanamrut.git
cd Ask-Vachanamrut
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-key-here
```

The vector index (`data/chroma/`) is already built and committed, so you
can run the app immediately without any indexing step:

```bash
streamlit run app.py
```

### Making changes

If you change the chunking logic, the embedding model, or the source text
itself, rebuild the index:

```bash
python -m scripts.build_index          # full rebuild
python -m scripts.build_index --rebuild   # drop and recreate the collection
```

Run the test suite (fast; mocks all external services and models):

```bash
pytest
```

Live end-to-end tests hit the real Chroma index and the real Gemini API
(needs `GEMINI_API_KEY`) and are excluded by default:

```bash
pytest -m live
```

Check retrieval/generation quality against the grounded 25-question eval
set:

```bash
python -m scripts.run_eval                # full pass (retrieval + generation)
python -m scripts.run_eval --no-generation   # fast, free, retrieval-only pass
```

### Re-running earlier pipeline steps (rarely needed)

The raw text and its normalized/chunked/embedded outputs are all already
committed, so these steps are only needed if you're changing the pipeline
itself, not for regular use.

```bash
# Re-extract text from the source PDF (needs its own dependencies):
pip install -r src/pdf_ingestion/requirements.txt
python src/pdf_ingestion/extract_pdf.py

# Re-normalize raw text into structured discourse JSON:
python -m scripts.normalize
```

## Data source

The scripture text used here is a specific English translation of the
Vachanamrut. If you fork this project to publish your own deployment,
verify you have the right to redistribute whichever translation you use --
this repo does not grant that right on your behalf.
