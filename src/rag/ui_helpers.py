'''Pure formatting helpers for the Streamlit UI (`app.py`).

Kept separate from `app.py` so the formatting logic is unit-testable without
importing Streamlit or running a script -- `app.py` should only call these and
wire the results into `st.*` widgets.
'''

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from src.rag import config

_CITATION_SPLIT_RE = re.compile(r'(\[[^\[\]]+\])')

# Anirdesh's website interleaves Bhugol-Khagol as vachno 263 (between
# Gadhada III-39 and Amdavad-4), whereas this corpus appends it last at
# position 274 -- so Amdavad-4 through Jetalpur-5 (this corpus's positions
# 263-273) are shifted down by one to Anirdesh's 264-274.
_ANIRDESH_BHUGOL_KHAGOL_POSITION = 274
_ANIRDESH_BHUGOL_KHAGOL_VACHNO = 263
_ANIRDESH_SHIFT_RANGE = (263, 273)

THEME_CSS = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Geist:wght@300;400;500;600;700&display=swap');

html, body, .stApp, .stApp * {
    font-family: 'Geist', sans-serif;
}

/* Streamlit renders expander/alert/spinner chevrons etc. as ligature text in
   an icon font (e.g. "keyboard_double_arrow_right") -- the blanket Geist
   rule above was overriding that font-family, so the ligature names showed
   up as literal, unrendered text instead of icon glyphs. Explicitly
   restore their icon font. */
[data-testid="stIconMaterial"], [data-testid="stImageIcon"], [data-testid="stSpinnerIcon"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* The "running man" status widget Streamlit shows top-right during every
   script rerun (i.e. every query submission) -- decorative chrome, not part
   of this app's UI. */
[data-testid="stStatusWidget"] {
    display: none !important;
}

/* Streamlit's built-in "Press Enter to submit form" / "Press Enter to apply"
   hint below text inputs -- redundant once the label is hidden and a
   rotating example line replaces it as the visual cue. */
[data-testid="InputInstructions"] {
    display: none !important;
}

/* Base app chrome goes transparent so the fixed .space-bg layer behind it
   (injected separately, see SPACE_BACKGROUND_HTML) shows through. A solid
   black lives on .space-bg itself as the fallback if that ever fails to
   render, so the page can never end up on browser-default white. */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0.4) !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #000000 0%, #0a0612 35%, #170c28 65%, #030207 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.15);
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.app-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: 0.06em;
    text-align: center;
    background: linear-gradient(90deg, #a78bfa, #f5d0fe, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 40px rgba(167, 139, 250, 0.45);
    margin: 0.5rem 0 1rem 0;
}

/* Rotating example-question hint above the query box. Pure CSS: every span
   shares one 32s animation, staggered by animation-delay (i * 4s, set in
   ROTATING_EXAMPLES_HTML below) so exactly one is visible at a time. */
.rotating-examples {
    position: relative;
    height: 1.5rem;
    max-width: 34rem;
    margin: 0 auto 1.5rem auto;
    text-align: center;
    color: #9ca3af;
    font-size: 0.95rem;
    font-style: italic;
}

.rotating-examples span {
    position: absolute;
    left: 0;
    right: 0;
    opacity: 0;
    animation: rotateExamples 32s infinite;
}

@keyframes rotateExamples {
    0%    { opacity: 0; }
    1%    { opacity: 1; }
    11%   { opacity: 1; }
    12.5% { opacity: 0; }
    100%  { opacity: 0; }
}

.answer-box {
    background: rgba(8, 7, 18, 0.72);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(167, 139, 250, 0.25);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    line-height: 1.7;
    color: #e5e7eb;
    box-shadow: 0 0 40px rgba(139, 92, 246, 0.12);
}

.citation-badge {
    display: inline-block;
    background: rgba(251, 191, 36, 0.12);
    border: 1px solid rgba(251, 191, 36, 0.5);
    color: #fbbf24;
    border-radius: 999px;
    padding: 0.05rem 0.55rem;
    font-size: 0.85em;
    box-shadow: 0 0 8px rgba(251, 191, 36, 0.25);
}

[data-testid="stForm"], [data-testid="stExpander"] {
    background: rgba(8, 7, 18, 0.6) !important;
    backdrop-filter: blur(6px);
    border: 1px solid rgba(167, 139, 250, 0.2) !important;
    border-radius: 14px !important;
}

[data-testid="stTextInput"] input {
    background: rgba(5, 4, 12, 0.85) !important;
    color: #e5e7eb !important;
    border: 1px solid rgba(167, 139, 250, 0.35) !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #6b7280 !important;
}

[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: rgba(5, 4, 12, 0.85) !important;
    color: #e5e7eb !important;
    border-color: rgba(167, 139, 250, 0.35) !important;
}

[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] {
    color: #cbd5e1;
}

/* ---- flying-through-deep-space background (fixed, behind all content) --- */

.space-bg {
    position: fixed;
    inset: 0;
    z-index: -1;
    overflow: hidden;
    background: #000000;
}

/* Each depth is a pair (-a/-b) of identical layers, animations offset by
   half a cycle, so as one finishes zooming in and fading out the other is
   just starting -- a continuous "flying forward through space" loop with no
   visible reset jump. Near layers use a shorter duration and bigger end
   scale than far ones, so they appear to rush past faster/closer, giving the
   scene real depth. Dot colors mix white/pale-blue/warm-gold/soft-orange to
   read as a colorful field rather than a flat monochrome starfield. */
.warp-layer {
    position: absolute;
    inset: 0;
    background-repeat: repeat;
    transform-origin: 50% 50%;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
}

.warp-far-a, .warp-far-b {
    background-image:
        radial-gradient(1.5px 1.5px at 15% 25%, #ffffff, transparent),
        radial-gradient(1.5px 1.5px at 45% 60%, #cfe3ff, transparent),
        radial-gradient(1.5px 1.5px at 70% 20%, #ffe4b8, transparent),
        radial-gradient(1.5px 1.5px at 85% 75%, #ffffff, transparent),
        radial-gradient(1.5px 1.5px at 25% 85%, #cfe3ff, transparent),
        radial-gradient(1.5px 1.5px at 60% 45%, #ffd699, transparent);
    background-size: 500px 500px;
    animation-name: warpZoomFar;
    animation-duration: 16s;
}
.warp-far-b { animation-delay: 8s; }

.warp-mid-a, .warp-mid-b {
    background-image:
        radial-gradient(2px 2px at 20% 30%, #ffffff, transparent),
        radial-gradient(2px 2px at 50% 70%, #ffd699, transparent),
        radial-gradient(2px 2px at 75% 15%, #cfe3ff, transparent),
        radial-gradient(2px 2px at 30% 80%, #ffffff, transparent),
        radial-gradient(2px 2px at 90% 55%, #ffe4b8, transparent),
        radial-gradient(2px 2px at 10% 55%, #d9c2ff, transparent);
    background-size: 340px 340px;
    animation-name: warpZoomMid;
    animation-duration: 10s;
}
.warp-mid-b { animation-delay: 5s; }

.warp-near-a, .warp-near-b {
    background-image:
        radial-gradient(2.5px 2.5px at 25% 40%, #ffffff, transparent),
        radial-gradient(2.5px 2.5px at 55% 65%, #cfe3ff, transparent),
        radial-gradient(2.5px 2.5px at 80% 20%, #ffd699, transparent),
        radial-gradient(2.5px 2.5px at 15% 75%, #ffffff, transparent),
        radial-gradient(2.5px 2.5px at 65% 10%, #d9c2ff, transparent);
    background-size: 200px 200px;
    animation-name: warpZoomNear;
    animation-duration: 6s;
}
.warp-near-b { animation-delay: 3s; }

@keyframes warpZoomFar {
    0%   { transform: scale(1);   opacity: 0; }
    15%  { opacity: 0.5; }
    85%  { opacity: 0.5; }
    100% { transform: scale(2.2); opacity: 0; }
}
@keyframes warpZoomMid {
    0%   { transform: scale(1);   opacity: 0; }
    15%  { opacity: 0.7; }
    85%  { opacity: 0.7; }
    100% { transform: scale(2.6); opacity: 0; }
}
@keyframes warpZoomNear {
    0%   { transform: scale(1);   opacity: 0; }
    15%  { opacity: 0.9; }
    85%  { opacity: 0.9; }
    100% { transform: scale(3.2); opacity: 0; }
}

.galaxy {
    position: absolute;
    border-radius: 50%;
    filter: blur(35px);
    opacity: 0.55;
}

.galaxy-1 {
    width: 320px; height: 190px; top: 10%; left: 68%;
    transform: rotate(25deg);
    background: radial-gradient(ellipse at center, rgba(251, 191, 146, 0.55), rgba(139, 92, 246, 0.15) 60%, transparent 80%);
    animation: driftGalaxy 100s ease-in-out infinite alternate;
}
.galaxy-2 {
    width: 240px; height: 150px; top: 60%; left: 8%;
    transform: rotate(-15deg);
    background: radial-gradient(ellipse at center, rgba(167, 139, 250, 0.45), transparent 75%);
    animation: driftGalaxy 140s ease-in-out infinite alternate;
}
.galaxy-3 {
    width: 180px; height: 180px; top: 78%; left: 78%;
    background: radial-gradient(circle at center, rgba(253, 224, 71, 0.4), transparent 70%);
    animation: driftGalaxy 90s ease-in-out infinite alternate-reverse;
}
.galaxy-4 {
    width: 260px; height: 170px; top: 28%; left: 14%;
    transform: rotate(45deg);
    background: radial-gradient(ellipse at center, rgba(147, 197, 253, 0.5), rgba(59, 130, 246, 0.1) 65%, transparent 80%);
    animation: driftGalaxy 120s ease-in-out infinite alternate;
}
.galaxy-5 {
    width: 200px; height: 200px; top: 42%; left: 88%;
    background: radial-gradient(circle at center, rgba(244, 114, 182, 0.4), transparent 70%);
    animation: driftGalaxy 110s ease-in-out infinite alternate-reverse;
}
.galaxy-6 {
    width: 220px; height: 140px; top: 88%; left: 38%;
    transform: rotate(-30deg);
    background: radial-gradient(ellipse at center, rgba(94, 234, 212, 0.4), transparent 75%);
    animation: driftGalaxy 130s ease-in-out infinite alternate;
}

@keyframes driftGalaxy {
    from { transform: translate(0, 0) scale(1); }
    to { transform: translate(-90px, 50px) scale(1.3); }
}
</style>
'''

SPACE_BACKGROUND_HTML = '''
<div class="space-bg">
    <div class="warp-layer warp-far-a"></div>
    <div class="warp-layer warp-far-b"></div>
    <div class="warp-layer warp-mid-a"></div>
    <div class="warp-layer warp-mid-b"></div>
    <div class="warp-layer warp-near-a"></div>
    <div class="warp-layer warp-near-b"></div>
    <div class="galaxy galaxy-1"></div>
    <div class="galaxy galaxy-2"></div>
    <div class="galaxy galaxy-3"></div>
    <div class="galaxy galaxy-4"></div>
    <div class="galaxy galaxy-5"></div>
    <div class="galaxy galaxy-6"></div>
</div>
'''

EXAMPLE_QUESTIONS = [
    'What is the nature of maya?',
    'How do I find peace amid uncertainty?',
    'What does true surrender really mean?',
    'How can I let go of attachment to outcomes?',
    'What is the relationship between the soul and God?',
    'How do I stay steady when everything around me is changing?',
    'What is the purpose of devotion?',
    'How can I quiet a restless mind?',
]

ROTATING_EXAMPLES_HTML = '''
<div class="rotating-examples">
    {spans}
</div>
'''.format(spans='\n    '.join(
    f'<span style="animation-delay: {i * 4}s">Try asking: "{html.escape(q)}"</span>'
    for i, q in enumerate(EXAMPLE_QUESTIONS)
))


def format_answer_html(answer: str, cited: list[str]) -> str:
    '''Render `answer` as HTML with cited chunk-id brackets (e.g.
    "[Gadhada I-1__c003]") wrapped in glowing citation badges. Escapes
    everything else so the LLM output can't inject arbitrary HTML.
    '''
    cited_set = set(cited)
    parts = _CITATION_SPLIT_RE.split(answer)
    out = []
    for part in parts:
        if part.startswith('[') and part.endswith(']') and part[1:-1] in cited_set:
            out.append(f'<span class="citation-badge">{html.escape(part)}</span>')
        else:
            out.append(html.escape(part))
    return ''.join(out).replace('\n', '<br>')


def build_source_label(source: dict) -> str:
    '''Human-readable label for one retrieved source, e.g.
    "Gadhada I-1 -- On the nature of maya".
    '''
    if source['title']:
        return f"{source['header']} -- {source['title']}"
    return source['header']


def build_quote_preview(text: str, max_chars: int = 500) -> str:
    '''Trim a retrieved passage down to a short quote (a few sentences),
    cutting at the last sentence boundary before `max_chars` where one
    exists so the preview doesn't end mid-sentence.
    '''
    text = text.strip()
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    sentence_end = max(truncated.rfind('. '), truncated.rfind('? '), truncated.rfind('! '))
    if sentence_end > max_chars * 0.4:
        return truncated[:sentence_end + 1].strip()

    space = truncated.rfind(' ')
    if space > 0:
        truncated = truncated[:space]
    return truncated.strip() + '...'


def load_anirdesh_vachno_map(path: Path = config.DISCOURSES_PATH) -> dict[str, int]:
    '''Map each discourse's parent_id to its `vachno` query-param value on
    anirdesh.com. Anirdesh interleaves Bhugol-Khagol as vachno 263 (between
    Gadhada III-39 and Amdavad-4) rather than appending it last as this
    corpus does at position 274 -- see the module-level comment above.
    '''
    mapping: dict[str, int] = {}
    with path.open('r', encoding='utf-8') as f:
        position = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            position += 1
            parent_id = json.loads(line)['id']
            if position == _ANIRDESH_BHUGOL_KHAGOL_POSITION:
                mapping[parent_id] = _ANIRDESH_BHUGOL_KHAGOL_VACHNO
            elif _ANIRDESH_SHIFT_RANGE[0] <= position <= _ANIRDESH_SHIFT_RANGE[1]:
                mapping[parent_id] = position + 1
            else:
                mapping[parent_id] = position
    return mapping


def anirdesh_url(vachno: int) -> str:
    '''Link to the full discourse text on anirdesh.com for the given vachno.'''
    return f'https://www.anirdesh.com/vachanamrut/index.php?format=en&vachno={vachno}'
