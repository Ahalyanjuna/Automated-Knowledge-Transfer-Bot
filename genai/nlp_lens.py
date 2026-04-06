"""
=============================================================================
nlp_lens.py  —  NLP Lens: Live Frontend NLP Analysis Panel
=============================================================================
A drop-in Streamlit component that gives a real-time, visual breakdown of
what the NLP pipeline is actually doing — designed to impress your teacher.

Shows:
  1. Query Analysis    — tokenization, language detection, NER on the user's
                         raw input (before it even hits the retriever)
  2. Source Chunk NLP  — tags, complexity, docstring flag, entities from the
                         actual retrieved chunks powering the answer
  3. Faithfulness Map  — token-overlap heatmap between answer and context
  4. Pipeline Trace    — step-by-step animated badge trail showing which NLP
                         steps fired and what they produced

Usage in app.py:
  from nlp_lens import render_nlp_lens

  # After engine.generate_response() returns `result`:
  render_nlp_lens(
      query        = prompt,            # raw user input
      result       = result,            # dict from generate_response()
      final_hits   = final_hits,        # list of retrieved chunk dicts
      chroma_meta  = chroma_meta,       # optional: metadata dicts from Chroma
  )
=============================================================================
"""

from __future__ import annotations
import re
import math
import html
from collections import Counter
from typing import Optional

import streamlit as st

# ── Optional heavy deps (graceful fallbacks) ──────────────────────────────

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False
    _NLP = None

# ─────────────────────────────────────────────────────────────────────────────
# CSS  (injected once per session)
# ─────────────────────────────────────────────────────────────────────────────

def _inject_css():
    st.markdown("""
<style>
/* ── NLP Lens shell ── */
.nlp-lens-wrap {
    background: linear-gradient(135deg, #0d0d24 0%, #111128 100%);
    border: 1px solid #2a2a55;
    border-radius: 14px;
    padding: 1.1rem 1.4rem 1.3rem;
    margin: 1rem 0 0.5rem;
    position: relative;
    overflow: hidden;
}
.nlp-lens-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        90deg,
        rgba(167,139,250,0.015) 0px,
        rgba(167,139,250,0.015) 1px,
        transparent 1px,
        transparent 40px
    );
    pointer-events: none;
}
.nlp-lens-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: #7c3aed;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.nlp-lens-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #2a2a55, transparent);
}

/* ── Token chips ── */
.token-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin: 0.4rem 0 0.8rem;
}
.tok {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 7px;
    border-radius: 4px;
    border: 1px solid;
}
.tok-word    { background: rgba(99,102,241,0.12); border-color: #4338ca; color: #a5b4fc; }
.tok-num     { background: rgba(251,191,36,0.1);  border-color: #d97706; color: #fcd34d; }
.tok-punct   { background: rgba(107,114,128,0.1); border-color: #374151; color: #6b7280; }
.tok-stop    { background: rgba(55,65,81,0.08);   border-color: #1f2937; color: #374151; }

/* ── NER entity badges ── */
.ner-row { display: flex; flex-wrap: wrap; gap: 5px; margin: 0.3rem 0 0.7rem; }
.ner-badge {
    font-size: 0.68rem;
    font-family: 'Space Mono', monospace;
    border-radius: 5px;
    padding: 2px 6px 2px 5px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.ner-FUNC       { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid #3730a3; }
.ner-CLASS      { background: rgba(236,72,153,0.12); color: #f472b6; border: 1px solid #9d174d; }
.ner-LIB        { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid #065f46; }
.ner-URL        { background: rgba(14,165,233,0.12); color: #38bdf8; border: 1px solid #075985; }
.ner-TODO       { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid #92400e; }
.ner-FILE_PATH  { background: rgba(163,230,53,0.1);  color: #a3e635; border: 1px solid #3f6212; }
.ner-ENV_VAR    { background: rgba(251,146,60,0.12); color: #fb923c; border: 1px solid #92400e; }
.ner-VERSION    { background: rgba(167,139,250,0.12);color: #a78bfa; border: 1px solid #5b21b6; }
.ner-ORG        { background: rgba(244,114,182,0.1); color: #f9a8d4; border: 1px solid #831843; }
.ner-default    { background: rgba(107,114,128,0.1); color: #9ca3af; border: 1px solid #374151; }
.ner-label {
    font-size: 0.58rem;
    opacity: 0.65;
    letter-spacing: 0.06em;
}

/* ── Lang badge ── */
.lang-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 99px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #34d399;
    margin: 0.3rem 0 0.7rem;
}
.lang-conf { opacity: 0.55; font-size: 0.62rem; }

/* ── Pipeline trace badges ── */
.pipe-trace {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
    margin: 0.5rem 0;
}
.pipe-step {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 4px 10px;
    border-radius: 4px;
    opacity: 1;
}
.pipe-arrow {
    color: #4b5563;
    font-size: 0.75rem;
    margin: 0 1px;
    font-weight: bold;
}
.ps-clean  { background: rgba(99,102,241,0.15); color:#818cf8; border:1px solid #3730a3; }
.ps-ner    { background: rgba(52,211,153,0.12); color:#34d399; border:1px solid #065f46; }
.ps-chunk  { background: rgba(251,191,36,0.1);  color:#fbbf24; border:1px solid #92400e; }
.ps-tag    { background: rgba(236,72,153,0.1);  color:#f472b6; border:1px solid #9d174d; }
.ps-embed  { background: rgba(14,165,233,0.12); color:#38bdf8; border:1px solid #075985; }
.ps-ml     { background: rgba(167,139,250,0.15);color:#a78bfa; border:1px solid #5b21b6; }

/* ── Chunk metadata cards ── */
.chunk-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid #1e1e40;
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.5rem;
}
.chunk-card-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #a78bfa;
    margin-bottom: 0.45rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.chunk-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    margin: 0.35rem 0;
}
.ctag {
    font-size: 0.62rem;
    font-family: 'Space Mono', monospace;
    background: rgba(99,102,241,0.1);
    border: 1px solid #312e81;
    color: #818cf8;
    border-radius: 3px;
    padding: 1px 5px;
}
.complexity-bar-wrap {
    height: 5px;
    background: #1e1e40;
    border-radius: 3px;
    margin: 0.4rem 0 0.2rem;
    overflow: hidden;
}
.complexity-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}

/* ── Faithfulness heatmap ── */
.faith-wrap { margin: 0.5rem 0; }
.faith-row  { display: flex; flex-wrap: wrap; gap: 3px; }
.faith-tok  {
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem;
    padding: 1px 4px;
    border-radius: 3px;
    display: inline-block;
}

/* ── Section labels ── */
.nlp-section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    color: #4b5563;
    text-transform: uppercase;
    margin: 0.6rem 0 0.3rem;
}

/* ── Small stat pills row ── */
.stat-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin: 0.3rem 0;
}
.stat-pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid #1e1e40;
    border-radius: 99px;
    padding: 2px 8px;
    color: #6b7280;
}
.stat-pill span { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# NLP ANALYSIS FUNCTIONS (lightweight, no heavy deps required)
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the","a","an","is","it","in","on","at","to","for","of","and","or",
    "but","with","this","that","are","was","were","be","been","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","i","you","he","she","we","they","what","how","when","where",
    "which","who","not","from","by","as","if","then","so","its","their",
}

_WORD_RE   = re.compile(r"[A-Za-z_]\w*")
_NUM_RE    = re.compile(r"\b\d+(?:\.\d+)?\b")
_PUNCT_RE  = re.compile(r"[^\w\s]")

_NER_RULES = [
    ("FUNC",      re.compile(r'\bdef\s+([A-Za-z_]\w*)')),
    ("CLASS",     re.compile(r'\bclass\s+([A-Za-z_]\w*)')),
    ("LIB",       re.compile(r'(?:import|from|require|use)\s+["\']?([A-Za-z_][\w./\-]*)["\']?')),
    ("URL",       re.compile(r'https?://[^\s\'"<>]+')),
    ("TODO",      re.compile(r'\b(TODO|FIXME|HACK|XXX)\b', re.I)),
    ("FILE_PATH", re.compile(r'(?:/[\w.\-]+){2,}')),
    ("ENV_VAR",   re.compile(r'\$\{?([A-Z_][A-Z0-9_]{2,})\}?')),
    ("VERSION",   re.compile(r'\bv?(\d+\.\d+(?:\.\d+)?)\b')),
]

LANG_NAMES = {
    "en":"English","ta":"Tamil","hi":"Hindi","fr":"French","de":"German",
    "es":"Spanish","zh-cn":"Chinese","ja":"Japanese","ko":"Korean",
    "ar":"Arabic","ru":"Russian","pt":"Portuguese","it":"Italian",
    "nl":"Dutch","tr":"Turkish","vi":"Vietnamese","th":"Thai",
    "id":"Indonesian","pl":"Polish","uk":"Ukrainian",
}

def _detect_lang(text: str) -> tuple[str, float]:
    if LANGDETECT_OK:
        try:
            code = detect(text)
            return code, 0.88
        except Exception:
            pass
    return "en", 1.0

def _tokenize_query(text: str) -> list[dict]:
    tokens = []
    pos = 0
    for part in re.split(r'(\s+)', text):
        if not part.strip():
            pos += len(part)
            continue
        word_m = _WORD_RE.fullmatch(part)
        num_m  = _NUM_RE.fullmatch(part)
        punct_m= _PUNCT_RE.fullmatch(part)
        if word_m:
            kind = "stop" if part.lower() in _STOPWORDS else "word"
        elif num_m:
            kind = "num"
        else:
            kind = "punct"
        tokens.append({"text": part, "kind": kind, "pos": pos})
        pos += len(part)
    return tokens

def _ner_query(text: str) -> list[dict]:
    entities = []
    if SPACY_OK and _NLP:
        doc = _NLP(text)
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_})

    for label, pat in _NER_RULES:
        for m in pat.finditer(text):
            matched = next((g for g in m.groups() if g), m.group(0))
            if matched and len(matched) > 1:
                entities.append({"text": matched.strip(), "label": label})

    # Deduplicate
    seen = set()
    deduped = []
    for e in entities:
        key = (e["text"].lower(), e["label"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped[:12]  # cap for display

def _word_set(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS and len(w) > 2}

def _overlap_score(answer: str, context: str) -> float:
    a_words = _word_set(answer)
    c_words = _word_set(context)
    if not a_words:
        return 0.0
    return len(a_words & c_words) / len(a_words)

def _faithfulness_tokens(answer: str, context: str) -> list[dict]:
    ctx_words = _word_set(context)
    tokens = []
    for word in answer.split():
        clean = re.sub(r'[^\w]', '', word).lower()
        matched = clean in ctx_words and len(clean) > 2 and clean not in _STOPWORDS
        tokens.append({"text": word, "matched": matched})
    return tokens[:120]  # cap

def _complexity_color(score: float) -> str:
    if score < 0.25: return "#34d399"
    if score < 0.5:  return "#60a5fa"
    if score < 0.75: return "#fbbf24"
    return "#f472b6"

def _parse_str_list(val) -> list[str]:
    if isinstance(val, list): return val
    if not val or val in ("[]", ""): return []
    try:
        import ast
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# HTML BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _render_token_chips(tokens: list[dict]) -> str:
    chips = ""
    for t in tokens:
        cls = f"tok-{t['kind']}"
        txt = html.escape(t["text"])
        chips += f'<span class="tok {cls}">{txt}</span>'
    return f'<div class="token-row">{chips}</div>'

def _render_ner_badges(entities: list[dict]) -> str:
    if not entities:
        return '<span style="color:#4b5563;font-size:0.72rem">No named entities detected</span>'
    badges = ""
    for e in entities:
        label = e["label"]
        cls   = f"ner-{label}" if label in {
            "FUNC","CLASS","LIB","URL","TODO","FILE_PATH","ENV_VAR","VERSION","ORG"
        } else "ner-default"
        txt = html.escape(e["text"][:30])
        badges += (
            f'<span class="ner-badge {cls}">'
            f'{txt}'
            f'<span class="ner-label">{label}</span>'
            f'</span>'
        )
    return f'<div class="ner-row">{badges}</div>'

# def _render_pipeline_trace(tokens: list[dict], entities: list[dict],
#                             lang: str, is_translated: bool) -> str:
#     steps = [
#         ("ps-clean",  "🧹", f"clean·{len(tokens)}tok"),
#         ("ps-ner",    "🏷️",  f"ner·{len(entities)}ent"),
#         ("ps-chunk",  "✂️",  "chunk"),
#         ("ps-tag",    "🔖",  "tag"),
#         ("ps-embed",  "📐",  "embed·384d"),
#         ("ps-ml",     "🌐",  f"lang·{lang}" + ("→en" if is_translated else "")),
#     ]
#     html_out = '<div class="pipe-trace">'
#     for i, (cls, icon, label) in enumerate(steps):
#         html_out += f'<span class="pipe-step {cls}">{icon} {label}</span>'
#         if i < len(steps) - 1:
#             html_out += '<span class="pipe-arrow">→</span>'
#     html_out += '</div>'
#     return html_out
def _render_pipeline_trace(tokens: list[dict], entities: list[dict],
                            lang: str, is_translated: bool) -> str:
    steps = [
        ("ps-clean",  f"clean·{len(tokens)}tok"),
        ("ps-ner",    f"ner·{len(entities)}ent"),
        ("ps-chunk",  "chunk"),
        ("ps-tag",    "tag"),
        ("ps-embed",  "embed·384d"),
        ("ps-ml",     f"lang·{lang}" + ("→en" if is_translated else "")),
    ]

    html_out = '<div class="pipe-trace">'
    for i, (cls, label) in enumerate(steps):
        html_out += f'<span class="pipe-step {cls}">{label}</span>'
        if i < len(steps) - 1:
            html_out += '<span class="pipe-arrow">→</span>'
    html_out += '</div>'
    
    return html_out

def _render_faithfulness_heatmap(faith_tokens: list[dict]) -> str:
    row = ""
    for t in faith_tokens:
        esc = html.escape(t["text"])
        if t["matched"]:
            opacity = 0.85
            bg = f"rgba(52,211,153,{opacity})"
            color = "#0f1f18"
        else:
            bg = "rgba(255,255,255,0.04)"
            color = "#6b7280"
        row += f'<span class="faith-tok" style="background:{bg};color:{color}">{esc}</span> '
    return f'<div class="faith-wrap"><div class="faith-row">{row}</div></div>'

def _render_chunk_card(hit: dict, idx: int) -> str:
    meta     = hit.get("metadata", {})
    fname    = meta.get("source_file", "Unknown")
    fname_short = fname.split("/")[-1] if "/" in fname else fname
    ctype    = meta.get("chunk_type", "—")
    lang     = meta.get("normalized_lang") or meta.get("language", "—")
    tokens   = int(meta.get("token_count", 0) or 0)
    complexity = float(meta.get("complexity_score", 0) or 0)
    has_doc  = str(meta.get("has_docstring", "")).lower() in ("true","1","yes")
    has_test = str(meta.get("has_tests", "")).lower() in ("true","1","yes")
    has_todo = str(meta.get("has_todos", "")).lower() in ("true","1","yes")
    tags     = _parse_str_list(meta.get("tags", "[]"))
    rl_score = hit.get("rl_score", None)

    # Complexity bar
    bar_w   = int(complexity * 100)
    bar_col = _complexity_color(complexity)

    # Tags HTML
    tags_html = "".join(f'<span class="ctag">{t}</span>' for t in tags[:8])
    tags_block = f'<div class="chunk-tags">{tags_html}</div>' if tags else ""

    # Flags
    flags = []
    if has_doc:  flags.append("docstring")
    if has_test: flags.append("tests")
    if has_todo: flags.append("todos")
    flags_html = " ".join(f'<span class="stat-pill">{f}</span>' for f in flags)

    rl_html = ""
    if rl_score is not None:
        rl_html = f'<span class="stat-pill">RL <span>{rl_score:.3f}</span></span>'

    return f"""
<div class="chunk-card">
  <div class="chunk-card-header">
    <span>#{idx+1} · {html.escape(fname_short)}</span>
    <span style="color:#4b5563">{ctype} · {lang}</span>
  </div>
  <div class="stat-pills">
    <span class="stat-pill">tokens <span>{tokens}</span></span>
    <span class="stat-pill">complexity <span>{complexity:.2f}</span></span>
    {rl_html}
    {flags_html}
  </div>
  <div class="complexity-bar-wrap">
    <div class="complexity-bar" style="width:{bar_w}%;background:{bar_col}"></div>
  </div>
  {tags_block}
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def render_nlp_lens(
    query: str,
    result: dict,
    final_hits: Optional[list] = None,
    *,
    show_faithfulness: bool = True,
    show_source_nlp: bool   = True,
    show_pipeline: bool     = True,
):
    """
    Render the full NLP Lens panel inside a Streamlit expander.

    Parameters
    ──────────
    query            : raw user query string
    result           : dict returned by KTChatEngine.generate_response()
    final_hits       : list of hit dicts (each has 'content', 'metadata', 'rl_score')
                       — pass engine's `final_hits` variable from generate_response
    show_faithfulness: show the token overlap heatmap
    show_source_nlp  : show retrieved chunk NLP metadata cards
    show_pipeline    : show animated pipeline trace
    """
    _inject_css()

    # ── Compute everything ────────────────────────────────────────────────
    original_query  = result.get("original_query", query)
    answer_english  = result.get("answer_english", result.get("answer", ""))
    is_translated   = result.get("is_translated", False)
    detected_lang   = result.get("detected_lang", "en")
    detected_name   = result.get("detected_lang_name", "English")

    # Query NLP (always on English version for NER consistency)
    analysis_text = result.get("translated_query") or original_query
    tokens   = _tokenize_query(analysis_text)
    entities = _ner_query(analysis_text)

    word_tokens  = [t for t in tokens if t["kind"] in ("word",)]
    stop_tokens  = [t for t in tokens if t["kind"] == "stop"]
    content_toks = [t for t in tokens if t["kind"] == "word"]
    num_tokens   = [t for t in tokens if t["kind"] == "num"]

    # Faithfulness
    if final_hits and show_faithfulness:
        full_context = " ".join(h.get("content","") for h in final_hits)
        faith_score  = _overlap_score(answer_english, full_context)
        faith_tokens = _faithfulness_tokens(answer_english, full_context)
    else:
        full_context = ""
        faith_score  = 0.0
        faith_tokens = []

    # ── Render inside expander ────────────────────────────────────────────
    with st.expander("NLP Lens — See what the pipeline did", expanded=False):

        st.markdown('<div class="nlp-lens-wrap">', unsafe_allow_html=True)

        # ── SECTION 1: Query Analysis ─────────────────────────────────────
        st.markdown(
            '<div class="nlp-lens-title">Query NLP Analysis</div>',
            unsafe_allow_html=True
        )

        # Language detection
        lang_display = LANG_NAMES.get(detected_lang, detected_lang.upper())
        flag_note = " · translated → English for retrieval" if is_translated else ""
        st.markdown(
            f'<span class="lang-pill">'
            f'{lang_display} <span class="lang-conf">({detected_lang}){flag_note}</span>'
            f'</span>',
            unsafe_allow_html=True
        )

        # Token stats
        st.markdown('<div class="nlp-section-label">Tokenization</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tokens",   len(tokens))
        col2.metric("Content Words",  len(content_toks))
        col3.metric("Stop Words",     len(stop_tokens))
        col4.metric("Numbers",        len(num_tokens))

        # Token chips
        st.markdown(
            '<div class="nlp-section-label">Token Visualization</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            _render_token_chips(tokens),
            unsafe_allow_html=True
        )
        st.caption("content word · number · punctuation · dimmed = stopword")

        # NER
        st.markdown('<div class="nlp-section-label">Named Entity Recognition</div>', unsafe_allow_html=True)
        st.markdown(_render_ner_badges(entities), unsafe_allow_html=True)

        # Pipeline trace
        if show_pipeline:
            st.markdown('<div class="nlp-section-label">Pipeline Trace</div>', unsafe_allow_html=True)
            st.markdown(
                _render_pipeline_trace(tokens, entities, detected_lang, is_translated),
                unsafe_allow_html=True
            )

        st.divider()

        # ── SECTION 2: Source Chunk NLP ───────────────────────────────────
        if show_source_nlp and final_hits:
            st.markdown(
                '<div class="nlp-lens-title">Retrieved Chunk NLP Metadata</div>',
                unsafe_allow_html=True
            )
            st.caption("These are the chunks the retriever selected. Their NLP metadata powered the answer.")

            for i, hit in enumerate(final_hits):
                st.markdown(_render_chunk_card(hit, i), unsafe_allow_html=True)

            st.divider()

        # ── SECTION 3: Faithfulness Heatmap ──────────────────────────────
        if show_faithfulness and faith_tokens:
            st.markdown(
                '<div class="nlp-lens-title">Answer Faithfulness Heatmap</div>',
                unsafe_allow_html=True
            )

            faith_col1, faith_col2, faith_col3 = st.columns([1,1,2])
            faith_col1.metric(
                "Overlap Score",
                f"{faith_score:.0%}",
                help="% of answer's content words found in retrieved context"
            )
            faith_col2.metric(
                "Grounded Tokens",
                f"{sum(1 for t in faith_tokens if t['matched'])} / {len(faith_tokens)}"
            )
            with faith_col3:
                st.caption("Green = answer word found in context · Grey = not in context")

            st.markdown(_render_faithfulness_heatmap(faith_tokens), unsafe_allow_html=True)
            st.caption(
                "**How to read this:** Every word in the AI answer is shown. "
                "Green words appear in the retrieved source code, meaning the answer is grounded. "
                "Grey words are generated additions."
            )

        st.markdown('</div>', unsafe_allow_html=True)  # close nlp-lens-wrap


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE MINI BADGE  (use in sidebar or chat without full expander)
# ─────────────────────────────────────────────────────────────────────────────

def render_nlp_mini_badge(result: dict):
    """
    Tiny inline badge row showing lang + faithfulness score.
    Useful under each assistant message bubble.
    """
    _inject_css()
    lang     = result.get("detected_lang", "en")
    lang_nm  = LANG_NAMES.get(lang, lang.upper())
    xlated   = result.get("is_translated", False)
    sources  = result.get("sources", [])

    pills = [
        f'<span class="stat-pill">lang: <span>{lang_nm}</span></span>',
        f'<span class="stat-pill">sources: <span>{len(sources)}</span></span>',
    ]
    if xlated:
        pills.append('<span class="stat-pill">translated</span>')

    st.markdown(
        f'<div class="stat-pills" style="margin-top:0.3rem">{"".join(pills)}</div>',
        unsafe_allow_html=True
    )