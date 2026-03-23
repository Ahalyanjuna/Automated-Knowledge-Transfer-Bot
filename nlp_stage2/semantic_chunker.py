"""
=============================================================================
semantic_chunker.py  —  Step 3: Semantic Chunking
=============================================================================
Breaks each NLPChunk's cleaned_text into coherent semantic segments.

Strategy selection (automatic, based on chunk_type / language):
  DOC chunks  (markdown, rst, doc_section, html_text …)
      → Split on headings (## …) or double-newlines (paragraphs)

  CODE chunks (function, method, class, code_block …)
      → Split on top-level function/class boundaries, or
        sliding window with overlap for large single functions

  DATA chunks (json_key, yaml_section, csv_batch …)
      → Each data chunk is already atomic — one segment

  NOTEBOOK cells
      → Already per-cell from Stage 1; large markdown cells are sub-split

  Fallback
      → Sliding window (max_tokens words, stride overlap)

Optional upgrade  (pip install sentence-transformers):
  use_similarity=True → cosine-similarity drop between consecutive
  sentence embeddings triggers a boundary.
=============================================================================
"""

from __future__ import annotations
import re
import logging
from dataclasses import asdict
from models import NLPChunk, SemanticSegment
import numpy as np
log = logging.getLogger("semantic_chunker")

# ── Optional sentence-transformers ────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_MODEL    = SentenceTransformer("all-MiniLM-L6-v2")
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False
    _ST_MODEL    = None

# ── Patterns ──────────────────────────────────────────────────────────────

_MD_HEADING   = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)
_BLANK_LINE   = re.compile(r'\n\s*\n')
_FUNC_DEF     = re.compile(
    r'^(?:async\s+)?(?:def|function|func|fn|'
    r'public|private|protected|static|void|int|str|bool|float)\s+\w+',
    re.MULTILINE,
)
_CLASS_DEF    = re.compile(r'^(?:class|struct|interface|impl|trait)\s+\w+', re.MULTILINE)
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# ── Chunk type sets ───────────────────────────────────────────────────────

_DOC_TYPES  = {
    "doc_section", "doc_paragraph", "markdown", "rst",
    "html_text", "html_section", "notebook_markdown_cell",
}
_CODE_TYPES = {
    "function", "method", "class", "code_block",
    "shell_function", "script_block",
    "notebook_code_cell", "sql_select", "sql_create",
}
_DATA_TYPES = {
    "json_key", "json_array_batch", "yaml_section",
    "toml_section", "csv_batch", "xml_element",
}


# ═══════════════════════════════════════════════════════════════════════════
# Splitting helpers
# ═══════════════════════════════════════════════════════════════════════════

def _split_by_pattern(text: str, pattern: re.Pattern) -> list[tuple[int, int]]:
    """Return (start, end) char spans between pattern matches."""
    boundaries = [m.start() for m in pattern.finditer(text)]
    if not boundaries:
        return [(0, len(text))]
    spans = []
    # Prepend content before first match
    if boundaries[0] > 0:
        spans.append((0, boundaries[0]))
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        spans.append((start, end))
    return spans


def _sliding_window(text: str, max_tokens: int = 200, stride: int = 50) -> list[tuple[int, int]]:
    """Word-based sliding window fallback."""
    words        = text.split()
    if not words:
        return [(0, len(text))]
    char_offsets = []
    pos = 0
    for w in words:
        idx = text.index(w, pos)
        char_offsets.append(idx)
        pos = idx + len(w)
    char_offsets.append(len(text))

    spans = []
    i = 0
    while i < len(words):
        j = min(i + max_tokens, len(words))
        s = char_offsets[i]
        e = char_offsets[j] if j < len(char_offsets) else len(text)
        spans.append((s, e))
        if j >= len(words):
            break
        i += max_tokens - stride
    return spans


def _semantic_similarity_split(text: str, threshold: float = 0.75) -> list[tuple[int, int]]:
    """
    Embedding-based boundary detection.
    Falls back to paragraph split if sentence-transformers not available.
    """
    if not ST_AVAILABLE or not _ST_MODEL:
        return _split_by_pattern(text, _BLANK_LINE)

    sentences = _SENTENCE_END.split(text)
    if len(sentences) < 3:
        return [(0, len(text))]

    embeddings = _ST_MODEL.encode(sentences, normalize_embeddings=True)
    sims = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    boundaries = [0]
    pos = 0
    for sent, sim in zip(sentences[:-1], sims):
        pos += len(sent) + 1
        if sim < threshold:
            boundaries.append(pos)
    boundaries.append(len(text))

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _spans_to_segments(text: str, spans: list[tuple[int, int]],
                        chunk_id: str) -> list[SemanticSegment]:
    segs = []
    for i, (s, e) in enumerate(spans):
        body = text[s:e].strip()
        if not body:
            continue
        segs.append(SemanticSegment(
            text=body,
            start_char=s,
            end_char=e,
            segment_id=f"{chunk_id}__seg_{i + 1}",
        ))
    return segs


# ═══════════════════════════════════════════════════════════════════════════
# SemanticChunker
# ═══════════════════════════════════════════════════════════════════════════

class SemanticChunker:
    """
    Call `.chunk(nlp_chunk)` to populate:
      semantic_segments (List[SemanticSegment] as dicts)
      segment_count
    """

    def __init__(self,
                 max_tokens:           int   = 250,
                 stride:               int   = 50,
                 similarity_threshold: float = 0.75,
                 use_similarity:       bool  = False):
        self.max_tokens = max_tokens
        self.stride     = stride
        self.threshold  = similarity_threshold
        self.use_sim    = use_similarity and ST_AVAILABLE
        if use_similarity and not ST_AVAILABLE:
            log.warning(
                "sentence-transformers not installed. "
                "pip install sentence-transformers  — falling back to rule-based splitting."
            )

    def _choose_strategy(self, chunk: NLPChunk) -> str:
        ct = chunk.chunk_type
        if ct in _DOC_TYPES:  return "doc"
        if ct in _CODE_TYPES: return "code"
        if ct in _DATA_TYPES: return "data"
        return "fallback"

    def chunk(self, nlp_chunk: NLPChunk) -> NLPChunk:
        text  = nlp_chunk.cleaned_text or nlp_chunk.content
        cid   = nlp_chunk.chunk_id
        strat = self._choose_strategy(nlp_chunk)

        if strat == "doc":
            if _MD_HEADING.search(text):
                spans = _split_by_pattern(text, _MD_HEADING)
            elif self.use_sim:
                spans = _semantic_similarity_split(text, self.threshold)
            else:
                spans = _split_by_pattern(text, _BLANK_LINE)

        elif strat == "code":
            boundaries = sorted(set(
                [m.start() for m in _FUNC_DEF.finditer(text)] +
                [m.start() for m in _CLASS_DEF.finditer(text)]
            ))
            if len(boundaries) > 1:
                spans = [
                    (boundaries[i], boundaries[i + 1] if i + 1 < len(boundaries) else len(text))
                    for i in range(len(boundaries))
                ]
                if boundaries[0] > 0:
                    spans.insert(0, (0, boundaries[0]))
            else:
                spans = _sliding_window(text, self.max_tokens, self.stride)

        elif strat == "data":
            spans = [(0, len(text))]   # data chunks are already atomic

        else:  # fallback
            if self.use_sim:
                spans = _semantic_similarity_split(text, self.threshold)
            else:
                spans = _sliding_window(text, self.max_tokens, self.stride)

        segs = _spans_to_segments(text, spans, cid)
        nlp_chunk.semantic_segments = [asdict(s) for s in segs]
        nlp_chunk.segment_count     = len(segs)
        return nlp_chunk

    def chunk_all(self, chunks: list[NLPChunk]) -> list[NLPChunk]:
        return [self.chunk(c) for c in chunks]


def make_semantic_chunker(**kwargs) -> SemanticChunker:
    return SemanticChunker(**kwargs)
