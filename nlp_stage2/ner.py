"""
=============================================================================
ner.py  —  Step 2: Named Entity Recognition (NER)
=============================================================================
Two-tier approach:
  Tier 1 (always on) — Rule-based regex NER
      FUNC, CLASS, LIB, URL, EMAIL, TODO, FILE_PATH, DECORATOR,
      SQL_TABLE, CSS_SELECTOR, ENV_VAR, VERSION, GQL_TYPE

  Tier 2 (optional)  — spaCy model NER (en_core_web_sm or larger)
      ORG, PERSON, GPE, PRODUCT, TECH  (general entities from spaCy)

Both tiers are merged, de-duplicated, and stored as List[Entity] dicts.

Install spaCy tier:
  pip install spacy
  python -m spacy download en_core_web_sm
=============================================================================
"""

from __future__ import annotations
import re
import logging
from dataclasses import asdict
from models import NLPChunk, Entity

log = logging.getLogger("ner")

# ── spaCy (optional) ──────────────────────────────────────────────────────

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    _NLP = None

# ── Rule-based patterns (label, compiled regex) ───────────────────────────

_RULES: list[tuple[str, re.Pattern]] = [
    # Python / JS / TS function definitions
    ("FUNC",         re.compile(
        r'\bdef\s+([A-Za-z_]\w*)'
        r'|([A-Za-z_]\w*)\s*(?:=\s*)?(?:async\s+)?function\b'
        r'|\b([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\('
    )),
    # class definitions (multi-language)
    ("CLASS",        re.compile(r'\bclass\s+([A-Za-z_]\w*)')),
    # import / require / use statements → library names
    ("LIB",          re.compile(
        r'(?:import|from|require|use)\s+["\']?([A-Za-z_][\w./\-]*)["\']?'
    )),
    # URLs
    ("URL",          re.compile(r'https?://[^\s\'"<>]+')),
    # Email addresses
    ("EMAIL",        re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')),
    # TODO / FIXME / HACK / NOTE annotations
    ("TODO",         re.compile(r'\b(TODO|FIXME|HACK|NOQA|NOTE|XXX)\b[:\s].*')),
    # File paths (Unix/Windows)
    ("FILE_PATH",    re.compile(r'(?:/[\w.\-]+){2,}|[A-Za-z]:\\[\w\\.\-]+')),
    # Decorators (@decorator)
    ("DECORATOR",    re.compile(r'@([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)')),
    # SQL table names after FROM / JOIN / INTO / UPDATE
    ("SQL_TABLE",    re.compile(
        r'(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([A-Za-z_]\w*)', re.IGNORECASE
    )),
    # Environment variable references
    ("ENV_VAR",      re.compile(
        r'\$\{?([A-Z_][A-Z0-9_]{2,})\}?'
        r'|os\.environ\[["\']([A-Z_][A-Z0-9_]+)["\']\]'
    )),
    # Semantic versioning
    ("VERSION",      re.compile(r'\bv?(\d+\.\d+(?:\.\d+)?(?:[-+]\w+)?)\b')),
    # CSS selectors (simple)
    ("CSS_SELECTOR", re.compile(
        r'(?:^|[\n;])\s*([.#]?[A-Za-z][\w\-]*(?:\s*[>+~]\s*[.#]?[A-Za-z][\w\-]*)*)\s*\{'
    )),
    # GraphQL type definitions
    ("GQL_TYPE",     re.compile(
        r'\b(?:type|interface|enum|input)\s+([A-Z][A-Za-z_]*)\b'
    )),
]

# spaCy labels we forward from the general model
_SPACY_FORWARD = {"ORG", "PERSON", "GPE", "PRODUCT", "NORP", "WORK_OF_ART", "LAW"}


# ── Tier 1: Rule-based ────────────────────────────────────────────────────

def _rule_ner(text: str) -> list[Entity]:
    entities: list[Entity] = []
    for label, pat in _RULES:
        for m in pat.finditer(text):
            matched = next((g for g in m.groups() if g), m.group(0))
            matched = matched.strip()
            if not matched or len(matched) < 2:
                continue
            entities.append(Entity(
                text=matched, label=label,
                start=m.start(), end=m.end(),
                score=1.0,
            ))
    return entities


# ── Tier 2: spaCy ────────────────────────────────────────────────────────

def _spacy_ner(text: str) -> list[Entity]:
    if not SPACY_AVAILABLE or not _NLP:
        return []
    doc = _NLP(text[:100_000])   # spaCy char limit guard
    return [
        Entity(
            text=ent.text, label=ent.label_,
            start=ent.start_char, end=ent.end_char,
            score=0.9,
        )
        for ent in doc.ents
        if ent.label_ in _SPACY_FORWARD
    ]


# ── Deduplication ────────────────────────────────────────────────────────

def _dedup(entities: list[Entity]) -> list[Entity]:
    """Remove duplicates (same text + label), keep highest score."""
    seen: dict[tuple[str, str], Entity] = {}
    for e in entities:
        key = (e.text.lower(), e.label)
        if key not in seen or e.score > seen[key].score:
            seen[key] = e
    return list(seen.values())


# ── NERTagger ─────────────────────────────────────────────────────────────

class NERTagger:
    """
    Call `.tag(chunk)` to populate chunk.entities (List[Entity] as dicts).
    """

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        if use_spacy and not SPACY_AVAILABLE:
            log.warning(
                "spaCy not available. "
                "Run: pip install spacy && python -m spacy download en_core_web_sm. "
                "Falling back to rule-based NER only."
            )

    def tag(self, chunk: NLPChunk) -> NLPChunk:
        text = chunk.cleaned_text or chunk.content
        entities = _rule_ner(text)
        if self.use_spacy:
            entities += _spacy_ner(text)
        entities = _dedup(entities)
        chunk.entities = [asdict(e) for e in entities]
        return chunk

    def tag_all(self, chunks: list[NLPChunk]) -> list[NLPChunk]:
        return [self.tag(c) for c in chunks]


def make_ner(use_spacy: bool = True) -> NERTagger:
    return NERTagger(use_spacy=use_spacy)
