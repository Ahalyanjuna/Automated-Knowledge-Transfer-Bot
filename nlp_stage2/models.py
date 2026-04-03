"""
=============================================================================
models.py  —  Shared data model for Stage 2: NLP Preprocessing Pipeline
=============================================================================
NLPChunk extends the raw Chunk from Stage 1 with every field produced
across all six NLP steps. All modules read/write this single object.
=============================================================================
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Chunk:
    """Mirror of Stage-1 Chunk so models.py is self-contained."""
    chunk_id:    str
    content:     str
    chunk_type:  str
    file_type:   str
    language:    Optional[str]
    source_file: str
    start_line:  Optional[int] = None
    end_line:    Optional[int] = None
    name:        Optional[str] = None
    docstring:   Optional[str] = None
    parameters:  list          = field(default_factory=list)
    return_type: Optional[str] = None
    parent:      Optional[str] = None
    imports:     list          = field(default_factory=list)
    extra:       dict          = field(default_factory=dict)


@dataclass
class Entity:
    text:  str
    label: str           # FUNC | CLASS | LIB | TECH | URL | VAR | ORG
    start: int           # char offset in cleaned_text
    end:   int
    score: float = 1.0   # 1.0 = rule-based, 0-1 = model confidence


@dataclass
class SemanticSegment:
    text:       str
    start_char: int
    end_char:   int
    segment_id: str      # "<chunk_id>__seg_N"


@dataclass
class NLPChunk:
    # ── Stage-1 origin ────────────────────────────────────────────────────
    chunk_id:    str
    content:     str
    chunk_type:  str
    file_type:   str
    language:    Optional[str]
    source_file: str
    start_line:  Optional[int]
    end_line:    Optional[int]
    name:        Optional[str]
    docstring:   Optional[str]
    parameters:  list
    return_type: Optional[str]
    parent:      Optional[str]
    imports:     list
    extra:       dict

    # ── Step 1: Cleaning ──────────────────────────────────────────────────
    cleaned_text:    str           = ""
    tokens:          list          = field(default_factory=list)
    token_count:     int           = 0
    normalized_lang: Optional[str] = None

    # ── Step 2: NER ───────────────────────────────────────────────────────
    entities:        list          = field(default_factory=list)  # List[Entity] as dicts

    # ── Step 3: Semantic Chunking ─────────────────────────────────────────
    semantic_segments: list        = field(default_factory=list)  # List[SemanticSegment] as dicts
    segment_count:     int         = 0

    # ── Step 4: Metadata Tagging ──────────────────────────────────────────
    tags:             list         = field(default_factory=list)
    complexity_score: float        = 0.0
    has_docstring:    bool         = False
    has_tests:        bool         = False
    has_todos:        bool         = False
    api_surface:      list         = field(default_factory=list)  # public names

    # ── Step 5: Embedding ─────────────────────────────────────────────────
    embedding:        list         = field(default_factory=list)  # List[float]
    embedding_model:  Optional[str]= None
    embedding_dim:    int          = 0

    # ── Step 6: Multilingual ──────────────────────────────────────────────
    detected_lang:      Optional[str] = None   # ISO 639-1 e.g. "en", "fr"
    detected_lang_conf: float         = 0.0
    translated_text:    Optional[str] = None
    is_translated:      bool          = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_chunk(cls, c: Chunk) -> "NLPChunk":
        return cls(
            chunk_id=c.chunk_id, content=c.content,
            chunk_type=c.chunk_type, file_type=c.file_type,
            language=c.language, source_file=c.source_file,
            start_line=c.start_line, end_line=c.end_line,
            name=c.name, docstring=c.docstring,
            parameters=c.parameters, return_type=c.return_type,
            parent=c.parent, imports=c.imports, extra=c.extra,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "NLPChunk":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
