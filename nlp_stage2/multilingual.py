"""
=============================================================================
multilingual.py  —  Step 6: Multilingual Knowledge Base Support
=============================================================================
Two sub-tasks:

  A) Language Detection
       Primary   : langdetect   (pip install langdetect)
       Secondary : langid       (pip install langid)
       Fallback  : Unicode script-range heuristic (always available)

  B) Translation → English  (only when detected_lang != "en")
       Primary   : deep-translator / GoogleTranslator (pip install deep-translator)
       Secondary : argostranslate  offline           (pip install argostranslate)
       Fallback  : No-op (chunk kept as-is, is_translated stays False)

Applies ONLY to doc / comment / docstring content.
Code chunks: language detection runs, but translation is skipped by default
because source code is already language-agnostic.

Configuration:
  translate_langs      : set of ISO-639-1 codes to translate (None = all non-en)
  skip_code            : bool — skip translation for code chunks (default True)
  min_text_length      : int  — skip detection for very short texts
  confidence_threshold : float — minimum detection confidence to trust result
=============================================================================
"""

from __future__ import annotations
import logging
from typing import Optional
from models import NLPChunk

log = logging.getLogger("multilingual")

# ── langdetect ────────────────────────────────────────────────────────────

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42    # deterministic results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# ── langid ────────────────────────────────────────────────────────────────

try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False

# ── deep-translator (Google Translate, online, free tier) ─────────────────

try:
    from deep_translator import GoogleTranslator
    DEEP_TRANS_AVAILABLE = True
except ImportError:
    DEEP_TRANS_AVAILABLE = False

# ── argostranslate (offline) ──────────────────────────────────────────────

try:
    import argostranslate.translate as _argo
    ARGO_AVAILABLE = True
except ImportError:
    ARGO_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Heuristic language detector (always available, zero dependencies)
# ═══════════════════════════════════════════════════════════════════════════

_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),   # Devanagari
    "ar": (0x0600, 0x06FF),   # Arabic
    "zh": (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    "ja": (0x3040, 0x30FF),   # Hiragana + Katakana
    "ko": (0xAC00, 0xD7AF),   # Hangul
    "ru": (0x0400, 0x04FF),   # Cyrillic
    "el": (0x0370, 0x03FF),   # Greek
    "he": (0x0590, 0x05FF),   # Hebrew
    "th": (0x0E00, 0x0E7F),   # Thai
}


def _heuristic_detect(text: str) -> tuple[str, float]:
    """Character-range heuristic. Returns (iso_code, confidence)."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
    if counts:
        top  = max(counts, key=counts.__getitem__)
        conf = min(counts[top] / max(len(text), 1) * 5, 0.95)
        return top, round(conf, 3)
    return "en", 0.5   # default: assume English


def detect_language(text: str) -> tuple[str, float]:
    """Detect language using best available library."""
    text = text.strip()
    if not text:
        return "en", 1.0

    if LANGDETECT_AVAILABLE:
        try:
            lang = detect(text)
            return lang, 0.9
        except Exception:
            pass

    if LANGID_AVAILABLE:
        try:
            lang, conf = langid.classify(text)
            return lang, round(abs(float(conf)), 3)
        except Exception:
            pass

    return _heuristic_detect(text)


# ═══════════════════════════════════════════════════════════════════════════
# Translation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _translate_google(text: str, source_lang: str) -> Optional[str]:
    """Translate via Google Translate (deep-translator). 5 000-char limit per call."""
    try:
        translator = GoogleTranslator(source=source_lang, target="en")
        if len(text) > 4800:
            parts = [text[i: i + 4800] for i in range(0, len(text), 4800)]
            return " ".join(translator.translate(p) for p in parts)
        return translator.translate(text)
    except Exception as e:
        log.warning(f"[Multilingual] Google translate error: {e}")
        return None


def _translate_argo(text: str, source_lang: str) -> Optional[str]:
    """Translate via argostranslate (offline)."""
    try:
        installed = _argo.get_installed_languages()
        src = next((l for l in installed if l.code == source_lang), None)
        tgt = next((l for l in installed if l.code == "en"), None)
        if src and tgt:
            return src.get_translation(tgt).translate(text)
    except Exception as e:
        log.warning(f"[Multilingual] Argostranslate error: {e}")
    return None


def translate_to_english(text: str, source_lang: str) -> Optional[str]:
    """Try Google Translate → Argostranslate → return None on failure."""
    if DEEP_TRANS_AVAILABLE:
        result = _translate_google(text, source_lang)
        if result:
            return result
    if ARGO_AVAILABLE:
        result = _translate_argo(text, source_lang)
        if result:
            return result
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Chunk type classification for translation gating
# ═══════════════════════════════════════════════════════════════════════════

_DOC_CHUNK_TYPES = {
    "doc_section", "doc_paragraph", "html_text",
    "notebook_markdown_cell", "shell_comment",
    "markdown", "rst", "text",
}


def _text_for_detection(chunk: NLPChunk) -> str:
    """Prefer human-readable content for language detection."""
    doc = chunk.docstring or ""
    txt = chunk.cleaned_text or chunk.content
    return (doc + " " + txt).strip()[:2000]


# ═══════════════════════════════════════════════════════════════════════════
# MultilingualProcessor
# ═══════════════════════════════════════════════════════════════════════════

class MultilingualProcessor:
    """
    Call `.process(chunk)` to populate:
      detected_lang, detected_lang_conf, translated_text, is_translated
    """

    def __init__(self,
                 translate_langs:      Optional[set] = None,
                 skip_code:            bool          = True,
                 min_text_length:      int           = 20,
                 confidence_threshold: float         = 0.6):
        """
        translate_langs      : ISO-639-1 codes to translate (None = all non-en)
        skip_code            : skip translation for pure code chunks
        min_text_length      : skip detection for texts shorter than this
        confidence_threshold : minimum detection confidence to act on
        """
        self.translate_langs = translate_langs
        self.skip_code       = skip_code
        self.min_len         = min_text_length
        self.conf_threshold  = confidence_threshold

    def process(self, chunk: NLPChunk) -> NLPChunk:
        text = _text_for_detection(chunk)

        if len(text) < self.min_len:
            chunk.detected_lang      = "en"
            chunk.detected_lang_conf = 1.0
            return chunk

        lang, conf = detect_language(text)
        chunk.detected_lang      = lang
        chunk.detected_lang_conf = conf

        # Gate: should we translate?
        if lang == "en":                                                   return chunk
        if conf < self.conf_threshold:                                     return chunk
        if self.skip_code and chunk.chunk_type not in _DOC_CHUNK_TYPES:   return chunk
        if self.translate_langs is not None and lang not in self.translate_langs:
            return chunk

        # Translate
        src_text   = chunk.cleaned_text or chunk.content
        translated = translate_to_english(src_text[:5000], lang)
        if translated and translated.strip() != src_text.strip():
            chunk.translated_text = translated
            chunk.is_translated   = True
        else:
            log.debug(f"[Multilingual] No translation available for {chunk.chunk_id} ({lang})")

        return chunk

    def process_all(self, chunks: list[NLPChunk]) -> list[NLPChunk]:
        return [self.process(c) for c in chunks]


def make_multilingual(**kwargs) -> MultilingualProcessor:
    return MultilingualProcessor(**kwargs)
