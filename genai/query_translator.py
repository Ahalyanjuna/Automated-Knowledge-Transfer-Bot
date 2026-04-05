"""
=============================================================================
query_translator.py  —  Runtime Query & Response Translation
=============================================================================
Used by chat_engine.py at query time (not the NLP ingestion pipeline).

Functions:
  detect_query_language(text)       → (iso_code, lang_name, confidence)
  translate_to_english(text, lang)  → str | None
  translate_from_english(text, target_lang) → str | None

Backends (same priority as multilingual.py):
  1. deep-translator / GoogleTranslator  (pip install deep-translator)
  2. Fallback: returns None (caller handles gracefully)
=============================================================================
"""

from __future__ import annotations
import logging
from typing import Optional

log = logging.getLogger("query_translator")

# ── langdetect ────────────────────────────────────────────────────────────
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# ── deep-translator ────────────────────────────────────────────────────────
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANS_AVAILABLE = True
except ImportError:
    DEEP_TRANS_AVAILABLE = False

# ── langid fallback ───────────────────────────────────────────────────────
try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False


# ── ISO 639-1 code → human-readable name map ─────────────────────────────
LANG_NAMES: dict[str, str] = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "fi": "Finnish",
    "el": "Greek",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
}

# ── Script-range heuristic (zero-dependency fallback) ─────────────────────
_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),
    "ar": (0x0600, 0x06FF),
    "zh": (0x4E00, 0x9FFF),
    "ja": (0x3040, 0x30FF),
    "ko": (0xAC00, 0xD7AF),
    "ru": (0x0400, 0x04FF),
    "el": (0x0370, 0x03FF),
    "he": (0x0590, 0x05FF),
    "th": (0x0E00, 0x0E7F),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
    "bn": (0x0980, 0x09FF),
    "gu": (0x0A80, 0x0AFF),
    "pa": (0x0A00, 0x0A7F),
}


def _heuristic_detect(text: str) -> tuple[str, float]:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
    if counts:
        top = max(counts, key=counts.__getitem__)
        conf = min(counts[top] / max(len(text), 1) * 5, 0.95)
        return top, round(conf, 3)
    return "en", 0.5


def detect_query_language(text: str) -> tuple[str, str, float]:
    """
    Detect the language of a query string.
    Returns: (iso_code, human_readable_name, confidence)
    Example: ("ta", "Tamil", 0.9)
    """
    text = text.strip()
    if not text:
        return "en", "English", 1.0

    iso_code = "en"
    confidence = 0.5

    if LANGDETECT_AVAILABLE:
        try:
            iso_code = detect(text)
            confidence = 0.9
        except Exception:
            pass
    elif LANGID_AVAILABLE:
        try:
            iso_code, conf = langid.classify(text)
            confidence = round(abs(float(conf)), 3)
        except Exception:
            iso_code, confidence = _heuristic_detect(text)
    else:
        iso_code, confidence = _heuristic_detect(text)

    lang_name = LANG_NAMES.get(iso_code, iso_code.upper())
    return iso_code, lang_name, confidence


def translate_to_english(text: str, source_lang: str) -> Optional[str]:
    """
    Translate text → English.
    Returns translated string or None if translation failed/unavailable.
    """
    if source_lang == "en":
        return text

    if not DEEP_TRANS_AVAILABLE:
        log.warning("[QueryTranslator] deep-translator not installed. pip install deep-translator")
        return None

    try:
        translator = GoogleTranslator(source=source_lang, target="en")
        if len(text) > 4800:
            parts = [text[i: i + 4800] for i in range(0, len(text), 4800)]
            return " ".join(translator.translate(p) for p in parts)
        return translator.translate(text)
    except Exception as e:
        log.warning(f"[QueryTranslator] translate_to_english failed: {e}")
        return None


def translate_from_english(text: str, target_lang: str) -> Optional[str]:
    """
    Translate English text → target language.
    Returns translated string or None if translation failed/unavailable.
    """
    if target_lang == "en":
        return text

    if not DEEP_TRANS_AVAILABLE:
        log.warning("[QueryTranslator] deep-translator not installed.")
        return None

    try:
        translator = GoogleTranslator(source="en", target=target_lang)
        # Split into chunks if response is long
        if len(text) > 4800:
            parts = [text[i: i + 4800] for i in range(0, len(text), 4800)]
            return " ".join(translator.translate(p) for p in parts)
        return translator.translate(text)
    except Exception as e:
        log.warning(f"[QueryTranslator] translate_from_english failed ({target_lang}): {e}")
        return None