"""
=============================================================================
cleaner.py  —  Step 1: Text Cleaning & Normalisation
=============================================================================
Responsibilities
  • Strip / collapse whitespace, dedent code blocks
  • Remove ANSI escape codes, null bytes, non-printable chars
  • Normalise quotes, dashes, unicode ligatures
  • Language-aware comment stripping (Python, JS, SQL, Shell, CSS…)
  • Tokenise → fill NLPChunk.tokens, .token_count, .normalized_lang
  • Canonical language name mapping
=============================================================================
"""

from __future__ import annotations
import re
import textwrap
import unicodedata
from typing import Optional
from models import NLPChunk

# ── Language name aliases → canonical form ────────────────────────────────

_LANG_ALIASES: dict[str, str] = {
    "py": "python", "pyw": "python",
    "js": "javascript", "mjs": "javascript", "jsx": "javascript",
    "ts": "typescript", "tsx": "typescript",
    "sh": "shell", "bash": "shell", "zsh": "shell", "fish": "shell",
    "md": "markdown", "mdx": "markdown", "rmd": "markdown",
    "yml": "yaml",
    "rs": "rust",
    "rb": "ruby",
    "kt": "kotlin", "kts": "kotlin",
    "ex": "elixir", "exs": "elixir",
    "tf": "terraform", "hcl": "terraform",
    "ipynb": "jupyter",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
}

# ── Comment patterns per language ────────────────────────────────────────

_COMMENT_PATTERNS: dict[str, list[str]] = {
    "python":     [r'"""[\s\S]*?"""', r"'''[\s\S]*?'''", r'#[^\n]*'],
    "javascript": [r'/\*[\s\S]*?\*/', r'//[^\n]*'],
    "typescript": [r'/\*[\s\S]*?\*/', r'//[^\n]*'],
    "java":       [r'/\*[\s\S]*?\*/', r'//[^\n]*'],
    "c":          [r'/\*[\s\S]*?\*/', r'//[^\n]*'],
    "cpp":        [r'/\*[\s\S]*?\*/', r'//[^\n]*'],
    "go":         [r'/\*[\s\S]*?\*/', r'//[^\n]*'],
    "rust":       [r'/\*[\s\S]*?\*/', r'//[^\n]*'],
    "css":        [r'/\*[\s\S]*?\*/'],
    "scss":       [r'/\*[\s\S]*?\*/', r'//[^\n]*'],
    "shell":      [r'#[^\n]*'],
    "sql":        [r'--[^\n]*', r'/\*[\s\S]*?\*/'],
    "ruby":       [r'#[^\n]*', r'=begin[\s\S]*?=end'],
    "php":        [r'/\*[\s\S]*?\*/', r'//[^\n]*', r'#[^\n]*'],
    "kotlin":     [r'/\*[\s\S]*?\*/', r'//[^\n]*'],
    "swift":      [r'/\*[\s\S]*?\*/', r'//[^\n]*'],
    "r":          [r'#[^\n]*'],
}

# ── Compiled patterns ─────────────────────────────────────────────────────

_ANSI_RE    = re.compile(r'\x1b\[[0-9;]*[mGKHF]')
_NULL_RE    = re.compile(r'\x00+')
_CTRL_RE    = re.compile(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SPACE_RE   = re.compile(r'[^\S\n]+')        # horizontal whitespace collapse
_BLANK_RE   = re.compile(r'\n{3,}')          # 3+ blank lines → 2
_WORD_TOKEN = re.compile(r'[A-Za-z_]\w*|[0-9]+(?:\.[0-9]+)?|[^\w\s]')


def _normalize_lang(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    l = lang.lower().lstrip(".")
    return _LANG_ALIASES.get(l, l)


def _unicode_normalize(text: str) -> str:
    """NFC → collapse fancy unicode to nearest ASCII where safe."""
    text = unicodedata.normalize("NFC", text)
    # smart quotes → straight
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # en/em dash → hyphen
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # ellipsis → ...
    text = text.replace("\u2026", "...")
    return text


def _strip_comments(text: str, lang: Optional[str]) -> str:
    """Remove single-line and block comments for the given language."""
    if lang not in _COMMENT_PATTERNS:
        return text
    for pat in _COMMENT_PATTERNS[lang]:
        text = re.sub(pat, " ", text, flags=re.DOTALL)
    return text


def _tokenize(text: str) -> list[str]:
    return [m.group() for m in _WORD_TOKEN.finditer(text) if m.group().strip()]


def _is_doc_chunk(chunk_type: str) -> bool:
    return chunk_type in {
        "doc_section", "doc_paragraph", "markdown", "rst",
        "html_text", "notebook_markdown_cell",
    }


class TextCleaner:
    """
    Stateless cleaner. Call `.clean(chunk)` to populate:
      cleaned_text, tokens, token_count, normalized_lang
    """

    def __init__(self, strip_comments: bool = False):
        """
        strip_comments : if True, remove inline code comments.
                         Disabled by default to preserve docstrings / context.
        """
        self.strip_comments = strip_comments

    def clean(self, chunk: NLPChunk) -> NLPChunk:
        lang   = _normalize_lang(chunk.language)
        source = chunk.content

        # 1. ANSI / control chars
        source = _ANSI_RE.sub("", source)
        source = _NULL_RE.sub("", source)
        source = _CTRL_RE.sub("", source)

        # 2. Unicode normalization
        source = _unicode_normalize(source)

        # 3. Dedent code blocks
        if not _is_doc_chunk(chunk.chunk_type):
            source = textwrap.dedent(source)

        # 4. Strip comments (optional, never strip from doc chunks)
        if self.strip_comments and not _is_doc_chunk(chunk.chunk_type):
            source = _strip_comments(source, lang)

        # 5. Whitespace normalization
        source = _SPACE_RE.sub(" ", source)     # collapse horizontal spaces
        source = _BLANK_RE.sub("\n\n", source)  # max 2 consecutive blank lines
        source = source.strip()

        # 6. Tokenize
        tokens = _tokenize(source)

        chunk.cleaned_text    = source
        chunk.tokens          = tokens
        chunk.token_count     = len(tokens)
        chunk.normalized_lang = lang
        return chunk

    def clean_all(self, chunks: list[NLPChunk]) -> list[NLPChunk]:
        return [self.clean(c) for c in chunks]


def make_cleaner(strip_comments: bool = False) -> TextCleaner:
    return TextCleaner(strip_comments=strip_comments)
