"""
pip install pyyaml chardet beautifulsoup4 pymupdf tree-sitter tree-sitter-javascript tree-sitter-java tree-sitter-c tree-sitter-cpp tree-sitter-go tree-sitter-rust
=============================================================================
STAGE 1 — INPUT INGESTION & PARSING  (Universal Edition)
KT Bot: Automated Knowledge Transfer System
=============================================================================
Handles EVERY file type found in a GitHub repo.

Dedicated parsers
─────────────────
  .py              → Python AST  (functions, classes, imports, docstrings)
  .js / .ts / .jsx / .tsx → Tree-sitter JS/TS  (functions, classes, arrow fns)
  .java            → Tree-sitter Java  (methods, classes)
  .c / .cpp / .h   → Tree-sitter C/C++  (functions, structs)
  .go              → Tree-sitter Go  (functions, types)
  .rs              → Tree-sitter Rust  (functions, impls)
  .html / .htm     → BeautifulSoup  (tags, scripts, styles, text)
  .css / .scss / .sass / .less → CSS rule-block parser  (selectors, properties)
  .json            → JSON structure parser  (keys, nested objects)
  .yaml / .yml     → YAML structure parser  (top-level keys / sections)
  .toml            → TOML section parser
  .xml / .svg      → XML element parser
  .sh / .bash / .zsh → Shell function + comment parser
  .sql             → SQL statement parser  (CREATE, SELECT, INSERT …)
  .md / .mdx       → Heading-based section splitter
  .rst             → reStructuredText underline-section splitter
  .txt             → Paragraph splitter
  .pdf             → PyMuPDF page-block extractor
  .csv / .tsv      → Row-group chunker with header awareness
  .ipynb           → Jupyter notebook cell extractor
  .env / .ini / .cfg / .conf → Config key-value parser
  Dockerfile       → Instruction-block parser
  Makefile         → Target-block parser
  .tf / .hcl / .rb / .php / .swift / .kt / .r / .graphql
                   → Generic 60-line block chunker (graceful fallback)
  Any other text file → Auto-encoding detect + line-block chunker
  Binary / images / lock files → Skipped automatically

Output
──────
  List[dict]  — Chunk objects as JSON/JSONL, ready for Stage 2 (NLP Pipeline)

Usage
─────
  python ingestion.py --repo_url https://github.com/ArulKevin2004/VisionCortex
  python ingestion.py --local_path /path/to/repo --output chunks.jsonl
  python ingestion.py --local_path /path/to/repo --max_file_size_kb 1000
=============================================================================
"""

import os
import re
import ast
import csv
import json
import argparse
import tempfile
import subprocess
import logging
from io import StringIO
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field

# ── Optional deps (graceful degradation) ─────────────────────────────────

try:
    import yaml          # pip install pyyaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import chardet       # pip install chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

try:
    import fitz          # pip install pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from bs4 import BeautifulSoup   # pip install beautifulsoup4
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import tomllib       # stdlib Python 3.11+
    TOML_AVAILABLE = True
except ImportError:
    try:
        import tomli as tomllib   # pip install tomli  (Python <3.11)
        TOML_AVAILABLE = True
    except ImportError:
        TOML_AVAILABLE = False

try:
    from tree_sitter import Language, Parser as TSParser
    import tree_sitter_javascript as ts_js
    import tree_sitter_java       as ts_java
    import tree_sitter_c          as ts_c
    import tree_sitter_cpp        as ts_cpp
    import tree_sitter_go         as ts_go
    import tree_sitter_rust       as ts_rust
    TREESITTER_AVAILABLE = True
    _TS_LANGS = {
        "javascript": Language(ts_js.language()),
        "typescript": Language(ts_js.language()),
        "java":       Language(ts_java.language()),
        "c":          Language(ts_c.language()),
        "cpp":        Language(ts_cpp.language()),
        "go":         Language(ts_go.language()),
        "rust":       Language(ts_rust.language()),
    }
except Exception:
    TREESITTER_AVAILABLE = False
    _TS_LANGS = {}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingestion")


# ══════════════════════════════════════════════════════════════════════════
# 1. DATA MODEL
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    chunk_id:    str
    content:     str
    chunk_type:  str            # function|class|method|css_rule|html_section|…
    file_type:   str            # code|doc|config|data|notebook|script|infra
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
    extra:       dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════
# 2. FILE TYPE REGISTRY
# ══════════════════════════════════════════════════════════════════════════

# extension  →  (file_type, language, parser_key)
FILE_REGISTRY: dict[str, tuple[str, str, str]] = {
    # Python
    ".py":        ("code",     "python",      "python"),
    ".pyw":       ("code",     "python",      "python"),
    # JavaScript / TypeScript
    ".js":        ("code",     "javascript",  "treesitter"),
    ".jsx":       ("code",     "javascript",  "treesitter"),
    ".mjs":       ("code",     "javascript",  "treesitter"),
    ".ts":        ("code",     "typescript",  "treesitter"),
    ".tsx":       ("code",     "typescript",  "treesitter"),
    # Java
    ".java":      ("code",     "java",        "treesitter"),
    # C / C++
    ".c":         ("code",     "c",           "treesitter"),
    ".h":         ("code",     "c",           "treesitter"),
    ".cpp":       ("code",     "cpp",         "treesitter"),
    ".cc":        ("code",     "cpp",         "treesitter"),
    ".hpp":       ("code",     "cpp",         "treesitter"),
    # Go
    ".go":        ("code",     "go",          "treesitter"),
    # Rust
    ".rs":        ("code",     "rust",        "treesitter"),
    # Web
    ".html":      ("code",     "html",        "html"),
    ".htm":       ("code",     "html",        "html"),
    ".css":       ("code",     "css",         "css"),
    ".scss":      ("code",     "scss",        "css"),
    ".sass":      ("code",     "scss",        "css"),
    ".less":      ("code",     "less",        "css"),
    # Shell
    ".sh":        ("script",   "shell",       "shell"),
    ".bash":      ("script",   "shell",       "shell"),
    ".zsh":       ("script",   "shell",       "shell"),
    ".fish":      ("script",   "shell",       "shell"),
    # SQL
    ".sql":       ("code",     "sql",         "sql"),
    # Data / Config
    ".json":      ("data",     "json",        "json"),
    ".yaml":      ("config",   "yaml",        "yaml"),
    ".yml":       ("config",   "yaml",        "yaml"),
    ".toml":      ("config",   "toml",        "toml"),
    ".ini":       ("config",   "ini",         "ini"),
    ".cfg":       ("config",   "ini",         "ini"),
    ".conf":      ("config",   "ini",         "ini"),
    ".env":       ("config",   "env",         "ini"),
    ".xml":       ("data",     "xml",         "xml"),
    ".svg":       ("data",     "xml",         "xml"),
    ".csv":       ("data",     "csv",         "csv"),
    ".tsv":       ("data",     "csv",         "csv"),
    # Docs
    ".md":        ("doc",      "markdown",    "markdown"),
    ".mdx":       ("doc",      "markdown",    "markdown"),
    ".rst":       ("doc",      "rst",         "rst"),
    ".txt":       ("doc",      "text",        "text"),
    ".pdf":       ("doc",      "pdf",         "pdf"),
    # Notebooks
    ".ipynb":     ("notebook", "jupyter",     "notebook"),
    # Infrastructure
    ".dockerfile":("infra",    "dockerfile",  "dockerfile"),
    ".tf":        ("infra",    "terraform",   "generic_code"),
    ".hcl":       ("infra",    "hcl",         "generic_code"),
    # Other languages (generic block chunker)
    ".rb":        ("code",     "ruby",        "generic_code"),
    ".php":       ("code",     "php",         "generic_code"),
    ".swift":     ("code",     "swift",       "generic_code"),
    ".kt":        ("code",     "kotlin",      "generic_code"),
    ".kts":       ("code",     "kotlin",      "generic_code"),
    ".r":         ("code",     "r",           "generic_code"),
    ".rmd":       ("doc",      "markdown",    "markdown"),
    ".graphql":   ("code",     "graphql",     "generic_code"),
    ".gql":       ("code",     "graphql",     "generic_code"),
    ".proto":     ("code",     "protobuf",    "generic_code"),
    ".dart":      ("code",     "dart",        "generic_code"),
    ".lua":       ("code",     "lua",         "generic_code"),
    ".scala":     ("code",     "scala",       "generic_code"),
    ".ex":        ("code",     "elixir",      "generic_code"),
    ".exs":       ("code",     "elixir",      "generic_code"),
    ".cs":        ("code",     "csharp",      "generic_code"),
    ".fs":        ("code",     "fsharp",      "generic_code"),
    ".vue":       ("code",     "vue",         "html"),    # treat Vue SFC as HTML
    ".svelte":    ("code",     "svelte",      "html"),
}

# Exact filenames (case-insensitive) → registry entry
FILENAME_REGISTRY: dict[str, tuple[str, str, str]] = {
    "dockerfile":      ("infra",   "dockerfile", "dockerfile"),
    "makefile":        ("infra",   "makefile",   "makefile"),
    "gnumakefile":     ("infra",   "makefile",   "makefile"),
    ".env":            ("config",  "env",         "ini"),
    ".env.example":    ("config",  "env",         "ini"),
    ".env.local":      ("config",  "env",         "ini"),
    ".gitignore":      ("config",  "text",        "text"),
    ".dockerignore":   ("config",  "text",        "text"),
    ".editorconfig":   ("config",  "ini",         "ini"),
    "requirements.txt":("config",  "text",        "text"),
    "pipfile":         ("config",  "toml",        "toml"),
    "gemfile":         ("config",  "text",        "text"),
    "procfile":        ("config",  "text",        "text"),
    "rakefile":        ("code",    "ruby",        "generic_code"),
    "justfile":        ("infra",   "makefile",    "makefile"),
    "cmakelists.txt":  ("infra",   "text",        "text"),
}

# Extensions to always skip
SKIP_EXTENSIONS = {
    # Images
    ".png",".jpg",".jpeg",".gif",".bmp",".ico",".webp",".tiff",".psd",
    # Audio / Video
    ".mp4",".mp3",".mov",".avi",".wav",".ogg",".flac",
    # Archives
    ".zip",".tar",".gz",".rar",".7z",".bz2",".xz",
    # Compiled / Binary
    ".exe",".dll",".so",".dylib",".bin",".class",".pyc",".pyo",".wasm",
    # Source maps & minified
    ".map",
    # Lock files (by extension)
    ".lock",".sum",
    # Fonts
    ".ttf",".woff",".woff2",".eot",".otf",
    # Misc
    ".db",".sqlite",".pickle",".pkl",".npy",".npz",
}

SKIP_DIRS = {
    ".git","__pycache__","node_modules",".venv","venv","env",
    "dist","build",".next",".nuxt",".idea",".vscode",
    "migrations","coverage",".pytest_cache",".mypy_cache",
    "vendor","third_party","extern","deps","bower_components",
    ".tox","eggs",".eggs","htmlcov",
}

SKIP_FILENAME_PATTERNS = [
    re.compile(r".*\.min\.(js|css)$",      re.I),
    re.compile(r".*-lock\.(json|yaml)$",   re.I),
    re.compile(r"^package-lock\.json$",    re.I),
    re.compile(r"^yarn\.lock$",            re.I),
    re.compile(r"^poetry\.lock$",          re.I),
    re.compile(r"^Pipfile\.lock$",         re.I),
    re.compile(r"^composer\.lock$",        re.I),
    re.compile(r"^Cargo\.lock$",           re.I),
    re.compile(r".*\.bundle\.js$",         re.I),
    re.compile(r".*\.chunk\.js$",          re.I),
]


# ══════════════════════════════════════════════════════════════════════════
# 3. HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _safe_read(path: Path) -> Optional[str]:
    """Read any file with auto-encoding detection. Returns None for binary."""
    raw = path.read_bytes()
    # Binary sniff: null bytes in first 8KB = binary
    if b"\x00" in raw[:8000]:
        return None
    if CHARDET_AVAILABLE:
        detected = chardet.detect(raw)
        enc = detected.get("encoding") or "utf-8"
    else:
        enc = "utf-8"
    try:
        return raw.decode(enc, errors="replace")
    except Exception:
        return raw.decode("utf-8", errors="replace")


def _chunk_id(rel_path: str, kind: str, name: str, counter: int) -> str:
    stem = re.sub(r"[^\w]", "_", Path(rel_path).stem)[:25]
    safe = re.sub(r"[^\w]", "_", str(name))[:25]
    return f"{stem}__{kind}__{safe}__{counter}"


def _line_block_chunks(source: str, rel_path: str, language: str,
                        file_type: str, block_size: int = 60) -> list[Chunk]:
    """Universal fallback: split source into N-line blocks."""
    lines  = source.splitlines()
    chunks = []
    for i, start in enumerate(range(0, len(lines), block_size)):
        block = "\n".join(lines[start: start + block_size])
        if not block.strip():
            continue
        chunks.append(Chunk(
            chunk_id=f"{re.sub(r'[^\\w]','_',Path(rel_path).stem)}__block__{i+1}",
            content=block[:4000],
            chunk_type="code_block", file_type=file_type, language=language,
            source_file=rel_path,
            start_line=start + 1, end_line=min(start + block_size, len(lines)),
            name=None, docstring=None,
            parameters=[], return_type=None, parent=None, imports=[],
        ))
    return chunks


# ══════════════════════════════════════════════════════════════════════════
# 4. DEDICATED PARSERS
# ══════════════════════════════════════════════════════════════════════════

# ── 4a. Python ────────────────────────────────────────────────────────────

class PythonParser:
    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return _line_block_chunks(source, rel_path, "python", "code")

        imports = self._imports(tree)
        counter = [0]

        def mk(kind, name):
            counter[0] += 1
            return _chunk_id(rel_path, kind, name, counter[0])

        # Annotate parent pointers
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child._parent = node  # type: ignore[attr-defined]

        chunks = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                src = ast.get_source_segment(source, node) or ""
                chunks.append(Chunk(
                    chunk_id=mk("class", node.name), content=src,
                    chunk_type="class", file_type="code", language="python",
                    source_file=rel_path,
                    start_line=node.lineno, end_line=node.end_lineno,
                    name=node.name, docstring=ast.get_docstring(node),
                    parameters=[], return_type=None, parent=None, imports=imports,
                ))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent_node = getattr(node, "_parent", None)
                parent_name = parent_node.name if isinstance(parent_node, ast.ClassDef) else None
                params = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
                ret = None
                if node.returns:
                    try: ret = ast.unparse(node.returns)
                    except: pass
                src = ast.get_source_segment(source, node) or ""
                chunks.append(Chunk(
                    chunk_id=mk("function", node.name), content=src,
                    chunk_type="method" if parent_name else "function",
                    file_type="code", language="python",
                    source_file=rel_path,
                    start_line=node.lineno, end_line=node.end_lineno,
                    name=node.name, docstring=ast.get_docstring(node),
                    parameters=params, return_type=ret, parent=parent_name,
                    imports=imports,
                ))

        return chunks or _line_block_chunks(source, rel_path, "python", "code")

    @staticmethod
    def _imports(tree) -> list[str]:
        result = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                result += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                result.append(n.module)
        return list(set(result))


# ── 4b. Tree-sitter (JS/TS/Java/C/C++/Go/Rust) ───────────────────────────

class TreeSitterParser:
    FUNCTION_NODES = {
        "function_declaration", "function_definition",
        "method_declaration",   "method_definition",
        "arrow_function",       "function_expression",
        "func_literal",         "fn_item",
    }
    CLASS_NODES = {
        "class_declaration", "class_definition",
        "struct_item",       "impl_item",
        "type_declaration",  "interface_declaration",
    }

    def __init__(self, language: str):
        self.language = language
        self._parser  = None
        if TREESITTER_AVAILABLE and language in _TS_LANGS:
            try:
                self._parser = TSParser(_TS_LANGS[language])
            except Exception as e:
                log.warning(f"[TreeSitter] Init failed for {language}: {e}")

    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        if not self._parser:
            return _line_block_chunks(source, rel_path, self.language, "code")

        src_bytes = source.encode("utf-8", errors="replace")
        tree      = self._parser.parse(src_bytes)
        chunks    = []
        counter   = [0]

        def txt(node):
            return src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

        def walk(node, parent_name=None):
            ntype = node.type
            if ntype in self.FUNCTION_NODES:
                nn   = node.child_by_field_name("name")
                name = txt(nn) if nn else "<anonymous>"
                counter[0] += 1
                chunks.append(Chunk(
                    chunk_id=_chunk_id(rel_path, "function", name, counter[0]),
                    content=txt(node),
                    chunk_type="method" if parent_name else "function",
                    file_type="code", language=self.language,
                    source_file=rel_path,
                    start_line=node.start_point[0]+1,
                    end_line=node.end_point[0]+1,
                    name=name, docstring=None,
                    parameters=self._params(node, src_bytes),
                    return_type=None, parent=parent_name, imports=[],
                ))
                for child in node.children: walk(child, parent_name)
                return
            if ntype in self.CLASS_NODES:
                nn   = node.child_by_field_name("name")
                name = txt(nn) if nn else "<Anonymous>"
                counter[0] += 1
                chunks.append(Chunk(
                    chunk_id=_chunk_id(rel_path, "class", name, counter[0]),
                    content=txt(node),
                    chunk_type="class", file_type="code", language=self.language,
                    source_file=rel_path,
                    start_line=node.start_point[0]+1,
                    end_line=node.end_point[0]+1,
                    name=name, docstring=None,
                    parameters=[], return_type=None, parent=None, imports=[],
                ))
                for child in node.children: walk(child, name)
                return
            for child in node.children: walk(child, parent_name)

        walk(tree.root_node)
        return chunks or _line_block_chunks(source, rel_path, self.language, "code")

    @staticmethod
    def _params(node, src_bytes) -> list[str]:
        pn = node.child_by_field_name("parameters")
        if not pn: return []
        return [
            src_bytes[c.start_byte:c.end_byte].decode("utf-8","replace").strip()
            for c in pn.children
            if c.type not in ("(",")",",","formal_parameters")
            and src_bytes[c.start_byte:c.end_byte].decode().strip()
        ]


# ── 4c. HTML / Vue / Svelte ───────────────────────────────────────────────

class HTMLParser:
    """
    Splits HTML into:
      - <script> blocks  → JS chunks
      - <style> blocks   → CSS chunks
      - Semantic sections (<section>, <article>, <main>, <form>, etc.)
      - Remaining body text → doc_paragraph chunk
    Falls back to line-block chunking if BeautifulSoup not installed.
    """
    SEMANTIC_TAGS = ["section","article","main","header","footer",
                     "nav","aside","form","table","dialog","template"]

    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        if not BS4_AVAILABLE:
            log.warning("BeautifulSoup not installed (pip install beautifulsoup4). Using line-block fallback.")
            return _line_block_chunks(source, rel_path, "html", "code")

        soup    = BeautifulSoup(source, "html.parser")
        chunks  = []
        counter = [0]
        lines   = source.splitlines()

        def approx_line(snippet: str) -> Optional[int]:
            for i, l in enumerate(lines, 1):
                if snippet[:30] in l: return i
            return None

        def add(content, ctype, name, lang="html", ftype="code", extra=None):
            if not str(content).strip(): return
            counter[0] += 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, ctype, name or "block", counter[0]),
                content=str(content)[:4000],
                chunk_type=ctype, file_type=ftype, language=lang,
                source_file=rel_path,
                start_line=approx_line(str(content)[:30]),
                end_line=None, name=name, docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
                extra=extra or {},
            ))

        # Scripts
        for tag in soup.find_all("script"):
            js = tag.get_text()
            if js.strip():
                src_attr = tag.get("src") or "inline_script"
                add(js, "script_block", src_attr, lang="javascript")
            tag.decompose()

        # Styles
        for tag in soup.find_all("style"):
            css = tag.get_text()
            if css.strip():
                add(css, "style_block", "inline_style", lang="css")
            tag.decompose()

        # Semantic sections
        for tag_name in self.SEMANTIC_TAGS:
            for tag in soup.find_all(tag_name):
                tag_id   = tag.get("id") or (tag.get("class") or [None])[0] or tag_name
                add(str(tag), "html_section", str(tag_id),
                    extra={"tag": tag_name, "id": tag.get("id"), "class": tag.get("class")})
                tag.decompose()

        # Body text
        body = soup.find("body")
        if body:
            rem = body.get_text(separator="\n", strip=True)
            if rem.strip():
                add(rem, "html_text", "body_text", lang="html", ftype="doc")

        return chunks or _line_block_chunks(source, rel_path, "html", "code")


# ── 4d. CSS / SCSS / LESS ────────────────────────────────────────────────

class CSSParser:
    """
    Extracts:
      - CSS custom properties (variables) → one chunk
      - Every rule block (selector + properties) → one chunk
      - Classifies: @media, @keyframes, @mixin, pseudo-class rules
    """
    RULE_RE = re.compile(
        r'(/\*.*?\*/\s*)?([^{}/][^{}/]*?)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        re.DOTALL
    )
    VAR_RE = re.compile(r'(--[\w-]+)\s*:\s*([^;]+);')

    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        lang = ("scss" if rel_path.endswith((".scss",".sass"))
                else "less" if rel_path.endswith(".less") else "css")
        chunks  = []
        counter = [0]

        # CSS variables block
        vars_found = self.VAR_RE.findall(source)
        if vars_found:
            counter[0] += 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "css_variables", "vars", counter[0]),
                content="\n".join(f"{k}: {v};" for k, v in vars_found),
                chunk_type="css_variables", file_type="code", language=lang,
                source_file=rel_path, start_line=1, end_line=None,
                name="CSS Variables", docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
                extra={"count": len(vars_found)},
            ))

        for m in self.RULE_RE.finditer(source):
            selector = m.group(2).strip()
            body     = m.group(3).strip()
            if not selector or not body: continue
            full     = f"{selector} {{\n{body}\n}}"
            start_ln = source[:m.start()].count("\n") + 1
            ctype    = ("keyframe"     if selector.startswith("@keyframes") else
                        "media_query"  if selector.startswith("@media")     else
                        "mixin"        if selector.startswith("@mixin")      else
                        "at_rule"      if selector.startswith("@")           else
                        "pseudo_rule"  if (":" in selector and "::" not in selector) else
                        "css_rule")
            props = [l.strip() for l in body.splitlines()
                     if ":" in l and not l.strip().startswith(("//","/*"))]
            counter[0] += 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, ctype, selector[:40], counter[0]),
                content=full[:4000],
                chunk_type=ctype, file_type="code", language=lang,
                source_file=rel_path,
                start_line=start_ln, end_line=start_ln + body.count("\n") + 1,
                name=selector[:100], docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
                extra={"properties": props, "prop_count": len(props)},
            ))

        return chunks or _line_block_chunks(source, rel_path, lang, "code")


# ── 4e. JSON ──────────────────────────────────────────────────────────────

class JSONParser:
    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        try:
            data = json.loads(source)
        except json.JSONDecodeError as e:
            log.warning(f"[JSON] {rel_path}: {e}")
            return _line_block_chunks(source, rel_path, "json", "data")

        chunks  = []
        counter = [0]
        BATCH   = 20

        if isinstance(data, list):
            for i in range(0, max(len(data), 1), BATCH):
                batch = data[i:i+BATCH]
                counter[0] += 1
                chunks.append(Chunk(
                    chunk_id=_chunk_id(rel_path, "json_array", f"items_{i}", counter[0]),
                    content=json.dumps(batch, indent=2, ensure_ascii=False)[:4000],
                    chunk_type="json_array_batch", file_type="data", language="json",
                    source_file=rel_path, start_line=None, end_line=None,
                    name=f"items[{i}:{i+BATCH}]", docstring=None,
                    parameters=[], return_type=None, parent=None, imports=[],
                    extra={"total": len(data), "batch_start": i},
                ))
        elif isinstance(data, dict):
            for key, value in data.items():
                counter[0] += 1
                chunks.append(Chunk(
                    chunk_id=_chunk_id(rel_path, "json_key", str(key), counter[0]),
                    content=json.dumps({key: value}, indent=2, ensure_ascii=False)[:4000],
                    chunk_type="json_key", file_type="data", language="json",
                    source_file=rel_path, start_line=None, end_line=None,
                    name=str(key), docstring=None,
                    parameters=[], return_type=None, parent=None, imports=[],
                    extra={"value_type": type(value).__name__},
                ))
        else:
            counter[0] += 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "json_value", "root", counter[0]),
                content=json.dumps(data, indent=2, ensure_ascii=False)[:4000],
                chunk_type="json_value", file_type="data", language="json",
                source_file=rel_path, start_line=None, end_line=None,
                name="root", docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
            ))
        return chunks


# ── 4f. YAML ──────────────────────────────────────────────────────────────

class YAMLParser:
    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        if not YAML_AVAILABLE:
            return _line_block_chunks(source, rel_path, "yaml", "config")
        try:
            data = yaml.safe_load(source)
        except yaml.YAMLError as e:
            log.warning(f"[YAML] {rel_path}: {e}")
            return _line_block_chunks(source, rel_path, "yaml", "config")

        if not isinstance(data, dict):
            return _line_block_chunks(source, rel_path, "yaml", "config")

        chunks = []
        for i, (key, value) in enumerate(data.items(), 1):
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "yaml_key", str(key), i),
                content=yaml.dump({key: value}, default_flow_style=False)[:4000],
                chunk_type="yaml_section", file_type="config", language="yaml",
                source_file=rel_path, start_line=None, end_line=None,
                name=str(key), docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
                extra={"value_type": type(value).__name__},
            ))
        return chunks or _line_block_chunks(source, rel_path, "yaml", "config")


# ── 4g. TOML ──────────────────────────────────────────────────────────────

class TOMLParser:
    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        if not TOML_AVAILABLE:
            return _line_block_chunks(source, rel_path, "toml", "config")
        try:
            data = tomllib.loads(source)
        except Exception as e:
            log.warning(f"[TOML] {rel_path}: {e}")
            return _line_block_chunks(source, rel_path, "toml", "config")

        chunks = []
        for i, (key, value) in enumerate(data.items(), 1):
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "toml_section", key, i),
                content=f"[{key}]\n{json.dumps(value, indent=2, default=str)[:3000]}",
                chunk_type="toml_section", file_type="config", language="toml",
                source_file=rel_path, start_line=None, end_line=None,
                name=key, docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
            ))
        return chunks or _line_block_chunks(source, rel_path, "toml", "config")


# ── 4h. INI / CFG / .env ─────────────────────────────────────────────────

class INIParser:
    SECTION_RE = re.compile(r"^\[(.+)\]", re.MULTILINE)

    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        sections = self.SECTION_RE.split(source)
        chunks   = []
        counter  = 0

        if len(sections) <= 1:
            counter += 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "config_block", "root", counter),
                content=source[:4000],
                chunk_type="config_block", file_type="config", language="ini",
                source_file=rel_path, start_line=1, end_line=source.count("\n")+1,
                name=Path(rel_path).name, docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
            ))
            return chunks

        it = iter(sections[1:])
        for name, body in zip(it, it):
            counter += 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "ini_section", name, counter),
                content=f"[{name}]\n{body.strip()}"[:4000],
                chunk_type="ini_section", file_type="config", language="ini",
                source_file=rel_path, start_line=None, end_line=None,
                name=name, docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
            ))
        return chunks or _line_block_chunks(source, rel_path, "ini", "config")


# ── 4i. XML / SVG ────────────────────────────────────────────────────────

class XMLParser:
    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(source)
        except Exception as e:
            log.warning(f"[XML] {rel_path}: {e}")
            return _line_block_chunks(source, rel_path, "xml", "data")

        chunks = []
        for i, child in enumerate(root, 1):
            tag     = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            content = __import__("xml.etree.ElementTree",
                                  fromlist=["ElementTree"]).tostring(
                          child, encoding="unicode")[:4000]
            name    = child.get("id") or child.get("name") or tag
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "xml_element", name, i),
                content=content,
                chunk_type="xml_element", file_type="data", language="xml",
                source_file=rel_path, start_line=None, end_line=None,
                name=name, docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
                extra={"tag": tag, "attribs": dict(child.attrib)},
            ))
        return chunks or _line_block_chunks(source, rel_path, "xml", "data")


# ── 4j. CSV ───────────────────────────────────────────────────────────────

class CSVParser:
    BATCH = 50

    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        try:
            rows = list(csv.reader(StringIO(source)))
        except Exception:
            return _line_block_chunks(source, rel_path, "csv", "data")

        if not rows: return []
        header    = rows[0]
        data_rows = rows[1:]
        chunks    = []

        for i in range(0, max(len(data_rows), 1), self.BATCH):
            batch = data_rows[i:i+self.BATCH]
            out   = StringIO()
            csv.writer(out).writerow(header)
            csv.writer(out).writerows(batch)
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "csv_batch", f"rows_{i}", i//self.BATCH+1),
                content=out.getvalue()[:4000],
                chunk_type="csv_batch", file_type="data", language="csv",
                source_file=rel_path,
                start_line=i+2, end_line=i+2+len(batch),
                name=f"rows[{i+1}:{i+len(batch)+1}]", docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
                extra={"columns": header, "row_count": len(batch)},
            ))
        return chunks


# ── 4k. Jupyter Notebook ─────────────────────────────────────────────────

class NotebookParser:
    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        try:
            nb = json.loads(source)
        except json.JSONDecodeError:
            return _line_block_chunks(source, rel_path, "jupyter", "notebook")

        chunks = []
        for i, cell in enumerate(nb.get("cells", []), 1):
            ctype   = cell.get("cell_type", "code")
            content = "".join(cell.get("source", []))
            if not content.strip(): continue
            # Collect text outputs for code cells
            out_text = ""
            for o in cell.get("outputs", []):
                if "text" in o:
                    out_text += "".join(o["text"])
                elif "data" in o and "text/plain" in o["data"]:
                    out_text += "".join(o["data"]["text/plain"])
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, f"cell_{ctype}", f"cell_{i}", i),
                content=content[:3000],
                chunk_type=f"notebook_{ctype}_cell",
                file_type="notebook",
                language="python" if ctype == "code" else "markdown",
                source_file=rel_path, start_line=None, end_line=None,
                name=f"Cell {i} ({ctype})", docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
                extra={"cell_index": i, "output_preview": out_text[:500]},
            ))
        return chunks or _line_block_chunks(source, rel_path, "jupyter", "notebook")


# ── 4l. Shell ─────────────────────────────────────────────────────────────

class ShellParser:
    FUNC_RE = re.compile(
        r'^(?:function\s+)?(\w+)\s*\(\s*\)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}',
        re.MULTILINE | re.DOTALL
    )
    COMMENT_BLOCK_RE = re.compile(r'((?:^#[^\n]*\n){3,})', re.MULTILINE)

    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        chunks  = []
        counter = [0]

        for m in self.FUNC_RE.finditer(source):
            name = m.group(1)
            counter[0] += 1
            start = source[:m.start()].count("\n") + 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "shell_function", name, counter[0]),
                content=m.group(0),
                chunk_type="shell_function", file_type="script", language="shell",
                source_file=rel_path,
                start_line=start, end_line=start + m.group(2).count("\n") + 1,
                name=name, docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
            ))

        for m in self.COMMENT_BLOCK_RE.finditer(source):
            counter[0] += 1
            start = source[:m.start()].count("\n") + 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "comment", f"L{start}", counter[0]),
                content=m.group(1).strip(),
                chunk_type="shell_comment", file_type="script", language="shell",
                source_file=rel_path, start_line=start, end_line=None,
                name=f"Comment L{start}", docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
            ))

        return chunks or _line_block_chunks(source, rel_path, "shell", "script")


# ── 4m. SQL ───────────────────────────────────────────────────────────────

class SQLParser:
    STMT_RE = re.compile(
        r'((?:--[^\n]*\n)*'
        r'(?:CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|SELECT|WITH|GRANT|REVOKE|BEGIN|COMMIT|TRUNCATE)'
        r'[^;]*;)',
        re.IGNORECASE | re.DOTALL
    )

    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        chunks = []
        for i, m in enumerate(self.STMT_RE.finditer(source), 1):
            stmt  = m.group(1).strip()
            if not stmt: continue
            first = stmt.lstrip("- \n").split()[0].upper() if stmt.split() else "SQL"
            nm    = re.search(r'(?:TABLE|VIEW|INDEX|FUNCTION|PROCEDURE)\s+(\w+)', stmt, re.I)
            name  = nm.group(1) if nm else f"stmt_{i}"
            start = source[:m.start()].count("\n") + 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "sql_stmt", name, i),
                content=stmt[:4000],
                chunk_type=f"sql_{first.lower()}", file_type="code", language="sql",
                source_file=rel_path,
                start_line=start, end_line=start + stmt.count("\n"),
                name=name, docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
                extra={"stmt_type": first},
            ))
        return chunks or _line_block_chunks(source, rel_path, "sql", "code")


# ── 4n. Markdown ─────────────────────────────────────────────────────────

class MarkdownParser:
    HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)', re.MULTILINE)

    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        matches = list(self.HEADING_RE.finditer(source))
        if not matches:
            return self._para_chunks(source, rel_path)
        chunks = []
        for i, m in enumerate(matches):
            s_pos   = m.start()
            e_pos   = matches[i+1].start() if i+1 < len(matches) else len(source)
            body    = source[s_pos:e_pos].strip()
            level   = len(m.group(1))
            heading = m.group(2).strip()
            s_ln    = source[:s_pos].count("\n") + 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, f"h{level}_section", heading, i+1),
                content=body[:4000],
                chunk_type="doc_section", file_type="doc", language="markdown",
                source_file=rel_path, start_line=s_ln, end_line=s_ln+body.count("\n"),
                name=heading, docstring=body[:200],
                parameters=[], return_type=None, parent=None, imports=[],
                extra={"heading_level": level},
            ))
        return chunks

    def _para_chunks(self, source: str, rel_path: str) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=_chunk_id(rel_path, "paragraph", f"p{i}", i+1),
                content=p[:4000], chunk_type="doc_paragraph",
                file_type="doc", language="markdown",
                source_file=rel_path, start_line=None, end_line=None,
                name=None, docstring=p[:200],
                parameters=[], return_type=None, parent=None, imports=[],
            )
            for i, p in enumerate(p for p in source.split("\n\n") if p.strip())
        ]


# ── 4o. RST ───────────────────────────────────────────────────────────────

class RSTParser:
    SECTION_RE = re.compile(r'^([^\n]+)\n([=\-~#*^"`.\']{3,})\n', re.MULTILINE)

    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        matches = list(self.SECTION_RE.finditer(source))
        if not matches:
            return _line_block_chunks(source, rel_path, "rst", "doc")
        chunks = []
        for i, m in enumerate(matches):
            s_pos   = m.start()
            e_pos   = matches[i+1].start() if i+1 < len(matches) else len(source)
            body    = source[s_pos:e_pos].strip()
            heading = m.group(1).strip()
            s_ln    = source[:s_pos].count("\n") + 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "rst_section", heading, i+1),
                content=body[:4000], chunk_type="doc_section",
                file_type="doc", language="rst",
                source_file=rel_path, start_line=s_ln, end_line=None,
                name=heading, docstring=body[:200],
                parameters=[], return_type=None, parent=None, imports=[],
            ))
        return chunks


# ── 4p. Plain Text ────────────────────────────────────────────────────────

class PlainTextParser:
    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=_chunk_id(rel_path, "paragraph", f"p{i}", i+1),
                content=p[:4000], chunk_type="doc_paragraph",
                file_type="doc", language="text",
                source_file=rel_path, start_line=None, end_line=None,
                name=None, docstring=p[:200],
                parameters=[], return_type=None, parent=None, imports=[],
            )
            for i, p in enumerate(p for p in source.split("\n\n") if p.strip())
        ]


# ── 4q. PDF ───────────────────────────────────────────────────────────────

class PDFParser:
    def parse(self, file_path: str, rel_path: str) -> list[Chunk]:
        if not PYMUPDF_AVAILABLE:
            log.warning("PyMuPDF not installed (pip install pymupdf). Skipping PDF.")
            return []
        doc    = fitz.open(file_path)
        chunks = []
        counter= 0
        for page_num, page in enumerate(doc, 1):
            for block in page.get_text("blocks"):
                text = block[4].strip()
                if not text or len(text) < 20: continue
                counter += 1
                is_heading = (len(text) < 120 and text.endswith((":","\n"))) or text.isupper()
                chunks.append(Chunk(
                    chunk_id=f"{Path(rel_path).stem}__p{page_num}__b{counter}",
                    content=text[:4000],
                    chunk_type="doc_section" if is_heading else "doc_paragraph",
                    file_type="doc", language="pdf",
                    source_file=rel_path, start_line=page_num, end_line=page_num,
                    name=text[:80] if is_heading else None, docstring=text[:200],
                    parameters=[], return_type=None, parent=None, imports=[],
                    extra={"page": page_num},
                ))
        doc.close()
        return chunks


# ── 4r. Dockerfile ────────────────────────────────────────────────────────

class DockerfileParser:
    INSTR_RE = re.compile(
        r'^(FROM|RUN|COPY|ADD|ENV|ARG|WORKDIR|EXPOSE|CMD|ENTRYPOINT|LABEL|VOLUME|USER|HEALTHCHECK|SHELL)\b',
        re.MULTILINE | re.IGNORECASE
    )

    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        pos = [m.start() for m in self.INSTR_RE.finditer(source)]
        if not pos: return _line_block_chunks(source, rel_path, "dockerfile", "infra")
        chunks = []
        for i, sp in enumerate(pos):
            ep    = pos[i+1] if i+1 < len(pos) else len(source)
            block = source[sp:ep].strip()
            instr = block.split()[0].upper()
            s_ln  = source[:sp].count("\n") + 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "docker_instr", instr, i+1),
                content=block,
                chunk_type="dockerfile_instruction", file_type="infra", language="dockerfile",
                source_file=rel_path, start_line=s_ln, end_line=s_ln+block.count("\n"),
                name=instr, docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
                extra={"instruction": instr},
            ))
        return chunks


# ── 4s. Makefile ──────────────────────────────────────────────────────────

class MakefileParser:
    TARGET_RE = re.compile(r'^([\w.\-/]+)\s*:', re.MULTILINE)

    def parse(self, source: str, rel_path: str) -> list[Chunk]:
        matches = list(self.TARGET_RE.finditer(source))
        if not matches: return _line_block_chunks(source, rel_path, "makefile", "infra")
        chunks = []
        for i, m in enumerate(matches):
            sp     = m.start()
            ep     = matches[i+1].start() if i+1 < len(matches) else len(source)
            block  = source[sp:ep].strip()
            target = m.group(1)
            s_ln   = source[:sp].count("\n") + 1
            chunks.append(Chunk(
                chunk_id=_chunk_id(rel_path, "make_target", target, i+1),
                content=block,
                chunk_type="makefile_target", file_type="infra", language="makefile",
                source_file=rel_path, start_line=s_ln, end_line=s_ln+block.count("\n"),
                name=target, docstring=None,
                parameters=[], return_type=None, parent=None, imports=[],
            ))
        return chunks


# ══════════════════════════════════════════════════════════════════════════
# 5. REPO LOADER
# ══════════════════════════════════════════════════════════════════════════

class RepoLoader:
    def __init__(self, repo_url=None, local_path=None):
        if repo_url and local_path:
            raise ValueError("Provide repo_url OR local_path, not both.")
        if not repo_url and not local_path:
            raise ValueError("Provide at least one of repo_url or local_path.")
        self.repo_url   = repo_url
        self.local_path = local_path

    def load(self) -> Path:
        if self.local_path:
            p = Path(self.local_path).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"Path not found: {p}")
            log.info(f"Using local repo: {p}")
            return p
        return self._clone()

    def _clone(self) -> Path:
        tmp = tempfile.mkdtemp(prefix="kt_repo_")
        log.info(f"Cloning {self.repo_url} → {tmp}")
        r = subprocess.run(["git","clone","--depth","1",self.repo_url, tmp],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Git clone failed:\n{r.stderr}")
        log.info("Clone complete.")
        return Path(tmp)

    def collect_files(self, root: Path) -> list[tuple[Path,str,str,str]]:
        """Return (abs_path, file_type, language, parser_key) for every processable file."""
        results = []
        for p in root.rglob("*"):
            if not p.is_file(): continue
            if any(part in SKIP_DIRS for part in p.parts): continue
            if any(pat.match(p.name) for pat in SKIP_FILENAME_PATTERNS): continue

            ext  = p.suffix.lower()
            name = p.name.lower()

            if ext in SKIP_EXTENSIONS: continue

            entry = FILE_REGISTRY.get(ext) or FILENAME_REGISTRY.get(name)
            if entry:
                ft, lang, parser_key = entry
            else:
                # Unknown → will be read as text and line-chunked
                ft, lang, parser_key = "code", ext.lstrip(".") or "unknown", "generic_text"

            results.append((p, ft, lang, parser_key))

        log.info(f"Found {len(results)} files to process.")
        return results


# ══════════════════════════════════════════════════════════════════════════
# 6. ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════

class IngestionPipeline:
    def __init__(self, repo_url=None, local_path=None, max_file_size_kb=500):
        self.loader    = RepoLoader(repo_url=repo_url, local_path=local_path)
        self.max_bytes = max_file_size_kb * 1024

        self._parsers: dict = {
            "python":     PythonParser(),
            "html":       HTMLParser(),
            "css":        CSSParser(),
            "json":       JSONParser(),
            "yaml":       YAMLParser(),
            "toml":       TOMLParser(),
            "ini":        INIParser(),
            "xml":        XMLParser(),
            "csv":        CSVParser(),
            "notebook":   NotebookParser(),
            "shell":      ShellParser(),
            "sql":        SQLParser(),
            "markdown":   MarkdownParser(),
            "rst":        RSTParser(),
            "text":       PlainTextParser(),
            "pdf":        PDFParser(),
            "dockerfile": DockerfileParser(),
            "makefile":   MakefileParser(),
        }
        self._ts_cache: dict[str, TreeSitterParser] = {}

    def _ts(self, lang: str) -> TreeSitterParser:
        if lang not in self._ts_cache:
            self._ts_cache[lang] = TreeSitterParser(lang)
        return self._ts_cache[lang]

    def run(self) -> list[Chunk]:
        root  = self.loader.load()
        files = self.loader.collect_files(root)

        all_chunks:  list[Chunk]    = []
        type_counts: dict[str, int] = {}
        skipped_size = skipped_bin = errors = 0

        for abs_path, file_type, language, parser_key in files:
            rel_path = str(abs_path.relative_to(root))

            if abs_path.stat().st_size > self.max_bytes:
                log.warning(f"  → too large, skipping: {rel_path}")
                skipped_size += 1
                continue

            try:
                chunks = self._dispatch(abs_path, rel_path, file_type, language, parser_key)
                if chunks is None:
                    skipped_bin += 1
                    continue
                all_chunks.extend(chunks)
                type_counts[language] = type_counts.get(language, 0) + len(chunks)
                log.info(f"  ✓ {rel_path:<55} [{language}]  {len(chunks)} chunks")
            except Exception as e:
                log.error(f"  ✗ {rel_path}: {e}")
                errors += 1

        log.info("\n" + "="*65)
        log.info("INGESTION COMPLETE")
        log.info(f"  Total chunks     : {len(all_chunks)}")
        log.info(f"  Files skipped    : size={skipped_size}  binary={skipped_bin}  errors={errors}")
        log.info("  Chunks by language:")
        for lang, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
            log.info(f"    {lang:<22} {cnt}")
        log.info("="*65 + "\n")
        return all_chunks

    def _dispatch(self, abs_path: Path, rel_path: str,
                  file_type: str, language: str, parser_key: str) -> Optional[list[Chunk]]:
        # PDF is special — needs file path not source
        if parser_key == "pdf":
            return self._parsers["pdf"].parse(str(abs_path), rel_path)

        source = _safe_read(abs_path)
        if source is None:
            log.debug(f"  Binary, skipping: {rel_path}")
            return None
        if not source.strip():
            return []

        if parser_key in self._parsers:
            return self._parsers[parser_key].parse(source, rel_path)
        if parser_key == "treesitter":
            return self._ts(language).parse(source, rel_path)
        # generic_code / generic_text / anything unknown
        return _line_block_chunks(source, rel_path, language, file_type)


# ══════════════════════════════════════════════════════════════════════════
# 7. SERIALISATION
# ══════════════════════════════════════════════════════════════════════════

def chunks_to_json(chunks: list[Chunk], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(chunks)} chunks → {path}")

def chunks_to_jsonl(chunks: list[Chunk], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    log.info(f"Saved {len(chunks)} chunks (JSONL) → {path}")


# ══════════════════════════════════════════════════════════════════════════
# 8. CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Stage 1 — Universal repo ingestion & parsing.")
    ap.add_argument("--repo_url",         default=None,  help="GitHub URL")
    ap.add_argument("--local_path",       default=None,  help="Local repo path")
    ap.add_argument("--output",           default="chunks.json")
    ap.add_argument("--max_file_size_kb", type=int, default=500)
    args = ap.parse_args()

    pipeline = IngestionPipeline(
        repo_url=args.repo_url,
        local_path=args.local_path,
        max_file_size_kb=args.max_file_size_kb,
    )
    chunks = pipeline.run()

    (chunks_to_jsonl if args.output.endswith(".jsonl") else chunks_to_json)(chunks, args.output)

    print(f"\n--- Sample chunk (1 of {len(chunks)}) ---")
    if chunks:
        print(json.dumps(asdict(chunks[0]), indent=2)[:1500])


if __name__ == "__main__":
    main()