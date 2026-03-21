"""
=============================================================================
tagger.py  —  Step 4: Metadata Tagging
=============================================================================
Enriches each NLPChunk with:
  tags             — domain/topic labels (auth, async, crud, test, …)
  complexity_score — 0.0 (trivial) → 1.0 (complex) heuristic
  has_docstring    — bool
  has_tests        — bool
  has_todos        — bool
  api_surface      — list of publicly accessible names

Tag taxonomy (25 domains):
  auth, async, crud, error_handling, logging, caching, config,
  database, networking, filesystem, testing, security, serialization,
  validation, concurrency, ml, ui, routing, middleware, deployment,
  documentation, migration, parsing, streaming, utils
=============================================================================
"""

from __future__ import annotations
import re
import math
from models import NLPChunk


# ═══════════════════════════════════════════════════════════════════════════
# Tag keyword rules  {tag: [keywords …]}
# ═══════════════════════════════════════════════════════════════════════════

_TAG_RULES: dict[str, list[str]] = {
    "auth": [
        "login", "logout", "authenticate", "authorize", "jwt",
        "oauth", "token", "session", "password", "credential",
        "permission", "role", "rbac", "bearer",
    ],
    "async": [
        "async", "await", "promise", "future", "coroutine",
        "asyncio", "aiohttp", "concurrent", "thread", "executor",
    ],
    "crud": [
        "create", "read", "update", "delete", "insert", "select",
        "upsert", "findone", "findall", "save", "remove",
    ],
    "error_handling": [
        "try", "except", "catch", "finally", "raise", "throw",
        "error", "exception", "traceback", "fallback", "retry",
    ],
    "logging": [
        "logger", "logging", "log.info", "log.debug", "log.error",
        "log.warning", "print(", "console.log", "sentry",
    ],
    "caching": [
        "cache", "redis", "memcached", "lru_cache", "ttl",
        "invalidate", "evict", "cached_property",
    ],
    "config": [
        "config", "settings", "env", "environ", "dotenv",
        "configparser", "argparse", ".env",
    ],
    "database": [
        "database", "db", "sql", "orm", "sqlalchemy", "django.db",
        "mongoose", "prisma", "sequelize", "cursor", "transaction",
    ],
    "networking": [
        "http", "https", "request", "response", "socket", "tcp",
        "udp", "grpc", "websocket", "fetch", "axios", "httpx",
    ],
    "filesystem": [
        "open(", "read(", "write(", "pathlib", "os.path",
        "shutil", "glob", "makedirs", "walk(",
    ],
    "testing": [
        "test_", "def test", "assert", "unittest", "pytest",
        "mock", "fixture", "describe(", "it(", "expect(",
    ],
    "security": [
        "encrypt", "decrypt", "hash", "salt", "hmac", "aes",
        "rsa", "ssl", "tls", "sanitize", "csrf", "xss",
    ],
    "serialization": [
        "json", "pickle", "marshal", "serialize", "deserialize",
        "encode", "decode", "yaml.dump", "yaml.load",
    ],
    "validation": [
        "validate", "validator", "pydantic", "schema", "required",
        "isinstance", "assert", "constraint", "regex",
    ],
    "concurrency": [
        "thread", "mutex", "lock", "semaphore", "queue",
        "multiprocessing", "parallel", "race", "deadlock",
    ],
    "ml": [
        "model", "train", "predict", "tensor", "numpy", "pandas",
        "sklearn", "torch", "tensorflow", "keras", "embedding",
        "loss", "optimizer", "epoch", "batch_size",
    ],
    "ui": [
        "render", "component", "props", "state", "useState",
        "useEffect", "template", "html", "css", "dom", "react",
        "vue", "svelte", "angular",
    ],
    "routing": [
        "route", "router", "endpoint", "url", "path", "get(",
        "post(", "put(", "delete(", "app.get", "app.post",
    ],
    "middleware": [
        "middleware", "interceptor", "hook", "filter", "decorator",
        "before_request", "after_request",
    ],
    "deployment": [
        "dockerfile", "docker", "k8s", "kubernetes", "helm",
        "deploy", "release", "ci", "cd", "github actions",
    ],
    "documentation": [
        "docstring", "readme", "changelog", "comment", "sphinx",
        "jsdoc", "mkdocs", "openapi", "swagger",
    ],
    "migration": [
        "migration", "migrate", "alembic", "flyway", "schema",
        "alter table", "add column", "drop column",
    ],
    "parsing": [
        "parse", "parser", "ast", "tokenize", "lex", "grammar",
        "regex", "xpath", "beautifulsoup", "lxml",
    ],
    "streaming": [
        "stream", "generator", "yield", "iterable", "pipe",
        "kafka", "rabbitmq", "pubsub", "event",
    ],
    "utils": [
        "util", "helper", "common", "shared", "misc", "tools",
        "format", "convert",
    ],
}

# Pre-compile one pattern per tag
_TAG_RE: dict[str, re.Pattern] = {
    tag: re.compile("|".join(re.escape(kw) for kw in kws), re.IGNORECASE)
    for tag, kws in _TAG_RULES.items()
}

# ── Other patterns ────────────────────────────────────────────────────────

_PUBLIC_DEF_RE = re.compile(
    r'(?:^|\n)\s*(?:export\s+)?(?:async\s+)?'
    r'(?:def|function|class|func|fn)\s+([A-Za-z][A-Za-z0-9_]*)'
)
_DOCSTRING_RE  = re.compile(r'(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|/\*[\s\S]*?\*/)')
_TODO_RE       = re.compile(r'\b(TODO|FIXME|HACK|XXX)\b', re.IGNORECASE)
_TEST_RE       = re.compile(
    r'\b(def test_|it\(|describe\(|test\(|@pytest|unittest\.)', re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════════
# Complexity heuristic  (0.0 – 1.0)
# ═══════════════════════════════════════════════════════════════════════════

def _complexity(text: str, token_count: int) -> float:
    """
    Lightweight McCabe-like proxy.
    Counts branching keywords + estimates max nesting depth + token volume.
    """
    branches = len(re.findall(
        r'\b(if|elif|else|for|while|try|except|catch|switch|case|match|with)\b',
        text, re.IGNORECASE,
    ))

    # Estimate max indent depth (assume 4-space or 1-tab indent)
    depths = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            depths.append(indent // 4 if "    " in line else indent)
    max_depth = max(depths) if depths else 0

    # Log-scaled token volume (2000 tokens ≈ 1.0)
    vol = math.log1p(token_count) / math.log1p(2000)

    raw = (branches * 0.05) + (max_depth * 0.05) + (vol * 0.5)
    return round(min(raw, 1.0), 3)


# ═══════════════════════════════════════════════════════════════════════════
# MetadataTagger
# ═══════════════════════════════════════════════════════════════════════════

class MetadataTagger:
    """
    Call `.tag(chunk)` to populate:
      tags, complexity_score, has_docstring, has_tests, has_todos, api_surface
    """

    def __init__(self, min_tag_matches: int = 1):
        """
        min_tag_matches : minimum keyword hits required to apply a tag.
                          Raise to 2 for stricter tagging.
        """
        self.min_matches = min_tag_matches

    def tag(self, chunk: NLPChunk) -> NLPChunk:
        text = chunk.cleaned_text or chunk.content

        # ── Domain tags ───────────────────────────────────────────────────
        tags = [
            tag for tag, pat in _TAG_RE.items()
            if len(pat.findall(text)) >= self.min_matches
        ]

        # ── Boolean flags ─────────────────────────────────────────────────
        has_docstring = bool(_DOCSTRING_RE.search(text)) or bool(chunk.docstring)
        has_tests     = bool(_TEST_RE.search(text))
        has_todos     = bool(_TODO_RE.search(text))

        # ── API surface (public names only — no leading underscore) ───────
        api_surface = list({
            m.group(1)
            for m in _PUBLIC_DEF_RE.finditer(text)
            if not m.group(1).startswith("_")
        })

        # ── Complexity ────────────────────────────────────────────────────
        complexity = _complexity(text, chunk.token_count)

        chunk.tags             = sorted(set(tags))
        chunk.complexity_score = complexity
        chunk.has_docstring    = has_docstring
        chunk.has_tests        = has_tests
        chunk.has_todos        = has_todos
        chunk.api_surface      = api_surface
        return chunk

    def tag_all(self, chunks: list[NLPChunk]) -> list[NLPChunk]:
        return [self.tag(c) for c in chunks]


def make_tagger(**kwargs) -> MetadataTagger:
    return MetadataTagger(**kwargs)
