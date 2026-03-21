"""
=============================================================================
embedder.py  —  Step 5: Embedding Generation
=============================================================================
Generates dense vector representations for each NLPChunk.

Backends (tried in priority order — first available wins):
  1. sentence-transformers   pip install sentence-transformers
       default: "all-MiniLM-L6-v2"  (384-dim, fast, offline)
       swap to "all-mpnet-base-v2"  for higher quality

  2. OpenAI API              pip install openai  + set OPENAI_API_KEY
       model: text-embedding-3-small  (1536-dim)

  3. Cohere API              pip install cohere  + set COHERE_API_KEY
       model: embed-multilingual-v3.0  (1024-dim) — best for Step 6

  4. HuggingFace Transformers  pip install transformers torch
       mean-pool over last hidden state

  5. TF-IDF                  always available (sklearn), offline fallback
       sparse → truncated dense vector (256-dim)

Text used for embedding (priority order):
  name  +  docstring  +  cleaned_text  (truncated to max_chars)
=============================================================================
"""

from __future__ import annotations
import os
import logging
from typing import Optional
from models import NLPChunk

log = logging.getLogger("embedder")

# ── Backend availability flags ────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import openai as _openai
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import cohere as _cohere
    COHERE_AVAILABLE = bool(os.getenv("COHERE_API_KEY"))
except ImportError:
    COHERE_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ═══════════════════════════════════════════════════════════════════════════
# Backend classes
# ═══════════════════════════════════════════════════════════════════════════

class _STBackend:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self.name   = f"sentence-transformers/{model_name}"
        self.dim    = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()


class _OpenAIBackend:
    def __init__(self, model: str = "text-embedding-3-small"):
        self._client = _openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model  = model
        self.name    = f"openai/{model}"
        self.dim     = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]


class _CohereBackend:
    def __init__(self, model: str = "embed-multilingual-v3.0"):
        self._client = _cohere.Client(os.getenv("COHERE_API_KEY"))
        self._model  = model
        self.name    = f"cohere/{model}"
        self.dim     = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embed(
            texts=texts, model=self._model, input_type="search_document"
        )
        return [list(e) for e in resp.embeddings]


class _HFBackend:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._tok   = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self.name   = f"hf/{model_name}"
        self.dim    = self._model.config.hidden_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        enc = self._tok(texts, padding=True, truncation=True,
                        max_length=512, return_tensors="pt")
        with torch.no_grad():
            out = self._model(**enc)
        mask  = enc["attention_mask"].unsqueeze(-1).float()
        vecs  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return (vecs / norms).numpy().tolist()


class _TFIDFBackend:
    """Always-available offline fallback. Fits on the current batch."""
    name = "tfidf/fallback"
    dim  = 256

    def __init__(self):
        self._vec     = TfidfVectorizer(max_features=self.dim, sublinear_tf=True)
        self._fitted  = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            self._vec.fit(texts)
            self._fitted = True
        mat   = self._vec.transform(texts).toarray().astype(float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (mat / norms).tolist()


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build text to embed
# ═══════════════════════════════════════════════════════════════════════════

def _build_text(chunk: NLPChunk, max_chars: int) -> str:
    """Highest-signal content first: name → docstring → cleaned_text."""
    parts = []
    if chunk.name:
        parts.append(chunk.name)
    if chunk.docstring:
        parts.append(chunk.docstring[:300])
    parts.append(chunk.cleaned_text or chunk.content)
    return " \n".join(parts)[:max_chars]


# ═══════════════════════════════════════════════════════════════════════════
# Embedder
# ═══════════════════════════════════════════════════════════════════════════

class Embedder:
    """
    Call `.embed(chunk)` or `.embed_all(chunks)` to populate:
      embedding, embedding_model, embedding_dim
    """

    BACKENDS = ["sentence_transformers", "openai", "cohere", "huggingface", "tfidf"]

    def __init__(self,
                 backend:    str           = "auto",
                 model_name: Optional[str] = None,
                 batch_size: int           = 64,
                 max_chars:  int           = 2000):
        """
        backend    : "auto" | "sentence_transformers" | "openai" |
                     "cohere" | "huggingface" | "tfidf"
        model_name : override default model for the chosen backend.
        batch_size : chunks per API / model call.
        max_chars  : input text truncation limit.
        """
        self._backend   = self._select(backend, model_name)
        self.batch_size = batch_size
        self.max_chars  = max_chars
        log.info(f"[Embedder] backend={self._backend.name}  dim={self._backend.dim}")

    @staticmethod
    def _select(pref: str, model_name: Optional[str]):
        want = lambda name: pref == name or pref == "auto"
        if want("sentence_transformers") and ST_AVAILABLE:
            return _STBackend(model_name or "all-MiniLM-L6-v2")
        if want("openai") and OPENAI_AVAILABLE:
            return _OpenAIBackend(model_name or "text-embedding-3-small")
        if want("cohere") and COHERE_AVAILABLE:
            return _CohereBackend(model_name or "embed-multilingual-v3.0")
        if want("huggingface") and HF_AVAILABLE:
            return _HFBackend(model_name or "sentence-transformers/all-MiniLM-L6-v2")
        if pref not in ("auto", "tfidf"):
            log.warning(f"[Embedder] Backend '{pref}' not available — falling back to TF-IDF.")
        return _TFIDFBackend()

    def embed(self, chunk: NLPChunk) -> NLPChunk:
        return self.embed_all([chunk])[0]

    def embed_all(self, chunks: list[NLPChunk]) -> list[NLPChunk]:
        texts = [_build_text(c, self.max_chars) for c in chunks]
        for i in range(0, len(chunks), self.batch_size):
            batch_c = chunks[i: i + self.batch_size]
            batch_t = texts[i:  i + self.batch_size]
            try:
                vecs = self._backend.embed(batch_t)
            except Exception as e:
                log.error(f"[Embedder] Batch {i // self.batch_size} failed: {e}")
                vecs = [[] for _ in batch_c]
            for chunk, vec in zip(batch_c, vecs):
                chunk.embedding       = vec
                chunk.embedding_model = self._backend.name
                chunk.embedding_dim   = len(vec)
        return chunks


def make_embedder(**kwargs) -> Embedder:
    return Embedder(**kwargs)
