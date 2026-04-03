"""
=============================================================================
pipeline.py  —  Stage 2: NLP Preprocessing Pipeline  (Orchestrator)
=============================================================================
pip install scikit-learn numpy chardet sentence-transformers langdetect langid deep-translator spacy pyyaml pymupdf beautifulsoup4 openai cohere
>> python -m spacy download en_core_web_sm
python nlp_stage2/nlp_pipeline.py --input chunks.json --output nlp_chunks.json

Wires all six NLP steps together in sequence:

  Stage 1 output  (chunks.json / List[Chunk])
       │
       ▼  NLPChunk.from_chunk()
  List[NLPChunk]
       │
       ▼  Step 1 — TextCleaner          (cleaner.py)
  cleaned_text · tokens · token_count · normalized_lang
       │
       ▼  Step 2 — NERTagger            (ner.py)
  entities
       │
       ▼  Step 3 — SemanticChunker      (semantic_chunker.py)
  semantic_segments · segment_count
       │
       ▼  Step 4 — MetadataTagger       (tagger.py)
  tags · complexity_score · has_docstring · has_tests · has_todos · api_surface
       │
       ▼  Step 5 — Embedder             (embedder.py)
  embedding · embedding_model · embedding_dim
       │
       ▼  Step 6 — MultilingualProcessor (multilingual.py)
  detected_lang · detected_lang_conf · translated_text · is_translated
       │
       ▼  Serialise
  nlp_chunks.json  /  nlp_chunks.jsonl

─────────────────────────────────────────────────────────────────────────────
CLI usage:
  python pipeline.py --input chunks.json --output nlp_chunks.json
  python pipeline.py --input chunks.json --output nlp_chunks.jsonl \
      --no_spacy --no_translate --embedding_backend tfidf
  python pipeline.py --input chunks.json --skip_steps embed multilingual

Python API usage:
  from pipeline import NLPPipeline, load_chunks_json, save_nlp_chunks

  raw    = load_chunks_json("chunks.json")
  pipe   = NLPPipeline(embedding_backend="sentence_transformers")
  result = pipe.run(raw)
  save_nlp_chunks(result, "nlp_chunks.json")
=============================================================================
"""

from __future__ import annotations
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

from models           import Chunk, NLPChunk
from cleaner          import make_cleaner
from ner              import make_ner
from semantic_chunker import make_semantic_chunker
from tagger           import make_tagger
from embedder         import make_embedder
from multilingual     import make_multilingual

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("nlp_pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# NLPPipeline
# ═══════════════════════════════════════════════════════════════════════════

class NLPPipeline:
    """
    End-to-end NLP preprocessing pipeline.

    Parameters
    ──────────
    strip_comments      Strip inline code comments during cleaning (default False)
    use_spacy           Enable spaCy NER tier 2 (default True)
    use_similarity      Use embedding similarity for semantic chunking (default False)
    embedding_backend   "auto"|"sentence_transformers"|"openai"|"cohere"|
                        "huggingface"|"tfidf"
    embedding_model     Override default model for the chosen backend
    embedding_batch     Batch size for embedding API / model calls
    translate_langs     Set of ISO-639-1 codes to translate (None = all non-en)
    skip_code_translate Skip translation for pure code chunks (default True)
    skip_steps          Set of step names to skip entirely:
                        {"clean","ner","chunk","tag","embed","multilingual"}
    """

    STEP_NAMES = ("clean", "ner", "chunk", "tag", "embed", "multilingual")

    def __init__(
        self,
        strip_comments:      bool          = False,
        use_spacy:           bool          = True,
        use_similarity:      bool          = False,
        embedding_backend:   str           = "auto",
        embedding_model:     Optional[str] = None,
        embedding_batch:     int           = 64,
        translate_langs:     Optional[set] = None,
        skip_code_translate: bool          = True,
        skip_steps:          set           = frozenset(),
    ):
        self.skip = set(skip_steps)

        self.cleaner      = make_cleaner(strip_comments=strip_comments)
        self.ner          = make_ner(use_spacy=use_spacy)
        self.chunker      = make_semantic_chunker(use_similarity=use_similarity)
        self.tagger       = make_tagger()
        self.embedder     = make_embedder(
                                backend=embedding_backend,
                                model_name=embedding_model,
                                batch_size=embedding_batch,
                            )
        self.multilingual = make_multilingual(
                                translate_langs=translate_langs,
                                skip_code=skip_code_translate,
                            )

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, raw_chunks: list[Chunk]) -> list[NLPChunk]:
        """Process Stage-1 Chunk objects → fully enriched NLPChunks."""
        log.info(f"NLP Pipeline starting — {len(raw_chunks)} chunks")
        t0 = time.time()

        nlp_chunks: list[NLPChunk] = [NLPChunk.from_chunk(c) for c in raw_chunks]

        if "clean" not in self.skip:
            log.info("Step 1/6 — Text Cleaning & Normalisation")
            nlp_chunks = self.cleaner.clean_all(nlp_chunks)

        if "ner" not in self.skip:
            log.info("Step 2/6 — Named Entity Recognition")
            nlp_chunks = self.ner.tag_all(nlp_chunks)

        if "chunk" not in self.skip:
            log.info("Step 3/6 — Semantic Chunking")
            nlp_chunks = self.chunker.chunk_all(nlp_chunks)

        if "tag" not in self.skip:
            log.info("Step 4/6 — Metadata Tagging")
            nlp_chunks = self.tagger.tag_all(nlp_chunks)

        if "embed" not in self.skip:
            log.info("Step 5/6 — Embedding Generation")
            nlp_chunks = self.embedder.embed_all(nlp_chunks)

        if "multilingual" not in self.skip:
            log.info("Step 6/6 — Multilingual Processing")
            nlp_chunks = self.multilingual.process_all(nlp_chunks)

        self._summary(nlp_chunks, round(time.time() - t0, 2))
        return nlp_chunks

    # ── Summary ───────────────────────────────────────────────────────────

    @staticmethod
    def _summary(chunks: list[NLPChunk], elapsed: float) -> None:
        total_ents = sum(len(c.entities)   for c in chunks)
        total_segs = sum(c.segment_count   for c in chunks)
        embedded   = sum(1 for c in chunks if c.embedding)
        translated = sum(1 for c in chunks if c.is_translated)
        langs: dict[str, int] = {}
        for c in chunks:
            if c.detected_lang:
                langs[c.detected_lang] = langs.get(c.detected_lang, 0) + 1

        log.info("=" * 60)
        log.info("NLP PIPELINE COMPLETE")
        log.info(f"  Chunks processed   : {len(chunks)}")
        log.info(f"  Total entities     : {total_ents}")
        log.info(f"  Semantic segments  : {total_segs}")
        log.info(f"  Embedded chunks    : {embedded}")
        log.info(f"  Detected langs     : {dict(sorted(langs.items(), key=lambda x: -x[1]))}")
        log.info(f"  Translated chunks  : {translated}")
        log.info(f"  Wall time          : {elapsed}s")
        log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_chunks_json(path: str) -> list[Chunk]:
    """Load Stage-1 output from a JSON or JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)

    fields = set(Chunk.__dataclass_fields__)
    return [Chunk(**{k: v for k, v in d.items() if k in fields}) for d in records]


def save_nlp_chunks(chunks: list[NLPChunk], path: str) -> None:
    """Serialise NLPChunks to JSON or JSONL."""
    dicts = [c.to_dict() for c in chunks]
    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for d in dicts:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dicts, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(chunks)} NLP chunks → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 2 — NLP Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",              default="chunks.json",
                    help="Stage-1 output file (.json or .jsonl)")
    ap.add_argument("--output",             default="nlp_chunks.json",
                    help="Output file (.json or .jsonl)")
    ap.add_argument("--strip_comments",     action="store_true",
                    help="Strip inline code comments during cleaning")
    ap.add_argument("--no_spacy",           action="store_true",
                    help="Disable spaCy NER (rule-based only)")
    ap.add_argument("--no_translate",       action="store_true",
                    help="Disable multilingual translation step")
    ap.add_argument("--use_similarity",     action="store_true",
                    help="Use embedding similarity for semantic splitting")
    ap.add_argument("--embedding_backend",  default="auto",
                    choices=["auto", "sentence_transformers", "openai",
                             "cohere", "huggingface", "tfidf"],
                    help="Embedding backend to use")
    ap.add_argument("--embedding_model",    default=None,
                    help="Override default model name for the chosen backend")
    ap.add_argument("--embedding_batch",    type=int, default=64,
                    help="Batch size for embedding calls")
    ap.add_argument("--skip_steps",         nargs="*", default=[],
                    choices=list(NLPPipeline.STEP_NAMES),
                    help="Pipeline steps to skip entirely")
    ap.add_argument("--log_level",          default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.getLogger().setLevel(args.log_level)

    skip = set(args.skip_steps)
    if args.no_translate:
        skip.add("multilingual")

    raw_chunks = load_chunks_json(args.input)
    log.info(f"Loaded {len(raw_chunks)} chunks from {args.input}")

    pipeline = NLPPipeline(
        strip_comments    = args.strip_comments,
        use_spacy         = not args.no_spacy,
        use_similarity    = args.use_similarity,
        embedding_backend = args.embedding_backend,
        embedding_model   = args.embedding_model,
        embedding_batch   = args.embedding_batch,
        skip_steps        = skip,
    )

    nlp_chunks = pipeline.run(raw_chunks)
    save_nlp_chunks(nlp_chunks, args.output)

    # Print a sample (hide the embedding vector for readability)
    print(f"\n--- Sample NLP chunk (1 of {len(nlp_chunks)}) ---")
    if nlp_chunks:
        sample = nlp_chunks[0].to_dict()
        sample.pop("embedding", None)
        print(json.dumps(sample, indent=2, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
