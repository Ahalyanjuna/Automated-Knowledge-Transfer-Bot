```bash
Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

python ingestion.py --repo_url https://github.com/ArulKevin2004/VisionCortex

cd nlp_stage2
python nlp_pipeline.py --input ../chunks.json --output ../output/nlp_chunks.json
cd ..

cd genai 
python load_chunks_to_chroma.py
cd ..

streamlit run app.py
```
# 🧠 KT Bot — Automated Knowledge Transfer System

> Transform any software repository into an intelligent, multilingual Q&A assistant powered by NLP, vector search, and reinforcement learning.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green.svg)](https://www.trychroma.com)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](#)

---

## 📋 Table of Contents

- [What is KT Bot?](#what-is-kt-bot)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [NLP Pipeline](#nlp-pipeline)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Admin Features](#admin-features)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)

---

## What is KT Bot?

KT Bot is a production-grade **Automated Knowledge Transfer System** that turns any codebase into a searchable, conversational knowledge base.  

It solves a real problem: onboarding new developers takes weeks because understanding a large codebase is hard. KT Bot lets anyone — developer, project manager, or data scientist — ask questions about a project in plain language (in **any language**) and get precise, role-aware answers grounded in the actual source code.

```
"How does authentication work in this project?"
→ KT Bot retrieves the exact code, explains it for your role, in your language.
```

---

## Key Features

| Feature | Description |
|---|---|
| 🗂️ **Universal Ingestion** | Parses 30+ file types — Python, JS/TS, Java, C/C++, Go, Rust, HTML, CSS, SQL, YAML, Notebooks, PDFs, and more |
| 🔬 **6-Step NLP Pipeline** | Cleaning → NER → Semantic Chunking → Tagging → Embedding → Multilingual |
| 🔍 **Semantic Vector Search** | ChromaDB with cosine similarity on 384-dim sentence-transformer embeddings |
| 🤖 **RL Re-ranking** | Deep Q-Network continuously improves retrieval from thumbs-up/down feedback |
| 🌐 **Multilingual** | Ask in Tamil, Hindi, French, Japanese — any of 100+ languages. Answers back in your language |
| 🎭 **Role-Aware Answers** | Different explanations for Developer / Project Manager / Data Scientist |
| 📊 **NLP Dashboard** | Rich analytics: chunk types, tag clouds, complexity heatmaps, PCA embedding plots |
| 🔬 **NLP Lens** | Real-time per-query NLP breakdown: tokenization, NER, pipeline trace, faithfulness heatmap |
| 📄 **Auto Documentation** | Generate technical reports from any GitHub repo |
| 📋 **RAGAS Evaluation** | Batch retrieval quality testing |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  OFFLINE INGESTION PIPELINE                 │
│                                                             │
│  GitHub/Local  →  ingestion.py  →  nlp_pipeline.py         │
│  (30+ parsers)    (List[Chunk])   (6-step NLP enrichment)   │
│                        ↓                                    │
│              load_chunks_to_chroma.py                       │
│              (chroma_<project>/ vector DB)                  │
└─────────────────────────────────────────────────────────────┘
                          ↓ (one-time setup)
┌─────────────────────────────────────────────────────────────┐
│                   ONLINE SERVING LAYER                      │
│                                                             │
│  User Query  →  detect lang  →  translate to EN            │
│       ↓                                                     │
│  KTRetriever  →  ChromaDB cosine search  →  top-10 hits    │
│       ↓                                                     │
│  RLAgent  →  DQN re-rank  →  top-3 context chunks          │
│       ↓                                                     │
│  Groq LLM  →  role-aware prompt  →  English answer         │
│       ↓                                                     │
│  translate back  →  Streamlit UI  →  User                  │
│       ↓                                                     │
│  👍/👎 feedback  →  rl_feedback.db  →  train DQN           │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
kt-bot/
├── app.py                    # Streamlit web application (main entry point)
├── ingestion.py              # Stage 1: Universal repo ingestion & parsing
├── nlp_pipeline.py           # Stage 2: NLP pipeline orchestrator
├── models.py                 # Shared dataclasses: Chunk, NLPChunk, Entity
├── cleaner.py                # NLP Step 1: Text cleaning & normalisation
├── ner.py                    # NLP Step 2: Named entity recognition
├── semantic_chunker.py       # NLP Step 3: Semantic segmentation
├── tagger.py                 # NLP Step 4: Metadata tagging (25 domains)
├── embedder.py               # NLP Step 5: Embedding generation (5 backends)
├── multilingual.py           # NLP Step 6: Language detection & translation
├── retriever.py              # KTRetriever: ChromaDB semantic search
├── rl_agent.py               # Deep Q-Network re-ranker
├── rl_logger.py              # SQLite feedback experience logger
├── train_rl.py               # RL model training from feedback DB
├── load_chunks_to_chroma.py  # Batch-load NLP chunks into ChromaDB
├── chat_engine.py            # KTChatEngine: full response generation
├── evaluator.py              # Single-query faithfulness/relevance scorer
├── ragas_test.py             # Batch RAGAS-lite evaluation
├── auto_doc.py               # GitHub repo → Markdown technical report
├── nlp_dashboard.py          # Streamlit NLP analytics dashboard
├── nlp_lens.py               # Live per-query NLP analysis panel
├── query_translator.py       # Runtime language detection & translation
└── .env                      # Environment variables (not committed)
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-org/kt-bot.git
cd kt-bot

pip install streamlit chromadb sentence-transformers groq langdetect
pip install deep-translator langid spacy pyyaml pymupdf beautifulsoup4
pip install reportlab plotly pandas torch scikit-learn
python -m spacy download en_core_web_sm
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
USERS_DB=users.db
RL_DB=rl_feedback.db
PROJECT_CONFIG={"myproject": {"repo": "https://github.com/user/myproject"}}
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Ingest a Project

```bash
# From GitHub
python ingestion.py --repo_url https://github.com/user/repo --output chunks.json

# From local path
python ingestion.py --local_path /path/to/repo --output chunks.json
```

### 4. Run the NLP Pipeline

```bash
python nlp_pipeline.py --input chunks.json --output nlp_chunks.json
```

### 5. Load into ChromaDB

Edit `load_chunks_to_chroma.py` to set:
```python
INPUT_JSON      = Path("nlp_chunks.json")
PERSIST_DIR     = Path("chroma_myproject")   # must match project name
COLLECTION_NAME = "myproject_rag"
```

Then run:
```bash
python load_chunks_to_chroma.py
```

### 6. Launch the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` — register an account (username `admin` gets admin access), select your project, and start asking questions!

---

## NLP Pipeline

The 6-step NLP pipeline transforms raw source chunks into semantically rich, searchable knowledge:

```
Raw Chunk
   │
   ▼  Step 1 — TextCleaner          (cleaner.py)
   │  • Strip ANSI codes, null bytes, control chars
   │  • Unicode normalization (NFC), smart quotes → ASCII
   │  • Dedent code, collapse whitespace, tokenize
   │
   ▼  Step 2 — NERTagger            (ner.py)
   │  • Rule-based: FUNC, CLASS, LIB, URL, TODO, FILE_PATH, ENV_VAR, VERSION…
   │  • Optional spaCy: ORG, PERSON, GPE, PRODUCT
   │
   ▼  Step 3 — SemanticChunker      (semantic_chunker.py)
   │  • Docs: split on headings / paragraphs
   │  • Code: split on function/class boundaries or sliding window
   │  • Data: preserve atomic (JSON key, YAML section, CSV batch)
   │
   ▼  Step 4 — MetadataTagger       (tagger.py)
   │  • 25 domain tags: auth, async, crud, ml, database, security, routing…
   │  • McCabe-like complexity score 0.0–1.0
   │  • Flags: has_docstring, has_tests, has_todos
   │  • Public API surface extraction
   │
   ▼  Step 5 — Embedder             (embedder.py)
   │  • Default: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
   │  • Fallback chain: OpenAI → Cohere → HuggingFace → TF-IDF
   │
   ▼  Step 6 — MultilingualProcessor (multilingual.py)
      • Detect language: langdetect → langid → unicode heuristic
      • Translate non-English doc chunks → English for embedding
      • Code chunks: detect only, no translation
```

### Supported File Types

| Category | Extensions | Parser |
|---|---|---|
| Python | `.py`, `.pyw` | Python AST |
| JS / TS | `.js`, `.jsx`, `.ts`, `.tsx`, `.mjs` | Tree-sitter |
| Java | `.java` | Tree-sitter |
| C / C++ | `.c`, `.h`, `.cpp`, `.hpp` | Tree-sitter |
| Go | `.go` | Tree-sitter |
| Rust | `.rs` | Tree-sitter |
| Web | `.html`, `.css`, `.scss`, `.less`, `.vue`, `.svelte` | BeautifulSoup / CSS regex |
| Data | `.json`, `.yaml`, `.toml`, `.xml`, `.csv`, `.tsv` | Dedicated parsers |
| Docs | `.md`, `.rst`, `.txt`, `.pdf` | Heading/paragraph splits |
| Shell | `.sh`, `.bash`, `.zsh` | Function + comment parser |
| SQL | `.sql` | Statement parser |
| Notebooks | `.ipynb` | Cell extractor |
| Infrastructure | `Dockerfile`, `Makefile`, `.tf`, `.hcl` | Instruction/target parsers |
| Other | any text file | 60-line block chunker |

> **Security:** `.env`, `.gitignore`, `.npmrc`, `.yarnrc`, and other sensitive files are automatically skipped.

---

## Configuration

### Multiple Projects

KT Bot supports multiple projects simultaneously. Each project needs:
1. A `chroma_<projectname>/` folder with its ChromaDB
2. An entry in `PROJECT_CONFIG` for report generation

```env
PROJECT_CONFIG={"projectA": {"repo": "https://github.com/..."}, "projectB": {"repo": "https://..."}}
```

### Embedding Backend

Change the embedding model in `nlp_pipeline.py`:

```bash
# Use TF-IDF (no GPU, always available)
python nlp_pipeline.py --embedding_backend tfidf

# Use OpenAI embeddings (requires OPENAI_API_KEY)
python nlp_pipeline.py --embedding_backend openai

# Use Cohere (multilingual, requires COHERE_API_KEY)
python nlp_pipeline.py --embedding_backend cohere
```

### Skip Pipeline Steps

```bash
# Skip translation and spaCy NER (faster)
python nlp_pipeline.py --no_spacy --no_translate

# Skip embedding (if you'll load pre-computed vectors)
python nlp_pipeline.py --skip_steps embed
```

---

## Usage Guide

### Asking Questions

1. Log in and select a project from the dropdown
2. Type your question in **any language**
3. The bot detects your language, translates internally, retrieves relevant code, and answers
4. Use the **🇬🇧 English toggle** to see the English version of the response
5. Give **👍 or 👎** feedback — this trains the RL model to improve future results

### NLP Lens

After each answer, expand **"🔬 NLP Lens"** to see:
- How your query was tokenized and which entities were detected
- The NLP metadata of each retrieved code chunk (tags, complexity, docstring flag)
- A **faithfulness heatmap** — which answer words were actually grounded in the retrieved context

---

## Admin Features

Log in with username `admin` to access additional tabs:

| Tab | Description |
|---|---|
| 💬 **Chat** | Same as user chat |
| 📄 **Report Generator** | Generate a Markdown technical report from any configured GitHub repo |
| 📊 **Evaluator** | Score any query on faithfulness (1–5) and relevance (1–5) |
| 🧪 **RAGAS Test** | Batch-test multiple queries and see source coverage statistics |
| 🔬 **NLP Dashboard** | Full analytics: chunk type distribution, tag clouds, complexity heatmaps, embedding PCA |
| 📋 **Feedback** | View feedback log, metrics, and trigger RL model training |

### Training the RL Model

1. Use the chat and give 👍/👎 feedback on several answers (10+ recommended)
2. Go to **📋 Feedback** tab
3. Select a project and click **"🧠 Train RL Model for Selected Project"**
4. Future retrievals for that project will use the updated model

---

## API Reference

### KTChatEngine

```python
from genai.chat_engine import KTChatEngine

engine = KTChatEngine(groq_api_key="gsk_...")

result = engine.generate_response(
    user_query="How does login work?",   # any language
    project_name="myproject",            # chroma_myproject must exist
    user_role="Developer",               # or "Project Manager", "Data Scientist"
    respond_in_original_lang=True,       # translate answer back to user's language
)

# result keys:
result["answer"]            # response in user's language
result["answer_english"]    # response always in English
result["sources"]           # list of source file paths
result["final_hits"]        # retrieved chunk dicts (content + metadata + rl_score)
result["is_translated"]     # True if query was non-English
result["detected_lang"]     # ISO 639-1 code, e.g. "ta"
result["detected_lang_name"]# human name, e.g. "Tamil"
```

### KTRetriever

```python
from genai.retriever import KTRetriever

retriever = KTRetriever(db_path="chroma_myproject")
hits = retriever.search("How is authentication handled?", top_k=5)
# hits: List[Dict] with id, score, content, metadata
```

### RLAgent

```python
from genai.rl_agent import RLAgent

agent = RLAgent(model_path="genai/rl_model_myproject.pth")
score = agent.get_q_value(query_vec, doc_vec)   # float
loss  = agent.update(query_vec, doc_vec, reward) # reward: +1.0 or -1.0
```

### IngestionPipeline (CLI)

```bash
python ingestion.py \
  --repo_url https://github.com/user/repo \
  --output chunks.json \
  --max_file_size_kb 500
```

### NLPPipeline (CLI)

```bash
python nlp_pipeline.py \
  --input chunks.json \
  --output nlp_chunks.json \
  --embedding_backend sentence_transformers \
  --no_spacy \
  --no_translate \
  --skip_steps multilingual
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web UI** | Streamlit |
| **LLM** | Groq API (llama-3.1-8b-instant) |
| **Vector DB** | ChromaDB (persistent, cosine similarity) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2, 384-dim) |
| **RL Model** | PyTorch DQN (768→256→128→64→1) |
| **NER** | Rule-based regex + spaCy (en_core_web_sm) |
| **Language Detection** | langdetect + langid + Unicode heuristic |
| **Translation** | deep-translator (Google Translate) + argostranslate (offline) |
| **Code Parsing** | Python AST + Tree-sitter (JS/TS/Java/C/C++/Go/Rust) |
| **HTML Parsing** | BeautifulSoup4 |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Visualisation** | Plotly, Pandas |
| **Auth & Storage** | SQLite (users.db, rl_feedback.db) |
| **Config** | python-dotenv |

---

## Troubleshooting

**No projects found in chat dropdown**  
→ Make sure your ChromaDB folder is named `chroma_<projectname>` in the project root.

**"Chroma DB not found" error**  
→ Run the ingestion → NLP pipeline → load_chunks_to_chroma steps first.

**Translation not working**  
→ Install `pip install deep-translator`. Requires internet access.

**spaCy model missing**  
→ Run `python -m spacy download en_core_web_sm`.

**RL model giving poor results**  
→ Collect more feedback (10–20 samples minimum) before training.

**Embeddings are 0 or wrong dimension**  
→ Ensure `sentence-transformers` is installed: `pip install sentence-transformers`.
