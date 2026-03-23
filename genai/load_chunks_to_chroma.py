import json
from pathlib import Path
import chromadb
from tqdm import tqdm  # optional progress bar
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_JSON      = Path("../output/nlp_chunks.json")          # adjust path if needed
PERSIST_DIR     = Path("../chroma_visioncortex")       # where DB files saved
COLLECTION_NAME = "visioncortex_rag"                  # choose a meaningful name
BATCH_SIZE      = 200                                 # tune based on RAM (500–1000 ok for most)

# ─── Load your chunks ────────────────────────────────────────────────────────
print(f"Loading {INPUT_JSON} ...")
with INPUT_JSON.open(encoding="utf-8") as f:
    chunks = json.load(f)

print(f"→ Loaded {len(chunks)} chunks")

# ─── Prepare data for Chroma ─────────────────────────────────────────────────
ids        = []
documents  = []
metadatas  = []
embeddings = []

for chunk in tqdm(chunks, desc="Preparing data"):
    chunk_id = chunk["chunk_id"]
    
    text = chunk.get("cleaned_text") or chunk.get("content", "")
    if not text.strip():
        print(f"Skipping empty chunk: {chunk_id}")
        continue

    ids.append(chunk_id)
    documents.append(text)

    # Rich metadata — remove heavy / redundant fields
    meta = {
        k: v
        for k, v in chunk.items()
        if k not in [
            "cleaned_text", "content", "embedding", "tokens",
            "semantic_segments", "docstring"
        ]
    }
    
    # Flatten 'extra' if present
    if "extra" in meta:
        meta.update(meta.pop("extra", {}))
    
    # ─── FIX EMPTY LISTS FOR CHROMA ────────────────────────────────
    for key in list(meta.keys()):
        if isinstance(meta[key], list) and len(meta[key]) == 0:
            meta[key] = None   # or del meta[key]
    
    metadatas.append(meta)

    emb = chunk.get("embedding")
    if emb is None or len(emb) != 384:
        print(f"Warning: Invalid/missing embedding for {chunk_id} — skipping")
        continue
    embeddings.append(emb)

print(f"→ Prepared {len(ids)} valid items (with embeddings)")

id_counter = defaultdict(int)
fixed_ids = []

for i, original_id in enumerate(ids):
    id_counter[original_id] += 1
    if id_counter[original_id] == 1:
        fixed_ids.append(original_id)
    else:
        new_id = f"{original_id}_{id_counter[original_id]}"
        fixed_ids.append(new_id)
        print(f"Renamed duplicate {original_id} → {new_id}")

# Replace the original ids list with the fixed one
ids = fixed_ids

# print summary
dupe_count = sum(1 for v in id_counter.values() if v > 1)
print(f"After fixing duplicates: {len(ids)} items remain "
      f"({dupe_count} original IDs had duplicates)")

# ─── Connect & Create Collection ─────────────────────────────────────────────
client = chromadb.PersistentClient(path=str(PERSIST_DIR))

# Cosine is usually best for sentence-transformers normalized embeddings
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}   # alternatives: "l2", "ip"
)

# ─── Add in batches ──────────────────────────────────────────────────────────
print("Adding to Chroma in batches...")

for start in tqdm(range(0, len(ids), BATCH_SIZE)):
    end = start + BATCH_SIZE
    
    batch_ids       = ids[start:end]
    batch_docs      = documents[start:end]
    batch_metas     = metadatas[start:end]
    batch_embs      = embeddings[start:end]

    collection.add(
        ids=batch_ids,
        documents=batch_docs,       # required even with precomputed embs
        metadatas=batch_metas,
        embeddings=batch_embs,      # ← your pre-computed vectors
    )

print(f"Done! Collection '{COLLECTION_NAME}' now contains {collection.count()} items")
print(f"DB saved persistently at: {PERSIST_DIR.resolve()}")