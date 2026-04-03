import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize the same embedding model used during ingestion
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to the local ChromaDB
# Ensure this path matches the PERSIST_DIR in your loading script
db_path = "./chroma_visioncortex" 
client = chromadb.PersistentClient(path=db_path)

try:
    collection = client.get_collection(name="visioncortex_rag")
    print(f"✅ Success! Connected to collection: visioncortex_rag")
    print(f"📊 Total items in DB: {collection.count()}")

    # 3. Perform a test query
    query = "How does the face recognition system work?"
    query_vector = model.encode(query).tolist()

    print(f"\n🔍 Searching for: '{query}'...")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2,
        include=["documents", "metadatas", "distances"]
    )

    # 4. Display Results
    for i in range(len(results['ids'][0])):
        print(f"\n--- Result #{i+1} (Distance: {results['distances'][0][i]:.4f}) ---")
        print(f"ID: {results['ids'][0][i]}")
        print(f"File: {results['metadatas'][0][i].get('source_file', 'N/A')}")
        print(f"Snippet: {results['documents'][0][i][:200]}...")

except Exception as e:
    print(f"❌ Error: {e}")