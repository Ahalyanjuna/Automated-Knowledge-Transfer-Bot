# import chromadb
# from sentence_transformers import SentenceTransformer
# from typing import List, Dict

# class KTRetriever:
#     def __init__(self, db_path: str = "../chroma_visioncortex", collection_name: str = "visioncortex_rag"):
#         """
#         Initializes the search engine.
#         """
#         # 1. Load the same embedding model used in ingestion
#         self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
#         # 2. Connect to ChromaDB
#         self.client = chromadb.PersistentClient(path=db_path)
#         self.collection = self.client.get_collection(name=collection_name)

#     def search(self, query: str, top_k: int = 3) -> List[Dict]:
#         """
#         Takes a natural language question and returns the best code/doc snippets.
#         """
#         # Convert user question into numbers (vector)
#         query_vector = self.model.encode(query).tolist()

#         # Query the database
#         results = self.collection.query(
#             query_embeddings=[query_vector],
#             n_results=top_k,
#             include=["documents", "metadatas", "distances"]
#         )

#         # Format the output into a clean list of dictionaries
#         formatted_results = []
#         for i in range(len(results['ids'][0])):
#             formatted_results.append({
#                 "id": results['ids'][0][i],
#                 "score": results['distances'][0][i], # Lower is better
#                 "content": results['documents'][0][i],
#                 "metadata": results['metadatas'][0][i]
#             })
            
#         return formatted_results

#     def get_context_string(self, hits: List[Dict]) -> str:
#         """
#         Converts search hits into a clean string for the AI to read.
#         """
#         context_parts = []
#         for i, hit in enumerate(hits, 1):
#             file_name = hit['metadata'].get('source_file', 'Unknown File')
#             content = hit['content']
#             context_parts.append(f"--- SOURCE {i}: {file_name} ---\n{content}\n")
        
#         return "\n".join(context_parts)

# # --- Simple Test Block ---
# if __name__ == "__main__":
#     # Note: path is "." if running from inside genai folder, ".." if from root
#     retriever = KTRetriever(db_path="../chroma_visioncortex")
#     test_query = "How do I register a new face?"
    
#     hits = retriever.search(test_query)
    
#     print(f"\nFound {len(hits)} relevant snippets for: '{test_query}'")
#     for hit in hits:
#         print(f"\n[File: {hit['metadata'].get('source_file')}] (Score: {hit['score']:.4f})")
#         print(f"Snippet: {hit['content'][:150]}...")

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional


class KTRetriever:
    def __init__(self, db_path: str, collection_name: Optional[str] = None):
        """
        Initializes the retriever for a selected Chroma DB.
        If collection_name is not provided, the first available collection is used.
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=db_path)

        if collection_name:
            self.collection = self.client.get_collection(name=collection_name)
        else:
            collections = self.client.list_collections()
            if not collections:
                raise ValueError(f"No collections found in Chroma DB: {db_path}")
            self.collection = self.client.get_collection(name=collections[0].name)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Takes a natural language question and returns the best code/doc snippets.
        """
        query_vector = self.model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "score": results["distances"][0][i],  # lower is better
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })

        return formatted_results

    def get_context_string(self, hits: List[Dict]) -> str:
        """
        Converts search hits into a clean string for the AI to read.
        """
        context_parts = []
        for i, hit in enumerate(hits, 1):
            file_name = hit["metadata"].get("source_file", "Unknown File")
            content = hit["content"]
            context_parts.append(f"--- SOURCE {i}: {file_name} ---\n{content}\n")

        return "\n".join(context_parts)


if __name__ == "__main__":
    retriever = KTRetriever(db_path="../chroma_visioncortex")
    test_query = "How do I register a new face?"

    hits = retriever.search(test_query)

    print(f"\nFound {len(hits)} relevant snippets for: '{test_query}'")
    for hit in hits:
        print(f"\n[File: {hit['metadata'].get('source_file')}] (Score: {hit['score']:.4f})")
        print(f"Snippet: {hit['content'][:150]}...")