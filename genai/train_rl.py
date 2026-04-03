import sqlite3
import json
import torch
import numpy as np
import os
# Fix imports for Streamlit compatibility
from genai.rl_agent import RLAgent
from genai.retriever import KTRetriever

def train_from_db():
    # 1. Setup Absolute Paths
    # This ensures we find the DBs whether running from root or genai/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    feedback_db_path = os.path.join(project_root, "rl_feedback.db")
    chroma_db_path = os.path.join(project_root, "chroma_visioncortex")
    model_path = os.path.join(current_dir, "rl_model.pth")

    # 2. Initialize Agent and Retriever
    print(f"🧠 Loading RL Agent from: {model_path}")
    agent = RLAgent(model_path=model_path)
    
    print(f"🔍 Connecting to Chroma at: {chroma_db_path}")
    retriever = KTRetriever(db_path=chroma_db_path)
    
    # 3. Connect to Feedback DB
    if not os.path.exists(feedback_db_path):
        print("❌ No feedback database found. Go give some feedback in the chat first!")
        return

    conn = sqlite3.connect(feedback_db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT query_vector, selected_chunk_ids, reward FROM experiences")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        print("❌ Table 'experiences' not found. Ask some questions in the app first!")
        conn.close()
        return

    if not rows:
        print("ℹ️ Internal DB is empty. No experiences to train on yet.")
        conn.close()
        return

    print(f"🚀 Feeding {len(rows)} experiences into the brain...")

    # 4. Training Loop
    for q_vec_json, chunk_ids_json, reward in rows:
        query_vec = np.array(json.loads(q_vec_json))
        chunk_ids = json.loads(chunk_ids_json)
        
        for cid in chunk_ids:
            # Fetch the actual document text from Chroma
            res = retriever.collection.get(ids=[cid], include=["documents"])
            if res["documents"]:
                doc_text = res["documents"][0]
                doc_vec = retriever.model.encode(doc_text)
                
                # Update the Neural Network
                agent.update(query_vec, doc_vec, reward)
    
    print("✨ Training complete! The RL Model has been updated and saved.")
    conn.close()

if __name__ == "__main__":
    train_from_db()