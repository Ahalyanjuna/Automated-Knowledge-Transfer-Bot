import sqlite3
import json
import numpy as np
import os
import plotly.express as px
from genai.rl_agent import RLAgent
from genai.retriever import KTRetriever


def train_from_db(project_name: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    feedback_db_path = os.path.join(project_root, "rl_feedback.db")
    chroma_db_path = os.path.join(project_root, f"chroma_{project_name}")
    model_path = os.path.join(current_dir, f"rl_model_{project_name}.pth")

    print(f"🧠 Loading RL Agent from: {model_path}")
    agent = RLAgent(model_path=model_path)

    if not os.path.exists(chroma_db_path):
        print(f"❌ Chroma DB not found for project: {project_name}")
        return

    print(f"🔍 Connecting to Chroma at: {chroma_db_path}")
    retriever = KTRetriever(db_path=chroma_db_path)

    if not os.path.exists(feedback_db_path):
        print("❌ No feedback database found. Go give some feedback in the chat first!")
        return

    conn = sqlite3.connect(feedback_db_path)
    cursor = conn.cursor()

    try:
        cols = [row[1] for row in cursor.execute("PRAGMA table_info(experiences)").fetchall()]
        has_project_col = "project_name" in cols

        if has_project_col:
            cursor.execute("""
                SELECT query_vector, selected_chunk_ids, reward
                FROM experiences
                WHERE project_name = ?
            """, (project_name,))
        else:
            # fallback for old DBs
            cursor.execute("""
                SELECT query_vector, selected_chunk_ids, reward
                FROM experiences
            """)

        rows = cursor.fetchall()

    except sqlite3.OperationalError as e:
        print(f"❌ DB error: {e}")
        conn.close()
        return

    if not rows:
        print(f"ℹ️ No feedback rows found for project '{project_name}'.")
        conn.close()
        return

    print(f"🚀 Feeding {len(rows)} experiences into the brain for project '{project_name}'...")

    for q_vec_json, chunk_ids_json, reward in rows:
        query_vec = np.array(json.loads(q_vec_json))
        chunk_ids = json.loads(chunk_ids_json)

        for cid in chunk_ids:
            try:
                res = retriever.collection.get(ids=[cid], include=["documents"])
                if res["documents"]:
                    doc_text = res["documents"][0]
                    doc_vec = retriever.model.encode(doc_text)
                    agent.update(query_vec, doc_vec, reward)
            except Exception as e:
                print(f"⚠️ Skipping chunk {cid}: {e}")

    print(f"✨ Training complete for project '{project_name}'. Saved model: {model_path}")
    conn.close()


if __name__ == "__main__":
    train_from_db("visioncortex")