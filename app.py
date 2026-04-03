import streamlit as st
from genai.chat_engine import KTChatEngine
from genai.auto_doc import AutoDocEngine
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# --- 1. Helper Functions ---

def get_live_stats():
    """Fetches stats directly from the SQLite file to bypass any session lag."""
    db_path = os.path.join(os.getcwd(), "rl_feedback.db")
    if not os.path.exists(db_path):
        return 0, 0.0
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), AVG(reward) FROM experiences")
        count, avg = cursor.fetchone()
        conn.close()
        return (count if count else 0), (avg if avg else 0.0)
    except:
        return 0, 0.0

def record_feedback(query, score):
    """Callback function that runs IMMEDIATELY on button click."""
    # We retrieve the engine from session state
    engine = st.session_state.engine
    # Encode and search to get context for RL logging
    q_vec = engine.retriever.model.encode(query).tolist()
    ids = [hit['id'] for hit in engine.retriever.search(query)]
    
    # Log to DB
    engine.logger.log_experience(query, q_vec, ids, score)
    st.toast(f"Feedback {score} recorded!")

# --- 2. Page Configuration & Styling ---
st.set_page_config(page_title="Universal KT Bot", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stChatInput { border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Engine Initialization ---

@st.cache_resource
def get_cached_engine(api_key):
    return KTChatEngine(api_key)

if "engine" not in st.session_state:
    st.session_state.engine = get_cached_engine(API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. Sidebar ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.info("Current Project: **VisionCortex**")
    
    st.divider()
    
    st.subheader("🛠️ Operations")
    if st.button("Generate Technical Manual"):
        with st.spinner("Analyzing project files..."):
            autodoc = AutoDocEngine(st.session_state.engine.client.api_key)
            status = autodoc.generate_full_manual("./")
            st.success(status)
            # Ensure path exists before downloading
            if os.path.exists("genai/TECHNICAL_GUIDE.md"):
                st.download_button("Download Manual", open("genai/TECHNICAL_GUIDE.md").read(), "TECHNICAL_GUIDE.md")

    st.divider()
    
    st.subheader("📈 RL Intelligence")
    # Fetch live stats every time the sidebar renders
    live_count, live_avg = get_live_stats()
    st.metric("Training Samples", live_count)
    st.metric("Avg Helpfulness", f"{live_avg:.2f}" if live_avg else "0.0")

    st.subheader("🧠 RL Brain Control")
    if st.button("Train RL Model"):
        with st.spinner("Teaching the brain from feedback..."):
            from genai.train_rl import train_from_db
            train_from_db() 
            st.success("Brain Updated!")
            st.rerun()

# --- 5. Main Chat UI ---
st.title("🧠 Automated Knowledge Transfer")
st.caption("Ask anything about the codebase. Powered by RAG + Reinforcement Learning.")

# Display Chat History
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.caption(f"Sources: {', '.join(message['sources'])}")
            
        # Display feedback buttons for assistant messages ONLY if it's the last message
        if message["role"] == "assistant" and i == len(st.session_state.messages) - 1:
            col1, col2 = st.columns([1, 8])
            # Find the original query for this response
            orig_query = st.session_state.messages[i-1]["content"] if i > 0 else ""
            
            with col1:
                st.button("👍", key=f"up_{i}", on_click=record_feedback, args=(orig_query, 1.0))
            with col2:
                st.button("👎", key=f"down_{i}", on_click=record_feedback, args=(orig_query, -1.0))

# Chat Input
if prompt := st.chat_input("Explain the database logic..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate Response
    result = st.session_state.engine.generate_response(prompt)
    
    # Store assistant message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": result["answer"], 
        "sources": result["sources"]
    })
    
    # Rerun to update chat display and show feedback buttons on the new message
    st.rerun()