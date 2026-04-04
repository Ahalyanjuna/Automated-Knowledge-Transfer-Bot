# import streamlit as st
# from genai.chat_engine import KTChatEngine
# from genai.auto_doc import AutoDocEngine
# import os
# import sqlite3
# from dotenv import load_dotenv

# load_dotenv()
# API_KEY = os.getenv("GROQ_API_KEY")

# # --- 1. Helper Functions ---

# def get_live_stats():
#     """Fetches stats directly from the SQLite file to bypass any session lag."""
#     db_path = os.path.join(os.getcwd(), "rl_feedback.db")
#     if not os.path.exists(db_path):
#         return 0, 0.0
#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
#         cursor.execute("SELECT COUNT(*), AVG(reward) FROM experiences")
#         count, avg = cursor.fetchone()
#         conn.close()
#         return (count if count else 0), (avg if avg else 0.0)
#     except:
#         return 0, 0.0

# def record_feedback(query, score):
#     """Callback function that runs IMMEDIATELY on button click."""
#     # We retrieve the engine from session state
#     engine = st.session_state.engine
#     # Encode and search to get context for RL logging
#     q_vec = engine.retriever.model.encode(query).tolist()
#     ids = [hit['id'] for hit in engine.retriever.search(query)]
    
#     # Log to DB
#     engine.logger.log_experience(query, q_vec, ids, score)
#     st.toast(f"Feedback {score} recorded!")

# # --- 2. Page Configuration & Styling ---
# st.set_page_config(page_title="Universal KT Bot", layout="wide", page_icon="🧠")

# st.markdown("""
#     <style>
#     .main { background-color: #f5f7f9; }
#     .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
#     .stChatInput { border-radius: 20px; }
#     </style>
#     """, unsafe_allow_html=True)

# # --- 3. Engine Initialization ---

# @st.cache_resource
# def get_cached_engine(api_key):
#     return KTChatEngine(api_key)

# if "engine" not in st.session_state:
#     st.session_state.engine = get_cached_engine(API_KEY)

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # --- 4. Sidebar ---
# with st.sidebar:
#     st.title("⚙️ Control Panel")
#     st.info("Current Project: **VisionCortex**")
    
#     st.divider()
    
#     st.subheader("🛠️ Operations")
#     if st.button("Generate Technical Manual"):
#         with st.spinner("Analyzing project files..."):
#             autodoc = AutoDocEngine(st.session_state.engine.client.api_key)
#             status = autodoc.generate_full_manual("./")
#             st.success(status)
#             # Ensure path exists before downloading
#             if os.path.exists("genai/TECHNICAL_GUIDE.md"):
#                 st.download_button("Download Manual", open("genai/TECHNICAL_GUIDE.md").read(), "TECHNICAL_GUIDE.md")

#     st.divider()
    
#     st.subheader("📈 RL Intelligence")
#     # Fetch live stats every time the sidebar renders
#     live_count, live_avg = get_live_stats()
#     st.metric("Training Samples", live_count)
#     st.metric("Avg Helpfulness", f"{live_avg:.2f}" if live_avg else "0.0")

#     st.subheader("🧠 RL Brain Control")
#     if st.button("Train RL Model"):
#         with st.spinner("Teaching the brain from feedback..."):
#             from genai.train_rl import train_from_db
#             train_from_db() 
#             st.success("Brain Updated!")
#             st.rerun()

# # --- 5. Main Chat UI ---
# st.title("🧠 Automated Knowledge Transfer")
# st.caption("Ask anything about the codebase. Powered by RAG + Reinforcement Learning.")

# # Display Chat History
# for i, message in enumerate(st.session_state.messages):
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if "sources" in message:
#             st.caption(f"Sources: {', '.join(message['sources'])}")
            
#         # Display feedback buttons for assistant messages ONLY if it's the last message
#         if message["role"] == "assistant" and i == len(st.session_state.messages) - 1:
#             col1, col2 = st.columns([1, 8])
#             # Find the original query for this response
#             orig_query = st.session_state.messages[i-1]["content"] if i > 0 else ""
            
#             with col1:
#                 st.button("👍", key=f"up_{i}", on_click=record_feedback, args=(orig_query, 1.0))
#             with col2:
#                 st.button("👎", key=f"down_{i}", on_click=record_feedback, args=(orig_query, -1.0))

# # Chat Input
# if prompt := st.chat_input("Explain the database logic..."):
#     # Add user message
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Generate Response
#     result = st.session_state.engine.generate_response(prompt)
    
#     # Store assistant message
#     st.session_state.messages.append({
#         "role": "assistant", 
#         "content": result["answer"], 
#         "sources": result["sources"]
#     })
    
#     # Rerun to update chat display and show feedback buttons on the new message
#     st.rerun()

#-------------------------------------------------------------------
# import streamlit as st
# import sqlite3
# import hashlib
# import os
# import re
# from datetime import datetime
# from dotenv import load_dotenv

# load_dotenv()
# API_KEY = os.getenv("GROQ_API_KEY")

# # ─── DB Helpers ───────────────────────────────────────────────────────────────

# USERS_DB = os.path.join(os.getcwd(), "Bot_users.db")
# RL_DB    = os.path.join(os.getcwd(), "rl_feedback.db")

# def get_users_conn():
#     return sqlite3.connect(USERS_DB)

# def hash_password(pw: str) -> str:
#     return hashlib.sha256(pw.encode()).hexdigest()

# def authenticate(username: str, password: str):
#     """Returns user row or None."""
#     with get_users_conn() as conn:
#         row = conn.execute(
#             "SELECT id, name, username, position FROM users WHERE username=? AND password=?",
#             (username, hash_password(password))
#         ).fetchone()
#     return row  # (id, name, username, position)

# def register_user(name, email, username, position, password):
#     try:
#         with get_users_conn() as conn:
#             conn.execute(
#                 "INSERT INTO users (name, email, username, position, password, created_at) VALUES (?,?,?,?,?,?)",
#                 (name, email, username, position, hash_password(password), datetime.now().isoformat())
#             )
#             conn.commit()
#         return True, "✅ Account created!"
#     except sqlite3.IntegrityError:
#         return False, "⚠️ Username or email already exists."
#     except Exception as e:
#         return False, f"❌ Error: {e}"

# def get_live_stats():
#     if not os.path.exists(RL_DB):
#         return 0, 0.0
#     try:
#         with sqlite3.connect(RL_DB) as conn:
#             count, avg = conn.execute("SELECT COUNT(*), AVG(reward) FROM experiences").fetchone()
#         return (count or 0), (avg or 0.0)
#     except:
#         return 0, 0.0

# def record_feedback(query, score):
#     engine = st.session_state.engine
#     q_vec = engine.retriever.model.encode(query).tolist()
#     ids   = [hit['id'] for hit in engine.retriever.search(query)]
#     engine.logger.log_experience(query, q_vec, ids, score)
#     st.toast(f"{'👍' if score > 0 else '👎'} Feedback recorded!")

# # ─── Page Config ──────────────────────────────────────────────────────────────

# st.set_page_config(page_title="KT Bot", layout="wide", page_icon="🧠")

# # ─── Inject CSS ───────────────────────────────────────────────────────────────

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

# html, body, [class*="css"] {
#     font-family: 'DM Sans', sans-serif;
# }

# /* ── Auth page ── */
# .auth-wrapper {
#     max-width: 420px;
#     margin: 4rem auto;
#     padding: 2.5rem;
#     background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
#     border: 1px solid #2d2d5e;
#     border-radius: 18px;
#     box-shadow: 0 0 60px rgba(100,60,255,0.18);
# }
# .auth-logo {
#     font-family: 'Space Mono', monospace;
#     font-size: 2.2rem;
#     font-weight: 700;
#     color: #a78bfa;
#     text-align: center;
#     letter-spacing: -1px;
#     margin-bottom: .25rem;
# }
# .auth-sub {
#     text-align: center;
#     color: #6b7280;
#     font-size: .85rem;
#     margin-bottom: 2rem;
# }

# /* ── Chat area ── */
# .user-badge {
#     display: inline-flex;
#     align-items: center;
#     gap: .5rem;
#     background: #1e1e3f;
#     border: 1px solid #3b3b7a;
#     border-radius: 99px;
#     padding: .35rem .9rem;
#     font-size: .8rem;
#     color: #c4b5fd;
#     font-family: 'Space Mono', monospace;
# }

# /* ── Buttons ── */
# div.stButton > button {
#     border-radius: 8px;
#     font-weight: 500;
#     transition: all .2s;
# }
# div.stButton > button:hover {
#     transform: translateY(-1px);
# }

# /* ── Metrics ── */
# [data-testid="stMetric"] {
#     background: #0f0f23;
#     border: 1px solid #2d2d5e;
#     border-radius: 10px;
#     padding: .8rem 1rem;
# }
# </style>
# """, unsafe_allow_html=True)

# # ─── Session Init ─────────────────────────────────────────────────────────────

# for key, default in [
#     ("authenticated", False),
#     ("user", None),          # (id, name, username, position)
#     ("messages", []),
#     ("engine", None),
#     ("auth_tab", "login"),
# ]:
#     if key not in st.session_state:
#         st.session_state[key] = default

# # ─── Auth Page ────────────────────────────────────────────────────────────────

# def show_auth():
#     col = st.columns([1, 1.6, 1])[1]
#     with col:
#         st.markdown('<div class="auth-logo">🧠 KT Bot</div>', unsafe_allow_html=True)
#         st.markdown('<div class="auth-sub">Knowledge Transfer · RAG + RL</div>', unsafe_allow_html=True)

#         tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

#         # ── Login ──
#         with tab_login:
#             with st.form("login_form"):
#                 username = st.text_input("Username", placeholder="admin")
#                 password = st.text_input("Password", type="password", placeholder="••••••••")
#                 submitted = st.form_submit_button("Sign In", use_container_width=True)

#             if submitted:
#                 if not username or not password:
#                     st.warning("Fill in all fields.")
#                 else:
#                     row = authenticate(username, password)
#                     if row:
#                         st.session_state.authenticated = True
#                         st.session_state.user = row  # (id, name, username, position)
#                         st.session_state.messages = []
#                         st.rerun()
#                     else:
#                         st.error("Invalid credentials.")

#         # ── Register ──
#         with tab_register:
#             with st.form("reg_form"):
#                 r_name     = st.text_input("Full Name")
#                 r_email    = st.text_input("Email")
#                 r_username = st.text_input("Username")
#                 r_position = st.text_input("Position / Role", placeholder="e.g. ML Engineer")
#                 r_pass     = st.text_input("Password", type="password")
#                 r_pass2    = st.text_input("Confirm Password", type="password")
#                 reg_btn    = st.form_submit_button("Create Account", use_container_width=True)

#             if reg_btn:
#                 if not all([r_name, r_email, r_username, r_position, r_pass]):
#                     st.warning("All fields are required.")
#                 elif r_pass != r_pass2:
#                     st.error("Passwords don't match.")
#                 elif len(r_pass) < 6:
#                     st.warning("Password must be at least 6 characters.")
#                 else:
#                     ok, msg = register_user(r_name, r_email, r_username, r_position, r_pass)
#                     if ok:
#                         st.success(msg + " Please switch to Login.")
#                     else:
#                         st.error(msg)

# # ─── Engine Loader ────────────────────────────────────────────────────────────

# @st.cache_resource
# def get_cached_engine(api_key):
#     from genai.chat_engine import KTChatEngine
#     return KTChatEngine(api_key)

# def ensure_engine():
#     if st.session_state.engine is None:
#         with st.spinner("Initialising AI engine…"):
#             st.session_state.engine = get_cached_engine(API_KEY)

# # ─── Admin Page ───────────────────────────────────────────────────────────────

# def show_admin():
#     ensure_engine()
#     engine = st.session_state.engine
#     user   = st.session_state.user   # (id, name, username, position)

#     # ── Sidebar ──
#     with st.sidebar:
#         st.markdown(f"""
#         <div class="user-badge">👑 {user[2]} &nbsp;·&nbsp; Admin</div>
#         """, unsafe_allow_html=True)
#         st.caption(f"Logged in as **{user[1]}**")

#         st.divider()

#         st.subheader("⚙️ Operations")
#         if st.button("📄 Generate Technical Manual", use_container_width=True):
#             from genai.auto_doc import AutoDocEngine
#             with st.spinner("Analysing project files…"):
#                 autodoc = AutoDocEngine(engine.client.api_key)
#                 status  = autodoc.generate_full_manual("./")
#                 st.success(status)
#                 guide = "genai/TECHNICAL_GUIDE.md"
#                 if os.path.exists(guide):
#                     st.download_button("⬇️ Download Manual", open(guide).read(), "TECHNICAL_GUIDE.md")

#         st.divider()

#         st.subheader("📈 RL Dashboard")
#         live_count, live_avg = get_live_stats()
#         c1, c2 = st.columns(2)
#         c1.metric("Samples", live_count)
#         c2.metric("Avg Reward", f"{live_avg:.2f}")

#         if st.button("🧠 Train RL Model", use_container_width=True):
#             from genai.train_rl import train_from_db
#             with st.spinner("Training from feedback…"):
#                 train_from_db()
#                 st.success("Brain Updated!")
#                 st.rerun()

#         st.divider()

#         st.subheader("📋 Feedback Log")
#         _show_feedback_log()

#         st.divider()
#         if st.button("🚪 Logout", use_container_width=True):
#             _logout()

#     # ── Main ──
#     st.markdown('<span style="font-family:Space Mono;font-size:1.6rem;font-weight:700;color:#a78bfa">🧠 KT Bot</span> <span style="color:#6b7280;font-size:.85rem">— Admin View</span>', unsafe_allow_html=True)
#     st.caption("Full RAG + RL · ask anything about the codebase.")

#     _render_chat(is_admin=True)


# def _show_feedback_log():
#     """Compact table of last 10 feedback entries."""
#     if not os.path.exists(RL_DB):
#         st.caption("No feedback yet.")
#         return
#     try:
#         with sqlite3.connect(RL_DB) as conn:
#             rows = conn.execute(
#                 "SELECT query, reward, timestamp FROM experiences ORDER BY id DESC LIMIT 10"
#             ).fetchall()
#         if not rows:
#             st.caption("No feedback yet.")
#             return
#         for q, r, ts in rows:
#             icon = "👍" if r > 0 else "👎"
#             short_q = (q[:35] + "…") if len(q) > 38 else q
#             st.markdown(f"`{icon}` {short_q}")
#     except:
#         st.caption("Could not load log.")

# # ─── User Page ────────────────────────────────────────────────────────────────

# def show_user():
#     ensure_engine()
#     user = st.session_state.user  # (id, name, username, position)

#     with st.sidebar:
#         st.markdown(f"""
#         <div class="user-badge">👤 {user[2]}</div>
#         """, unsafe_allow_html=True)
#         st.caption(f"**{user[1]}** · {user[3]}")
#         st.divider()
#         st.info("💡 Ask anything about the codebase. Your feedback helps improve the bot!")
#         st.divider()
#         if st.button("🚪 Logout", use_container_width=True):
#             _logout()

#     st.markdown('<span style="font-family:Space Mono;font-size:1.6rem;font-weight:700;color:#a78bfa">🧠 KT Bot</span>', unsafe_allow_html=True)
#     st.caption(f"Welcome, **{user[1]}** · {user[3]}")

#     _render_chat(is_admin=False)

# # ─── Shared Chat Renderer ─────────────────────────────────────────────────────

# def _render_chat(is_admin: bool):
#     engine = st.session_state.engine

#     # Display history
#     for i, message in enumerate(st.session_state.messages):
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#             if "sources" in message:
#                 st.caption(f"📂 Sources: {', '.join(message['sources'])}")

#             # Feedback buttons on the last assistant message
#             if message["role"] == "assistant" and i == len(st.session_state.messages) - 1:
#                 orig_query = st.session_state.messages[i - 1]["content"] if i > 0 else ""
#                 col1, col2, col3 = st.columns([1, 1, 10])
#                 with col1:
#                     st.button("👍", key=f"up_{i}", on_click=record_feedback, args=(orig_query, 1.0))
#                 with col2:
#                     st.button("👎", key=f"down_{i}", on_click=record_feedback, args=(orig_query, -1.0))
#                 if is_admin:
#                     with col3:
#                         st.caption(f"RL score shown in console · query saved to feedback DB")

#     # Input
#     if prompt := st.chat_input("Ask about the codebase…"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.spinner("Thinking…"):
#             result = engine.generate_response(prompt)
#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": result["answer"],
#             "sources": result["sources"],
#         })
#         st.rerun()

# # ─── Helpers ──────────────────────────────────────────────────────────────────

# def _logout():
#     st.session_state.authenticated = False
#     st.session_state.user          = None
#     st.session_state.messages      = []
#     st.session_state.engine        = None
#     st.rerun()

# # ─── Router ───────────────────────────────────────────────────────────────────

# if not st.session_state.authenticated:
#     show_auth()
# else:
#     user = st.session_state.user  # (id, name, username, position)
#     if user[2] == "admin":        # username == "admin"
#         show_admin()
#     else:
#         show_user()

import streamlit as st
import sqlite3
import hashlib
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# ─── DB Helpers ───────────────────────────────────────────────────────────────

USERS_DB = os.path.join(os.getcwd(), "Bot_users.db")
RL_DB = os.path.join(os.getcwd(), "rl_feedback.db")


def get_users_conn():
    return sqlite3.connect(USERS_DB)


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def authenticate(username: str, password: str):
    """Returns user row or None."""
    with get_users_conn() as conn:
        row = conn.execute(
            "SELECT id, name, username, position FROM users WHERE username=? AND password=?",
            (username, hash_password(password))
        ).fetchone()
    return row  # (id, name, username, position)


def register_user(name, email, username, position, password):
    try:
        with get_users_conn() as conn:
            conn.execute(
                "INSERT INTO users (name, email, username, position, password, created_at) VALUES (?,?,?,?,?,?)",
                (name, email, username, position, hash_password(password), datetime.now().isoformat())
            )
            conn.commit()
        return True, "✅ Account created!"
    except sqlite3.IntegrityError:
        return False, "⚠️ Username or email already exists."
    except Exception as e:
        return False, f"❌ Error: {e}"


def get_available_projects():
    """
    Finds all folders in the project root named like chroma_projectname
    and returns ['projectname', ...]
    """
    root = os.getcwd()
    projects = []

    for name in os.listdir(root):
        full_path = os.path.join(root, name)
        if os.path.isdir(full_path) and name.startswith("chroma_"):
            project_name = name.replace("chroma_", "", 1)
            projects.append(project_name)

    return sorted(projects)


def get_live_stats(project_name=None):
    if not os.path.exists(RL_DB):
        return 0, 0.0

    try:
        with sqlite3.connect(RL_DB) as conn:
            # Check if project_name column exists
            cols = [row[1] for row in conn.execute("PRAGMA table_info(experiences)").fetchall()]
            has_project_col = "project_name" in cols

            if project_name and has_project_col:
                count, avg = conn.execute(
                    "SELECT COUNT(*), AVG(reward) FROM experiences WHERE project_name=?",
                    (project_name,)
                ).fetchone()
            else:
                count, avg = conn.execute(
                    "SELECT COUNT(*), AVG(reward) FROM experiences"
                ).fetchone()

        return (count or 0), (avg or 0.0)
    except Exception:
        return 0, 0.0


def record_feedback(query, score, project_name):
    engine = st.session_state.engine

    try:
        retriever = engine._get_retriever(project_name)
        q_vec = retriever.model.encode(query).tolist()
        ids = [hit["id"] for hit in retriever.search(query)]

        engine.logger.log_experience(project_name, query, q_vec, ids, score)
        st.toast(f"{'👍' if score > 0 else '👎'} Feedback recorded for {project_name}!")
    except Exception as e:
        st.error(f"Failed to record feedback: {e}")


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="KT Bot", layout="wide", page_icon="🧠")

# ─── Inject CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Auth page ── */
.auth-wrapper {
    max-width: 420px;
    margin: 4rem auto;
    padding: 2.5rem;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
    border: 1px solid #2d2d5e;
    border-radius: 18px;
    box-shadow: 0 0 60px rgba(100,60,255,0.18);
}
.auth-logo {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #a78bfa;
    text-align: center;
    letter-spacing: -1px;
    margin-bottom: .25rem;
}
.auth-sub {
    text-align: center;
    color: #6b7280;
    font-size: .85rem;
    margin-bottom: 2rem;
}

/* ── Chat area ── */
.user-badge {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    background: #1e1e3f;
    border: 1px solid #3b3b7a;
    border-radius: 99px;
    padding: .35rem .9rem;
    font-size: .8rem;
    color: #c4b5fd;
    font-family: 'Space Mono', monospace;
}

/* ── Buttons ── */
div.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all .2s;
}
div.stButton > button:hover {
    transform: translateY(-1px);
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #0f0f23;
    border: 1px solid #2d2d5e;
    border-radius: 10px;
    padding: .8rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Session Init ─────────────────────────────────────────────────────────────

for key, default in [
    ("authenticated", False),
    ("user", None),                # (id, name, username, position)
    ("messages", []),
    ("engine", None),
    ("auth_tab", "login"),
    ("selected_project", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Auth Page ────────────────────────────────────────────────────────────────

def show_auth():
    col = st.columns([1, 1.6, 1])[1]
    with col:
        st.markdown('<div class="auth-logo">🧠 KT Bot</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Knowledge Transfer · RAG + RL</div>', unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

        # ── Login ──
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="admin")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.warning("Fill in all fields.")
                else:
                    row = authenticate(username, password)
                    if row:
                        st.session_state.authenticated = True
                        st.session_state.user = row
                        st.session_state.messages = []
                        projects = get_available_projects()
                        st.session_state.selected_project = projects[0] if projects else None
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")

        # ── Register ──
        with tab_register:
            with st.form("reg_form"):
                r_name = st.text_input("Full Name")
                r_email = st.text_input("Email")
                r_username = st.text_input("Username")
                r_position = st.text_input("Position / Role", placeholder="e.g. Developer / Project Manager / Data Scientist")
                r_pass = st.text_input("Password", type="password")
                r_pass2 = st.text_input("Confirm Password", type="password")
                reg_btn = st.form_submit_button("Create Account", use_container_width=True)

            if reg_btn:
                if not all([r_name, r_email, r_username, r_position, r_pass]):
                    st.warning("All fields are required.")
                elif r_pass != r_pass2:
                    st.error("Passwords don't match.")
                elif len(r_pass) < 6:
                    st.warning("Password must be at least 6 characters.")
                else:
                    ok, msg = register_user(r_name, r_email, r_username, r_position, r_pass)
                    if ok:
                        st.success(msg + " Please switch to Login.")
                    else:
                        st.error(msg)

# ─── Engine Loader ────────────────────────────────────────────────────────────

@st.cache_resource
def get_cached_engine(api_key):
    from genai.chat_engine import KTChatEngine
    return KTChatEngine(api_key)


def ensure_engine():
    if st.session_state.engine is None:
        with st.spinner("Initialising AI engine…"):
            st.session_state.engine = get_cached_engine(API_KEY)

# ─── Admin Page ───────────────────────────────────────────────────────────────

def _render_admin_evaluator():
    st.subheader("📊 Single Query Evaluator")

    projects = get_available_projects()

    project_name = st.selectbox(
        "Project",
        projects,
        key="eval_project"
    )

    user_role = st.selectbox(
    "Test as role",
    ["Developer", "Project Manager", "Data Scientist"],
    key="eval_role"
    )

    user = st.session_state.user
    user_role = user[3] if user and len(user) > 3 else "Developer"

    eval_query = st.text_input(
        "Enter a query to evaluate",
        placeholder="Ask a question to test faithfulness and relevance..."
    )

    if st.button("Run Evaluator", use_container_width=True, key="run_evaluator_btn"):
        if not eval_query.strip():
            st.warning("Please enter a query.")
            return

        try:
            from genai.evaluator import KTEvaluator

            evaluator = KTEvaluator(
                api_key=API_KEY,
                project_name=project_name,
                user_role=user_role
            )

            with st.spinner("Running evaluator..."):
                result = evaluator.ask_and_evaluate(eval_query)

            st.markdown("### Answer")
            st.write(result["answer"])

            st.markdown("### Sources")
            if result["sources"]:
                st.write(", ".join(result["sources"]))
            else:
                st.write("No sources returned.")

            eval_data = result["evaluation"]

            c1, c2 = st.columns(2)
            c1.metric("Faithfulness", eval_data.get("faithfulness", "N/A"))
            c2.metric("Relevance", eval_data.get("relevance", "N/A"))

            st.markdown("### Reason")
            st.write(eval_data.get("reason", "No reason returned."))

        except Exception as e:
            st.error(f"Evaluator failed: {e}")


def _render_admin_ragas():
    st.subheader("🧪 RAGAS-Lite Test")

    projects = get_available_projects()

    project_name = st.selectbox(
        "Project",
        projects,
        key="ragas_project"
    )
 
    user_role = st.selectbox(
    "Test as role",
    ["Developer", "Project Manager", "Data Scientist"],
    key="ragas_role"
)

    st.caption("Enter one query per line, or leave blank to run the default golden questions.")

    query_block = st.text_area(
        "Test queries",
        placeholder="What database is used?\nHow is login handled?\nWhat happens if logs folder is missing?",
        height=180
    )

    if st.button("Run RAGAS Test", use_container_width=True, key="run_ragas_btn"):
        try:
            from genai.ragas_test import run_ragas_lite_test

            custom_queries = [q.strip() for q in query_block.splitlines() if q.strip()]

            with st.spinner("Running RAGAS-Lite test..."):
                df = run_ragas_lite_test(
                    project_name=project_name,
                    user_role=user_role,
                    test_queries=custom_queries if custom_queries else None
                )

            st.markdown("### Test Results")
            st.dataframe(df, use_container_width=True)

            if not df.empty:
                min_sources = df["Source_Count"].min()
                if min_sources > 0:
                    st.success("All test queries returned at least one source.")
                else:
                    st.warning("Some test queries returned zero sources.")

        except Exception as e:
            st.error(f"RAGAS test failed: {e}")

def _render_admin_feedback():
    st.subheader("📋 Project Feedback & RL Training")

    projects = get_available_projects()
    if not projects:
        st.warning("No chroma_* project folders found in the root directory.")
        return

    feedback_project = st.selectbox(
        "Project",
        projects,
        key="feedback_project"
    )

    live_count, live_avg = get_live_stats(feedback_project)

    c1, c2 = st.columns(2)
    c1.metric("Samples", live_count)
    c2.metric("Avg Reward", f"{live_avg:.2f}")

    st.markdown("### Recent Feedback")
    _show_feedback_log(feedback_project)

    st.markdown("### RL Training")
    if st.button("🧠 Train RL Model for Selected Project", use_container_width=True, key="train_feedback_project_btn"):
        from genai.train_rl import train_from_db

        with st.spinner(f"Training RL model for {feedback_project}..."):
            train_from_db(feedback_project)

            # Optional: clear cached RL agent for this project so the next query reloads the new model
            engine = st.session_state.engine
            if engine is not None and hasattr(engine, "rl_agent_cache"):
                engine.rl_agent_cache.pop(feedback_project, None)

            st.success(f"RL model trained for project: {feedback_project}")
            st.rerun()

def show_admin():
    # ensure_engine()
    # engine = st.session_state.engine
    # user = st.session_state.user

    # with st.sidebar:
    #     st.markdown(f"""
    #     <div class="user-badge">👑 {user[2]} &nbsp;·&nbsp; Admin</div>
    #     """, unsafe_allow_html=True)
    #     st.caption(f"Logged in as **{user[1]}**")


    #     st.subheader("⚙️ Operations")
    #     if st.button("📄 Generate Technical Manual", use_container_width=True):
    #         from genai.auto_doc import AutoDocEngine
    #         with st.spinner("Analysing project files…"):
    #             autodoc = AutoDocEngine(engine.client.api_key)
    #             status = autodoc.generate_full_manual("./")
    #             st.success(status)
    #             guide = "TECHNICAL_GUIDE.md"
    #             if os.path.exists(guide):
    #                 with open(guide, "r", encoding="utf-8") as f:
    #                     st.download_button("⬇️ Download Manual", f.read(), "TECHNICAL_GUIDE.md")

    #     st.divider()

    #     st.subheader("📈 RL Dashboard")
    #     live_count, live_avg = get_live_stats(st.session_state.selected_project)
    #     c1, c2 = st.columns(2)
    #     c1.metric("Samples", live_count)
    #     c2.metric("Avg Reward", f"{live_avg:.2f}")

    #     if st.button("🧠 Train RL Model", use_container_width=True):
    #         from genai.train_rl import train_from_db
    #         selected_project = st.session_state.get("selected_project")

    #         if not selected_project:
    #             st.warning("Please select a project first.")
    #         else:
    #             with st.spinner(f"Training from feedback for {selected_project}…"):
    #                 train_from_db(selected_project)
    #                 st.success(f"Brain updated for {selected_project}!")
    #                 st.rerun()

    #     st.divider()

    #     st.subheader("📋 Feedback Log")
    #     _show_feedback_log(st.session_state.selected_project)

    #     st.divider()
    #     if st.button("🚪 Logout", use_container_width=True):
    #         _logout()

    # st.markdown(
    #     '<span style="font-family:Space Mono;font-size:1.6rem;font-weight:700;color:#a78bfa">🧠 KT Bot</span> <span style="color:#6b7280;font-size:.85rem">— Admin View</span>',
    #     unsafe_allow_html=True
    # )
    # st.caption("Full RAG + RL · ask anything about the selected project codebase.")

    # tab_chat, tab_eval, tab_ragas = st.tabs(["💬 Chat", "📊 Evaluator", "🧪 RAGAS Test"])

    # with tab_chat:
    #     _render_chat(is_admin=True)

    # with tab_eval:
    #     _render_admin_evaluator()

    # with tab_ragas:
    #     _render_admin_ragas()
    ensure_engine()
    user = st.session_state.user

    with st.sidebar:
        st.markdown(f"""
        <div class="user-badge">👑 {user[2]} &nbsp;·&nbsp; Admin</div>
        """, unsafe_allow_html=True)
        st.caption(f"Logged in as **{user[1]}**")

        st.divider()
        st.info("Admin tools are available in separate tabs: Chat, Evaluator, RAGAS Test, and Feedback.")
        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            _logout()

    st.markdown(
        '<span style="font-family:Space Mono;font-size:1.6rem;font-weight:700;color:#a78bfa">🧠 KT Bot</span> <span style="color:#6b7280;font-size:.85rem">— Admin View</span>',
        unsafe_allow_html=True
    )
    st.caption("Full RAG + RL · ask, evaluate, test, and manage project-specific feedback.")

    tab_chat, tab_eval, tab_ragas, tab_feedback = st.tabs(
        ["💬 Chat", "📊 Evaluator", "🧪 RAGAS Test", "📋 Feedback"]
    )

    with tab_chat:
        _render_chat(is_admin=True)

    with tab_eval:
        _render_admin_evaluator()

    with tab_ragas:
        _render_admin_ragas()

    with tab_feedback:
        _render_admin_feedback()


def _show_feedback_log(project_name=None):
    if not os.path.exists(RL_DB):
        st.caption("No feedback yet.")
        return

    try:
        with sqlite3.connect(RL_DB) as conn:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(experiences)").fetchall()]
            has_project_col = "project_name" in cols

            if project_name and has_project_col:
                rows = conn.execute(
                    "SELECT query, reward, timestamp FROM experiences WHERE project_name=? ORDER BY id DESC LIMIT 10",
                    (project_name,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT query, reward, timestamp FROM experiences ORDER BY id DESC LIMIT 10"
                ).fetchall()

        if not rows:
            st.caption("No feedback yet.")
            return

        for q, r, ts in rows:
            icon = "👍" if r > 0 else "👎"
            short_q = (q[:35] + "…") if len(q) > 38 else q
            st.markdown(f"`{icon}` {short_q}")

    except Exception:
        st.caption("Could not load log.")

# ─── User Page ────────────────────────────────────────────────────────────────

def show_user():
    ensure_engine()
    user = st.session_state.user

    with st.sidebar:
        st.markdown(f"""
        <div class="user-badge">👤 {user[2]}</div>
        """, unsafe_allow_html=True)
        st.caption(f"**{user[1]}** · {user[3]}")

        # st.divider()

        # projects = get_available_projects()
        # if projects:
        #     if st.session_state.selected_project not in projects:
        #         st.session_state.selected_project = projects[0]

        #     st.session_state.selected_project = st.selectbox(
        #         "Project",
        #         options=projects,
        #         index=projects.index(st.session_state.selected_project),
        #         help="Choose which project knowledge base to ask about"
        #     )
        # else:
        #     st.warning("No chroma_* project folders found in the root directory.")

        # st.divider()
        st.info("💡 Ask anything about the selected codebase. Your feedback helps improve the bot.")
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            _logout()

    st.markdown('<span style="font-family:Space Mono;font-size:1.6rem;font-weight:700;color:#a78bfa">🧠 KT Bot</span>', unsafe_allow_html=True)
    st.caption(f"Welcome, **{user[1]}** · {user[3]}")

    _render_chat(is_admin=False)

# ─── Shared Chat Renderer ─────────────────────────────────────────────────────

def _render_chat(is_admin: bool):
    engine = st.session_state.engine
    user = st.session_state.user
    user_role = user[3] if user and len(user) > 3 else "User"

    # Display history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "sources" in message and message["sources"]:
                st.caption(f"📂 Sources: {', '.join(message['sources'])}")

            if "project" in message and message["project"]:
                st.caption(f"🧩 Project: {message['project']}")

            if message["role"] == "assistant" and i == len(st.session_state.messages) - 1:
                orig_query = st.session_state.messages[i - 1]["content"] if i > 0 else ""
                project_name = message.get("project")
                col1, col2, col3 = st.columns([1, 1, 10])

                with col1:
                    st.button(
                        "👍",
                        key=f"up_{i}",
                        on_click=record_feedback,
                        args=(orig_query, 1.0, project_name)
                    )

                with col2:
                    st.button(
                        "👎",
                        key=f"down_{i}",
                        on_click=record_feedback,
                        args=(orig_query, -1.0, project_name)
                    )

                if is_admin:
                    with col3:
                        st.caption("RL score shown in console · project-aware feedback saved to DB")

    st.divider()

    projects = get_available_projects()
    if not projects:
        st.error("No project vector DBs found. Expected folders like chroma_projectname in the root.")
        return

    if st.session_state.selected_project not in projects:
        st.session_state.selected_project = projects[0]

    with st.form("project_chat_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 6])

        with col1:
            selected_project = st.selectbox(
                "Project",
                options=projects,
                index=projects.index(st.session_state.selected_project),
                help="Choose the project knowledge base to query"
            )

        with col2:
            prompt = st.text_input(
                "Ask your question",
                placeholder="Ask about the selected project..."
            )

        submitted = st.form_submit_button("Ask", use_container_width=True)

    st.session_state.selected_project = selected_project

    st.caption(f"Role-aware mode: **{user_role}** · Selected project: **{selected_project}**")

    if submitted:
        if not prompt.strip():
            st.warning("Please enter a question.")
            return

        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "project": selected_project
        })

        with st.spinner("Thinking…"):
            result = engine.generate_response(
                user_query=prompt,
                project_name=selected_project,
                user_role=user_role
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "project": selected_project
        })

        st.rerun()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _logout():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.messages = []
    st.session_state.engine = None
    st.session_state.selected_project = None
    st.rerun()

# ─── Router ───────────────────────────────────────────────────────────────────

if not st.session_state.authenticated:
    show_auth()
else:
    user = st.session_state.user
    if user[2] == "admin":
        show_admin()
    else:
        show_user()
