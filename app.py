import streamlit as st
import sqlite3
import hashlib
import os
from datetime import datetime
from dotenv import load_dotenv
import json

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

USERS_DB = os.path.join(os.getcwd(), os.getenv("USERS_DB"))
RL_DB = os.path.join(os.getcwd(), os.getenv("RL_DB"))

PROJECT_CONFIG = json.loads(os.getenv("PROJECT_CONFIG"))

# ── NEW: import NLP dashboard + NLP Lens ──────────────────────────────────────
from genai.nlp_dashboard import render_nlp_dashboard
from genai.nlp_lens import render_nlp_lens, render_nlp_mini_badge


def get_users_conn():
    return sqlite3.connect(USERS_DB)


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def authenticate(username: str, password: str):
    with get_users_conn() as conn:
        row = conn.execute(
            "SELECT id, name, username, position FROM users WHERE username=? AND password=?",
            (username, hash_password(password))
        ).fetchone()
    return row


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
    root = os.getcwd()
    projects = []
    for name in os.listdir(root):
        full_path = os.path.join(root, name)
        if os.path.isdir(full_path) and name.startswith("chroma_"):
            project_name = name.replace("chroma_", "", 1)
            projects.append(project_name)
    return sorted(projects)

def get_report_projects():
    return sorted(PROJECT_CONFIG.keys())


def get_repo_url(project_name: str) -> str | None:
    cfg = PROJECT_CONFIG.get(project_name, {})
    return cfg.get("repo")


def get_generated_report_path(project_name: str, is_admin: bool) -> str:
    suffix = "admin" if is_admin else "user"
    return os.path.join("genai", f"TECHNICAL_GUIDE_{project_name}_{suffix}.md")

def get_live_stats(project_name=None):
    if not os.path.exists(RL_DB):
        return 0, 0.0
    try:
        with sqlite3.connect(RL_DB) as conn:
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

/* ── Translation display boxes ── */
.translation-box {
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    line-height: 1.6;
}
.original-query-box {
    background: rgba(167, 139, 250, 0.08);
    border: 1px solid rgba(167, 139, 250, 0.25);
    color: #c4b5fd;
}
.translated-query-box {
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.25);
    color: #6ee7b7;
}
.lang-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.1rem 0.5rem;
    border-radius: 99px;
    margin-right: 0.4rem;
    letter-spacing: 0.05em;
}
.lang-badge-original {
    background: rgba(167, 139, 250, 0.2);
    color: #a78bfa;
}
.lang-badge-english {
    background: rgba(52, 211, 153, 0.2);
    color: #34d399;
}
.response-lang-note {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 0.3rem;
    font-style: italic;
}

div.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all .2s;
}
div.stButton > button:hover {
    transform: translateY(-1px);
}

/* Disable typing cursor in all selectboxes — dropdown only, pointer cursor */
[data-baseweb="select"] input {
    pointer-events: none !important;
    caret-color: transparent !important;
    cursor: pointer !important;
}
[data-baseweb="select"] {
    cursor: pointer !important;
}
[data-baseweb="select"] * {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session Init ─────────────────────────────────────────────────────────────

for key, default in [
    ("authenticated", False),
    ("user", None),
    ("messages", []),
    ("engine", None),
    ("auth_tab", "login"),
    ("selected_project", None),
    ("response_in_original_lang", True),
    # NEW: stores last result dict so NLP Lens can re-render from history
    ("last_nlp_result", None),
    ("last_nlp_hits", []),
    ("last_nlp_query", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Auth Page ────────────────────────────────────────────────────────────────

def show_auth():
    col = st.columns([1, 1.6, 1])[1]
    with col:
        st.markdown('<div class="auth-logo">🧠 KT Bot</div>', unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

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

# ─── NLP Dashboard Helper ─────────────────────────────────────────────────────

def _render_nlp_dashboard_tab(project_name: str):
    """Wrapper that resolves paths and calls render_nlp_dashboard."""
    db_path = os.path.join(os.getcwd(), f"chroma_{project_name}")
    render_nlp_dashboard(
        project_name=project_name,
        db_path=db_path,
        rl_db=RL_DB,
    )

# ─── Report Generator ────────────────────────────────────────────────────────

def _render_report_generator(is_admin: bool):
    st.subheader("📄 Report Generator")

    projects = get_report_projects()
    if not projects:
        st.warning("No projects configured for report generation.")
        return

    report_project = st.selectbox(
        "Project",
        projects,
        key="report_project_admin" if is_admin else "report_project_user"
    )

    user = st.session_state.user
    audience_role = user[3] if user and len(user) > 3 else ("admin" if is_admin else "user")
    repo_url = get_repo_url(report_project)

    if st.button(
        "Generate Report",
        use_container_width=True,
        key="generate_report_admin" if is_admin else "generate_report_user"
    ):
        if not repo_url:
            st.error("No repository URL configured for the selected project.")
            return

        try:
            from genai.auto_doc import AutoDocEngine

            output_path = get_generated_report_path(report_project, is_admin)

            with st.spinner("Generating report from repository..."):
                autodoc = AutoDocEngine(API_KEY)
                ok, msg, report_text = autodoc.generate_from_github(
                    github_url=repo_url,
                    audience_role=audience_role
                )

            if ok and report_text:
                st.success(msg)
                st.markdown("### Preview")
                st.markdown(report_text)
                st.download_button(
                    "⬇️ Download Report",
                    data=report_text,
                    file_name=f"{report_project}_report.md",
                    mime="text/markdown"
                )
            else:
                st.error(msg)

        except Exception as e:
            st.error(f"Failed to generate report: {e}")

# ─── Admin-only sub-tabs ─────────────────────────────────────────────────────

def _render_admin_evaluator():
    st.subheader("📊 Single Query Evaluator")

    projects = get_available_projects()

    project_name = st.selectbox("Project", projects, key="eval_project")
    user_role = st.selectbox("Test as role", ["Developer", "Project Manager", "Data Scientist"], key="eval_role")
    eval_query = st.text_input("Enter a query to evaluate", placeholder="Ask a question to test faithfulness and relevance...")

    if st.button("Run Evaluator", use_container_width=True, key="run_evaluator_btn"):
        if not eval_query.strip():
            st.warning("Please enter a query.")
            return

        try:
            from genai.evaluator import KTEvaluator

            evaluator = KTEvaluator(api_key=API_KEY, project_name=project_name, user_role=user_role)

            with st.spinner("Running evaluator..."):
                result = evaluator.ask_and_evaluate(eval_query)

            # ── Scores row ────────────────────────────────────────────────
            eval_data = result["evaluation"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Faithfulness", f"{eval_data.get('faithfulness', 'N/A')} / 5")
            c2.metric("Relevance",    f"{eval_data.get('relevance', 'N/A')} / 5")
            avg = None
            try:
                avg = round((int(eval_data.get("faithfulness", 0)) + int(eval_data.get("relevance", 0))) / 2, 1)
            except Exception:
                pass
            if avg is not None:
                c3.metric("Overall Avg", f"{avg} / 5")

            # ── Evaluation reason ─────────────────────────────────────────
            st.markdown("**🧠 Evaluation Reasoning**")
            st.info(eval_data.get("reason", "No reason returned."))

            # ── Answer ────────────────────────────────────────────────────
            st.markdown("**💬 Answer**")
            st.markdown(result["answer"])

            # ── Sources with details ──────────────────────────────────────
            st.markdown("**📂 Retrieved Sources**")
            sources = result.get("sources", [])
            final_hits = result.get("final_hits", [])

            if final_hits:
                for i, hit in enumerate(final_hits):
                    fname = hit["metadata"].get("source_file", sources[i] if i < len(sources) else "Unknown")
                    ctype = hit["metadata"].get("chunk_type", "—")
                    lang  = hit["metadata"].get("normalized_lang") or hit["metadata"].get("language", "—")
                    tokens = hit["metadata"].get("token_count", "—")
                    tags_raw = hit["metadata"].get("tags", "[]")
                    try:
                        import ast
                        tags = ast.literal_eval(tags_raw) if isinstance(tags_raw, str) else tags_raw
                    except Exception:
                        tags = []
                    rl_score = hit.get("rl_score", None)

                    with st.expander(f"Source {i+1}: `{fname.split('/')[-1]}`", expanded=(i == 0)):
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.caption(f"**Type:** {ctype}")
                        sc2.caption(f"**Lang:** {lang}")
                        sc3.caption(f"**Tokens:** {tokens}")
                        if rl_score is not None:
                            sc4.caption(f"**RL Score:** {rl_score:.4f}")
                        if tags:
                            st.caption("**Tags:** " + " · ".join(f"`{t}`" for t in tags[:8]))
                        st.code(hit.get("content", "")[:600], language=lang if lang != "—" else None)
            elif sources:
                for s in sources:
                    st.markdown(f"- `{s}`")
            else:
                st.write("No sources returned.")

        except Exception as e:
            st.error(f"Evaluator failed: {e}")


def _render_admin_ragas():
    st.subheader("🧪 RAGAS Test")

    projects = get_available_projects()
    project_name = st.selectbox("Project", projects, key="ragas_project")
    user_role = st.selectbox("Test as role", ["Developer", "Project Manager", "Data Scientist"], key="ragas_role")

    st.caption("Enter one query per line")
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

            if df.empty:
                st.warning("No results returned.")
                return

            # ── Summary metrics row ───────────────────────────────────────
            total    = len(df)
            sourced  = int((df["Source_Count"] > 0).sum())
            avg_srcs = round(df["Source_Count"].mean(), 1)
            unique_files = len(set(
                f.strip()
                for row in df["Sources"].dropna()
                for f in row.split(",") if f.strip()
            ))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Queries Tested", total)
            c2.metric("With Sources",   f"{sourced}/{total}")
            c3.metric("Avg Sources",    avg_srcs)
            c4.metric("Unique Files Hit", unique_files)

            if sourced == total:
                st.success("✅ All queries returned at least one source.")
            else:
                st.warning(f"⚠️ {total - sourced} query(ies) returned zero sources.")

            # ── Per-query expandable results ──────────────────────────────
            st.markdown("**🔍 Query-by-Query Breakdown**")
            for _, row in df.iterrows():
                src_count = row["Source_Count"]
                icon = "✅" if src_count > 0 else "❌"
                with st.expander(f"{icon} {row['Question'][:80]}", expanded=False):
                    st.markdown(f"**💬 Answer**")
                    st.markdown(row["Answer"])

                    st.markdown(f"**📂 Sources** ({src_count} file{'s' if src_count != 1 else ''})")
                    if row["Sources"]:
                        for sf in row["Sources"].split(","):
                            sf = sf.strip()
                            if sf:
                                st.markdown(f"- `{sf}`")
                    else:
                        st.caption("No sources retrieved.")

            # ── Full dataframe ────────────────────────────────────────────
            with st.expander("📋 Raw Results Table", expanded=False):
                st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"RAGAS test failed: {e}")

def _render_admin_feedback():
    st.subheader("📋 Project Feedback & RL Training")

    projects = get_available_projects()
    if not projects:
        st.warning("No chroma_* project folders found in the root directory.")
        return

    feedback_project = st.selectbox("Project", projects, key="feedback_project")

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

            engine = st.session_state.engine
            if engine is not None and hasattr(engine, "rl_agent_cache"):
                engine.rl_agent_cache.pop(feedback_project, None)

            st.success(f"RL model trained for project: {feedback_project}")
            st.rerun()

# ─── Admin Page ───────────────────────────────────────────────────────────────

def show_admin():
    ensure_engine()
    user = st.session_state.user
    projects = get_available_projects()

    with st.sidebar:
        st.markdown(f'<div class="user-badge">👑 {user[2]} &nbsp;·&nbsp; Admin</div>', unsafe_allow_html=True)
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            _logout()

    active_project = st.session_state.selected_project or (projects[0] if projects else "")

    st.markdown(
        '<span style="font-family:Space Mono;font-size:1.6rem;font-weight:700;color:#a78bfa">🧠 KT Bot</span>',
        unsafe_allow_html=True
    )

    (tab_chat, tab_report,
     tab_eval, tab_ragas, tab_nlp, tab_feedback) = st.tabs([
        "💬 Chat",
        "📄 Report Generator",
        "📊 Evaluator",
        "🧪 RAGAS Test",
        "🔬 NLP Dashboard",
        "📋 Feedback",
    ])

    with tab_chat:
        _render_chat(is_admin=True)

    with tab_report:
        _render_report_generator(is_admin=True)

    with tab_eval:
        _render_admin_evaluator()

    with tab_ragas:
        _render_admin_ragas()

    with tab_nlp:
        if projects:
            selected = st.selectbox(
                "📂 Active Project",
                projects,
                index=projects.index(st.session_state.selected_project)
                      if st.session_state.selected_project in projects else 0,
                key="nlp_tab_project",
            )
            if selected != st.session_state.selected_project:
                st.session_state.selected_project = selected
                st.rerun()
            _render_nlp_dashboard_tab(selected)
        else:
            st.warning("No projects ingested yet. Run the NLP pipeline first.")

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
            short_q = (q[:100] + "…") if len(q) > 100 else q
            st.markdown(f"`{icon}` {short_q}")

    except Exception:
        st.caption("Could not load log.")

# ─── User Page ────────────────────────────────────────────────────────────────

def show_user():
    ensure_engine()
    user = st.session_state.user
    projects = get_available_projects()

    with st.sidebar:
        st.markdown(f'<div class="user-badge">👤 {user[2]}</div>', unsafe_allow_html=True)
        st.divider()
        st.caption(f"\n**{user[1]}** · {user[3]}")
        st.divider()

        st.divider()
        st.info("💡 Ask anything about the selected codebase in any language. Your feedback helps improve the bot.")
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            _logout()

    st.markdown(
        '<span style="font-family:Space Mono;font-size:1.6rem;font-weight:700;color:#a78bfa">🧠 KT Bot</span>',
        unsafe_allow_html=True
    )

    tab_chat, tab_report = st.tabs([
        "💬 Chat",
        "📄 Report Generator",
    ])

    with tab_chat:
        _render_chat(is_admin=False)

    with tab_report:
        _render_report_generator(is_admin=False)

# ─── Translation Display Helpers ──────────────────────────────────────────────

def _render_translation_info(message: dict):
    """Render the original + translated query display for a user message."""
    if not message.get("is_translated"):
        return

    lang_name = message.get("detected_lang_name", "Unknown")
    original = message.get("original_query", message["content"])
    translated = message.get("translated_query", "")

    st.markdown(
        f'<div class="translation-box original-query-box">'
        f'<span class="lang-badge lang-badge-original">🌐 {lang_name}</span>'
        f'<strong>Original:</strong> {original}'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="translation-box translated-query-box">'
        f'<span class="lang-badge lang-badge-english">🇬🇧 English</span>'
        f'<strong>Translated:</strong> {translated}'
        f'</div>',
        unsafe_allow_html=True
    )


def _render_response_with_toggle(message: dict, msg_index: int, is_admin: bool):
    """Render the assistant response with a language toggle if applicable."""
    is_translated = message.get("is_translated", False)
    answer_original = message.get("content", "")
    answer_english = message.get("answer_english", answer_original)
    lang_name = message.get("detected_lang_name", "English")

    if is_translated and answer_english and answer_english != answer_original:
        toggle_key = f"lang_toggle_{msg_index}"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = False

        col_text, col_toggle = st.columns([8, 2])

        with col_toggle:
            show_english = st.toggle(
                "🇬🇧 English",
                key=toggle_key,
                help=f"Toggle between {lang_name} and English response"
            )

        with col_text:
            if show_english:
                st.markdown(answer_english)
                st.markdown(
                    f'<div class="response-lang-note">📝 Showing English response</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(answer_original)
                st.markdown(
                    f'<div class="response-lang-note">🌐 Showing {lang_name} response · Toggle to see English</div>',
                    unsafe_allow_html=True
                )
    else:
        st.markdown(answer_original)

    # ── NEW: mini NLP badge under every assistant reply ────────────────────
    render_nlp_mini_badge(message)


# ─── Shared Chat Renderer ─────────────────────────────────────────────────────

def _render_chat(is_admin: bool):
    engine = st.session_state.engine
    user = st.session_state.user
    user_role = user[3] if user and len(user) > 3 else "User"

    # ── Display history ────────────────────────────────────────────────────
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):

            if message["role"] == "user":
                st.markdown(message["content"])
                _render_translation_info(message)

            else:
                # Assistant message — render with toggle + mini badge
                _render_response_with_toggle(message, i, is_admin)

                if "sources" in message and message["sources"]:
                    st.caption(f"📂 Sources: {', '.join(message['sources'])}")

                if "project" in message and message["project"]:
                    st.caption(f"🧩 Project: {message['project']}")

                # Feedback buttons only for the last assistant message
                if i == len(st.session_state.messages) - 1:
                    orig_query = st.session_state.messages[i - 1].get("translated_query") \
                                 or st.session_state.messages[i - 1]["content"] \
                                 if i > 0 else ""
                    project_name = message.get("project")
                    col1, col2, col3 = st.columns([1, 1, 10])

                    with col1:
                        st.button(
                            "👍", key=f"up_{i}",
                            on_click=record_feedback,
                            args=(orig_query, 1.0, project_name)
                        )
                    with col2:
                        st.button(
                            "👎", key=f"down_{i}",
                            on_click=record_feedback,
                            args=(orig_query, -1.0, project_name)
                        )

                    if is_admin:
                        with col3:
                            st.caption("RL score shown in console · project-aware feedback saved to DB")

    # ── NEW: NLP Lens panel — shown after last reply, persists until next query
    if st.session_state.last_nlp_result:
        render_nlp_lens(
            query=st.session_state.last_nlp_query,
            result=st.session_state.last_nlp_result,
            final_hits=st.session_state.last_nlp_hits,
        )

    st.divider()

    # ── Project + Query form ───────────────────────────────────────────────
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
                "Ask your question (any language)",
                placeholder="Ask anything… / எதாவது கேளுங்கள்… / कुछ भी पूछें…"
            )

        submitted = st.form_submit_button("Ask", use_container_width=True)

    st.session_state.selected_project = selected_project

    st.caption(f"Role-aware mode: **{user_role}** · Selected project: **{selected_project}** · 🌐 Multilingual enabled")

    if submitted:
        if not prompt.strip():
            st.warning("Please enter a question.")
            return

        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "project": selected_project,
            "is_translated": False,
        })

        with st.spinner("Thinking…"):
            result = engine.generate_response(
                user_query=prompt,
                project_name=selected_project,
                user_role=user_role,
                respond_in_original_lang=True,
            )

        # Update user message with translation metadata
        last_user_msg = st.session_state.messages[-1]
        last_user_msg["is_translated"] = result["is_translated"]
        last_user_msg["translated_query"] = result.get("translated_query")
        last_user_msg["original_query"] = result["original_query"]
        last_user_msg["detected_lang"] = result["detected_lang"]
        last_user_msg["detected_lang_name"] = result["detected_lang_name"]

        # Add assistant message (include NLP badge fields)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "answer_english": result["answer_english"],
            "sources": result["sources"],
            "project": selected_project,
            "is_translated": result["is_translated"],
            "detected_lang": result["detected_lang"],
            "detected_lang_name": result["detected_lang_name"],
        })

        # ── NEW: stash NLP Lens data in session so it survives rerun ──────
        st.session_state.last_nlp_query  = prompt
        st.session_state.last_nlp_result = result
        # final_hits is returned by the updated chat_engine (see note below)
        st.session_state.last_nlp_hits   = result.get("final_hits", [])

        st.rerun()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _logout():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.messages = []
    st.session_state.engine = None
    st.session_state.selected_project = None
    st.session_state.last_nlp_result = None
    st.session_state.last_nlp_hits = []
    st.session_state.last_nlp_query = ""
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