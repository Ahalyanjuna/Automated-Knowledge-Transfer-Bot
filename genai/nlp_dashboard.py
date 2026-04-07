"""
nlp_dashboard.py  —  NLP Analytics Dashboard
=============================================
Auto-generates a rich NLP analysis panel for any selected project
by reading chunk metadata directly from the Chroma vector DB.

Covers:
  • Corpus Overview     — chunk counts, file types, languages
  • Chunk Type Dist     — function / class / doc_section / etc.
  • NLP Tag Cloud       — domain tags from tagger.py (auth, ml, crud …)
  • Complexity Heatmap  — per-file complexity scores
  • NER Entity Report   — top entities by label
  • Embedding Explorer  — 2-D UMAP/PCA projection of embeddings
  • Docstring Coverage  — has_docstring ratio by file
  • Token Distribution  — histogram of token counts
  • RAG Eval Summary    — faithfulness / relevance trends from RL feedback DB
"""

from __future__ import annotations
import os
import json
import ast
import sqlite3
from collections import Counter, defaultdict
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ── optional heavy deps ────────────────────────────────────────────────────
try:
    import chromadb
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=120, show_spinner=False)
def load_project_metadata(db_path: str) -> pd.DataFrame:
    """
    Pull ALL chunk metadata from the Chroma collection for the project.
    Returns a flat DataFrame — one row per chunk.
    """
    if not CHROMA_OK:
        return pd.DataFrame()

    try:
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        if not collections:
            return pd.DataFrame()
        col = client.get_collection(name=collections[0].name)

        total = col.count()
        if total == 0:
            return pd.DataFrame()

        # Fetch in batches to avoid memory issues
        BATCH = 500
        all_ids, all_docs, all_metas = [], [], []
        for offset in range(0, total, BATCH):
            res = col.get(
                limit=BATCH,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            all_ids   += res["ids"]
            all_docs  += res["documents"]
            all_metas += res["metadatas"]

        rows = []
        embeddings_list = []
        for cid, doc, meta in zip(all_ids, all_docs, all_metas):
            row = dict(meta)
            row["chunk_id"] = cid
            row["content_len"] = len(doc)

            # Safe parse list/str fields stored as strings
            for field in ["tags", "parameters", "imports", "api_surface"]:
                val = row.get(field, "[]")
                if isinstance(val, str):
                    try:
                        row[field] = ast.literal_eval(val)
                    except Exception:
                        row[field] = []

            rows.append(row)

        df = pd.DataFrame(rows)

        # Normalise numeric columns
        for col_name in ["token_count", "complexity_score", "segment_count",
                         "embedding_dim", "detected_lang_conf"]:
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0)

        for col_name in ["has_docstring", "has_tests", "has_todos", "is_translated"]:
            if col_name in df.columns:
                df[col_name] = df[col_name].map(
                    lambda x: x if isinstance(x, bool)
                    else str(x).lower() in ("true", "1", "yes")
                )

        return df

    except Exception as e:
        st.error(f"Dashboard data load failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_project_embeddings(db_path: str, max_samples: int = 800) -> tuple:
    """Returns (embeddings np.array, labels list) for PCA plot."""
    if not CHROMA_OK:
        return np.array([]), []

    try:
        client = chromadb.PersistentClient(path=db_path)
        cols = client.list_collections()
        if not cols:
            return np.array([]), []
        col = client.get_collection(name=cols[0].name)

        total = min(col.count(), max_samples)
        res = col.get(limit=total, offset=0, include=["embeddings", "metadatas"])

        embs = [e for e in res["embeddings"] if e and len(e) > 0]
        metas = res["metadatas"][:len(embs)]

        labels = [m.get("chunk_type", "unknown") for m in metas]
        files  = [m.get("source_file", "")       for m in metas]
        return np.array(embs, dtype=np.float32), labels, files

    except Exception:
        return np.array([]), [], []


# ═══════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════

PALETTE = [
    "#a78bfa", "#34d399", "#f472b6", "#60a5fa", "#fbbf24",
    "#fb7185", "#818cf8", "#2dd4bf", "#a3e635", "#e879f9",
    "#38bdf8", "#4ade80", "#f97316", "#c084fc", "#e2e8f0",
]

BG = "rgba(0,0,0,0)"
PAPER = "rgba(15,15,35,0)"
FONT_COLOR = "#e2e8f0"

LAYOUT_BASE = dict(
    paper_bgcolor=PAPER,
    plot_bgcolor=BG,
    font=dict(color=FONT_COLOR, family="DM Sans"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def _fig_base(**kwargs):
    fig = go.Figure(**kwargs)
    fig.update_layout(**LAYOUT_BASE)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SECTION RENDERERS
# ═══════════════════════════════════════════════════════════════════════════

def _metric_row(df: pd.DataFrame):
    total      = len(df)
    languages  = df["normalized_lang"].nunique() if "normalized_lang" in df.columns else "—"
    files      = df["source_file"].nunique()     if "source_file"     in df.columns else "—"
    avg_tokens = int(df["token_count"].mean())   if "token_count"     in df.columns else 0
    tags_total = sum(len(t) for t in df["tags"]) if "tags"            in df.columns else 0
    doc_pct    = int(df["has_docstring"].mean() * 100) if "has_docstring" in df.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📦 Total Chunks",    f"{total:,}")
    c2.metric("📄 Source Files",    f"{files}")
    c3.metric("🌐 Languages",       f"{languages}")
    c4.metric("📝 Avg Tokens",      f"{avg_tokens}")
    c5.metric("🏷️ Tag Assignments", f"{tags_total:,}")
    c6.metric("📖 Docstring Cov.",  f"{doc_pct}%")


def _chunk_type_chart(df: pd.DataFrame):
    if "chunk_type" not in df.columns:
        return
    counts = df["chunk_type"].value_counts().reset_index()
    counts.columns = ["Chunk Type", "Count"]

    fig = px.bar(
        counts.head(15), x="Count", y="Chunk Type", orientation="h",
        color="Chunk Type", color_discrete_sequence=PALETTE,
        title="Chunk Type Distribution"
    )
    fig.update_layout(**LAYOUT_BASE, showlegend=False,
                      yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)


def _language_pie(df: pd.DataFrame):
    col = "normalized_lang" if "normalized_lang" in df.columns else "language"
    if col not in df.columns:
        return
    counts = df[col].value_counts().reset_index()
    counts.columns = ["Language", "Count"]

    fig = px.pie(
        counts, names="Language", values="Count",
        color_discrete_sequence=PALETTE,
        title="Chunks by Programming Language",
        hole=0.45,
    )
    fig.update_layout(**LAYOUT_BASE)
    fig.update_traces(textposition="inside", textinfo="label+percent")
    st.plotly_chart(fig, use_container_width=True)


def _file_type_pie(df: pd.DataFrame):
    if "file_type" not in df.columns:
        return
    counts = df["file_type"].value_counts().reset_index()
    counts.columns = ["File Type", "Count"]

    fig = px.pie(
        counts, names="File Type", values="Count",
        color_discrete_sequence=PALETTE,
        title="Chunks by File Type",
        hole=0.45,
    )
    fig.update_layout(**LAYOUT_BASE)
    st.plotly_chart(fig, use_container_width=True)


def _tag_cloud(df: pd.DataFrame):
    if "tags" not in df.columns:
        return

    tag_counter: Counter = Counter()
    for tags in df["tags"]:
        if isinstance(tags, list):
            tag_counter.update(tags)

    if not tag_counter:
        st.info("No domain tags found in this project.")
        return

    tags_df = pd.DataFrame(tag_counter.most_common(25), columns=["Tag", "Count"])

    fig = px.bar(
        tags_df, x="Tag", y="Count",
        color="Count", color_continuous_scale="Purples",
        title="NLP Domain Tag Frequency (Top 25)"
    )
    fig.update_layout(**LAYOUT_BASE, coloraxis_showscale=False,
                      xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    # Tag co-occurrence mini heatmap (top 10 tags)
    top_tags = [t for t, _ in tag_counter.most_common(10)]
    matrix = np.zeros((len(top_tags), len(top_tags)), dtype=int)
    for tags in df["tags"]:
        if isinstance(tags, list):
            present = [t for t in top_tags if t in tags]
            for i, a in enumerate(top_tags):
                for j, b in enumerate(top_tags):
                    if a in present and b in present:
                        matrix[i][j] += 1

    fig2 = go.Figure(go.Heatmap(
        z=matrix, x=top_tags, y=top_tags,
        colorscale="Purples", showscale=False,
        hovertemplate="<b>%{x}</b> + <b>%{y}</b>: %{z} chunks<extra></extra>",
    ))
    fig2.update_layout(**LAYOUT_BASE, title="Tag Co-occurrence (Top 10 Tags)",
                       xaxis_tickangle=-35)
    st.plotly_chart(fig2, use_container_width=True)


def _complexity_heatmap(df: pd.DataFrame):
    if "complexity_score" not in df.columns or "source_file" not in df.columns:
        return

    file_complexity = (
        df.groupby("source_file")["complexity_score"]
        .agg(["mean", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "Avg Complexity", "max": "Max Complexity", "count": "Chunks"})
        .sort_values("Avg Complexity", ascending=False)
        .head(20)
    )
    # Shorten long paths
    file_complexity["File"] = file_complexity["source_file"].apply(
        lambda x: x.split("/")[-1] if "/" in x else x
    )

    fig = px.bar(
        file_complexity, x="File", y="Avg Complexity",
        color="Max Complexity", color_continuous_scale="Reds",
        hover_data=["Chunks", "Max Complexity"],
        title="Code Complexity per File (avg · colour = max)"
    )
    fig.update_layout(**LAYOUT_BASE, xaxis_tickangle=-40,
                      coloraxis_colorbar=dict(title="Max"))
    st.plotly_chart(fig, use_container_width=True)

    # Complexity bucket breakdown
    if len(df) > 5:
        bins = pd.cut(df["complexity_score"],
                      bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                      labels=["Trivial", "Simple", "Moderate", "Complex", "Very Complex"])
        bucket_counts = bins.value_counts().sort_index().reset_index()
        bucket_counts.columns = ["Complexity Band", "Chunks"]
        col_colors = ["#34d399", "#60a5fa", "#fbbf24", "#f97316", "#f472b6"]
        fig2 = px.bar(bucket_counts, x="Complexity Band", y="Chunks",
                      color="Complexity Band",
                      color_discrete_sequence=col_colors,
                      title="Complexity Band Distribution")
        fig2.update_layout(**LAYOUT_BASE, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


def _token_distribution(df: pd.DataFrame):
    if "token_count" not in df.columns:
        return

    fig = px.histogram(
        df[df["token_count"] > 0], x="token_count", nbins=50,
        color_discrete_sequence=["#a78bfa"],
        title="Token Count Distribution across Chunks"
    )
    fig.update_layout(**LAYOUT_BASE, bargap=0.05)
    fig.update_xaxes(title="Token Count")
    fig.update_yaxes(title="Number of Chunks")
    st.plotly_chart(fig, use_container_width=True)

    # Box plot by chunk type
    if "chunk_type" in df.columns:
        top_types = df["chunk_type"].value_counts().head(8).index.tolist()
        sub = df[df["chunk_type"].isin(top_types)]
        fig2 = px.box(
            sub, x="chunk_type", y="token_count",
            color="chunk_type", color_discrete_sequence=PALETTE,
            title="Token Count by Chunk Type"
        )
        fig2.update_layout(**LAYOUT_BASE, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)


def _docstring_coverage(df: pd.DataFrame):
    if "has_docstring" not in df.columns or "source_file" not in df.columns:
        return

    coverage = (
        df.groupby("source_file")["has_docstring"]
        .agg(lambda x: round(x.mean() * 100, 1))
        .reset_index()
        .rename(columns={"has_docstring": "Docstring %"})
        .sort_values("Docstring %", ascending=True)
        .tail(20)
    )
    coverage["File"] = coverage["source_file"].apply(
        lambda x: x.split("/")[-1] if "/" in x else x
    )
    coverage["Color"] = coverage["Docstring %"].apply(
        lambda v: "#34d399" if v >= 70 else ("#fbbf24" if v >= 40 else "#f472b6")
    )

    fig = go.Figure(go.Bar(
        x=coverage["Docstring %"], y=coverage["File"],
        orientation="h",
        marker_color=coverage["Color"],
        hovertemplate="<b>%{y}</b><br>Coverage: %{x}%<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, title="Docstring Coverage per File (%)",
                      xaxis=dict(range=[0, 100], title="% chunks with docstrings"))
    st.plotly_chart(fig, use_container_width=True)

    # Summary flags
    has_tests  = int(df["has_tests"].sum())  if "has_tests"  in df.columns else 0
    has_todos  = int(df["has_todos"].sum())  if "has_todos"  in df.columns else 0
    has_docs   = int(df["has_docstring"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("📖 Chunks with Docstrings", has_docs)
    c2.metric("🧪 Chunks with Tests",      has_tests)
    c3.metric("📌 Chunks with TODOs",      has_todos)


def _ner_entity_report(df: pd.DataFrame):
    if "entities" not in df.columns:
        return

    entity_counter: dict[str, Counter] = defaultdict(Counter)
    for ents_raw in df["entities"]:
        ents = ents_raw
        if isinstance(ents, str):
            try:
                ents = ast.literal_eval(ents)
            except Exception:
                continue
        if not isinstance(ents, list):
            continue
        for e in ents:
            if isinstance(e, dict):
                label = e.get("label", "UNK")
                text  = e.get("text", "")[:40]
                entity_counter[label][text] += 1

    if not entity_counter:
        st.info("No NER entities found. Ensure the NLP pipeline ran with NER enabled.")
        return

    label_totals = {k: sum(v.values()) for k, v in entity_counter.items()}
    totals_df = pd.DataFrame(
        sorted(label_totals.items(), key=lambda x: -x[1]),
        columns=["Entity Label", "Count"]
    )

    fig = px.bar(
        totals_df, x="Entity Label", y="Count",
        color="Entity Label", color_discrete_sequence=PALETTE,
        title="Named Entities by Label (all chunks)"
    )
    fig.update_layout(**LAYOUT_BASE, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Per-label top entities
    st.markdown("#### Top Entities per Label")
    label_cols = st.columns(min(len(entity_counter), 4))
    for idx, (label, counter) in enumerate(
        sorted(entity_counter.items(), key=lambda x: -sum(x[1].values()))[:8]
    ):
        col = label_cols[idx % len(label_cols)]
        with col:
            st.markdown(f"**`{label}`**")
            for ent, cnt in counter.most_common(6):
                st.markdown(f"<small>• {ent} `{cnt}`</small>", unsafe_allow_html=True)


def _embedding_pca(db_path: str):
    if not PCA_OK or not PLOTLY_OK:
        st.info("Install scikit-learn + plotly for the embedding visualisation.")
        return

    with st.spinner("Running PCA on embeddings…"):
        embs, labels, files = load_project_embeddings(db_path)

    if len(embs) < 10:
        st.info("Not enough embeddings to plot (need ≥ 10 chunks).")
        return

    n_components = min(2, embs.shape[1])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embs)

    plot_df = pd.DataFrame({
        "PC1":   coords[:, 0],
        "PC2":   coords[:, 1],
        "Type":  labels,
        "File":  [f.split("/")[-1] for f in files],
    })

    fig = px.scatter(
        plot_df, x="PC1", y="PC2",
        color="Type", hover_data=["File"],
        color_discrete_sequence=PALETTE,
        title=f"Embedding Space — PCA 2D ({len(embs)} chunks)",
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(**LAYOUT_BASE)
    st.plotly_chart(fig, use_container_width=True)

    var = pca.explained_variance_ratio_
    st.caption(
        f"PC1 explains **{var[0]*100:.1f}%** variance · "
        f"PC2 explains **{var[1]*100:.1f}%** variance · "
        f"Total: **{sum(var)*100:.1f}%**"
    )


def _rl_feedback_panel(rl_db: str, project_name: str):
    if not os.path.exists(rl_db):
        st.info("No RL feedback recorded yet. Start chatting and giving 👍/👎.")
        return

    try:
        with sqlite3.connect(rl_db) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(experiences)").fetchall()]
            has_proj = "project_name" in cols

            if has_proj:
                rows = conn.execute(
                    "SELECT query, reward, timestamp FROM experiences WHERE project_name=? ORDER BY id",
                    (project_name,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT query, reward, timestamp FROM experiences ORDER BY id"
                ).fetchall()

        if not rows:
            st.info("No feedback yet for this project.")
            return

        fb_df = pd.DataFrame(rows, columns=["Query", "Reward", "Timestamp"])
        fb_df["Timestamp"] = pd.to_datetime(fb_df["Timestamp"], errors="coerce")
        fb_df["Feedback"]  = fb_df["Reward"].apply(lambda r: "👍 Positive" if r > 0 else "👎 Negative")

        total   = len(fb_df)
        pos     = int((fb_df["Reward"] > 0).sum())
        neg     = total - pos
        avg_r   = fb_df["Reward"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Feedback", total)
        c2.metric("👍 Positive", pos)
        c3.metric("👎 Negative", neg)
        c4.metric("Avg Reward", f"{avg_r:.2f}")

        # Reward trend
        fb_df["Cumulative Avg"] = fb_df["Reward"].expanding().mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(fb_df))), y=fb_df["Reward"],
            marker_color=fb_df["Reward"].apply(lambda r: "#34d399" if r > 0 else "#f472b6"),
            name="Reward", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(fb_df))), y=fb_df["Cumulative Avg"],
            mode="lines", name="Cumulative Avg",
            line=dict(color="#a78bfa", width=2),
        ))
        fig.update_layout(**LAYOUT_BASE, title="RL Feedback Reward Trend",
                          xaxis_title="Feedback #", yaxis_title="Reward")
        st.plotly_chart(fig, use_container_width=True)

        # Pie
        fig2 = px.pie(
            fb_df, names="Feedback",
            color="Feedback",
            color_discrete_map={"👍 Positive": "#34d399", "👎 Negative": "#f472b6"},
            title="Feedback Ratio", hole=0.5,
        )
        fig2.update_layout(**LAYOUT_BASE)
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load RL feedback: {e}")


def _api_surface_report(df: pd.DataFrame):
    if "api_surface" not in df.columns:
        return

    all_apis: Counter = Counter()
    for apis in df["api_surface"]:
        if isinstance(apis, list):
            all_apis.update(apis)

    if not all_apis:
        st.info("No public API names detected.")
        return

    api_df = pd.DataFrame(all_apis.most_common(20), columns=["Function/Class", "Occurrences"])

    fig = px.bar(
        api_df, x="Occurrences", y="Function/Class", orientation="h",
        color="Occurrences", color_continuous_scale="Purples",
        title="Top Public API Surface (functions & classes)"
    )
    fig.update_layout(**LAYOUT_BASE, showlegend=False,
                      yaxis=dict(categoryorder="total ascending"),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def render_nlp_dashboard(project_name: str, db_path: str, rl_db: str):
    """
    Call this from app.py inside the NLP Dashboard tab.
    project_name : e.g. "visioncortex"
    db_path      : e.g. "/abs/path/to/chroma_visioncortex"
    rl_db        : absolute path to rl_feedback.db
    """
    if not PLOTLY_OK:
        st.warning("Install plotly for charts: `pip install plotly`")

    if not os.path.exists(db_path):
        st.error(f"Chroma DB not found at `{db_path}`. Ingest the project first.")
        return

    # ── Load data ──────────────────────────────────────────────────────────
    with st.spinner(f"Loading NLP metadata for **{project_name}**…"):
        df = load_project_metadata(db_path)

    if df.empty:
        st.warning("No chunk metadata found. Run the NLP pipeline and load to Chroma first.")
        return

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown(
        f"<h3 style='font-family:Space Mono;color:#a78bfa;margin-bottom:0'>🔬 NLP Analytics · {project_name}</h3>",
        unsafe_allow_html=True
    )
    st.caption(f"{len(df):,} chunks · auto-generated from ChromaDB metadata")

    # ── KPI row ────────────────────────────────────────────────────────────
    _metric_row(df)
    st.divider()

    # ── Tabs for sections ─────────────────────────────────────────────────
    t1, t2, t3 = st.tabs([
        "📊 Corpus",
        "🏷️ Tags & NER",
        "⚙️ Complexity",
    ])

    with t1:
        st.subheader("Corpus Overview")
        col1, col2 = st.columns(2)
        with col1:
            _chunk_type_chart(df)
        with col2:
            _language_pie(df)

        _token_distribution(df)

    with t2:
        st.subheader("Domain Tags")
        _tag_cloud(df)
        st.divider()
        st.subheader("Named Entity Recognition")
        _ner_entity_report(df)
        st.divider()
        st.subheader("Public API Surface")
        _api_surface_report(df)

    with t3:
        st.subheader("Code Complexity Analysis")
        _complexity_heatmap(df)

    # ── Raw data explorer ─────────────────────────────────────────────────
    with st.expander("🔍 Raw Chunk Explorer", expanded=False):
        display_cols = [c for c in [
            "chunk_id", "chunk_type", "normalized_lang", "source_file",
            "token_count", "complexity_score", "has_docstring",
            "has_tests", "has_todos", "tags",
        ] if c in df.columns]

        search = st.text_input("Filter by source file or chunk type", key="dash_search")
        fdf = df[display_cols].copy()
        if search:
            mask = fdf.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
            fdf = fdf[mask]
        st.dataframe(fdf, use_container_width=True, height=350)
        st.caption(f"Showing {len(fdf):,} of {len(df):,} chunks")