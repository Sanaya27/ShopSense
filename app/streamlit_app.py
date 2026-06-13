"""
ShopSense — Semantic Product Retrieval System
Amazon ML School Portfolio Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ShopSense · Semantic Product Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0D0F14;
    color: #E8EAF0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #13161E !important;
    border-right: 1px solid #1E2330;
}
[data-testid="stSidebar"] * {
    color: #C8CCDA !important;
}

/* ── Hero ── */
.hero-container {
    background: linear-gradient(135deg, #0D0F14 0%, #111420 50%, #0D0F14 100%);
    border: 1px solid #1E2330;
    border-radius: 16px;
    padding: 52px 48px 44px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,153,0,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #FF9900;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 54px;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -1.5px;
    margin-bottom: 10px;
    background: linear-gradient(90deg, #FFFFFF 60%, #FF9900 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 17px;
    color: #8891A8;
    font-weight: 400;
    margin-bottom: 22px;
    max-width: 560px;
    line-height: 1.6;
}
.hero-badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 4px;
}
.hero-badge {
    background: #1A1D28;
    border: 1px solid #2A2F42;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
    color: #8891A8;
    font-family: 'JetBrains Mono', monospace;
}
.hero-badge span {
    color: #FF9900;
    margin-right: 5px;
}

/* ── Search Box ── */
.search-wrapper {
    background: #13161E;
    border: 1.5px solid #2A2F42;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
    transition: border-color 0.2s;
}
.search-wrapper:focus-within {
    border-color: #FF9900;
}
.search-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4A5168;
    margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace;
}

/* Streamlit input override */
[data-testid="stTextInput"] input {
    background: #0D0F14 !important;
    border: 1px solid #2A2F42 !important;
    border-radius: 8px !important;
    color: #E8EAF0 !important;
    font-size: 16px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 14px 18px !important;
    caret-color: #FF9900;
}
[data-testid="stTextInput"] input:focus {
    border-color: #FF9900 !important;
    box-shadow: 0 0 0 3px rgba(255,153,0,0.12) !important;
}

/* ── Button ── */
.stButton > button {
    background: #FF9900 !important;
    color: #0D0F14 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 32px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.15s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: #FFB733 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(255,153,0,0.30) !important;
}

/* ── Metric Cards (Sidebar) ── */
.metric-card {
    background: #1A1D28;
    border: 1px solid #2A2F42;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.metric-card-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4A5168;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
}
.metric-card-value {
    font-size: 26px;
    font-weight: 800;
    color: #FF9900;
    line-height: 1;
    letter-spacing: -0.5px;
}
.metric-card-sub {
    font-size: 11px;
    color: #4A5168;
    margin-top: 3px;
}

/* Eval metric row */
.eval-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #1E2330;
}
.eval-row:last-child { border-bottom: none; }
.eval-metric-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #8891A8;
    font-weight: 600;
}
.eval-metric-val {
    font-size: 15px;
    font-weight: 700;
    color: #FF9900;
}
.eval-metric-bar-bg {
    width: 80px;
    height: 4px;
    background: #1E2330;
    border-radius: 4px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
    margin-left: 10px;
}
.eval-metric-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #FF6600, #FF9900);
}

/* ── Result Cards ── */
.result-card {
    background: #13161E;
    border: 1px solid #1E2330;
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 14px;
    transition: border-color 0.15s, transform 0.15s;
    position: relative;
    overflow: hidden;
}
.result-card:hover {
    border-color: #2A2F42;
    transform: translateY(-1px);
}
.result-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 12px 0 0 12px;
}
.rank-1::before { background: #FF9900; }
.rank-2::before { background: #C0C0C0; }
.rank-3::before { background: #CD7F32; }
.rank-other::before { background: #2A2F42; }

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 10px;
}
.card-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #4A5168;
    white-space: nowrap;
    margin-top: 3px;
}
.card-title {
    font-size: 15px;
    font-weight: 600;
    color: #E8EAF0;
    line-height: 1.4;
    flex: 1;
}
.card-score-pill {
    background: rgba(255,153,0,0.10);
    border: 1px solid rgba(255,153,0,0.25);
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #FF9900;
    white-space: nowrap;
}
.card-brand {
    font-size: 12px;
    color: #4A5168;
    font-weight: 500;
    margin-bottom: 0;
}
.card-brand span {
    color: #8891A8;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 36px 0 18px;
}
.section-header-line {
    flex: 1;
    height: 1px;
    background: #1E2330;
}
.section-header-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #4A5168;
    white-space: nowrap;
}

/* ── Pipeline ── */
.pipeline-container {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0;
    background: #13161E;
    border: 1px solid #1E2330;
    border-radius: 12px;
    padding: 28px 24px;
    margin-bottom: 24px;
}
.pipeline-step {
    text-align: center;
    padding: 0 16px;
}
.pipeline-step-icon {
    font-size: 22px;
    margin-bottom: 8px;
}
.pipeline-step-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #FF9900;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.pipeline-step-sub {
    font-size: 11px;
    color: #4A5168;
    max-width: 100px;
}
.pipeline-arrow {
    font-size: 20px;
    color: #2A2F42;
    padding: 0 4px;
    margin-bottom: 20px;
    align-self: flex-end;
}

/* ── Highlight Tags ── */
.highlights-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin-top: 4px;
}
.highlight-chip {
    background: #1A1D28;
    border: 1px solid #2A2F42;
    border-radius: 8px;
    padding: 14px 18px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.highlight-chip-icon { font-size: 18px; }
.highlight-chip-text {
    font-size: 13px;
    font-weight: 600;
    color: #C8CCDA;
    line-height: 1.3;
}
.highlight-chip-sub {
    font-size: 11px;
    color: #4A5168;
    margin-top: 2px;
}

/* ── Dataset Info ── */
.dataset-block {
    background: #13161E;
    border: 1px solid #1E2330;
    border-radius: 12px;
    padding: 24px 26px;
    margin-bottom: 24px;
}
.dataset-block p {
    color: #8891A8;
    font-size: 14px;
    line-height: 1.7;
    margin: 0;
}
.dataset-block strong {
    color: #E8EAF0;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 60px 24px;
    color: #4A5168;
}
.empty-state-icon { font-size: 44px; margin-bottom: 14px; }
.empty-state-text { font-size: 15px; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1A1D28 !important;
    border: 1px solid #2A2F42 !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D0F14; }
::-webkit-scrollbar-thumb { background: #2A2F42; border-radius: 3px; }

/* ── Sidebar divider ── */
.sidebar-divider {
    border: none;
    border-top: 1px solid #1E2330;
    margin: 18px 0;
}
.sidebar-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #4A5168 !important;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data + Model Loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_faiss_index():
    return faiss.read_index("models/shopsense.faiss")


@st.cache_data(show_spinner=False)
def load_products():
    df = pd.read_csv("data/shopsense_products_search.csv")
    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    return df


# ─────────────────────────────────────────────
# Search Logic
# ─────────────────────────────────────────────
def semantic_search(query: str, model, index, products_df: pd.DataFrame, top_k: int = 10):
    """Encode query, search FAISS, return ranked results."""
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx < 0 or idx >= len(products_df):
            continue
        row = products_df.iloc[idx]
        results.append({
            "rank": rank,
            "title":       row.get("product_title",       row.get("title",       "N/A")),
            "brand":       row.get("product_brand",       row.get("brand",       "—")),
            "description": row.get("product_description", row.get("description", "")),
            "score":       float(dist),
        })
    return results


# ─────────────────────────────────────────────
# UI Components
# ─────────────────────────────────────────────
def render_hero():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-eyebrow">⬡ Amazon ML School · Portfolio Project</div>
        <div class="hero-title">ShopSense</div>
        <div class="hero-subtitle">
            Dense semantic retrieval over 20,000 Amazon products using
            Sentence Transformers and FAISS vector search — built on the ESCI dataset.
        </div>
        <div class="hero-badge-row">
            <div class="hero-badge"><span>◉</span>all-MiniLM-L6-v2</div>
            <div class="hero-badge"><span>◉</span>FAISS IVF Index</div>
            <div class="hero-badge"><span>◉</span>Amazon ESCI Dataset</div>
            <div class="hero-badge"><span>◉</span>Recall@10 = 0.636</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_result_card(r: dict):
    rank = r["rank"]
    rank_class = {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "rank-other")
    rank_label = {1: "🥇 #1", 2: "🥈 #2", 3: "🥉 #3"}.get(rank, f"# {rank}")

    title      = str(r["title"])[:120] + ("…" if len(str(r["title"])) > 120 else "")
    brand      = str(r["brand"]) if r["brand"] != "nan" else "—"
    score      = r["score"]
    desc       = str(r["description"])

    st.markdown(f"""
    <div class="result-card {rank_class}">
        <div class="card-header">
            <div class="card-rank">{rank_label}</div>
            <div class="card-title">{title}</div>
            <div class="card-score-pill">{score:.4f}</div>
        </div>
        <div class="card-brand">Brand: <span>{brand}</span></div>
    </div>
    """, unsafe_allow_html=True)

    if desc and desc != "nan" and len(desc) > 10:
        with st.expander("View description"):
            st.markdown(
                f'<p style="font-size:13px;color:#8891A8;line-height:1.7;">{desc[:800]}</p>',
                unsafe_allow_html=True,
            )


def section_header(label: str):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-header-line"></div>
        <div class="section-header-label">{label}</div>
        <div class="section-header-line"></div>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline():
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step">
            <div class="pipeline-step-icon">💬</div>
            <div class="pipeline-step-label">Query</div>
            <div class="pipeline-step-sub">Natural language input</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">
            <div class="pipeline-step-icon">🧠</div>
            <div class="pipeline-step-label">Encoder</div>
            <div class="pipeline-step-sub">all-MiniLM-L6-v2 · 384-dim</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">
            <div class="pipeline-step-icon">📐</div>
            <div class="pipeline-step-label">Normalize</div>
            <div class="pipeline-step-sub">L2 unit sphere</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">
            <div class="pipeline-step-icon">⚡</div>
            <div class="pipeline-step-label">FAISS</div>
            <div class="pipeline-step-sub">Inner product ANN search</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">
            <div class="pipeline-step-icon">🏆</div>
            <div class="pipeline-step-label">Top-K</div>
            <div class="pipeline-step-sub">Ranked by similarity score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_highlights():
    st.markdown("""
    <div class="highlights-grid">
        <div class="highlight-chip">
            <div class="highlight-chip-icon">🔍</div>
            <div>
                <div class="highlight-chip-text">Semantic Search</div>
                <div class="highlight-chip-sub">Meaning over keywords</div>
            </div>
        </div>
        <div class="highlight-chip">
            <div class="highlight-chip-icon">🧬</div>
            <div>
                <div class="highlight-chip-text">Vector Embeddings</div>
                <div class="highlight-chip-sub">384-dim dense representations</div>
            </div>
        </div>
        <div class="highlight-chip">
            <div class="highlight-chip-icon">⚡</div>
            <div>
                <div class="highlight-chip-text">FAISS Indexing</div>
                <div class="highlight-chip-sub">Sub-linear ANN retrieval</div>
            </div>
        </div>
        <div class="highlight-chip">
            <div class="highlight-chip-icon">📊</div>
            <div>
                <div class="highlight-chip-text">IR Evaluation</div>
                <div class="highlight-chip-sub">Recall, MRR, NDCG metrics</div>
            </div>
        </div>
        <div class="highlight-chip">
            <div class="highlight-chip-icon">🏷️</div>
            <div>
                <div class="highlight-chip-text">ESCI Labels</div>
                <div class="highlight-chip-sub">Exact · Substitute · Complement · Irrelevant</div>
            </div>
        </div>
        <div class="highlight-chip">
            <div class="highlight-chip-icon">🚀</div>
            <div>
                <div class="highlight-chip-text">Two-Tower Ready</div>
                <div class="highlight-chip-sub">Extensible to fine-tuned bi-encoder</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_dataset_info():
    st.markdown("""
    <div class="dataset-block">
        <p>
            The <strong>Amazon ESCI Dataset</strong> is a large-scale, human-annotated benchmark for product search
            released by Amazon as part of KDD Cup 2022. Each query-product pair carries one of four
            relevance labels: <strong>Exact</strong> (direct match), <strong>Substitute</strong> (similar but not identical),
            <strong>Complement</strong> (used together), or <strong>Irrelevant</strong>.<br><br>
            ShopSense uses this dataset to build and evaluate a semantic retrieval pipeline, treating
            <em>Exact</em> matches as positives and measuring how well the system surfaces them within the
            top-10 ranked results across all 13,712 unique queries.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 0 20px;">
            <div style="font-size:22px;font-weight:800;color:#FF9900;letter-spacing:-0.5px;">ShopSense</div>
            <div style="font-size:11px;color:#4A5168;font-family:'JetBrains Mono',monospace;letter-spacing:1.5px;margin-top:2px;">SEMANTIC RETRIEVAL</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-label">DATASET STATS</div>', unsafe_allow_html=True)

        stats = [
            ("Products",           "20,000",  "Amazon catalogue entries"),
            ("Query-Product Pairs","52,687",  "Annotated relevance pairs"),
            ("Unique Queries",     "13,712",  "Natural language queries"),
        ]
        for label, value, sub in stats:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-label">{label}</div>
                <div class="metric-card-value">{value}</div>
                <div class="metric-card-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">EVAL METRICS (@10)</div>', unsafe_allow_html=True)

        metrics = [
            ("Recall@10", 0.6360),
            ("MRR@10",    0.3847),
            ("NDCG@10",   0.3710),
        ]
        items_html = ""
        for name, val in metrics:
            bar_pct = int(val * 100)
            items_html += f"""
            <div class="eval-row">
                <div>
                    <div class="eval-metric-name">{name}</div>
                </div>
                <div style="display:flex;align-items:center;gap:10px;">
                    <div class="eval-metric-val">{val:.4f}</div>
                    <div class="eval-metric-bar-bg">
                        <div class="eval-metric-bar-fill" style="width:{bar_pct}%;"></div>
                    </div>
                </div>
            </div>
            """
        st.markdown(f"""
        <div style="background:#1A1D28;border:1px solid #2A2F42;border-radius:10px;padding:16px 18px;">
            {items_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">MODEL</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#1A1D28;border:1px solid #2A2F42;border-radius:10px;padding:14px 16px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#FF9900;font-weight:600;">all-MiniLM-L6-v2</div>
            <div style="font-size:11px;color:#4A5168;margin-top:4px;">Sentence Transformers · 384-dim</div>
            <div style="font-size:11px;color:#4A5168;margin-top:2px;">FAISS Inner Product Index</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
def main():
    render_sidebar()

    # Load resources (cached)
    with st.spinner("Loading model and index…"):
        model    = load_model()
        index    = load_faiss_index()
        products = load_products()

    # Hero
    render_hero()

    # ── Search ──
    section_header("SEMANTIC SEARCH")

    st.markdown('<div class="search-wrapper"><div class="search-label">Enter a product query</div>', unsafe_allow_html=True)
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            label="query_input",
            label_visibility="collapsed",
            placeholder='e.g. "wireless noise cancelling headphones for travel"',
            key="query_input",
        )
    with col_btn:
        search_clicked = st.button("Search", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Results ──
    if search_clicked and query.strip():
        section_header("TOP 10 RESULTS")
        with st.spinner("Searching…"):
            t0      = time.time()
            results = semantic_search(query.strip(), model, index, products, top_k=10)
            elapsed = time.time() - t0

        st.markdown(
            f'<p style="font-size:12px;color:#4A5168;font-family:JetBrains Mono,monospace;'
            f'margin-bottom:18px;">Retrieved {len(results)} results in {elapsed*1000:.1f} ms</p>',
            unsafe_allow_html=True,
        )
        for r in results:
            render_result_card(r)

    elif search_clicked and not query.strip():
        st.warning("Please enter a query to search.")

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <div class="empty-state-text" style="color:#4A5168;">
                Enter a natural language query above to retrieve semantically similar products.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Architecture ──
    section_header("RETRIEVAL PIPELINE")
    render_pipeline()

    # ── Project Highlights ──
    section_header("PROJECT HIGHLIGHTS")
    render_highlights()

    # ── Dataset ──
    section_header("ABOUT THE DATASET")
    render_dataset_info()

    # ── Footer ──
    st.markdown("""
    <div style="text-align:center;padding:40px 0 20px;color:#2A2F42;
                font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:1px;">
        SHOPSENSE · AMAZON ML SCHOOL APPLICATION · BUILT WITH SENTENCE TRANSFORMERS + FAISS
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()