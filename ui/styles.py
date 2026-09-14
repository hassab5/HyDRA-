"""
ui/styles.py — CSS theming for HyDRA
=====================================
All custom CSS lives here. Import and call inject_css() once from app.py.
"""

import streamlit as st


def inject_css():
    """Inject the full HyDRA theme into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Design Tokens ───────────────────────────────────── */
:root {
    --bg-primary:   #0a0e1a;
    --accent-blue:  #6366f1;
    --accent-cyan:  #22d3ee;
    --accent-green: #10b981;
    --text-secondary: #94a3b8;
}

/* ── App Background ──────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 50%, #1a0a2e 100%);
}
.main .block-container {
    max-width: 1400px;
    padding: 1rem 2rem;
}

/* ── Sidebar ─────────────────────────────────────────── */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    border-right: 1px solid rgba(99, 102, 241, .2);
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(17, 24, 39, .8);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(99, 102, 241, .15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 13px;
    color: var(--text-secondary);
    transition: all .3s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: white !important;
}

/* ── Metrics ─────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: rgba(26, 31, 54, .8);
    border: 1px solid rgba(99, 102, 241, .15);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
}

/* ── Typography ──────────────────────────────────────── */
h1, h2, h3 { font-family: 'Inter', sans-serif !important; }

/* ── Hero Section ────────────────────────────────────── */
.hero-title {
    font-size: 2.5rem; font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #22d3ee, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: .5rem;
}
.hero-sub {
    color: var(--text-secondary);
    text-align: center; font-size: 1.1rem; margin-bottom: 2rem;
}

/* ── Cards ───────────────────────────────────────────── */
.card {
    background: rgba(26, 31, 54, .6);
    border: 1px solid rgba(99, 102, 241, .12);
    border-radius: 16px; padding: 24px;
    backdrop-filter: blur(12px); margin-bottom: 1rem;
}

/* ── Step Tracker Cards ──────────────────────────────── */
.step-card-done {
    background: rgba(16, 185, 129, .12);
    border: 1px solid rgba(16, 185, 129, .4);
    border-radius: 10px; padding: 10px;
    text-align: center; color: #10b981;
}
.step-card-run {
    background: rgba(99, 102, 241, .15);
    border: 1px solid rgba(99, 102, 241, .5);
    border-radius: 10px; padding: 10px;
    text-align: center; color: #818cf8;
    animation: pulse 1.5s infinite;
}
.step-card-pause {
    background: rgba(251, 191, 36, .10);
    border: 1px solid rgba(251, 191, 36, .4);
    border-radius: 10px; padding: 10px;
    text-align: center; color: #fbbf24;
}
.step-card-idle {
    background: rgba(30, 41, 59, .4);
    border: 1px solid rgba(30, 41, 59, .8);
    border-radius: 10px; padding: 10px;
    text-align: center; color: #475569;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .6; } }

/* ── Status Banners ──────────────────────────────────── */
.banner-pause {
    background: rgba(251, 191, 36, .10);
    border: 1px solid rgba(251, 191, 36, .4);
    border-radius: 10px; padding: 12px 20px;
    color: #fbbf24; font-weight: 600; margin: 8px 0;
}
.banner-done {
    background: rgba(16, 185, 129, .10);
    border: 1px solid rgba(16, 185, 129, .4);
    border-radius: 10px; padding: 12px 20px;
    color: #10b981; font-weight: 600; margin: 8px 0;
}
</style>
"""
