"""
Mohamed Sayed Hybrid Drug Design App
======================================
Dual-Pathway Hybrid Drug Discovery Pipeline
Combining FBDD + Pharmacophore-Guided Database Screening
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
import logging
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
try:
    from rdkit.Chem import Draw
except ImportError:
    Draw = None
import py3Dmol
from stmol import showmol

from modules.hybrid_orchestrator import HybridPipeline, parse_actives_file, parse_pdb_file
from modules.pharmacophore import FEATURE_COLORS, generate_viewer_html, load_pharmacophore_any_format
from modules.admet import generate_radar_chart, generate_comparison_chart, score_final
from modules.fragment_engine import highlight_parent_fragments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="HyDRA — Hybrid Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #1a1f36;
    --accent-blue: #6366f1;
    --accent-cyan: #22d3ee;
    --accent-green: #10b981;
    --accent-pink: #ec4899;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #1e293b;
}
.stApp { background: linear-gradient(135deg, var(--bg-primary) 0%, #0f172a 50%, #1a0a2e 100%); }
.main .block-container { max-width: 1400px; padding-top: 5rem; padding: 1rem 2rem; }
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: rgba(17,24,39,0.8); border-radius: 12px;
    padding: 4px; border: 1px solid rgba(99,102,241,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 8px 16px; font-family: 'Inter', sans-serif;
    font-weight: 500; font-size: 13px; color: var(--text-secondary);
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-blue), #7c3aed) !important;
    color: white !important;
}
div[data-testid="stMetric"] {
    background: rgba(26,31,54,0.8); border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px; padding: 16px; backdrop-filter: blur(10px);
}
.stDataFrame { border-radius: 12px; overflow: hidden; }
h1, h2, h3 { font-family: 'Inter', sans-serif !important; }
.hero-title {
    font-size: 2.5rem; font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #22d3ee, #10b981);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.5rem;
}
.hero-sub {
    color: var(--text-secondary); text-align: center;
    font-size: 1.1rem; margin-bottom: 2rem;
}
.card {
    background: rgba(26,31,54,0.6); border: 1px solid rgba(99,102,241,0.12);
    border-radius: 16px; padding: 24px; backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}
.step-banner {
    background: rgba(34,211,238,0.08); border: 1px solid rgba(34,211,238,0.3);
    border-radius: 10px; padding: 10px 18px; margin: 6px 0; color: #22d3ee;
}
.pause-banner {
    background: rgba(251,191,36,0.10); border: 1px solid rgba(251,191,36,0.4);
    border-radius: 10px; padding: 10px 18px; margin: 6px 0; color: #fbbf24; font-weight: 600;
}
.done-banner {
    background: rgba(16,185,129,0.10); border: 1px solid rgba(16,185,129,0.4);
    border-radius: 10px; padding: 10px 18px; margin: 6px 0; color: #10b981; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════════
_STEP_NAMES = {
    0: "Not started",
    1: "Pharmacophore built",
    2: "Classifier trained",
    3: "Fragment screening done",
    4: "Database screening done",
    5: "Complete",
}
_STEP_COUNT = 4  # steps 1-4

def _init_state():
    defaults = {
        # Pipeline object
        "pipeline": None,
        # Status flags
        "pipeline_run": False,       # fully complete
        "pipeline_running": False,   # currently in a step
        "pipeline_paused": False,    # paused between steps
        "pipeline_step": 0,          # 0=not started … 5=done
        "pipeline_error": None,
        # Pause signal: set by Pause button, checked between steps
        "pause_requested": False,
        # Auto-advance: set True to trigger next step on rerun
        "pipeline_auto_advance": False,
        # Input cache (so resume doesn't need uploads again)
        "cached_active_mols": None,
        "cached_active_smiles": None,
        "cached_pdb_block": None,
        "cached_pharmit_json": None,
        "cached_frag_path": None,
        "cached_params": None,
        # Download cache — pre-generated after completion
        "dl_frag_csv": None,
        "dl_db_csv": None,
        "dl_combined_csv": None,
        "dl_admet_csv": None,
        "dl_origins_csv": None,
        "dl_docking_csv": None,
        "dl_pharm_json": None,
        "dl_viewer_html": None,
        "dl_zip": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── DEMO DATA ──────────────────────────────────────────────
EGFR_DEMO_SMILES = [
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
    "COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1",
    "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
    "Nc1ccc(-c2cc3c(Nc4cccc(Cl)c4)ncnc3[nH]2)cc1",
    "COc1cc(Nc2ncnc3cc(OC)c(OC)cc23)ccc1NC(=O)C=C",
    "c1ccc(Nc2ncnc3ccccc23)cc1",
]
EGFR_DEMO_PDB = """HEADER    TRANSFERASE                             01-JUL-03   1M17
TITLE     CRYSTAL STRUCTURE OF EGFR KINASE DOMAIN (DEMO EXCERPT)
ATOM      1  N   MET A  696      18.168  54.537  36.088  1.00 41.78           N
ATOM      2  CA  MET A  696      18.399  53.816  34.827  1.00 41.03           C
ATOM      3  C   MET A  696      19.886  53.747  34.555  1.00 39.52           C
ATOM      4  O   MET A  696      20.594  54.748  34.599  1.00 40.57           O
ATOM     20  N   LEU A  718      25.898  51.312  30.088  1.00 29.80           N
ATOM     21  CA  LEU A  718      26.645  50.225  29.490  1.00 28.18           C
ATOM     30  N   VAL A  726      29.128  48.013  25.800  1.00 23.71           N
ATOM     31  CA  VAL A  726      30.003  47.291  24.882  1.00 22.55           C
ATOM     40  N   ALA A  743      31.250  44.188  22.395  1.00 18.43           N
ATOM     41  CA  ALA A  743      31.750  42.835  22.264  1.00 18.62           C
ATOM     50  N   LYS A  745      33.680  42.003  23.775  1.00 20.74           N
ATOM     60  N   THR A  790      30.851  42.810  18.085  1.00 18.75           N
ATOM     61  CA  THR A  790      30.180  41.612  17.593  1.00 18.07           C
ATOM     70  N   MET A  793      32.621  38.723  19.105  1.00 17.63           N
ATOM     80  N   LEU A  788      28.512  44.150  17.655  1.00 19.56           N
ATOM     90  N   ASP A  855      27.115  38.965  23.200  1.00 20.95           N
ATOM    100  N   PHE A  856      24.831  38.230  21.881  1.00 20.42           N
ATOM    110  N   GLU A  762      36.221  44.862  20.042  1.00 21.78           N
ATOM    120  N   ARG A  841      22.985  40.352  25.610  1.00 26.51           N
ATOM    130  N   TRP A  880      24.112  35.525  27.180  1.00 20.06           N
ATOM    140  N   TYR A  869      27.812  33.245  24.388  1.00 18.43           N
ATOM    150  N   HIS A  781      26.050  47.218  15.330  1.00 21.22           N
HETATM  200  C1  ERL A  900      28.500  42.100  20.500  1.00 15.00           C
HETATM  201  C2  ERL A  900      29.200  41.300  21.200  1.00 15.00           C
HETATM  202  N1  ERL A  900      30.100  40.800  20.400  1.00 15.00           N
HETATM  203  C3  ERL A  900      30.500  41.200  19.200  1.00 15.00           C
HETATM  207  O2  ERL A  900      31.500  40.500  18.500  1.00 15.00           O
END
"""


def load_demo_data():
    mols, smiles = [], []
    for smi in EGFR_DEMO_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            smiles.append(smi)
    return mols, smiles, EGFR_DEMO_PDB


def mol_to_png_bytes(smi, size=(300, 200)):
    if Draw is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─── DOWNLOAD CACHE ───────────────────────────────────────────
def _cache_downloads(pipeline):
    """Pre-generate all download data once, stored in session_state.
    Downloads tab reads these keys — never calls pipeline methods directly."""
    try:
        st.session_state.dl_frag_csv     = pipeline.export_fragment_results_csv()
        st.session_state.dl_db_csv       = pipeline.export_database_results_csv()
        st.session_state.dl_combined_csv = pipeline.export_combined_leaderboard_csv()
        st.session_state.dl_admet_csv    = pipeline.export_admet_full_csv()
        st.session_state.dl_origins_csv  = pipeline.export_fragment_origins_csv()
        st.session_state.dl_docking_csv  = pipeline.export_docking_scores_csv()
        st.session_state.dl_pharm_json   = pipeline.export_pharmacophore_json()
        st.session_state.dl_viewer_html  = pipeline.export_viewer_html()
        try:
            st.session_state.dl_zip = pipeline.create_zip()
        except Exception as e:
            logger.warning(f"ZIP pre-generation failed: {e}")
    except Exception as e:
        logger.error(f"Download cache failed: {e}")


# ═══════════════════════════════════════════════════
#  CORE STEP RUNNER  (no threading — runs synchronously)
# ═══════════════════════════════════════════════════
def _run_current_step(pipeline, bar, txt):
    """
    Run exactly ONE pipeline step synchronously.
    Progress bar and status text are updated live via callback.

    Returns:
        "done"    → pipeline fully complete
        "paused"  → user requested pause; state is saved
        "running" → step finished, more steps remain (caller should rerun)
        "error"   → exception occurred
    """
    step = st.session_state.pipeline_step

    def cb(frac, msg):
        """Live progress update — called from within the step function."""
        bar.progress(min(float(frac), 1.0))
        txt.markdown(f"**{msg}**")

    try:
        if step == 0:
            cb(0.00, "🔬 Step 1/4 — Building pharmacophore model…")
            pipeline.step0_pharmacophore(lambda f, m: cb(f * 0.10, m))
            st.session_state.pipeline_step = 1

        elif step == 1:
            cb(0.10, "🤖 Step 2/4 — Training ML classifier…")
            pipeline.step1_classifier(lambda f, m: cb(0.10 + f * 0.10, m))
            st.session_state.pipeline_step = 2

        elif step == 2:
            cb(0.20, "🧬 Step 3/4 — Fragment-based drug design…")
            pipeline.pathway1_fbdd(lambda f, m: cb(0.20 + f * 0.35, m))
            st.session_state.pipeline_step = 3

        elif step == 3:
            cb(0.55, "🗄️ Step 4/4 — Database screening & docking…")
            pipeline.pathway2_screening(lambda f, m: cb(0.55 + f * 0.35, m))
            st.session_state.pipeline_step = 4

        elif step == 4:
            cb(0.92, "📊 Combining results from both pathways…")
            pipeline.combine_results()
            cb(1.00, "✅ Pipeline complete!")
            st.session_state.pipeline_step = 5
            # Pipeline done
            st.session_state.pipeline_run     = True
            st.session_state.pipeline_running = False
            st.session_state.pipeline_paused  = False
            _cache_downloads(pipeline)
            return "done"

        # Step finished — check if pause was requested
        if st.session_state.pause_requested:
            st.session_state.pause_requested  = False
            st.session_state.pipeline_paused  = True
            st.session_state.pipeline_running = False
            current = st.session_state.pipeline_step
            cb(current / 5, f"⏸️ Paused — {_STEP_NAMES.get(current, '')} complete")
            return "paused"

        return "running"

    except Exception as e:
        logger.exception("Pipeline step error")
        st.session_state.pipeline_running = False
        st.session_state.pipeline_error   = str(e)
        return "error"


# ═══════════════════════════════════════════════════
#  AUTO-ADVANCE LOGIC  (top of script, before sidebar)
# ═══════════════════════════════════════════════════
# If auto_advance is set, run the next step immediately on this rerun.
# This creates a "self-advancing" loop: each step triggers the next rerun.
_auto_bar = None
_auto_txt = None

if st.session_state.pipeline_auto_advance and not st.session_state.pipeline_paused:
    st.session_state.pipeline_auto_advance = False
    pipeline = st.session_state.pipeline
    if pipeline and st.session_state.pipeline_step < 5:
        # Create live placeholders for progress
        _auto_bar = st.empty()
        _auto_txt = st.empty()
        bar = _auto_bar.progress(st.session_state.pipeline_step / 5)
        txt = _auto_txt.empty()
        result = _run_current_step(pipeline, bar, txt)
        if result == "running":
            # More steps — trigger another rerun
            st.session_state.pipeline_auto_advance = True
        # rerun to refresh full UI state
        st.rerun()


# ═══════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem">🧬 HyDRA</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="font-size:0.85rem">Hybrid Drug Discovery<br>Research Accelerator</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Pipeline Parameters")
    exhaustiveness = st.slider("Docking Exhaustiveness", 1, 16, 4,
                               help="Higher = more accurate but slower. Capped at 16 for HF Spaces CPU.")
    box_x = st.slider("Box Size X (Å)", 10.0, 40.0, 20.0)
    box_y = st.slider("Box Size Y (Å)", 10.0, 40.0, 20.0)
    box_z = st.slider("Box Size Z (Å)", 10.0, 40.0, 20.0)
    frag_thresh    = st.slider("Fragment Classifier Threshold", 0.5, 1.0, 0.70, 0.05)
    evolved_thresh = st.slider("Evolved Mol Threshold",        0.5, 1.0, 0.80, 0.05)
    db_thresh      = st.slider("Database Hit Threshold",       0.5, 1.0, 0.80, 0.05)

    st.divider()
    st.markdown("### 📊 Pipeline Status")

    step = st.session_state.pipeline_step

    if st.session_state.pipeline_running:
        st.warning("⚙️ Pipeline running…")
        # Step progress indicator
        for s in range(1, 6):
            icon = "✅" if s < step else ("🔄" if s == step else "⬜")
            name = {1:"Pharmacophore", 2:"Classifier", 3:"FBDD", 4:"DB Screening", 5:"Combine"}[s]
            st.caption(f"{icon} {name}")

    elif st.session_state.pipeline_paused:
        st.markdown('<div class="pause-banner">⏸️ Pipeline paused</div>', unsafe_allow_html=True)
        for s in range(1, 6):
            icon = "✅" if s < step else ("⏸️" if s == step else "⬜")
            name = {1:"Pharmacophore", 2:"Classifier", 3:"FBDD", 4:"DB Screening", 5:"Combine"}[s]
            st.caption(f"{icon} {name}")

    elif st.session_state.pipeline_run:
        st.markdown('<div class="done-banner">✅ Pipeline complete</div>', unsafe_allow_html=True)
        pipeline = st.session_state.pipeline
        if pipeline:
            try:
                summary = pipeline.get_summary()
                st.metric("Pharmacophore Features", summary["pharmacophore"]["n_features"])
                st.metric("Fragment Leads",  summary["pathway1"]["n_leads"])
                st.metric("Database Leads",  summary["pathway2"]["n_leads"])
                st.metric("Total Leads",     summary["combined"]["n_total_leads"])
            except Exception:
                pass

    elif st.session_state.pipeline_error:
        st.error(f"❌ {st.session_state.pipeline_error}")

    else:
        st.info("⏳ Pipeline not yet run")


# ═══════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════
tabs = st.tabs([
    "🚀 Launch", "🔬 Pharmacophore", "🧬 Fragment Results",
    "🗄️ Database Results", "📊 Combined Ranking",
    "🔗 Fragment Origins", "🧬 3D Viewer", "📥 Downloads"
])

# ─── TAB 1: LAUNCH ──────────────────────────────────
with tabs[0]:
    st.markdown('<div class="hero-title">🧬 HyDRA</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Dual-Pathway Hybrid Drug Discovery Pipeline<br>Fragment-Based Design × Pharmacophore-Guided Screening</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📁 Required Input")
        pdb_file = st.file_uploader("Protein Structure (.pdb) — **REQUIRED**",
                                    type=["pdb"], key="pdb_upload",
                                    disabled=st.session_state.pipeline_running)
        st.markdown("#### 📁 Active Compounds")
        actives_file = st.file_uploader("Known Binders (.smi, .sdf, .txt)",
                                        type=["smi","sdf","txt"], key="actives_upload",
                                        disabled=st.session_state.pipeline_running)

    with col2:
        st.markdown("#### 📁 Optional Inputs")
        fragment_file = st.file_uploader("Custom Fragment Library (.sdf)",
                                         type=["sdf"], key="frag_upload",
                                         disabled=st.session_state.pipeline_running)
        st.markdown("#### 📁 Upload Pharmacophore")
        pharmit_file = st.file_uploader(
            "Pharmacophore Model (multiple formats)",
            type=["json","ph4","lig","pml","mol2","xyz","csv","txt","sdf","tsv"],
            key="pharmit_upload",
            disabled=st.session_state.pipeline_running,
            help="Supported: Pharmit JSON, MOE .ph4/.lig, PyMOL .pml, .mol2, .xyz, CSV/TSV/TXT, SDF"
        )
        st.markdown("""
        <div class="card" style="border-left:3px solid #6366f1;">
            <strong>🌐 Pharmit Web Server</strong><br>
            <small>Create & search pharmacophore models online:</small><br>
            <a href="https://pharmit.csb.pitt.edu" target="_blank"
               style="color:#22d3ee;text-decoration:none;font-weight:600;">
                🔗 pharmit.csb.pitt.edu
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Step tracker display ──────────────────────────────────
    step = st.session_state.pipeline_step
    if step > 0:
        step_cols = st.columns(5)
        step_info = [
            ("🔬", "Pharmacophore"),
            ("🤖", "Classifier"),
            ("🧬", "FBDD"),
            ("🗄️", "DB Screen"),
            ("📊", "Combine"),
        ]
        for i, (icon, name) in enumerate(step_info, start=1):
            with step_cols[i - 1]:
                if i < step:
                    st.success(f"{icon} {name}\n\n✅ Done")
                elif i == step and st.session_state.pipeline_running:
                    st.warning(f"{icon} {name}\n\n🔄 Running…")
                elif i == step and st.session_state.pipeline_paused:
                    st.warning(f"{icon} {name}\n\n⏸️ Paused")
                else:
                    st.markdown(f"""
                    <div style="border:1px solid #1e293b;border-radius:8px;padding:10px;text-align:center;color:#64748b;">
                    {icon} {name}<br><small>⬜ Pending</small>
                    </div>""", unsafe_allow_html=True)
        st.divider()

    # ── Progress bar (always visible when running or paused) ──
    prog_bar  = st.empty()
    prog_text = st.empty()

    if st.session_state.pipeline_running:
        prog_bar.progress(min(step / 5, 1.0))
        prog_text.markdown(f"**🔄 Step {step}/4 in progress…**")
    elif st.session_state.pipeline_paused:
        prog_bar.progress(min(step / 5, 1.0))
        prog_text.markdown(f'<div class="pause-banner">⏸️ Paused after: {_STEP_NAMES.get(step, "")} — click ▶️ Resume to continue</div>', unsafe_allow_html=True)
    elif st.session_state.pipeline_run:
        prog_bar.progress(1.0)
        prog_text.markdown('<div class="done-banner">✅ Pipeline complete! Check the tabs above for results.</div>', unsafe_allow_html=True)

    # ── Buttons row ──────────────────────────────────────────
    col_demo, col_run, col_pause, col_reset, col_est = st.columns([1, 1, 1, 1, 1])

    with col_demo:
        demo_btn = st.button("🧪 Load EGFR Demo", use_container_width=True, type="secondary",
                             disabled=st.session_state.pipeline_running,
                             key="demo_btn")
        if demo_btn:
            dm, ds, dp = load_demo_data()
            st.session_state.demo_mols   = dm
            st.session_state.demo_smiles = ds
            st.session_state.demo_pdb    = dp
            st.success(f"Loaded {len(dm)} EGFR inhibitors + PDB")

    with col_run:
        is_paused = st.session_state.pipeline_paused
        run_label = "▶️ Resume" if is_paused else (
                    "▶️ Run Pipeline" if not st.session_state.pipeline_run else "🔁 Re-run")
        run_btn = st.button(run_label, use_container_width=True, type="primary",
                            disabled=st.session_state.pipeline_running,
                            key="run_btn")

    with col_pause:
        pause_btn = st.button("⏸️ Pause", use_container_width=True, type="secondary",
                              disabled=not st.session_state.pipeline_running,
                              key="pause_btn")
        if pause_btn:
            st.session_state.pause_requested = True
            st.info("⏸️ Pause requested — will stop after current step finishes.")

    with col_reset:
        reset_btn = st.button("🗑️ Reset", use_container_width=True, type="secondary",
                              disabled=st.session_state.pipeline_running,
                              key="reset_btn")
        if reset_btn:
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    with col_est:
        st.markdown("""
        <div class="card" style="text-align:center;">
        <strong>⏱️ Runtime</strong><br>
        <small>~5–15 min (CPU)</small>
        </div>""", unsafe_allow_html=True)

    # ── RUN / RESUME handler ─────────────────────────────────
    if run_btn:
        error_msg = None

        # If resuming, use cached inputs + existing pipeline
        if is_paused and st.session_state.pipeline is not None:
            pipeline = st.session_state.pipeline
        else:
            # Gather inputs
            active_mols   = []
            active_smiles = []
            pdb_block     = ""
            pharmit_json  = None
            frag_path     = None

            # Demo data
            if hasattr(st.session_state, "demo_mols") and st.session_state.demo_mols:
                active_mols   = st.session_state.demo_mols
                active_smiles = st.session_state.demo_smiles
                pdb_block     = st.session_state.demo_pdb

            # File uploads override demo
            if pdb_file:
                pdb_block = parse_pdb_file(pdb_file.read())
            if actives_file:
                content   = actives_file.read().decode("utf-8")
                parsed    = parse_actives_file(content, actives_file.name)
                active_mols   = [p[0] for p in parsed]
                active_smiles = [p[1] for p in parsed]
            if pharmit_file:
                raw = pharmit_file.read()
                feats = load_pharmacophore_any_format(raw, pharmit_file.name)
                if feats:
                    from modules.pharmacophore import export_pharmit_json as _epj
                    pharmit_json = _epj(feats)
                    st.success(f"Loaded {len(feats)} pharmacophore features")
                else:
                    st.warning("Could not parse pharmacophore file.")
            if fragment_file:
                import tempfile as _tf
                tmp = _tf.NamedTemporaryFile(suffix=".sdf", delete=False)
                tmp.write(fragment_file.read())
                tmp.close()
                frag_path = tmp.name

            # Validate
            if not pdb_block:
                error_msg = "Please upload a PDB file or load demo data first!"
            elif not active_mols and not pharmit_json:
                error_msg = "Please provide active compounds or a pharmacophore JSON!"

            if not error_msg:
                params = {
                    "exhaustiveness": exhaustiveness,
                    "box_size": [box_x, box_y, box_z],
                    "num_modes": 5,
                    "frag_threshold": frag_thresh,
                    "evolved_threshold": evolved_thresh,
                    "db_threshold": db_thresh,
                    "data_dir": os.path.join(os.path.dirname(__file__), "data"),
                }
                pipeline = HybridPipeline(
                    active_mols=active_mols,
                    active_smiles=active_smiles,
                    pdb_block=pdb_block,
                    fragment_sdf_path=frag_path,
                    pharmit_json=pharmit_json,
                    params=params,
                )
                st.session_state.pipeline      = pipeline
                st.session_state.pipeline_step = 0
                st.session_state.pipeline_run  = False
                st.session_state.pipeline_error = None
                # Clear old download cache
                for k in ["dl_frag_csv","dl_db_csv","dl_combined_csv","dl_admet_csv",
                          "dl_origins_csv","dl_docking_csv","dl_pharm_json","dl_viewer_html","dl_zip"]:
                    st.session_state[k] = None

        if error_msg:
            st.error(f"❌ {error_msg}")
        else:
            # Mark as running and trigger first step via auto-advance
            st.session_state.pipeline_running     = True
            st.session_state.pipeline_paused      = False
            st.session_state.pause_requested      = False
            st.session_state.pipeline_auto_advance = True
            st.rerun()


# ─── TAB 2: PHARMACOPHORE ───────────────────────────
with tabs[1]:
    st.markdown("## 🔬 Pharmacophore Model")
    pipeline = st.session_state.pipeline

    if pipeline and getattr(pipeline, "pharmacophore_features", None):
        features = pipeline.pharmacophore_features
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Feature Table")
            feat_data = [{"Type": f["type"],
                          "X": round(f["center"][0], 2),
                          "Y": round(f["center"][1], 2),
                          "Z": round(f["center"][2], 2),
                          "Radius": f.get("radius", 1.5),
                          "Frequency": f.get("frequency", 1.0)}
                         for f in features]
            st.dataframe(pd.DataFrame(feat_data), use_container_width=True)
            if st.session_state.dl_pharm_json:
                st.download_button("📥 Download Pharmacophore JSON",
                    st.session_state.dl_pharm_json,
                    "pharmacophore_model.json", "application/json",
                    key="dl_pharm_tab2")
        with col2:
            st.markdown("### 3D Viewer")
            if pipeline.pdb_block:
                view = py3Dmol.view(width=600, height=450)
                view.addModel(pipeline.pdb_block, "pdb")
                view.setStyle({}, {"cartoon": {"color": "spectrum", "opacity": 0.7}})
                for f in features:
                    color = FEATURE_COLORS.get(f["type"], "#FFFFFF")
                    view.addSphere({"center": {"x": f["center"][0], "y": f["center"][1], "z": f["center"][2]},
                                    "radius": f.get("radius", 1.5), "color": color, "opacity": 0.5})
                view.zoomTo()
                showmol(view, height=450, width=600)

        if getattr(pipeline, "classifier_metrics", None):
            st.markdown("### 🤖 Classifier Metrics")
            m = pipeline.classifier_metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy",  f"{m.get('accuracy',0):.3f}")
            c2.metric("ROC-AUC",   f"{m.get('roc_auc',0):.3f}")
            c3.metric("Precision", f"{m.get('precision',0):.3f}")
            c4.metric("Recall",    f"{m.get('recall',0):.3f}")
            c5.metric("F1 Score",  f"{m.get('f1',0):.3f}")
    else:
        st.info("Run the pipeline first (at least through Step 1) to see pharmacophore results.")


# ─── TAB 3: FRAGMENT RESULTS ────────────────────────
with tabs[2]:
    st.markdown("## 🧬 Fragment-Based Drug Design Results")
    pipeline = st.session_state.pipeline

    if pipeline and getattr(pipeline, "fragment_leads", None):
        if pipeline.pathway1_results:
            r = pipeline.pathway1_results
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Library Size",       r.get("n_library", 0))
            c2.metric("Scored Fragments",   r.get("n_scored", 0))
            c3.metric("Evolved Structures", r.get("n_evolved", 0))
            c4.metric("Final Leads",        len(pipeline.fragment_leads))

        st.markdown("### 🏆 Top Fragment-Derived Leads")
        lead_data = []
        for i, lead in enumerate(pipeline.fragment_leads):
            admet = lead.get("admet", {})
            lead_data.append({
                "Rank": i + 1,
                "SMILES":     lead.get("smiles","")[:60] + "…",
                "Origin":     lead.get("provenance","")[:50],
                "Docking":    f"{lead.get('docking_score',0):.2f}",
                "Classifier": f"{lead.get('classifier_score',0):.3f}",
                "MPO":        f"{lead.get('mpo_score',0):.3f}",
                "Final":      f"{lead.get('final_score',0):.4f}",
                "MW": admet.get("MW",""), "QED": admet.get("QED",""),
            })
        st.dataframe(pd.DataFrame(lead_data), use_container_width=True, height=500)

        st.markdown("### 📊 ADMET Profiles")
        sel = st.selectbox("Select Lead for ADMET Detail", range(len(pipeline.fragment_leads)),
                           format_func=lambda i: f"Lead {i+1}: {pipeline.fragment_leads[i].get('smiles','')[:40]}…")
        if sel is not None:
            lead  = pipeline.fragment_leads[sel]
            admet = lead.get("admet", {})
            if admet:
                ca, cb_ = st.columns(2)
                with ca:
                    st.plotly_chart(generate_radar_chart(admet, f"Lead {sel+1}"), use_container_width=True)
                with cb_:
                    smi = lead.get("smiles","")
                    mol = Chem.MolFromSmiles(smi)
                    if mol and Draw:
                        st.image(Draw.MolToImage(mol, size=(400, 300)), caption=f"Lead {sel+1}")
                    st.markdown(f"**SMILES:** `{smi}`")
                    st.markdown(f"**Provenance:** {lead.get('provenance','')}")
                    st.markdown(f"**Docking:** {lead.get('docking_score',0):.2f} kcal/mol")
    else:
        st.info("Run the pipeline first (through Step 3 — FBDD) to see fragment results.")


# ─── TAB 4: DATABASE RESULTS ────────────────────────
with tabs[3]:
    st.markdown("## 🗄️ Database Screening Results")
    pipeline = st.session_state.pipeline

    if pipeline and getattr(pipeline, "database_leads", None):
        if pipeline.pathway2_results:
            r  = pipeline.pathway2_results
            sc = r.get("source_counts", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Screened",  r.get("total_screened", 0))
            c2.metric("Above Threshold", r.get("total_filtered", 0))
            c3.metric("Sources Used",    len(sc))
            c4.metric("Final Leads",     len(pipeline.database_leads))
            if sc:
                fig = px.pie(names=list(sc.keys()), values=list(sc.values()),
                             color_discrete_sequence=px.colors.qualitative.Set3, hole=0.4)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#94a3b8", height=350)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🏆 Top Database Leads")
        db_data = []
        for i, lead in enumerate(pipeline.database_leads):
            admet = lead.get("admet", {})
            db_data.append({
                "Rank": i + 1,
                "SMILES":     lead.get("smiles","")[:60] + "…",
                "Source":     lead.get("source",""),
                "Docking":    f"{lead.get('docking_score',0):.2f}",
                "Classifier": f"{lead.get('classifier_score',0):.3f}",
                "MPO":        f"{lead.get('mpo_score',0):.3f}",
                "Final":      f"{lead.get('final_score',0):.4f}",
                "MW": admet.get("MW",""), "QED": admet.get("QED",""),
            })
        st.dataframe(pd.DataFrame(db_data), use_container_width=True, height=500)

        sel_db = st.selectbox("Select DB Lead for ADMET", range(len(pipeline.database_leads)),
                              format_func=lambda i: f"Lead {i+1}: {pipeline.database_leads[i].get('smiles','')[:40]}…",
                              key="db_admet_sel")
        if sel_db is not None:
            lead  = pipeline.database_leads[sel_db]
            admet = lead.get("admet", {})
            if admet:
                ca, cb_ = st.columns(2)
                with ca:
                    st.plotly_chart(generate_radar_chart(admet, f"DB Lead {sel_db+1}"), use_container_width=True)
                with cb_:
                    smi = lead.get("smiles","")
                    mol = Chem.MolFromSmiles(smi)
                    if mol and Draw:
                        st.image(Draw.MolToImage(mol, size=(400,300)), caption=f"DB Lead {sel_db+1}")
                    st.markdown(f"**Source:** {lead.get('source','')}")
                    st.markdown(f"**Docking:** {lead.get('docking_score',0):.2f} kcal/mol")
    else:
        st.info("Run the pipeline first (through Step 4 — DB Screening) to see database results.")


# ─── TAB 5: COMBINED RANKING ────────────────────────
with tabs[4]:
    st.markdown("## 📊 Combined Leaderboard")
    pipeline = st.session_state.pipeline

    if pipeline and getattr(pipeline, "combined_leaderboard", None):
        lb = pipeline.combined_leaderboard
        st.markdown(f"### All {len(lb)} Leads — Ranked by Final Score")
        lb_data = []
        for e in lb:
            admet = e.get("admet", {})
            lb_data.append({
                "Rank":        e["rank"],
                "Pathway":     e["pathway"],
                "SMILES":      e.get("smiles","")[:50] + "…",
                "Docking":     e.get("docking_score", 0),
                "Classifier":  e.get("classifier_score", 0),
                "MPO":         e.get("mpo_score", 0),
                "Final Score": e.get("final_score", 0),
                "MW":   admet.get("MW",""), "cLogP": admet.get("cLogP",""),
                "TPSA": admet.get("TPSA",""), "QED":  admet.get("QED",""),
            })
        st.dataframe(pd.DataFrame(lb_data), use_container_width=True, height=600)

        st.markdown("### 🔀 Multi-Parameter Visualization")
        pc = [{"Docking Score": e.get("docking_score",0),
               "MW":   e.get("admet",{}).get("MW",0),
               "cLogP": e.get("admet",{}).get("cLogP",0),
               "TPSA": e.get("admet",{}).get("TPSA",0),
               "QED":  e.get("admet",{}).get("QED",0),
               "Pathway": 0 if e.get("pathway_label")=="Fragment" else 1}
              for e in lb]
        if pc:
            fig = px.parallel_coordinates(pd.DataFrame(pc), color="Pathway",
                color_continuous_scale=["#6366f1","#ec4899"],
                dimensions=["Docking Score","MW","cLogP","TPSA","QED"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#94a3b8", height=450)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📈 Docking vs Classifier Score")
        sc_data = [{"Vina Score": e.get("docking_score",0),
                    "Classifier Prob": e.get("classifier_score",0),
                    "Pathway": e.get("pathway_label",""),
                    "SMILES": e.get("smiles","")[:30]}
                   for e in lb]
        if sc_data:
            fig = px.scatter(pd.DataFrame(sc_data), x="Vina Score", y="Classifier Prob",
                             color="Pathway", hover_data=["SMILES"],
                             color_discrete_map={"Fragment":"#6366f1","Database":"#ec4899"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#94a3b8", height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the full pipeline to see combined results.")


# ─── TAB 6: FRAGMENT ORIGINS ────────────────────────
with tabs[5]:
    st.markdown("## 🔗 Fragment Provenance & Origins")
    pipeline = st.session_state.pipeline

    if pipeline and getattr(pipeline, "fragment_leads", None):
        for i, lead in enumerate(pipeline.fragment_leads[:20]):
            with st.expander(f"Lead {i+1}: {lead.get('smiles','')[:50]}…", expanded=(i < 3)):
                st.markdown(f"**Provenance:** `{lead.get('provenance','')}`")
                st.markdown(f"**Method:** {lead.get('method','')}")
                st.markdown(f"**Docking Score:** {lead.get('docking_score',0):.2f} kcal/mol")
                st.markdown(f"**Final Score:** {lead.get('final_score',0):.4f}")
                smi = lead.get("smiles","")
                mol = Chem.MolFromSmiles(smi)
                if mol and Draw:
                    st.image(Draw.MolToImage(mol, size=(500, 300)), caption="Evolved Molecule")
    else:
        st.info("Run the pipeline first to see fragment origins.")


# ─── TAB 7: 3D VIEWER ───────────────────────────────
with tabs[6]:
    st.markdown("## 🧬 Interactive 3D Molecular Viewer")
    pipeline = st.session_state.pipeline

    if pipeline and getattr(pipeline, "combined_leaderboard", None) and pipeline.pdb_block:
        lb = pipeline.combined_leaderboard
        sel_mol = st.selectbox("Select Molecule", range(len(lb)),
            format_func=lambda i: f"#{lb[i]['rank']} {lb[i]['pathway']} — {lb[i].get('smiles','')[:40]}…")
        if sel_mol is not None:
            entry = lb[sel_mol]
            view  = py3Dmol.view(width=800, height=550)
            view.addModel(pipeline.pdb_block, "pdb")
            view.setStyle({}, {"cartoon": {"color": "spectrum", "opacity": 0.6}})
            view.addSurface(py3Dmol.VDW, {"opacity": 0.1, "color": "white"})
            if getattr(pipeline, "pharmacophore_features", None):
                for f in pipeline.pharmacophore_features:
                    color = FEATURE_COLORS.get(f["type"], "#FFFFFF")
                    view.addSphere({"center": {"x": f["center"][0], "y": f["center"][1], "z": f["center"][2]},
                                    "radius": f.get("radius",1.5), "color": color, "opacity": 0.35})
            smi = entry.get("smiles","")
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    mol3d = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol3d, AllChem.ETKDGv3())
                    AllChem.MMFFOptimizeMolecule(mol3d)
                    view.addModel(Chem.MolToMolBlock(mol3d), "mol")
                    view.setStyle({"model": -1}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.15}})
                except Exception:
                    pass
            view.zoomTo()
            showmol(view, height=550, width=800)
            st.markdown(f"**SMILES:** `{smi}`")
            st.markdown(f"**Pathway:** {entry.get('pathway','')}")
    else:
        st.info("Run the full pipeline to use the 3D viewer.")


# ─── TAB 8: DOWNLOADS ───────────────────────────────
with tabs[7]:
    st.markdown("## 📥 Download All Results")

    has_results = st.session_state.pipeline_run and any([
        st.session_state.dl_frag_csv,
        st.session_state.dl_combined_csv,
    ])

    if has_results:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 📄 CSV Reports")
            if st.session_state.dl_frag_csv:
                st.download_button("⬇️ Fragment Results CSV",
                    st.session_state.dl_frag_csv, "fragment_results.csv", "text/csv",
                    key="dl_frag_btn")
            if st.session_state.dl_db_csv:
                st.download_button("⬇️ Database Results CSV",
                    st.session_state.dl_db_csv, "database_results.csv", "text/csv",
                    key="dl_db_btn")
            if st.session_state.dl_combined_csv:
                st.download_button("⬇️ Combined Leaderboard CSV",
                    st.session_state.dl_combined_csv, "combined_leaderboard.csv", "text/csv",
                    key="dl_combined_btn")
        with c2:
            st.markdown("### 📊 Detailed Reports")
            if st.session_state.dl_admet_csv:
                st.download_button("⬇️ ADMET Full Report",
                    st.session_state.dl_admet_csv, "admet_full_report.csv", "text/csv",
                    key="dl_admet_btn")
            if st.session_state.dl_origins_csv:
                st.download_button("⬇️ Fragment Origins CSV",
                    st.session_state.dl_origins_csv, "fragment_origins.csv", "text/csv",
                    key="dl_origins_btn")
            if st.session_state.dl_docking_csv:
                st.download_button("⬇️ Docking Scores CSV",
                    st.session_state.dl_docking_csv, "docking_scores_all.csv", "text/csv",
                    key="dl_docking_btn")
        with c3:
            st.markdown("### 📦 Full Package")
            if st.session_state.dl_pharm_json:
                st.download_button("⬇️ Pharmacophore JSON",
                    st.session_state.dl_pharm_json, "pharmacophore_model.json", "application/json",
                    key="dl_pharm_btn")
            if st.session_state.dl_viewer_html:
                st.download_button("⬇️ 3D Viewer HTML",
                    st.session_state.dl_viewer_html, "pharmacophore_viewer.html", "text/html",
                    key="dl_viewer_btn")

        st.divider()
        st.markdown("### 🗜️ Download Everything as ZIP")
        if st.session_state.dl_zip:
            st.download_button("⬇️ Download ZIP Archive",
                st.session_state.dl_zip, "hydra_results.zip", "application/zip",
                use_container_width=True, key="dl_zip_btn")
        else:
            if st.button("📦 Generate ZIP", type="primary", use_container_width=True, key="gen_zip_btn"):
                pipeline = st.session_state.pipeline
                if pipeline:
                    with st.spinner("Packaging…"):
                        try:
                            st.session_state.dl_zip = pipeline.create_zip()
                            st.rerun()
                        except Exception as e:
                            st.error(f"ZIP failed: {e}")

    elif st.session_state.pipeline_paused:
        step = st.session_state.pipeline_step
        st.warning(f"⏸️ Pipeline paused at step {step}/4. Resume to complete and unlock downloads.")
    else:
        st.info("Run the full pipeline to download results.")


# ─── FOOTER ──────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.8rem; padding:1rem;">
    🧬 <strong>HyDRA</strong> — Hybrid Drug Discovery Research Accelerator<br>
    Developed by <strong>Mohamed Sayed</strong> | Powered by RDKit, AutoDock Vina, scikit-learn<br>
    © 2024 — For research use only
</div>
""", unsafe_allow_html=True)
