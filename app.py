"""
Mohamed Sayed Hybrid Drug Design App
======================================
Dual-Pathway Hybrid Drug Discovery Pipeline
Combining FBDD + Pharmacophore-Guided Database Screening
"""

import streamlit as st
import pandas as pd
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
from modules.admet import generate_radar_chart, score_final
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

# ─── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root {
    --bg-primary:#0a0e1a; --accent-blue:#6366f1; --accent-cyan:#22d3ee;
    --accent-green:#10b981; --text-secondary:#94a3b8;
}
.stApp{background:linear-gradient(135deg,#0a0e1a 0%,#0f172a 50%,#1a0a2e 100%);}
.main .block-container{max-width:1400px;padding:1rem 2rem;}
div[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0f172a 0%,#1e1b4b 100%);
    border-right:1px solid rgba(99,102,241,.2);
}
.stTabs [data-baseweb="tab-list"]{
    gap:0;background:rgba(17,24,39,.8);border-radius:12px;
    padding:4px;border:1px solid rgba(99,102,241,.15);
}
.stTabs [data-baseweb="tab"]{
    border-radius:8px;padding:8px 16px;font-family:'Inter',sans-serif;
    font-weight:500;font-size:13px;color:var(--text-secondary);transition:all .3s;
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(135deg,#6366f1,#7c3aed)!important;color:white!important;
}
div[data-testid="stMetric"]{
    background:rgba(26,31,54,.8);border:1px solid rgba(99,102,241,.15);
    border-radius:12px;padding:16px;backdrop-filter:blur(10px);
}
h1,h2,h3{font-family:'Inter',sans-serif!important;}
.hero-title{
    font-size:2.5rem;font-weight:700;
    background:linear-gradient(135deg,#6366f1,#22d3ee,#10b981);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    text-align:center;margin-bottom:.5rem;
}
.hero-sub{color:var(--text-secondary);text-align:center;font-size:1.1rem;margin-bottom:2rem;}
.card{
    background:rgba(26,31,54,.6);border:1px solid rgba(99,102,241,.12);
    border-radius:16px;padding:24px;backdrop-filter:blur(12px);margin-bottom:1rem;
}
.step-card-done{background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.4);
    border-radius:10px;padding:10px;text-align:center;color:#10b981;}
.step-card-run{background:rgba(99,102,241,.15);border:1px solid rgba(99,102,241,.5);
    border-radius:10px;padding:10px;text-align:center;color:#818cf8;animation:pulse 1.5s infinite;}
.step-card-pause{background:rgba(251,191,36,.10);border:1px solid rgba(251,191,36,.4);
    border-radius:10px;padding:10px;text-align:center;color:#fbbf24;}
.step-card-idle{background:rgba(30,41,59,.4);border:1px solid rgba(30,41,59,.8);
    border-radius:10px;padding:10px;text-align:center;color:#475569;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.6;}}
.banner-pause{background:rgba(251,191,36,.10);border:1px solid rgba(251,191,36,.4);
    border-radius:10px;padding:12px 20px;color:#fbbf24;font-weight:600;margin:8px 0;}
.banner-done{background:rgba(16,185,129,.10);border:1px solid rgba(16,185,129,.4);
    border-radius:10px;padding:12px 20px;color:#10b981;font-weight:600;margin:8px 0;}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────
def _init():
    defaults = {
        "pipeline": None,
        "pipeline_run": False,
        "pipeline_paused": False,
        "pipeline_step": 0,       # 0=not started, 1-4=step index, 5=done
        "pipeline_error": None,
        "pause_requested": False,
        "do_advance": False,       # trigger next step on this rerun
        # cached downloads
        "dl_frag_csv":None,"dl_db_csv":None,"dl_combined_csv":None,
        "dl_admet_csv":None,"dl_origins_csv":None,"dl_docking_csv":None,
        "dl_pharm_json":None,"dl_viewer_html":None,"dl_zip":None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

STEP_LABELS = {1:"Pharmacophore",2:"Classifier",3:"FBDD",4:"DB Screen",5:"Combine"}

# ─── DEMO DATA ────────────────────────────────────────────────
EGFR_SMILES = [
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
    "COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1",
    "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
    "Nc1ccc(-c2cc3c(Nc4cccc(Cl)c4)ncnc3[nH]2)cc1",
    "COc1cc(Nc2ncnc3cc(OC)c(OC)cc23)ccc1NC(=O)C=C",
    "c1ccc(Nc2ncnc3ccccc23)cc1",
]
EGFR_PDB = """HEADER    TRANSFERASE                             01-JUL-03   1M17
ATOM      1  N   MET A  696      18.168  54.537  36.088  1.00 41.78           N
ATOM      2  CA  MET A  696      18.399  53.816  34.827  1.00 41.03           C
ATOM      3  C   MET A  696      19.886  53.747  34.555  1.00 39.52           C
ATOM     20  N   LEU A  718      25.898  51.312  30.088  1.00 29.80           N
ATOM     21  CA  LEU A  718      26.645  50.225  29.490  1.00 28.18           C
ATOM     30  N   VAL A  726      29.128  48.013  25.800  1.00 23.71           N
ATOM     40  N   ALA A  743      31.250  44.188  22.395  1.00 18.43           N
ATOM     50  N   LYS A  745      33.680  42.003  23.775  1.00 20.74           N
ATOM     60  N   THR A  790      30.851  42.810  18.085  1.00 18.75           N
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
HETATM  207  O2  ERL A  900      31.500  40.500  18.500  1.00 15.00           O
END
"""

def load_demo():
    mols, smiles = [], []
    for smi in EGFR_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol); smiles.append(smi)
    return mols, smiles, EGFR_PDB

# ─── DOWNLOAD CACHE ───────────────────────────────────────────
def _cache_dl(pipeline):
    try:
        st.session_state.dl_frag_csv     = pipeline.export_fragment_results_csv()
        st.session_state.dl_db_csv       = pipeline.export_database_results_csv()
        st.session_state.dl_combined_csv = pipeline.export_combined_leaderboard_csv()
        st.session_state.dl_admet_csv    = pipeline.export_admet_full_csv()
        st.session_state.dl_origins_csv  = pipeline.export_fragment_origins_csv()
        st.session_state.dl_docking_csv  = pipeline.export_docking_scores_csv()
        st.session_state.dl_pharm_json   = pipeline.export_pharmacophore_json()
        st.session_state.dl_viewer_html  = pipeline.export_viewer_html()
        try:    st.session_state.dl_zip  = pipeline.create_zip()
        except: pass
    except Exception as e:
        logger.error(f"Download cache: {e}")

# ─── STEP RUNNER ─────────────────────────────────────────────
def _run_step(pipeline, bar, txt):
    """
    Run ONE pipeline step synchronously.
    bar  = st.progress() object  — updated live during execution
    txt  = st.empty()   object   — updated live during execution
    Returns: "done" | "paused" | "continue" | "error"
    """
    step = st.session_state.pipeline_step

    def cb(frac, msg):
        bar.progress(min(float(frac), 1.0))
        txt.markdown(f"**{msg}**")

    try:
        if step == 0:
            cb(0.02, "🔬 Step 1/4 — Building pharmacophore model…")
            pipeline.step0_pharmacophore(lambda f,m: cb(f*0.10, m))
            st.session_state.pipeline_step = 1

        elif step == 1:
            cb(0.12, "🤖 Step 2/4 — Training ML classifier…")
            pipeline.step1_classifier(lambda f,m: cb(0.10 + f*0.10, m))
            st.session_state.pipeline_step = 2

        elif step == 2:
            cb(0.22, "🧬 Step 3/4 — Fragment-based drug design…")
            pipeline.pathway1_fbdd(lambda f,m: cb(0.20 + f*0.35, m))
            st.session_state.pipeline_step = 3

        elif step == 3:
            cb(0.57, "🗄️ Step 4/4 — Database screening & docking…")
            pipeline.pathway2_screening(lambda f,m: cb(0.55 + f*0.35, m))
            st.session_state.pipeline_step = 4

        elif step == 4:
            cb(0.93, "📊 Combining results from both pathways…")
            pipeline.combine_results()
            cb(1.00, "✅ Pipeline complete!")
            st.session_state.pipeline_step = 5
            st.session_state.pipeline_run  = True
            st.session_state.pipeline_paused = False
            st.session_state.do_advance    = False
            _cache_dl(pipeline)
            return "done"

        # After step: check pause
        if st.session_state.pause_requested:
            st.session_state.pause_requested = False
            st.session_state.pipeline_paused = True
            st.session_state.do_advance      = False
            current = st.session_state.pipeline_step
            cb(current / 5, f"⏸️ Paused — {STEP_LABELS.get(current,'')} complete")
            return "paused"

        return "continue"

    except Exception as e:
        logger.exception("Step error")
        st.session_state.pipeline_error  = str(e)
        st.session_state.do_advance      = False
        return "error"


# ════════════════════════════════════════════════════════════
#  SIDEBAR  — always renders first, on every rerun
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem">🧬 HyDRA</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="font-size:.85rem">Hybrid Drug Discovery<br>Research Accelerator</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Pipeline Parameters")
    exhaustiveness = st.slider("Docking Exhaustiveness", 1, 16, 4)
    box_x = st.slider("Box X (Å)", 10.0, 40.0, 20.0)
    box_y = st.slider("Box Y (Å)", 10.0, 40.0, 20.0)
    box_z = st.slider("Box Z (Å)", 10.0, 40.0, 20.0)
    frag_thresh    = st.slider("Fragment Threshold",  0.5, 1.0, 0.70, 0.05)
    evolved_thresh = st.slider("Evolved Threshold",   0.5, 1.0, 0.80, 0.05)
    db_thresh      = st.slider("DB Hit Threshold",    0.5, 1.0, 0.80, 0.05)

    st.divider()
    st.markdown("### 📊 Pipeline Status")

    step = st.session_state.pipeline_step
    is_running = st.session_state.do_advance and not st.session_state.pipeline_paused
    is_paused  = st.session_state.pipeline_paused
    is_done    = st.session_state.pipeline_run

    if is_running:
        st.warning("⚙️ Running…")
    elif is_paused:
        st.markdown('<div class="banner-pause">⏸️ Paused</div>', unsafe_allow_html=True)
    elif is_done:
        st.markdown('<div class="banner-done">✅ Complete</div>', unsafe_allow_html=True)
        pipeline_obj = st.session_state.pipeline
        if pipeline_obj:
            try:
                s = pipeline_obj.get_summary()
                st.metric("Pharm. Features", s["pharmacophore"]["n_features"])
                st.metric("Fragment Leads",  s["pathway1"]["n_leads"])
                st.metric("Database Leads",  s["pathway2"]["n_leads"])
                st.metric("Total Leads",     s["combined"]["n_total_leads"])
            except: pass
    elif st.session_state.pipeline_error:
        st.error(f"❌ {st.session_state.pipeline_error}")
    else:
        st.info("⏳ Not yet run")

    # Per-step checklist
    if step > 0:
        st.divider()
        icons = {True: "✅", False: "⬜"}
        run_icon = "🔄" if is_running else ("⏸️" if is_paused else "⬜")
        for s_num, s_name in [(1,"Pharmacophore"),(2,"Classifier"),(3,"FBDD"),(4,"DB Screen"),(5,"Combine")]:
            if s_num < step:
                st.caption(f"✅ {s_name}")
            elif s_num == step and not is_done:
                st.caption(f"{run_icon} {s_name}")
            else:
                st.caption(f"⬜ {s_name}")


# ════════════════════════════════════════════════════════════
#  MAIN TABS
# ════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🚀 Launch","🔬 Pharmacophore","🧬 Fragment Results",
    "🗄️ Database Results","📊 Combined Ranking",
    "🔗 Fragment Origins","🧬 3D Viewer","📥 Downloads"
])

# ─── TAB 1: LAUNCH ──────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="hero-title">🧬 HyDRA</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Dual-Pathway Hybrid Drug Discovery Pipeline<br>Fragment-Based Design × Pharmacophore-Guided Screening</div>', unsafe_allow_html=True)

    # ── File uploads ─────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📁 Required")
        pdb_file = st.file_uploader("Protein (.pdb) — REQUIRED", type=["pdb"], key="pdb_up")
        st.markdown("#### 📁 Active Compounds")
        actives_file = st.file_uploader("Binders (.smi/.sdf/.txt)", type=["smi","sdf","txt"], key="act_up")
    with c2:
        st.markdown("#### 📁 Optional")
        fragment_file = st.file_uploader("Fragment Library (.sdf)", type=["sdf"], key="frag_up")
        st.markdown("#### 📁 Pharmacophore")
        pharmit_file = st.file_uploader(
            "Model (.json/.ph4/.pml/…)",
            type=["json","ph4","lig","pml","mol2","xyz","csv","txt","sdf","tsv"],
            key="pharm_up"
        )
        st.markdown("""
        <div class="card" style="border-left:3px solid #6366f1;padding:12px;">
        <strong>🌐 Pharmit</strong>&nbsp;
        <a href="https://pharmit.csb.pitt.edu" target="_blank" style="color:#22d3ee;">
        pharmit.csb.pitt.edu</a>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Step tracker (visible after run starts) ───────────────
    step = st.session_state.pipeline_step
    if step > 0 or st.session_state.pipeline_run:
        sc = st.columns(5)
        for idx, (snum, sname, sicon) in enumerate([
            (1,"Pharmacophore","🔬"),(2,"Classifier","🤖"),
            (3,"FBDD","🧬"),(4,"DB Screen","🗄️"),(5,"Combine","📊")
        ]):
            with sc[idx]:
                if snum < step or st.session_state.pipeline_run:
                    st.markdown(f'<div class="step-card-done">{sicon}<br><b>{sname}</b><br>✅ Done</div>', unsafe_allow_html=True)
                elif snum == step and not st.session_state.pipeline_paused:
                    st.markdown(f'<div class="step-card-run">{sicon}<br><b>{sname}</b><br>🔄 Running…</div>', unsafe_allow_html=True)
                elif snum == step and st.session_state.pipeline_paused:
                    st.markdown(f'<div class="step-card-pause">{sicon}<br><b>{sname}</b><br>⏸️ Paused</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="step-card-idle">{sicon}<br><b>{sname}</b><br>⬜ Pending</div>', unsafe_allow_html=True)
        st.markdown("")

    # ── Progress bar & status text (live-updated by _run_step) ─
    prog_bar  = st.progress(min(step / 5, 1.0) if step > 0 else 0.0)
    prog_text = st.empty()

    # Static status messages when not actively running
    if st.session_state.pipeline_run:
        prog_text.markdown('<div class="banner-done">✅ Pipeline complete — check the result tabs above!</div>', unsafe_allow_html=True)
    elif st.session_state.pipeline_paused:
        prog_text.markdown(f'<div class="banner-pause">⏸️ Paused after: {STEP_LABELS.get(step,"")} — click ▶️ Resume to continue</div>', unsafe_allow_html=True)
    elif st.session_state.pipeline_error:
        prog_text.error(f"❌ {st.session_state.pipeline_error}")

    # ── Buttons ──────────────────────────────────────────────
    bc1, bc2, bc3, bc4, bc5 = st.columns(5)

    is_paused  = st.session_state.pipeline_paused
    is_running = st.session_state.do_advance and not is_paused
    is_done    = st.session_state.pipeline_run

    with bc1:
        demo_btn = st.button("🧪 EGFR Demo", use_container_width=True, disabled=is_running, key="demo_btn")
    with bc2:
        run_label = "▶️ Resume" if is_paused else ("▶️ Run Pipeline" if not is_done else "🔁 Re-run")
        run_btn = st.button(run_label, use_container_width=True, type="primary", disabled=is_running, key="run_btn")
    with bc3:
        pause_btn = st.button("⏸️ Pause", use_container_width=True, disabled=not is_running, key="pause_btn")
    with bc4:
        reset_btn = st.button("🗑️ Reset", use_container_width=True, disabled=is_running, key="reset_btn")
    with bc5:
        st.markdown('<div class="card" style="padding:8px;text-align:center;"><small>⏱️ ~5–15 min CPU</small></div>', unsafe_allow_html=True)

    # ── Button handlers ───────────────────────────────────────
    if demo_btn:
        dm, ds, dp = load_demo()
        st.session_state.demo_mols   = dm
        st.session_state.demo_smiles = ds
        st.session_state.demo_pdb    = dp
        st.success(f"✅ Loaded {len(dm)} EGFR inhibitors")

    if pause_btn and is_running:
        st.session_state.pause_requested = True
        st.info("⏸️ Pause requested — will stop after this step finishes.")

    if reset_btn:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if run_btn and not is_running:
        error_msg = None

        if is_paused and st.session_state.pipeline is not None:
            # Resume: reuse existing pipeline object
            pass
        else:
            # Fresh run: gather inputs
            active_mols, active_smiles, pdb_block, pharmit_json, frag_path = [], [], "", None, None

            if getattr(st.session_state, "demo_mols", None):
                active_mols   = st.session_state.demo_mols
                active_smiles = st.session_state.demo_smiles
                pdb_block     = st.session_state.demo_pdb

            if pdb_file:
                pdb_block = parse_pdb_file(pdb_file.read())
            if actives_file:
                parsed = parse_actives_file(actives_file.read().decode("utf-8"), actives_file.name)
                active_mols = [p[0] for p in parsed]; active_smiles = [p[1] for p in parsed]
            if pharmit_file:
                feats = load_pharmacophore_any_format(pharmit_file.read(), pharmit_file.name)
                if feats:
                    from modules.pharmacophore import export_pharmit_json as _epj
                    pharmit_json = _epj(feats)
                    st.success(f"✅ Loaded {len(feats)} pharmacophore features")
                else:
                    st.warning("⚠️ Could not parse pharmacophore file.")
            if fragment_file:
                import tempfile as _tf
                tmp = _tf.NamedTemporaryFile(suffix=".sdf", delete=False)
                tmp.write(fragment_file.read()); tmp.close()
                frag_path = tmp.name

            if not pdb_block:
                error_msg = "Please upload a PDB file or load demo data!"
            elif not active_mols and not pharmit_json:
                error_msg = "Please provide active compounds or a pharmacophore JSON!"

            if not error_msg:
                st.session_state.pipeline = HybridPipeline(
                    active_mols=active_mols, active_smiles=active_smiles,
                    pdb_block=pdb_block, fragment_sdf_path=frag_path,
                    pharmit_json=pharmit_json,
                    params={
                        "exhaustiveness": exhaustiveness,
                        "box_size": [box_x, box_y, box_z],
                        "num_modes": 5,
                        "frag_threshold": frag_thresh,
                        "evolved_threshold": evolved_thresh,
                        "db_threshold": db_thresh,
                        "data_dir": os.path.join(os.path.dirname(__file__), "data"),
                    }
                )
                st.session_state.pipeline_step    = 0
                st.session_state.pipeline_run     = False
                st.session_state.pipeline_error   = None
                for k in ["dl_frag_csv","dl_db_csv","dl_combined_csv","dl_admet_csv",
                          "dl_origins_csv","dl_docking_csv","dl_pharm_json","dl_viewer_html","dl_zip"]:
                    st.session_state[k] = None

        if error_msg:
            st.error(f"❌ {error_msg}")
        else:
            st.session_state.pipeline_paused  = False
            st.session_state.pause_requested  = False
            st.session_state.do_advance       = True
            st.rerun()   # rerun so the step tracker shows "step 0 = running"

    # ══════════════════════════════════════════════════════════
    # AUTO-ADVANCE — runs INSIDE tabs[0], AFTER all UI elements
    # This guarantees sidebar + tab structure are always rendered.
    # Each invocation runs ONE step, then reruns to show updated UI.
    # ══════════════════════════════════════════════════════════
    if st.session_state.do_advance and not st.session_state.pipeline_paused:
        pipeline_obj = st.session_state.pipeline
        if pipeline_obj and st.session_state.pipeline_step < 5:
            result = _run_step(pipeline_obj, prog_bar, prog_text)
            if result == "continue":
                # More steps to go — rerun to render updated step tracker
                st.session_state.do_advance = True
                st.rerun()
            elif result == "done":
                st.session_state.do_advance = False
                st.balloons()
                st.rerun()
            elif result == "paused":
                st.session_state.do_advance = False
                st.rerun()
            elif result == "error":
                st.session_state.do_advance = False
                st.rerun()


# ─── TAB 2: PHARMACOPHORE ────────────────────────────────────
with tabs[1]:
    st.markdown("## 🔬 Pharmacophore Model")
    p = st.session_state.pipeline
    if p and getattr(p, "pharmacophore_features", None):
        features = p.pharmacophore_features
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Feature Table")
            fd = [{"Type":f["type"],"X":round(f["center"][0],2),"Y":round(f["center"][1],2),
                   "Z":round(f["center"][2],2),"Radius":f.get("radius",1.5)} for f in features]
            st.dataframe(pd.DataFrame(fd), use_container_width=True)
            if st.session_state.dl_pharm_json:
                st.download_button("📥 Pharmacophore JSON", st.session_state.dl_pharm_json,
                    "pharmacophore.json","application/json", key="dl_pharm_t2")
        with c2:
            st.markdown("### 3D Viewer")
            if p.pdb_block:
                v = py3Dmol.view(width=600, height=450)
                v.addModel(p.pdb_block,"pdb")
                v.setStyle({},{"cartoon":{"color":"spectrum","opacity":0.7}})
                for f in features:
                    v.addSphere({"center":{"x":f["center"][0],"y":f["center"][1],"z":f["center"][2]},
                                 "radius":f.get("radius",1.5),"color":FEATURE_COLORS.get(f["type"],"#FFF"),"opacity":0.5})
                v.zoomTo(); showmol(v, height=450, width=600)
        if getattr(p,"classifier_metrics",None):
            st.markdown("### 🤖 Classifier Metrics")
            m = p.classifier_metrics
            mc = st.columns(5)
            for col, (k, lbl) in zip(mc, [("accuracy","Accuracy"),("roc_auc","ROC-AUC"),
                                           ("precision","Precision"),("recall","Recall"),("f1","F1")]):
                col.metric(lbl, f"{m.get(k,0):.3f}")
    else:
        st.info("Run through Step 1 (Pharmacophore) to see results here.")


# ─── TAB 3: FRAGMENT RESULTS ─────────────────────────────────
with tabs[2]:
    st.markdown("## 🧬 Fragment-Based Drug Design Results")
    p = st.session_state.pipeline
    if p and getattr(p,"fragment_leads",None):
        if p.pathway1_results:
            r = p.pathway1_results
            cc = st.columns(4)
            cc[0].metric("Library",  r.get("n_library",0))
            cc[1].metric("Scored",   r.get("n_scored",0))
            cc[2].metric("Evolved",  r.get("n_evolved",0))
            cc[3].metric("Leads",    len(p.fragment_leads))
        ld = [{"Rank":i+1,"SMILES":l.get("smiles","")[:55]+"…",
               "Docking":f"{l.get('docking_score',0):.2f}",
               "Classifier":f"{l.get('classifier_score',0):.3f}",
               "MPO":f"{l.get('mpo_score',0):.3f}",
               "Final":f"{l.get('final_score',0):.4f}",
               "MW":l.get("admet",{}).get("MW",""),
               "QED":l.get("admet",{}).get("QED","")}
              for i,l in enumerate(p.fragment_leads)]
        st.dataframe(pd.DataFrame(ld), use_container_width=True, height=480)
        sel = st.selectbox("ADMET detail", range(len(p.fragment_leads)),
              format_func=lambda i: f"Lead {i+1}: {p.fragment_leads[i].get('smiles','')[:40]}…")
        if sel is not None:
            lead = p.fragment_leads[sel]; admet = lead.get("admet",{})
            if admet:
                ca, cb_ = st.columns(2)
                with ca: st.plotly_chart(generate_radar_chart(admet,f"Lead {sel+1}"),use_container_width=True)
                with cb_:
                    smi = lead.get("smiles",""); mol = Chem.MolFromSmiles(smi)
                    if mol and Draw: st.image(Draw.MolToImage(mol,(400,300)),caption=f"Lead {sel+1}")
                    st.markdown(f"**SMILES:** `{smi}`")
                    st.markdown(f"**Docking:** {lead.get('docking_score',0):.2f} kcal/mol")
    else:
        st.info("Run through Step 3 (FBDD) to see fragment results.")


# ─── TAB 4: DATABASE RESULTS ─────────────────────────────────
with tabs[3]:
    st.markdown("## 🗄️ Database Screening Results")
    p = st.session_state.pipeline
    if p and getattr(p,"database_leads",None):
        if p.pathway2_results:
            r = p.pathway2_results; sc = r.get("source_counts",{})
            cc = st.columns(4)
            cc[0].metric("Screened",  r.get("total_screened",0))
            cc[1].metric("Filtered",  r.get("total_filtered",0))
            cc[2].metric("Sources",   len(sc))
            cc[3].metric("Leads",     len(p.database_leads))
            if sc:
                fig = px.pie(names=list(sc.keys()),values=list(sc.values()),
                             color_discrete_sequence=px.colors.qualitative.Set3,hole=0.4)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#94a3b8",height=320)
                st.plotly_chart(fig, use_container_width=True)
        ld = [{"Rank":i+1,"SMILES":l.get("smiles","")[:55]+"…","Source":l.get("source",""),
               "Docking":f"{l.get('docking_score',0):.2f}","Classifier":f"{l.get('classifier_score',0):.3f}",
               "MPO":f"{l.get('mpo_score',0):.3f}","Final":f"{l.get('final_score',0):.4f}",
               "MW":l.get("admet",{}).get("MW",""),"QED":l.get("admet",{}).get("QED","")}
              for i,l in enumerate(p.database_leads)]
        st.dataframe(pd.DataFrame(ld), use_container_width=True, height=480)
        sel = st.selectbox("ADMET detail", range(len(p.database_leads)),
              format_func=lambda i: f"Lead {i+1}: {p.database_leads[i].get('smiles','')[:40]}…",key="db_sel")
        if sel is not None:
            lead = p.database_leads[sel]; admet = lead.get("admet",{})
            if admet:
                ca,cb_ = st.columns(2)
                with ca: st.plotly_chart(generate_radar_chart(admet,f"DB Lead {sel+1}"),use_container_width=True)
                with cb_:
                    smi = lead.get("smiles",""); mol = Chem.MolFromSmiles(smi)
                    if mol and Draw: st.image(Draw.MolToImage(mol,(400,300)),caption=f"DB {sel+1}")
                    st.markdown(f"**Source:** {lead.get('source','')}")
                    st.markdown(f"**Docking:** {lead.get('docking_score',0):.2f} kcal/mol")
    else:
        st.info("Run through Step 4 (DB Screening) to see database results.")


# ─── TAB 5: COMBINED RANKING ─────────────────────────────────
with tabs[4]:
    st.markdown("## 📊 Combined Leaderboard")
    p = st.session_state.pipeline
    if p and getattr(p,"combined_leaderboard",None):
        lb = p.combined_leaderboard
        st.markdown(f"### All {len(lb)} Leads — Ranked by Final Score")
        lbd = [{"Rank":e["rank"],"Pathway":e["pathway"],"SMILES":e.get("smiles","")[:50]+"…",
                "Docking":e.get("docking_score",0),"Classifier":e.get("classifier_score",0),
                "MPO":e.get("mpo_score",0),"Final":e.get("final_score",0),
                "MW":e.get("admet",{}).get("MW",""),"cLogP":e.get("admet",{}).get("cLogP",""),
                "TPSA":e.get("admet",{}).get("TPSA",""),"QED":e.get("admet",{}).get("QED","")}
               for e in lb]
        st.dataframe(pd.DataFrame(lbd), use_container_width=True, height=580)
        pc = [{"Docking":e.get("docking_score",0),"MW":e.get("admet",{}).get("MW",0),
               "cLogP":e.get("admet",{}).get("cLogP",0),"TPSA":e.get("admet",{}).get("TPSA",0),
               "QED":e.get("admet",{}).get("QED",0),"Pathway":0 if e.get("pathway_label")=="Fragment" else 1}
              for e in lb]
        if pc:
            fig = px.parallel_coordinates(pd.DataFrame(pc),color="Pathway",
                color_continuous_scale=["#6366f1","#ec4899"],
                dimensions=["Docking","MW","cLogP","TPSA","QED"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#94a3b8",height=430)
            st.plotly_chart(fig, use_container_width=True)
        scd = [{"Vina":e.get("docking_score",0),"Classifier":e.get("classifier_score",0),
                "Pathway":e.get("pathway_label",""),"SMILES":e.get("smiles","")[:30]} for e in lb]
        if scd:
            fig2 = px.scatter(pd.DataFrame(scd),x="Vina",y="Classifier",color="Pathway",hover_data=["SMILES"],
                              color_discrete_map={"Fragment":"#6366f1","Database":"#ec4899"})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#94a3b8",height=380)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run the full pipeline to see combined results.")


# ─── TAB 6: FRAGMENT ORIGINS ─────────────────────────────────
with tabs[5]:
    st.markdown("## 🔗 Fragment Provenance & Origins")
    p = st.session_state.pipeline
    if p and getattr(p,"fragment_leads",None):
        for i, lead in enumerate(p.fragment_leads[:20]):
            with st.expander(f"Lead {i+1}: {lead.get('smiles','')[:50]}…", expanded=(i<3)):
                st.markdown(f"**Provenance:** `{lead.get('provenance','')}`")
                st.markdown(f"**Method:** {lead.get('method','')}")
                st.markdown(f"**Docking:** {lead.get('docking_score',0):.2f} kcal/mol")
                st.markdown(f"**Final Score:** {lead.get('final_score',0):.4f}")
                mol = Chem.MolFromSmiles(lead.get("smiles",""))
                if mol and Draw:
                    st.image(Draw.MolToImage(mol,(500,300)),caption="Evolved Molecule")
    else:
        st.info("Run through Step 3 (FBDD) to see fragment origins.")


# ─── TAB 7: 3D VIEWER ────────────────────────────────────────
with tabs[6]:
    st.markdown("## 🧬 Interactive 3D Molecular Viewer")
    p = st.session_state.pipeline
    if p and getattr(p,"combined_leaderboard",None) and p.pdb_block:
        lb = p.combined_leaderboard
        sel = st.selectbox("Select Molecule", range(len(lb)),
            format_func=lambda i: f"#{lb[i]['rank']} {lb[i]['pathway']} — {lb[i].get('smiles','')[:40]}…")
        if sel is not None:
            e = lb[sel]
            v = py3Dmol.view(width=800, height=540)
            v.addModel(p.pdb_block,"pdb")
            v.setStyle({},{"cartoon":{"color":"spectrum","opacity":0.6}})
            v.addSurface(py3Dmol.VDW,{"opacity":0.1,"color":"white"})
            if getattr(p,"pharmacophore_features",None):
                for f in p.pharmacophore_features:
                    v.addSphere({"center":{"x":f["center"][0],"y":f["center"][1],"z":f["center"][2]},
                                 "radius":f.get("radius",1.5),"color":FEATURE_COLORS.get(f["type"],"#FFF"),"opacity":0.3})
            smi = e.get("smiles",""); mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    mol3d = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol3d, AllChem.ETKDGv3())
                    AllChem.MMFFOptimizeMolecule(mol3d)
                    v.addModel(Chem.MolToMolBlock(mol3d),"mol")
                    v.setStyle({"model":-1},{"stick":{"colorscheme":"greenCarbon","radius":0.15}})
                except: pass
            v.zoomTo(); showmol(v, height=540, width=800)
            st.markdown(f"**SMILES:** `{smi}`")
            st.markdown(f"**Pathway:** {e.get('pathway','')}")
    else:
        st.info("Run the full pipeline to use the 3D viewer.")


# ─── TAB 8: DOWNLOADS ────────────────────────────────────────
with tabs[7]:
    st.markdown("## 📥 Download All Results")
    has = st.session_state.pipeline_run and any([
        st.session_state.dl_frag_csv, st.session_state.dl_combined_csv])
    if has:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 📄 CSV Reports")
            for key, fname, label in [
                ("dl_frag_csv","fragment_results.csv","Fragment Results CSV"),
                ("dl_db_csv","database_results.csv","Database Results CSV"),
                ("dl_combined_csv","combined_leaderboard.csv","Combined Leaderboard CSV"),
            ]:
                if st.session_state[key]:
                    st.download_button(f"⬇️ {label}", st.session_state[key],
                        fname, "text/csv", key=f"btn_{key}")
        with c2:
            st.markdown("### 📊 Detailed Reports")
            for key, fname, label in [
                ("dl_admet_csv","admet_full.csv","ADMET Full Report"),
                ("dl_origins_csv","fragment_origins.csv","Fragment Origins CSV"),
                ("dl_docking_csv","docking_scores.csv","Docking Scores CSV"),
            ]:
                if st.session_state[key]:
                    st.download_button(f"⬇️ {label}", st.session_state[key],
                        fname, "text/csv", key=f"btn_{key}")
        with c3:
            st.markdown("### 📦 Package")
            if st.session_state.dl_pharm_json:
                st.download_button("⬇️ Pharmacophore JSON", st.session_state.dl_pharm_json,
                    "pharmacophore.json","application/json", key="btn_pharm")
            if st.session_state.dl_viewer_html:
                st.download_button("⬇️ 3D Viewer HTML", st.session_state.dl_viewer_html,
                    "viewer.html","text/html", key="btn_html")

        st.divider()
        st.markdown("### 🗜️ ZIP Archive")
        if st.session_state.dl_zip:
            st.download_button("⬇️ Download ZIP", st.session_state.dl_zip,
                "hydra_results.zip","application/zip",
                use_container_width=True, key="btn_zip")
        else:
            if st.button("📦 Generate ZIP", type="primary", use_container_width=True, key="gen_zip"):
                p = st.session_state.pipeline
                if p:
                    with st.spinner("Packaging…"):
                        try:
                            st.session_state.dl_zip = p.create_zip()
                            st.rerun()
                        except Exception as e:
                            st.error(f"ZIP failed: {e}")
    elif st.session_state.pipeline_paused:
        st.warning(f"⏸️ Paused at step {step}/4. Resume to complete and unlock downloads.")
    else:
        st.info("Run the full pipeline to download results.")


# ─── FOOTER ──────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#64748b;font-size:.8rem;padding:1rem;">
🧬 <strong>HyDRA</strong> — Hybrid Drug Discovery Research Accelerator<br>
Developed by <strong>Mohamed Sayed</strong> | Powered by RDKit · AutoDock Vina · scikit-learn<br>
© 2024 — For research use only
</div>""", unsafe_allow_html=True)
