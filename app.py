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
import plotly.graph_objects as go
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
.main .block-container { max-width: 1400px; padding-top: 5rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem;.
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
.stat-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600;
    background: rgba(99,102,241,0.15); color: #818cf8;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False
if "progress" not in st.session_state:
    st.session_state.progress = 0.0
if "status_msg" not in st.session_state:
    st.session_state.status_msg = ""

# ─── DEMO DATA ──────────────────────────────────────────────
EGFR_DEMO_SMILES = [
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",  # Gefitinib
    "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",        # Erlotinib
    "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",  # Lapatinib
    "COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1",    # Vandetanib
    "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",  # Osimertinib
    "Nc1ccc(-c2cc3c(Nc4cccc(Cl)c4)ncnc3[nH]2)cc1",       # similar core
    "COc1cc(Nc2ncnc3cc(OC)c(OC)cc23)ccc1NC(=O)C=C",      # analog
    "c1ccc(Nc2ncnc3ccccc23)cc1",                           # simple quinazoline
]

EGFR_DEMO_PDB = """HEADER    TRANSFERASE                             01-JUL-03   1M17
TITLE     CRYSTAL STRUCTURE OF EGFR KINASE DOMAIN (DEMO EXCERPT)
ATOM      1  N   MET A  696      18.168  54.537  36.088  1.00 41.78           N
ATOM      2  CA  MET A  696      18.399  53.816  34.827  1.00 41.03           C
ATOM      3  C   MET A  696      19.886  53.747  34.555  1.00 39.52           C
ATOM      4  O   MET A  696      20.594  54.748  34.599  1.00 40.57           O
ATOM      5  CB  MET A  696      17.621  54.463  33.686  1.00 42.39           C
ATOM     20  N   LEU A  718      25.898  51.312  30.088  1.00 29.80           N
ATOM     21  CA  LEU A  718      26.645  50.225  29.490  1.00 28.18           C
ATOM     22  C   LEU A  718      28.061  50.706  29.196  1.00 28.30           C
ATOM     23  O   LEU A  718      28.256  51.757  28.581  1.00 26.23           O
ATOM     30  N   VAL A  726      29.128  48.013  25.800  1.00 23.71           N
ATOM     31  CA  VAL A  726      30.003  47.291  24.882  1.00 22.55           C
ATOM     32  C   VAL A  726      31.478  47.456  25.223  1.00 23.84           C
ATOM     33  O   VAL A  726      31.804  47.876  26.337  1.00 23.31           O
ATOM     40  N   ALA A  743      31.250  44.188  22.395  1.00 18.43           N
ATOM     41  CA  ALA A  743      31.750  42.835  22.264  1.00 18.62           C
ATOM     50  N   LYS A  745      33.680  42.003  23.775  1.00 20.74           N
ATOM     51  CA  LYS A  745      34.556  40.868  23.495  1.00 21.18           C
ATOM     52  C   LYS A  745      35.952  41.344  23.102  1.00 22.28           C
ATOM     60  N   THR A  790      30.851  42.810  18.085  1.00 18.75           N
ATOM     61  CA  THR A  790      30.180  41.612  17.593  1.00 18.07           C
ATOM     62  C   THR A  790      30.875  40.323  18.022  1.00 18.28           C
ATOM     63  O   THR A  790      30.274  39.262  17.896  1.00 20.14           O
ATOM     70  N   MET A  793      32.621  38.723  19.105  1.00 17.63           N
ATOM     71  CA  MET A  793      33.271  37.473  18.729  1.00 17.22           C
ATOM     80  N   LEU A  788      28.512  44.150  17.655  1.00 19.56           N
ATOM     81  CA  LEU A  788      27.775  43.116  18.364  1.00 18.15           C
ATOM     90  N   ASP A  855      27.115  38.965  23.200  1.00 20.95           N
ATOM     91  CA  ASP A  855      26.412  37.801  23.715  1.00 21.97           C
ATOM     92  C   ASP A  855      25.105  37.527  22.973  1.00 22.64           C
ATOM    100  N   PHE A  856      24.831  38.230  21.881  1.00 20.42           N
ATOM    101  CA  PHE A  856      23.593  38.027  21.130  1.00 20.11           C
ATOM    110  N   GLU A  762      36.221  44.862  20.042  1.00 21.78           N
ATOM    111  CA  GLU A  762      36.688  44.095  18.891  1.00 22.23           C
ATOM    120  N   ARG A  841      22.985  40.352  25.610  1.00 26.51           N
ATOM    121  CA  ARG A  841      21.785  39.679  26.095  1.00 28.83           C
ATOM    130  N   TRP A  880      24.112  35.525  27.180  1.00 20.06           N
ATOM    131  CA  TRP A  880      23.558  34.266  27.677  1.00 20.18           C
ATOM    140  N   TYR A  869      27.812  33.245  24.388  1.00 18.43           N
ATOM    141  CA  TYR A  869      27.255  31.988  23.882  1.00 18.68           C
ATOM    150  N   HIS A  781      26.050  47.218  15.330  1.00 21.22           N
ATOM    151  CA  HIS A  781      25.255  46.188  14.680  1.00 20.94           C
HETATM  200  C1  ERL A  900      28.500  42.100  20.500  1.00 15.00           C
HETATM  201  C2  ERL A  900      29.200  41.300  21.200  1.00 15.00           C
HETATM  202  N1  ERL A  900      30.100  40.800  20.400  1.00 15.00           N
HETATM  203  C3  ERL A  900      30.500  41.200  19.200  1.00 15.00           C
HETATM  204  N2  ERL A  900      29.800  42.000  18.500  1.00 15.00           N
HETATM  205  C4  ERL A  900      28.900  42.500  19.200  1.00 15.00           C
HETATM  206  O1  ERL A  900      27.400  43.100  22.100  1.00 15.00           O
HETATM  207  O2  ERL A  900      31.500  40.500  18.500  1.00 15.00           O
END
"""


def load_demo_data():
    """Load EGFR demo data."""
    mols = []
    smiles = []
    for smi in EGFR_DEMO_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            smiles.append(smi)
    return mols, smiles, EGFR_DEMO_PDB


def mol_to_png_bytes(smi, size=(300, 200)):
    """Convert SMILES to PNG image bytes."""
    if Draw is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


# ═══════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem">🧬 HyDRA</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="font-size:0.85rem">Hybrid Drug Discovery<br>Research Accelerator</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Pipeline Parameters")
    exhaustiveness = st.slider("Docking Exhaustiveness", 1, 32, 8, help="Higher = more accurate but slower")
    box_x = st.slider("Box Size X (Å)", 10.0, 40.0, 20.0)
    box_y = st.slider("Box Size Y (Å)", 10.0, 40.0, 20.0)
    box_z = st.slider("Box Size Z (Å)", 10.0, 40.0, 20.0)
    frag_thresh = st.slider("Fragment Classifier Threshold", 0.5, 1.0, 0.70, 0.05)
    evolved_thresh = st.slider("Evolved Mol Threshold", 0.5, 1.0, 0.80, 0.05)
    db_thresh = st.slider("Database Hit Threshold", 0.5, 1.0, 0.80, 0.05)

    st.divider()
    st.markdown("### 📊 Pipeline Status")
    if st.session_state.pipeline_run:
        st.success("✅ Pipeline completed")
        pipeline = st.session_state.pipeline
        if pipeline:
            summary = pipeline.get_summary()
            st.metric("Pharmacophore Features", summary["pharmacophore"]["n_features"])
            st.metric("Fragment Leads", summary["pathway1"]["n_leads"])
            st.metric("Database Leads", summary["pathway2"]["n_leads"])
            st.metric("Total Leads", summary["combined"]["n_total_leads"])
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
        pdb_file = st.file_uploader("Protein Structure (.pdb) — **REQUIRED**", type=["pdb"], key="pdb_upload")

        st.markdown("#### 📁 Active Compounds")
        actives_file = st.file_uploader("Known Binders (.smi, .sdf, .txt)", type=["smi", "sdf", "txt"], key="actives_upload")

    with col2:
        st.markdown("#### 📁 Optional Inputs")
        fragment_file = st.file_uploader("Custom Fragment Library (.sdf)", type=["sdf"], key="frag_upload")

        st.markdown("#### 📁 Upload Pharmacophore")
        pharmit_file = st.file_uploader(
            "Pharmacophore Model (multiple formats supported)",
            type=["json", "ph4", "lig", "pml", "mol2", "xyz", "csv", "txt", "sdf", "tsv"],
            key="pharmit_upload",
            help="Supported formats: Pharmit JSON, MOE .ph4, .lig, PyMOL .pml, .mol2, .xyz, CSV/TSV/TXT, SDF"
        )

        # Pharmit web server link
        st.markdown("""
        <div class="card" style="border-left: 3px solid #6366f1;">
            <strong>🌐 Pharmit Web Server</strong><br>
            <small>Create, visualize, and search pharmacophore models online:</small><br>
            <a href="https://pharmit.csb.pitt.edu" target="_blank"
               style="color: #22d3ee; text-decoration: none; font-weight: 600;">
                🔗 https://pharmit.csb.pitt.edu
            </a><br>
            <small style="color: var(--text-secondary);">
                Upload your pharmacophore model there to search millions of compounds,
                or create a new model and download it here.
            </small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    col_demo, col_run, col_est = st.columns([1, 1, 1])

    with col_demo:
        demo_btn = st.button("🧪 Load EGFR Demo Data", use_container_width=True, type="secondary")
        if demo_btn:
            demo_mols, demo_smiles, demo_pdb = load_demo_data()
            st.session_state.demo_mols = demo_mols
            st.session_state.demo_smiles = demo_smiles
            st.session_state.demo_pdb = demo_pdb
            st.success(f"✅ Loaded {len(demo_mols)} EGFR inhibitors + PDB structure")

    with col_est:
        st.markdown("""
        <div class="card">
        <strong>⏱️ Estimated Runtime</strong><br>
        <small>~5-15 min (CPU, depending on library size)</small>
        </div>
        """, unsafe_allow_html=True)

    with col_run:
        run_btn = st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary")

    if run_btn:
        # Gather inputs
        active_mols = []
        active_smiles = []
        pdb_block = ""
        pharmit_json = None
        frag_path = None

        # Demo data check
        if hasattr(st.session_state, 'demo_mols') and st.session_state.demo_mols:
            active_mols = st.session_state.demo_mols
            active_smiles = st.session_state.demo_smiles
            pdb_block = st.session_state.demo_pdb

        # File uploads override demo
        if pdb_file:
            pdb_block = parse_pdb_file(pdb_file.read())
        if actives_file:
            content = actives_file.read().decode('utf-8')
            parsed = parse_actives_file(content, actives_file.name)
            active_mols = [p[0] for p in parsed]
            active_smiles = [p[1] for p in parsed]
        if pharmit_file:
            raw_content = pharmit_file.read()
            loaded_features = load_pharmacophore_any_format(raw_content, pharmit_file.name)
            if loaded_features:
                # Convert loaded features back to Pharmit JSON format for the pipeline
                from modules.pharmacophore import export_pharmit_json as _export_pj
                pharmit_json = _export_pj(loaded_features)
                st.success(f"✅ Loaded {len(loaded_features)} pharmacophore features from {pharmit_file.name}")
            else:
                st.warning(f"⚠️ Could not parse pharmacophore features from {pharmit_file.name}. Please check the file format.")
        if fragment_file:
            import tempfile as tf
            tmp = tf.NamedTemporaryFile(suffix='.sdf', delete=False)
            tmp.write(fragment_file.read())
            tmp.close()
            frag_path = tmp.name

        if not pdb_block:
            st.error("❌ Please upload a PDB file or load demo data first!")
        elif not active_mols and not pharmit_json:
            st.error("❌ Please provide active compounds or a pharmacophore JSON!")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(frac, msg):
                progress_bar.progress(min(frac, 1.0))
                status_text.markdown(f"**{msg}**")

            pipeline = HybridPipeline(
                active_mols=active_mols,
                active_smiles=active_smiles,
                pdb_block=pdb_block,
                fragment_sdf_path=frag_path,
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

            with st.spinner("Running dual-pathway pipeline..."):
                try:
                    pipeline.run_full_pipeline(progress_callback=update_progress)
                    st.session_state.pipeline = pipeline
                    st.session_state.pipeline_run = True
                    st.success("✅ Pipeline completed successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Pipeline error: {str(e)}")
                    logger.exception("Pipeline failed")


# ─── TAB 2: PHARMACOPHORE ───────────────────────────
with tabs[1]:
    st.markdown("## 🔬 Pharmacophore Model")
    pipeline = st.session_state.pipeline

    if pipeline and pipeline.pharmacophore_features:
        features = pipeline.pharmacophore_features
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Feature Table")
            feat_data = []
            for f in features:
                feat_data.append({
                    "Type": f["type"],
                    "X": round(f["center"][0], 2),
                    "Y": round(f["center"][1], 2),
                    "Z": round(f["center"][2], 2),
                    "Radius": f.get("radius", 1.5),
                    "Frequency": f.get("frequency", 1.0),
                })
            st.dataframe(pd.DataFrame(feat_data), use_container_width=True)

            if pipeline.pharmit_json:
                st.download_button("📥 Download Pharmacophore JSON",
                    json.dumps(pipeline.pharmit_json, indent=2),
                    "pharmacophore_model.json", "application/json")

        with col2:
            st.markdown("### 3D Viewer")
            if pipeline.pdb_block:
                view = py3Dmol.view(width=600, height=450)
                view.addModel(pipeline.pdb_block, "pdb")
                view.setStyle({}, {"cartoon": {"color": "spectrum", "opacity": 0.7}})
                for f in features:
                    color = FEATURE_COLORS.get(f["type"], "#FFFFFF")
                    view.addSphere({
                        "center": {"x": f["center"][0], "y": f["center"][1], "z": f["center"][2]},
                        "radius": f.get("radius", 1.5), "color": color, "opacity": 0.5,
                    })
                view.zoomTo()
                showmol(view, height=450, width=600)

        if pipeline.classifier_metrics:
            st.markdown("### 🤖 Classifier Metrics")
            metrics = pipeline.classifier_metrics
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            mc2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
            mc3.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            mc4.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            mc5.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
    else:
        st.info("Run the pipeline first to see pharmacophore results.")


# ─── TAB 3: FRAGMENT RESULTS ────────────────────────
with tabs[2]:
    st.markdown("## 🧬 Fragment-Based Drug Design Results")
    pipeline = st.session_state.pipeline

    if pipeline and pipeline.fragment_leads:
        # Top fragments summary
        if pipeline.pathway1_results:
            r = pipeline.pathway1_results
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Library Size", r.get("n_library", 0))
            mc2.metric("Scored Fragments", r.get("n_scored", 0))
            mc3.metric("Evolved Structures", r.get("n_evolved", 0))
            mc4.metric("Final Leads", len(pipeline.fragment_leads))

        st.markdown("### 🏆 Top 20 Fragment-Derived Leads")
        lead_data = []
        for i, lead in enumerate(pipeline.fragment_leads):
            admet = lead.get("admet", {})
            lead_data.append({
                "Rank": i + 1,
                "SMILES": lead.get("smiles", "")[:60] + "...",
                "Origin": lead.get("provenance", "")[:50],
                "Docking": f"{lead.get('docking_score', 0):.2f}",
                "Classifier": f"{lead.get('classifier_score', 0):.3f}",
                "MPO": f"{lead.get('mpo_score', 0):.3f}",
                "Final": f"{lead.get('final_score', 0):.4f}",
                "MW": admet.get("MW", ""),
                "QED": admet.get("QED", ""),
            })
        st.dataframe(pd.DataFrame(lead_data), use_container_width=True, height=500)

        # ADMET Radar Charts
        st.markdown("### 📊 ADMET Profiles")
        sel_idx = st.selectbox("Select Lead for ADMET Detail", range(len(pipeline.fragment_leads)),
                               format_func=lambda i: f"Lead {i+1}: {pipeline.fragment_leads[i].get('smiles', '')[:40]}...")
        if sel_idx is not None:
            lead = pipeline.fragment_leads[sel_idx]
            admet = lead.get("admet", {})
            if admet:
                col1, col2 = st.columns([1, 1])
                with col1:
                    fig = generate_radar_chart(admet, f"Lead {sel_idx+1} ADMET")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    smi = lead.get("smiles", "")
                    mol = Chem.MolFromSmiles(smi)
                    if mol and Draw:
                        img = Draw.MolToImage(mol, size=(400, 300))
                        st.image(img, caption=f"Lead {sel_idx+1}")
                    st.markdown(f"**SMILES:** `{smi}`")
                    st.markdown(f"**Provenance:** {lead.get('provenance', '')}")
                    st.markdown(f"**Docking Score:** {lead.get('docking_score', 0):.2f} kcal/mol")
    else:
        st.info("Run the pipeline first to see fragment results.")


# ─── TAB 4: DATABASE RESULTS ────────────────────────
with tabs[3]:
    st.markdown("## 🗄️ Database Screening Results")
    pipeline = st.session_state.pipeline

    if pipeline and pipeline.database_leads:
        if pipeline.pathway2_results:
            r = pipeline.pathway2_results
            sc = r.get("source_counts", {})
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Screened", r.get("total_screened", 0))
            mc2.metric("Above Threshold", r.get("total_filtered", 0))
            mc3.metric("Sources Used", len(sc))
            mc4.metric("Final Leads", len(pipeline.database_leads))

            # Source pie chart
            if sc:
                st.markdown("### 📊 Database Hit Distribution")
                fig = px.pie(
                    names=list(sc.keys()), values=list(sc.values()),
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.4
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font_color='#94a3b8', height=350)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🏆 Top 20 Database Leads")
        db_data = []
        for i, lead in enumerate(pipeline.database_leads):
            admet = lead.get("admet", {})
            db_data.append({
                "Rank": i + 1,
                "SMILES": lead.get("smiles", "")[:60] + "...",
                "Source": lead.get("source", ""),
                "Docking": f"{lead.get('docking_score', 0):.2f}",
                "Classifier": f"{lead.get('classifier_score', 0):.3f}",
                "MPO": f"{lead.get('mpo_score', 0):.3f}",
                "Final": f"{lead.get('final_score', 0):.4f}",
                "MW": admet.get("MW", ""),
                "QED": admet.get("QED", ""),
            })
        st.dataframe(pd.DataFrame(db_data), use_container_width=True, height=500)

        # ADMET for DB leads
        st.markdown("### 📊 ADMET Profiles")
        sel_db = st.selectbox("Select DB Lead", range(len(pipeline.database_leads)),
                              format_func=lambda i: f"Lead {i+1}: {pipeline.database_leads[i].get('smiles','')[:40]}...",
                              key="db_admet_sel")
        if sel_db is not None:
            lead = pipeline.database_leads[sel_db]
            admet = lead.get("admet", {})
            if admet:
                col1, col2 = st.columns([1, 1])
                with col1:
                    fig = generate_radar_chart(admet, f"DB Lead {sel_db+1}")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    smi = lead.get("smiles", "")
                    mol = Chem.MolFromSmiles(smi)
                    if mol and Draw:
                        img = Draw.MolToImage(mol, size=(400, 300))
                        st.image(img, caption=f"DB Lead {sel_db+1}")
                    st.markdown(f"**Source:** {lead.get('source', '')}")
                    st.markdown(f"**Docking:** {lead.get('docking_score', 0):.2f} kcal/mol")
    else:
        st.info("Run the pipeline first to see database results.")


# ─── TAB 5: COMBINED RANKING ────────────────────────
with tabs[4]:
    st.markdown("## 📊 Combined Leaderboard")
    pipeline = st.session_state.pipeline

    if pipeline and pipeline.combined_leaderboard:
        lb = pipeline.combined_leaderboard
        st.markdown(f"### All {len(lb)} Leads — Ranked by Final Score")

        lb_data = []
        for entry in lb:
            admet = entry.get("admet", {})
            lb_data.append({
                "Rank": entry["rank"],
                "Pathway": entry["pathway"],
                "SMILES": entry.get("smiles", "")[:50] + "...",
                "Docking": entry.get("docking_score", 0),
                "Classifier": entry.get("classifier_score", 0),
                "MPO": entry.get("mpo_score", 0),
                "Final Score": entry.get("final_score", 0),
                "MW": admet.get("MW", ""),
                "cLogP": admet.get("cLogP", ""),
                "TPSA": admet.get("TPSA", ""),
                "QED": admet.get("QED", ""),
            })
        st.dataframe(pd.DataFrame(lb_data), use_container_width=True, height=600)

        # Parallel coordinates
        st.markdown("### 🔀 Multi-Parameter Visualization")
        pc_data = []
        for entry in lb:
            admet = entry.get("admet", {})
            pc_data.append({
                "Docking Score": entry.get("docking_score", 0),
                "MW": admet.get("MW", 0),
                "cLogP": admet.get("cLogP", 0),
                "TPSA": admet.get("TPSA", 0),
                "QED": admet.get("QED", 0),
                "Pathway": 0 if entry.get("pathway_label") == "Fragment" else 1,
            })
        if pc_data:
            df_pc = pd.DataFrame(pc_data)
            fig = px.parallel_coordinates(df_pc,
                color="Pathway", color_continuous_scale=["#6366f1", "#ec4899"],
                dimensions=["Docking Score", "MW", "cLogP", "TPSA", "QED"])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#94a3b8', height=450)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter
        st.markdown("### 📈 Docking vs Classifier Score")
        scatter_data = []
        for entry in lb:
            scatter_data.append({
                "Vina Score": entry.get("docking_score", 0),
                "Classifier Prob": entry.get("classifier_score", 0),
                "Pathway": entry.get("pathway_label", ""),
                "SMILES": entry.get("smiles", "")[:30],
            })
        if scatter_data:
            df_sc = pd.DataFrame(scatter_data)
            fig = px.scatter(df_sc, x="Vina Score", y="Classifier Prob", color="Pathway",
                           hover_data=["SMILES"],
                           color_discrete_map={"Fragment": "#6366f1", "Database": "#ec4899"})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#94a3b8', height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the pipeline first to see combined results.")


# ─── TAB 6: FRAGMENT ORIGINS ────────────────────────
with tabs[5]:
    st.markdown("## 🔗 Fragment Provenance & Origins")
    pipeline = st.session_state.pipeline

    if pipeline and pipeline.fragment_leads:
        for i, lead in enumerate(pipeline.fragment_leads[:20]):
            with st.expander(f"Lead {i+1}: {lead.get('smiles','')[:50]}...", expanded=(i < 3)):
                prov = lead.get("provenance", "")
                st.markdown(f"**Provenance:** `{prov}`")
                st.markdown(f"**Method:** {lead.get('method', '')}")
                st.markdown(f"**Docking Score:** {lead.get('docking_score', 0):.2f} kcal/mol")
                st.markdown(f"**Final Score:** {lead.get('final_score', 0):.4f}")

                smi = lead.get("smiles", "")
                mol = Chem.MolFromSmiles(smi)
                if mol and Draw:
                    img = Draw.MolToImage(mol, size=(500, 300))
                    st.image(img, caption="Evolved Molecule")
    else:
        st.info("Run the pipeline first to see fragment origins.")


# ─── TAB 7: 3D VIEWER ───────────────────────────────
with tabs[6]:
    st.markdown("## 🧬 Interactive 3D Molecular Viewer")
    pipeline = st.session_state.pipeline

    if pipeline and pipeline.combined_leaderboard and pipeline.pdb_block:
        lb = pipeline.combined_leaderboard
        sel_mol = st.selectbox("Select Molecule",
            range(len(lb)),
            format_func=lambda i: f"#{lb[i]['rank']} {lb[i]['pathway']} — {lb[i].get('smiles','')[:40]}...")

        if sel_mol is not None:
            entry = lb[sel_mol]
            view = py3Dmol.view(width=800, height=550)
            view.addModel(pipeline.pdb_block, "pdb")
            view.setStyle({}, {"cartoon": {"color": "spectrum", "opacity": 0.6}})
            view.addSurface(py3Dmol.VDW, {"opacity": 0.1, "color": "white"})

            # Pharmacophore spheres
            if pipeline.pharmacophore_features:
                for f in pipeline.pharmacophore_features:
                    color = FEATURE_COLORS.get(f["type"], "#FFFFFF")
                    view.addSphere({
                        "center": {"x": f["center"][0], "y": f["center"][1], "z": f["center"][2]},
                        "radius": f.get("radius", 1.5), "color": color, "opacity": 0.35,
                    })

            # Add ligand if possible
            smi = entry.get("smiles", "")
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    mol3d = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol3d, AllChem.ETKDGv3())
                    AllChem.MMFFOptimizeMolecule(mol3d)
                    mol_block = Chem.MolToMolBlock(mol3d)
                    view.addModel(mol_block, "mol")
                    view.setStyle({"model": -1}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.15}})
                except Exception:
                    pass

            view.zoomTo()
            showmol(view, height=550, width=800)

            st.markdown(f"**SMILES:** `{smi}`")
            st.markdown(f"**Pathway:** {entry.get('pathway', '')}")
    else:
        st.info("Run the pipeline first to use the 3D viewer.")


# ─── TAB 8: DOWNLOADS ───────────────────────────────
with tabs[7]:
    st.markdown("## 📥 Download All Results")
    pipeline = st.session_state.pipeline

    if pipeline and hasattr(st.session_state, 'pipeline_run') and st.session_state.pipeline_run:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📄 CSV Reports")
            frag_csv = pipeline.export_fragment_results_csv()
            if frag_csv:
                st.download_button("Fragment Results CSV", frag_csv, "fragment_results.csv", "text/csv")
            db_csv = pipeline.export_database_results_csv()
            if db_csv:
                st.download_button("Database Results CSV", db_csv, "database_results.csv", "text/csv")
            combined_csv = pipeline.export_combined_leaderboard_csv()
            if combined_csv:
                st.download_button("Combined Leaderboard CSV", combined_csv, "combined_leaderboard.csv", "text/csv")

        with col2:
            st.markdown("### 📊 Detailed Reports")
            admet_csv = pipeline.export_admet_full_csv()
            if admet_csv:
                st.download_button("ADMET Full Report", admet_csv, "admet_full_report.csv", "text/csv")
            origins_csv = pipeline.export_fragment_origins_csv()
            if origins_csv:
                st.download_button("Fragment Origins CSV", origins_csv, "fragment_origins.csv", "text/csv")
            docking_csv = pipeline.export_docking_scores_csv()
            if docking_csv:
                st.download_button("Docking Scores CSV", docking_csv, "docking_scores_all.csv", "text/csv")

        with col3:
            st.markdown("### 📦 Full Package")
            pharm_json = pipeline.export_pharmacophore_json()
            st.download_button("Pharmacophore JSON", pharm_json, "pharmacophore_model.json", "application/json")
            viewer_html = pipeline.export_viewer_html()
            st.download_button("3D Viewer HTML", viewer_html, "pharmacophore_viewer.html", "text/html")

        st.divider()
        st.markdown("### 🗜️ Download Everything as ZIP")
        if st.button("📦 Generate ZIP Archive", type="primary", use_container_width=True):
            with st.spinner("Packaging all files..."):
                zip_data = pipeline.create_zip()
                st.download_button(
                    "⬇️ Download ZIP",
                    zip_data,
                    "hydra_results.zip",
                    "application/zip",
                    use_container_width=True,
                )
    else:
        st.info("Run the pipeline first to download results.")

# ─── FOOTER ──────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.8rem; padding:1rem;">
    🧬 <strong>HyDRA</strong> — Hybrid Drug Discovery Research Accelerator<br>
    Developed by <strong>Mohamed Sayed</strong> | Powered by RDKit, AutoDock Vina, scikit-learn<br>
    © 2024 — For research use only
</div>
""", unsafe_allow_html=True)
