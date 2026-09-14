"""
HyDRA — Hybrid Drug Discovery Research Accelerator
====================================================
Main Streamlit entry point.

Architecture:
    app.py              ← You are here (page config, sidebar, tab routing)
    ui/
        styles.py       ← All CSS theming
        state.py        ← Session-state init, reset, result/download caching
        runner.py       ← Synchronous step-by-step pipeline executor
        demo.py         ← Built-in EGFR demo data
        tabs/
            tab_pharmacophore.py
            tab_fragments.py
            tab_database.py
            tab_combined.py
            tab_origins.py
            tab_viewer3d.py
            tab_downloads.py
    modules/
        hybrid_orchestrator.py   ← Pipeline engine
        pharmacophore.py         ← Pharmacophore model
        admet.py                 ← ADMET scoring & radar charts
        fragment_engine.py       ← Fragment linking / growing
        ...
"""

import os
import logging
import streamlit as st

from modules.hybrid_orchestrator import HybridPipeline, parse_actives_file, parse_pdb_file
from modules.pharmacophore import load_pharmacophore_any_format

from ui.styles import inject_css
from ui.state  import (
    STEP_LABELS, init_session_state, reset_all, clear_results,
)
from ui.runner import run_current_step
from ui.demo   import load_demo

from ui.tabs import (
    tab_pharmacophore, tab_fragments, tab_database,
    tab_combined, tab_origins, tab_viewer3d, tab_downloads,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  1. PAGE CONFIG & THEME
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HyDRA — Hybrid Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
init_session_state()


# ═══════════════════════════════════════════════════════════════
#  2. SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="hero-title" style="font-size:1.5rem">🧬 HyDRA</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub" style="font-size:.85rem">'
        "Hybrid Drug Discovery<br>Research Accelerator</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Pipeline parameters ──────────────────────────────────
    st.markdown("### ⚙️ Pipeline Parameters")
    exhaustiveness = st.slider("Docking Exhaustiveness", 1, 16, 4)
    box_x = st.slider("Box X (Å)", 10.0, 40.0, 20.0)
    box_y = st.slider("Box Y (Å)", 10.0, 40.0, 20.0)
    box_z = st.slider("Box Z (Å)", 10.0, 40.0, 20.0)
    frag_thresh    = st.slider("Fragment Threshold",  0.5, 1.0, 0.70, 0.05)
    evolved_thresh = st.slider("Evolved Threshold",   0.5, 1.0, 0.80, 0.05)
    db_thresh      = st.slider("DB Hit Threshold",    0.5, 1.0, 0.80, 0.05)

    st.divider()

    # ── Live status ──────────────────────────────────────────
    st.markdown("### 📊 Pipeline Status")

    step       = st.session_state.pipeline_step
    is_running = st.session_state.do_advance and not st.session_state.pipeline_paused
    is_paused  = st.session_state.pipeline_paused
    is_done    = st.session_state.pipeline_run

    if is_running:
        st.warning("⚙️ Running…")
    elif is_paused:
        st.markdown('<div class="banner-pause">⏸️ Paused</div>', unsafe_allow_html=True)
    elif is_done:
        st.markdown('<div class="banner-done">✅ Complete</div>', unsafe_allow_html=True)
        s = st.session_state.res_summary
        if s:
            try:
                st.metric("Pharm. Features", s["pharmacophore"]["n_features"])
                st.metric("Fragment Leads",  s["pathway1"]["n_leads"])
                st.metric("Database Leads",  s["pathway2"]["n_leads"])
                st.metric("Total Leads",     s["combined"]["n_total_leads"])
            except Exception:
                pass
    elif st.session_state.pipeline_error:
        st.error(f"❌ {st.session_state.pipeline_error}")
    else:
        st.info("⏳ Not yet run")

    # Per-step checklist
    if step > 0:
        st.divider()
        run_icon = "🔄" if is_running else ("⏸️" if is_paused else "⬜")
        for s_num, s_name in STEP_LABELS.items():
            if s_num < step:
                st.caption(f"✅ {s_name}")
            elif s_num == step and not is_done:
                st.caption(f"{run_icon} {s_name}")
            else:
                st.caption(f"⬜ {s_name}")


# ═══════════════════════════════════════════════════════════════
#  3. MAIN AREA — 8 TABS
# ═══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🚀 Launch",         "🔬 Pharmacophore",   "🧬 Fragment Results",
    "🗄️ Database Results", "📊 Combined Ranking",
    "🔗 Fragment Origins", "🧬 3D Viewer",       "📥 Downloads",
])


# ── TAB 1: LAUNCH (pipeline controls) ───────────────────────
with tabs[0]:
    st.markdown('<div class="hero-title">🧬 HyDRA</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Dual-Pathway Hybrid Drug Discovery Pipeline'
        "<br>Fragment-Based Design × Pharmacophore-Guided Screening</div>",
        unsafe_allow_html=True,
    )

    # ── File uploads ─────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📁 Required")
        pdb_file = st.file_uploader("Protein (.pdb) — REQUIRED", type=["pdb"], key="pdb_up")
        st.markdown("#### 📁 Active Compounds")
        actives_file = st.file_uploader(
            "Binders (.smi/.sdf/.txt)", type=["smi", "sdf", "txt"], key="act_up",
        )
    with c2:
        st.markdown("#### 📁 Optional")
        fragment_file = st.file_uploader("Fragment Library (.sdf)", type=["sdf"], key="frag_up")
        st.markdown("#### 📁 Pharmacophore")
        pharmit_file = st.file_uploader(
            "Model (.json/.ph4/.pml/…)",
            type=["json", "ph4", "lig", "pml", "mol2", "xyz", "csv", "txt", "sdf", "tsv"],
            key="pharm_up",
        )
        st.markdown(
            '<div class="card" style="border-left:3px solid #6366f1;padding:12px;">'
            '<strong>🌐 Pharmit</strong>&nbsp;'
            '<a href="https://pharmit.csb.pitt.edu" target="_blank" style="color:#22d3ee;">'
            "pharmit.csb.pitt.edu</a></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Step tracker cards ───────────────────────────────────
    step = st.session_state.pipeline_step
    if step > 0 or st.session_state.pipeline_run:
        cols = st.columns(5)
        _step_defs = [
            (1, "Pharmacophore", "🔬"),
            (2, "Classifier",    "🤖"),
            (3, "FBDD",          "🧬"),
            (4, "DB Screen",     "🗄️"),
            (5, "Combine",       "📊"),
        ]
        for idx, (snum, sname, sicon) in enumerate(_step_defs):
            with cols[idx]:
                if snum < step or st.session_state.pipeline_run:
                    css = "step-card-done"; status = "✅ Done"
                elif snum == step and not st.session_state.pipeline_paused:
                    css = "step-card-run"; status = "🔄 Running…"
                elif snum == step and st.session_state.pipeline_paused:
                    css = "step-card-pause"; status = "⏸️ Paused"
                else:
                    css = "step-card-idle"; status = "⬜ Pending"
                st.markdown(
                    f'<div class="{css}">{sicon}<br><b>{sname}</b><br>{status}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("")

    # ── Progress bar & status text ───────────────────────────
    prog_bar  = st.progress(min(step / 5, 1.0) if step > 0 else 0.0)
    prog_text = st.empty()

    if st.session_state.pipeline_run:
        prog_text.markdown(
            '<div class="banner-done">✅ Pipeline complete — check the result tabs!</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.pipeline_paused:
        prog_text.markdown(
            f'<div class="banner-pause">⏸️ Paused after: '
            f'{STEP_LABELS.get(step, "")} — click ▶️ Resume to continue</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.pipeline_error:
        prog_text.error(f"❌ {st.session_state.pipeline_error}")

    # ── Action buttons ───────────────────────────────────────
    bc1, bc2, bc3, bc4, bc5 = st.columns(5)

    is_paused  = st.session_state.pipeline_paused
    is_running = st.session_state.do_advance and not is_paused
    is_done    = st.session_state.pipeline_run

    with bc1:
        demo_btn = st.button(
            "🧪 EGFR Demo", use_container_width=True, disabled=is_running, key="demo_btn",
        )
    with bc2:
        _has_progress = (
            st.session_state.pipeline is not None
            and 0 < st.session_state.pipeline_step < 5
        )
        run_label = (
            "▶️ Resume" if _has_progress
            else ("▶️ Run Pipeline" if not is_done else "🔁 Re-run")
        )
        run_btn = st.button(
            run_label, use_container_width=True, type="primary",
            disabled=is_running, key="run_btn",
        )
    with bc3:
        pause_btn = st.button(
            "⏸️ Pause", use_container_width=True, disabled=not is_running, key="pause_btn",
        )
    with bc4:
        reset_btn = st.button(
            "🗑️ Reset", use_container_width=True, disabled=is_running, key="reset_btn",
        )
    with bc5:
        st.markdown(
            '<div class="card" style="padding:8px;text-align:center;">'
            "<small>⏱️ ~5–15 min CPU</small></div>",
            unsafe_allow_html=True,
        )

    # ── Button handlers ──────────────────────────────────────
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
        reset_all()
        st.rerun()

    if run_btn and not is_running:
        error_msg = None

        # Resume vs. fresh run:  check pipeline_step (always reliable)
        has_progress = (
            st.session_state.pipeline is not None
            and 0 < st.session_state.pipeline_step < 5
        )

        if has_progress:
            # ── RESUME ───────────────────────────────────────
            logger.info(f"RESUME from step {st.session_state.pipeline_step}")
            st.toast(
                f"▶️ Resuming from step {st.session_state.pipeline_step}/4 "
                f"({STEP_LABELS.get(st.session_state.pipeline_step, '')})"
            )
        else:
            # ── FRESH RUN ────────────────────────────────────
            active_mols, active_smiles = [], []
            pdb_block, pharmit_json, frag_path = "", None, None

            if getattr(st.session_state, "demo_mols", None):
                active_mols   = st.session_state.demo_mols
                active_smiles = st.session_state.demo_smiles
                pdb_block     = st.session_state.demo_pdb

            if pdb_file:
                pdb_block = parse_pdb_file(pdb_file.read())
            if actives_file:
                parsed = parse_actives_file(
                    actives_file.read().decode("utf-8"), actives_file.name,
                )
                active_mols   = [p[0] for p in parsed]
                active_smiles = [p[1] for p in parsed]
            if pharmit_file:
                feats = load_pharmacophore_any_format(
                    pharmit_file.read(), pharmit_file.name,
                )
                if feats:
                    from modules.pharmacophore import export_pharmit_json
                    pharmit_json = export_pharmit_json(feats)
                    st.success(f"✅ Loaded {len(feats)} pharmacophore features")
                else:
                    st.warning("⚠️ Could not parse pharmacophore file.")
            if fragment_file:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
                tmp.write(fragment_file.read())
                tmp.close()
                frag_path = tmp.name

            if not pdb_block:
                error_msg = "Please upload a PDB file or load demo data!"
            elif not active_mols and not pharmit_json:
                error_msg = "Please provide active compounds or a pharmacophore JSON!"

            if not error_msg:
                st.session_state.pipeline = HybridPipeline(
                    active_mols=active_mols,
                    active_smiles=active_smiles,
                    pdb_block=pdb_block,
                    fragment_sdf_path=frag_path,
                    pharmit_json=pharmit_json,
                    params={
                        "exhaustiveness":   exhaustiveness,
                        "box_size":         [box_x, box_y, box_z],
                        "num_modes":        5,
                        "frag_threshold":   frag_thresh,
                        "evolved_threshold": evolved_thresh,
                        "db_threshold":     db_thresh,
                        "data_dir":         os.path.join(os.path.dirname(__file__), "data"),
                    },
                )
                clear_results()
                logger.info("FRESH RUN: new pipeline created")

        if error_msg:
            st.error(f"❌ {error_msg}")
        else:
            st.session_state.pipeline_paused  = False
            st.session_state.pause_requested  = False
            st.session_state.do_advance       = True
            st.rerun()

    # ── Auto-advance: run one step per rerun ─────────────────
    if st.session_state.do_advance and not st.session_state.pipeline_paused:
        pipeline_obj = st.session_state.pipeline
        if pipeline_obj and st.session_state.pipeline_step < 5:
            result = run_current_step(pipeline_obj, prog_bar, prog_text)
            if result == "continue":
                st.session_state.do_advance = True
                st.rerun()
            elif result == "done":
                st.session_state.do_advance = False
                st.balloons()
                st.rerun()
            else:  # "paused" or "error"
                st.session_state.do_advance = False
                st.rerun()


# ── TABS 2–8: Result displays ────────────────────────────────
with tabs[1]: tab_pharmacophore.render()
with tabs[2]: tab_fragments.render()
with tabs[3]: tab_database.render()
with tabs[4]: tab_combined.render()
with tabs[5]: tab_origins.render()
with tabs[6]: tab_viewer3d.render()
with tabs[7]: tab_downloads.render()


# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#64748b;font-size:.8rem;padding:1rem;">'
    "🧬 <strong>HyDRA</strong> — Hybrid Drug Discovery Research Accelerator<br>"
    "Developed by <strong>Mohamed Sayed</strong> | "
    "Powered by RDKit · AutoDock Vina · scikit-learn<br>"
    "© 2024 — For research use only</div>",
    unsafe_allow_html=True,
)
