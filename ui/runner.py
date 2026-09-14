"""
ui/runner.py — Synchronous step-by-step pipeline executor
==========================================================
Runs ONE pipeline step per Streamlit rerun, then calls st.rerun()
so the UI stays responsive throughout the multi-minute pipeline.

Design rationale:
- Threading does NOT work reliably on Hugging Face Spaces (freezes UI).
- Instead we use a "self-advancing loop": each rerun executes one step,
  updates progress, and reruns again.  The Pause button sets a flag
  that is checked BETWEEN steps (not during).
"""

import logging
import streamlit as st
from ui.state import STEP_LABELS, cache_results, cache_downloads

logger = logging.getLogger(__name__)


def run_current_step(pipeline, progress_bar, status_text):
    """
    Execute the step indicated by ``st.session_state.pipeline_step``.

    Args:
        pipeline:      HybridPipeline object (from session state)
        progress_bar:  ``st.progress()`` widget handle — updated live
        status_text:   ``st.empty()``  widget handle — updated live

    Returns one of:
        "continue"  — step finished, more steps remain
        "done"      — pipeline finished, results cached
        "paused"    — user requested a pause between steps
        "error"     — an exception occurred
    """
    step = st.session_state.pipeline_step

    # Tiny helper to push live updates into the progress bar + text
    def cb(frac, msg):
        progress_bar.progress(min(float(frac), 1.0))
        status_text.markdown(f"**{msg}**")

    try:
        # ── Step 0 → 1: Pharmacophore ────────────────────────
        if step == 0:
            cb(0.02, "🔬 Step 1/4 — Building pharmacophore model…")
            pipeline.step0_pharmacophore(lambda f, m: cb(f * 0.10, m))
            st.session_state.pipeline_step = 1

        # ── Step 1 → 2: Classifier ──────────────────────────
        elif step == 1:
            cb(0.12, "🤖 Step 2/4 — Training ML classifier…")
            pipeline.step1_classifier(lambda f, m: cb(0.10 + f * 0.10, m))
            st.session_state.pipeline_step = 2

        # ── Step 2 → 3: Fragment-Based Drug Design ──────────
        elif step == 2:
            cb(0.22, "🧬 Step 3/4 — Fragment-based drug design…")
            pipeline.pathway1_fbdd(lambda f, m: cb(0.20 + f * 0.35, m))
            st.session_state.pipeline_step = 3

        # ── Step 3 → 4: Database Screening ──────────────────
        elif step == 3:
            cb(0.57, "🗄️ Step 4/4 — Database screening & docking…")
            pipeline.pathway2_screening(lambda f, m: cb(0.55 + f * 0.35, m))
            st.session_state.pipeline_step = 4

        # ── Step 4 → 5: Combine & Finish ────────────────────
        elif step == 4:
            cb(0.93, "📊 Combining results from both pathways…")
            pipeline.combine_results()
            cb(1.00, "✅ Pipeline complete!")
            st.session_state.pipeline_step   = 5
            st.session_state.pipeline_run    = True
            st.session_state.pipeline_paused = False
            st.session_state.do_advance      = False
            cache_results(pipeline)
            cache_downloads(pipeline)
            return "done"

        # ── Check for pause signal (between steps) ──────────
        if st.session_state.pause_requested:
            st.session_state.pause_requested = False
            st.session_state.pipeline_paused = True
            st.session_state.do_advance      = False
            current = st.session_state.pipeline_step
            cb(current / 5, f"⏸️ Paused — {STEP_LABELS.get(current, '')} complete")
            return "paused"

        return "continue"

    except Exception as e:
        logger.exception("Pipeline step error")
        st.session_state.pipeline_error = str(e)
        st.session_state.do_advance     = False
        return "error"
