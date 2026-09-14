"""
ui/state.py — Session state management for HyDRA
==================================================
Handles initialization, caching pipeline results, and caching downloads.
"""

import logging
import streamlit as st

logger = logging.getLogger(__name__)

# Human-readable step labels
STEP_LABELS = {
    1: "Pharmacophore",
    2: "Classifier",
    3: "FBDD",
    4: "DB Screen",
    5: "Combine",
}

# All session-state keys with their default values
_DEFAULTS = {
    # Pipeline execution state
    "pipeline":         None,
    "pipeline_run":     False,
    "pipeline_paused":  False,
    "pipeline_step":    0,       # 0=not started, 1-4=running, 5=done
    "pipeline_error":   None,
    "pause_requested":  False,
    "do_advance":       False,

    # Cached DISPLAY results — simple serializable dicts.
    # These survive widget reruns even if the pipeline object is lost.
    "res_pharm_features":     None,   # list of dicts
    "res_classifier_metrics": None,   # dict
    "res_pdb_block":          None,   # str  (PDB text)
    "res_frag_leads":         None,   # list of dicts (no RDKit mol objects)
    "res_frag_stats":         None,   # dict
    "res_db_leads":           None,   # list of dicts (no RDKit mol objects)
    "res_db_stats":           None,   # dict
    "res_combined":           None,   # list of dicts (no RDKit mol objects)
    "res_summary":            None,   # dict

    # Cached download blobs (CSV / JSON / HTML / ZIP bytes)
    "dl_frag_csv":     None,
    "dl_db_csv":       None,
    "dl_combined_csv": None,
    "dl_admet_csv":    None,
    "dl_origins_csv":  None,
    "dl_docking_csv":  None,
    "dl_pharm_json":   None,
    "dl_viewer_html":  None,
    "dl_zip":          None,
}

# Keys to clear when starting a fresh pipeline run
RESET_KEYS = [k for k in _DEFAULTS if k.startswith("dl_") or k.startswith("res_")]


def init_session_state():
    """Set default values for any missing session-state keys."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_all():
    """Wipe every session-state key (full reset)."""
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def clear_results():
    """Clear only the result / download caches (used before a fresh run)."""
    for k in RESET_KEYS:
        st.session_state[k] = None
    st.session_state.pipeline_step  = 0
    st.session_state.pipeline_run   = False
    st.session_state.pipeline_error = None


# ── Result caching helpers ───────────────────────────────────

def _strip_mol(lead: dict) -> dict:
    """Return a copy of a lead dict without RDKit mol objects (not serializable)."""
    return {k: v for k, v in lead.items() if k != "mol"}


def cache_results(pipeline):
    """
    Extract all display-worthy data from the pipeline object
    and store as simple, serializable session-state entries.

    This is called once when the pipeline finishes so that all
    result-display tabs can work without touching the pipeline object.
    """
    try:
        if getattr(pipeline, "pharmacophore_features", None):
            st.session_state.res_pharm_features = pipeline.pharmacophore_features
        if getattr(pipeline, "classifier_metrics", None):
            st.session_state.res_classifier_metrics = pipeline.classifier_metrics
        if getattr(pipeline, "pdb_block", None):
            st.session_state.res_pdb_block = pipeline.pdb_block
        if getattr(pipeline, "fragment_leads", None):
            st.session_state.res_frag_leads = [_strip_mol(l) for l in pipeline.fragment_leads]
        if getattr(pipeline, "pathway1_results", None):
            st.session_state.res_frag_stats = pipeline.pathway1_results
        if getattr(pipeline, "database_leads", None):
            st.session_state.res_db_leads = [_strip_mol(l) for l in pipeline.database_leads]
        if getattr(pipeline, "pathway2_results", None):
            st.session_state.res_db_stats = pipeline.pathway2_results
        if getattr(pipeline, "combined_leaderboard", None):
            st.session_state.res_combined = [_strip_mol(e) for e in pipeline.combined_leaderboard]
        try:
            st.session_state.res_summary = pipeline.get_summary()
        except Exception:
            pass
    except Exception as e:
        logger.error(f"cache_results failed: {e}")


def cache_downloads(pipeline):
    """
    Pre-generate all downloadable files (CSV, JSON, HTML, ZIP)
    and store the raw bytes/strings in session state so that
    clicking a download button never triggers a page reset.
    """
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
        except Exception:
            pass
    except Exception as e:
        logger.error(f"cache_downloads failed: {e}")
