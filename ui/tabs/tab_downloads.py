"""
ui/tabs/tab_downloads.py — Download All Results tab
=====================================================
Pre-cached CSV, JSON, HTML, and ZIP downloads.
"""

import streamlit as st


def render():
    """Render Tab 8: Downloads."""
    st.markdown("## 📥 Download All Results")

    has_results = st.session_state.pipeline_run and any([
        st.session_state.dl_frag_csv,
        st.session_state.dl_combined_csv,
    ])

    if not has_results:
        if st.session_state.pipeline_paused:
            step = st.session_state.pipeline_step
            st.warning(f"⏸️ Paused at step {step}/4. Resume to complete and unlock downloads.")
        else:
            st.info("Run the full pipeline to download results.")
        return

    # ── Three-column download layout ─────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 📄 CSV Reports")
        for key, fname, label in [
            ("dl_frag_csv",     "fragment_results.csv",     "Fragment Results CSV"),
            ("dl_db_csv",       "database_results.csv",     "Database Results CSV"),
            ("dl_combined_csv", "combined_leaderboard.csv", "Combined Leaderboard CSV"),
        ]:
            if st.session_state[key]:
                st.download_button(
                    f"⬇️ {label}", st.session_state[key],
                    fname, "text/csv", key=f"btn_{key}",
                )

    with c2:
        st.markdown("### 📊 Detailed Reports")
        for key, fname, label in [
            ("dl_admet_csv",   "admet_full.csv",       "ADMET Full Report"),
            ("dl_origins_csv", "fragment_origins.csv",  "Fragment Origins CSV"),
            ("dl_docking_csv", "docking_scores.csv",    "Docking Scores CSV"),
        ]:
            if st.session_state[key]:
                st.download_button(
                    f"⬇️ {label}", st.session_state[key],
                    fname, "text/csv", key=f"btn_{key}",
                )

    with c3:
        st.markdown("### 📦 Package")
        if st.session_state.dl_pharm_json:
            st.download_button(
                "⬇️ Pharmacophore JSON", st.session_state.dl_pharm_json,
                "pharmacophore.json", "application/json", key="btn_pharm",
            )
        if st.session_state.dl_viewer_html:
            st.download_button(
                "⬇️ 3D Viewer HTML", st.session_state.dl_viewer_html,
                "viewer.html", "text/html", key="btn_html",
            )

    # ── ZIP archive ──────────────────────────────────────────
    st.divider()
    st.markdown("### 🗜️ ZIP Archive")
    if st.session_state.dl_zip:
        st.download_button(
            "⬇️ Download ZIP", st.session_state.dl_zip,
            "hydra_results.zip", "application/zip",
            use_container_width=True, key="btn_zip",
        )
    else:
        if st.button("📦 Generate ZIP", type="primary", use_container_width=True, key="gen_zip"):
            pipeline = st.session_state.pipeline
            if pipeline:
                with st.spinner("Packaging…"):
                    try:
                        st.session_state.dl_zip = pipeline.create_zip()
                        st.rerun()
                    except Exception as e:
                        st.error(f"ZIP failed: {e}")
