"""
ui/tabs/tab_pharmacophore.py — Pharmacophore Model tab
=======================================================
Shows the pharmacophore feature table, 3D viewer, and classifier metrics.
"""

import streamlit as st
import pandas as pd
import py3Dmol
from stmol import showmol
from modules.pharmacophore import FEATURE_COLORS


def render():
    """Render Tab 2: Pharmacophore Model."""
    st.markdown("## 🔬 Pharmacophore Model")

    features = st.session_state.res_pharm_features
    pdb_blk  = st.session_state.res_pdb_block

    if not features:
        st.info("Run through Step 1 (Pharmacophore) to see results here.")
        return

    # ── Layout: feature table + 3D viewer ────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Feature Table")
        rows = [
            {
                "Type":   f["type"],
                "X":      round(f["center"][0], 2),
                "Y":      round(f["center"][1], 2),
                "Z":      round(f["center"][2], 2),
                "Radius": f.get("radius", 1.5),
            }
            for f in features
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if st.session_state.dl_pharm_json:
            st.download_button(
                "📥 Pharmacophore JSON",
                st.session_state.dl_pharm_json,
                "pharmacophore.json",
                "application/json",
                key="dl_pharm_t2",
            )

    with c2:
        st.markdown("### 3D Viewer")
        if pdb_blk:
            viewer = py3Dmol.view(width=600, height=450)
            viewer.addModel(pdb_blk, "pdb")
            viewer.setStyle({}, {"cartoon": {"color": "spectrum", "opacity": 0.7}})
            for f in features:
                viewer.addSphere({
                    "center":  {"x": f["center"][0], "y": f["center"][1], "z": f["center"][2]},
                    "radius":  f.get("radius", 1.5),
                    "color":   FEATURE_COLORS.get(f["type"], "#FFF"),
                    "opacity": 0.5,
                })
            viewer.zoomTo()
            showmol(viewer, height=450, width=600)

    # ── Classifier metrics ───────────────────────────────────
    metrics = st.session_state.res_classifier_metrics
    if metrics:
        st.markdown("### 🤖 Classifier Metrics")
        cols = st.columns(5)
        for col, (key, label) in zip(
            cols,
            [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"),
             ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")],
        ):
            col.metric(label, f"{metrics.get(key, 0):.3f}")
