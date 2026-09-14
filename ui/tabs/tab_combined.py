"""
ui/tabs/tab_combined.py — Combined Leaderboard tab
====================================================
Shows the unified leaderboard, parallel-coordinates chart, and scatter plot.
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def render():
    """Render Tab 5: Combined Leaderboard."""
    st.markdown("## 📊 Combined Leaderboard")

    combined = st.session_state.res_combined
    if not combined:
        st.info("Run the full pipeline to see combined results.")
        return

    # ── Leaderboard table ────────────────────────────────────
    st.markdown(f"### All {len(combined)} Leads — Ranked by Final Score")
    rows = [
        {
            "Rank":       e["rank"],
            "Pathway":    e["pathway"],
            "SMILES":     e.get("smiles", "")[:50] + "…",
            "Docking":    e.get("docking_score", 0),
            "Classifier": e.get("classifier_score", 0),
            "MPO":        e.get("mpo_score", 0),
            "Final":      e.get("final_score", 0),
            "MW":         e.get("admet", {}).get("MW", ""),
            "cLogP":      e.get("admet", {}).get("cLogP", ""),
            "TPSA":       e.get("admet", {}).get("TPSA", ""),
            "QED":        e.get("admet", {}).get("QED", ""),
        }
        for e in combined
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=580)

    # ── Parallel-coordinates chart ───────────────────────────
    pc_data = [
        {
            "Docking": e.get("docking_score", 0),
            "MW":      e.get("admet", {}).get("MW", 0),
            "cLogP":   e.get("admet", {}).get("cLogP", 0),
            "TPSA":    e.get("admet", {}).get("TPSA", 0),
            "QED":     e.get("admet", {}).get("QED", 0),
            "Pathway": 0 if e.get("pathway_label") == "Fragment" else 1,
        }
        for e in combined
    ]
    if pc_data:
        fig = px.parallel_coordinates(
            pd.DataFrame(pc_data), color="Pathway",
            color_continuous_scale=["#6366f1", "#ec4899"],
            dimensions=["Docking", "MW", "cLogP", "TPSA", "QED"],
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8", height=430)
        st.plotly_chart(fig, use_container_width=True)

    # ── Scatter plot: Vina score vs. classifier score ────────
    sc_data = [
        {
            "Vina":       e.get("docking_score", 0),
            "Classifier": e.get("classifier_score", 0),
            "Pathway":    e.get("pathway_label", ""),
            "SMILES":     e.get("smiles", "")[:30],
        }
        for e in combined
    ]
    if sc_data:
        fig2 = px.scatter(
            pd.DataFrame(sc_data), x="Vina", y="Classifier",
            color="Pathway", hover_data=["SMILES"],
            color_discrete_map={"Fragment": "#6366f1", "Database": "#ec4899"},
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8", height=380)
        st.plotly_chart(fig2, use_container_width=True)
