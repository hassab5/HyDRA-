"""
ui/tabs/tab_database.py — Database Screening Results tab
=========================================================
Shows DB screening metrics, source pie chart, lead table, and ADMET details.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from rdkit import Chem
try:
    from rdkit.Chem import Draw
except ImportError:
    Draw = None

from modules.admet import generate_radar_chart


def render():
    """Render Tab 4: Database Results."""
    st.markdown("## 🗄️ Database Screening Results")

    leads = st.session_state.res_db_leads
    if not leads:
        st.info("Run through Step 4 (DB Screening) to see database results.")
        return

    # ── Summary metrics + source pie chart ───────────────────
    stats = st.session_state.res_db_stats
    if stats:
        sc = stats.get("source_counts", {})
        cc = st.columns(4)
        cc[0].metric("Screened", stats.get("total_screened", 0))
        cc[1].metric("Filtered", stats.get("total_filtered", 0))
        cc[2].metric("Sources",  len(sc))
        cc[3].metric("Leads",    len(leads))
        if sc:
            fig = px.pie(
                names=list(sc.keys()), values=list(sc.values()),
                color_discrete_sequence=px.colors.qualitative.Set3, hole=0.4,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8", height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Lead table ───────────────────────────────────────────
    rows = [
        {
            "Rank":       i + 1,
            "SMILES":     l.get("smiles", "")[:55] + "…",
            "Source":     l.get("source", ""),
            "Docking":    f"{l.get('docking_score', 0):.2f}",
            "Classifier": f"{l.get('classifier_score', 0):.3f}",
            "MPO":        f"{l.get('mpo_score', 0):.3f}",
            "Final":      f"{l.get('final_score', 0):.4f}",
            "MW":         l.get("admet", {}).get("MW", ""),
            "QED":        l.get("admet", {}).get("QED", ""),
        }
        for i, l in enumerate(leads)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=480)

    # ── ADMET detail ─────────────────────────────────────────
    sel = st.selectbox(
        "ADMET detail",
        range(len(leads)),
        format_func=lambda i: f"Lead {i+1}: {leads[i].get('smiles','')[:40]}…",
        key="db_sel",
    )
    if sel is not None:
        lead  = leads[sel]
        admet = lead.get("admet", {})
        if admet:
            ca, cb = st.columns(2)
            with ca:
                st.plotly_chart(
                    generate_radar_chart(admet, f"DB Lead {sel+1}"),
                    use_container_width=True,
                )
            with cb:
                smi = lead.get("smiles", "")
                mol = Chem.MolFromSmiles(smi)
                if mol and Draw:
                    st.image(Draw.MolToImage(mol, (400, 300)), caption=f"DB {sel+1}")
                st.markdown(f"**Source:** {lead.get('source', '')}")
                st.markdown(f"**Docking:** {lead.get('docking_score', 0):.2f} kcal/mol")
