"""
ui/tabs/tab_fragments.py — Fragment-Based Drug Design Results tab
=================================================================
Shows FBDD pathway metrics, lead table, ADMET radar charts, and 2D structures.
"""

import streamlit as st
import pandas as pd
from rdkit import Chem
try:
    from rdkit.Chem import Draw
except ImportError:
    Draw = None

from modules.admet import generate_radar_chart


def render():
    """Render Tab 3: Fragment Results."""
    st.markdown("## 🧬 Fragment-Based Drug Design Results")

    leads = st.session_state.res_frag_leads
    if not leads:
        st.info("Run through Step 3 (FBDD) to see fragment results.")
        return

    # ── Summary metrics ──────────────────────────────────────
    stats = st.session_state.res_frag_stats
    if stats:
        cc = st.columns(4)
        cc[0].metric("Library",  stats.get("n_library", 0))
        cc[1].metric("Scored",   stats.get("n_scored", 0))
        cc[2].metric("Evolved",  stats.get("n_evolved", 0))
        cc[3].metric("Leads",    len(leads))

    # ── Lead table ───────────────────────────────────────────
    rows = [
        {
            "Rank":       i + 1,
            "SMILES":     l.get("smiles", "")[:55] + "…",
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

    # ── ADMET detail for selected lead ───────────────────────
    sel = st.selectbox(
        "ADMET detail",
        range(len(leads)),
        format_func=lambda i: f"Lead {i+1}: {leads[i].get('smiles','')[:40]}…",
    )
    if sel is not None:
        lead  = leads[sel]
        admet = lead.get("admet", {})
        if admet:
            ca, cb = st.columns(2)
            with ca:
                st.plotly_chart(
                    generate_radar_chart(admet, f"Lead {sel+1}"),
                    use_container_width=True,
                )
            with cb:
                smi = lead.get("smiles", "")
                mol = Chem.MolFromSmiles(smi)
                if mol and Draw:
                    st.image(Draw.MolToImage(mol, (400, 300)), caption=f"Lead {sel+1}")
                st.markdown(f"**SMILES:** `{smi}`")
                st.markdown(f"**Docking:** {lead.get('docking_score', 0):.2f} kcal/mol")
