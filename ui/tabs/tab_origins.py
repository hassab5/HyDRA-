"""
ui/tabs/tab_origins.py — Fragment Provenance & Origins tab
===========================================================
Shows how each fragment lead was built, with 2D structure images.
"""

import streamlit as st
from rdkit import Chem
try:
    from rdkit.Chem import Draw
except ImportError:
    Draw = None


def render():
    """Render Tab 6: Fragment Origins."""
    st.markdown("## 🔗 Fragment Provenance & Origins")

    leads = st.session_state.res_frag_leads
    if not leads:
        st.info("Run through Step 3 (FBDD) to see fragment origins.")
        return

    for i, lead in enumerate(leads[:20]):
        with st.expander(
            f"Lead {i+1}: {lead.get('smiles','')[:50]}…",
            expanded=(i < 3),
        ):
            st.markdown(f"**Provenance:** `{lead.get('provenance', '')}`")
            st.markdown(f"**Method:** {lead.get('method', '')}")
            st.markdown(f"**Docking:** {lead.get('docking_score', 0):.2f} kcal/mol")
            st.markdown(f"**Final Score:** {lead.get('final_score', 0):.4f}")
            mol = Chem.MolFromSmiles(lead.get("smiles", ""))
            if mol and Draw:
                st.image(Draw.MolToImage(mol, (500, 300)), caption="Evolved Molecule")
