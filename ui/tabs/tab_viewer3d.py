"""
ui/tabs/tab_viewer3d.py — Interactive 3D Molecular Viewer tab
==============================================================
Protein + pharmacophore + selected ligand in an interactive py3Dmol viewer.
"""

import streamlit as st
import py3Dmol
from stmol import showmol
from rdkit import Chem
from rdkit.Chem import AllChem
from modules.pharmacophore import FEATURE_COLORS


def render():
    """Render Tab 7: 3D Viewer."""
    st.markdown("## 🧬 Interactive 3D Molecular Viewer")

    combined = st.session_state.res_combined
    pdb_blk  = st.session_state.res_pdb_block
    features = st.session_state.res_pharm_features

    if not combined or not pdb_blk:
        st.info("Run the full pipeline to use the 3D viewer.")
        return

    # ── Molecule selector ────────────────────────────────────
    sel = st.selectbox(
        "Select Molecule",
        range(len(combined)),
        format_func=lambda i: (
            f"#{combined[i]['rank']} {combined[i]['pathway']} — "
            f"{combined[i].get('smiles','')[:40]}…"
        ),
    )
    if sel is None:
        return

    entry = combined[sel]

    # ── Build 3D viewer ──────────────────────────────────────
    viewer = py3Dmol.view(width=800, height=540)

    # Protein
    viewer.addModel(pdb_blk, "pdb")
    viewer.setStyle({}, {"cartoon": {"color": "spectrum", "opacity": 0.6}})
    viewer.addSurface(py3Dmol.VDW, {"opacity": 0.1, "color": "white"})

    # Pharmacophore spheres
    if features:
        for f in features:
            viewer.addSphere({
                "center":  {"x": f["center"][0], "y": f["center"][1], "z": f["center"][2]},
                "radius":  f.get("radius", 1.5),
                "color":   FEATURE_COLORS.get(f["type"], "#FFF"),
                "opacity": 0.3,
            })

    # Ligand (generate 3D coords on the fly)
    smi = entry.get("smiles", "")
    mol = Chem.MolFromSmiles(smi)
    if mol:
        try:
            mol3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol3d, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol3d)
            viewer.addModel(Chem.MolToMolBlock(mol3d), "mol")
            viewer.setStyle(
                {"model": -1},
                {"stick": {"colorscheme": "greenCarbon", "radius": 0.15}},
            )
        except Exception:
            pass

    viewer.zoomTo()
    showmol(viewer, height=540, width=800)

    st.markdown(f"**SMILES:** `{smi}`")
    st.markdown(f"**Pathway:** {entry.get('pathway', '')}")
