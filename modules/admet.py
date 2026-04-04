"""
ADMET Profiling & Multi-Parameter Optimization (MPO) Scoring Module
===================================================================
Computes comprehensive ADMET properties for drug candidates:
  - Lipinski Ro5 (MW, cLogP, HBD, HBA)
  - Veber rules (TPSA, RotBonds)
  - QED (Quantitative Estimate of Drug-likeness)
  - Synthetic Accessibility Score (SAS)
  - CYP3A4 inhibition (SMARTS-based)
  - hERG cardiotoxicity (heuristic)
  - BBB penetration (heuristic)
  - Solubility (ESOL/Delaney model)
  - Oral bioavailability score
  - MPO weighted composite score
  - Plotly radar chart generation
"""

import numpy as np
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors, QED as RDKitQED, FilterCatalog
from rdkit.Chem import rdMolDescriptors
try:
    from rdkit.Chem.SA_Score import sascorer
except ImportError:
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError:
        # Fallback: implement a simplified SAS calculation
        class _FallbackSAScorer:
            @staticmethod
            def calculateScore(mol):
                """Simplified synthetic accessibility estimate."""
                from rdkit.Chem import rdMolDescriptors as _rd
                nAtoms = mol.GetNumHeavyAtoms()
                nRings = _rd.CalcNumRings(mol)
                nStereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                nSpiro = _rd.CalcNumSpiroAtoms(mol)
                nBridge = _rd.CalcNumBridgeheadAtoms(mol)
                score = 2.0 + nAtoms * 0.05 + nRings * 0.3 + nStereo * 0.5 + nSpiro * 0.8 + nBridge * 0.8
                return min(10.0, max(1.0, score))
        sascorer = _FallbackSAScorer()


# ─── SMARTS for CYP3A4 inhibition heuristics ───────────────────────────
CYP3A4_SMARTS = [
    Chem.MolFromSmarts("[#7]~1~[#6]~[#6]~[#7]~[#6]~[#6]~1"),   # imidazole-like
    Chem.MolFromSmarts("c1ccncc1"),                                # pyridine
    Chem.MolFromSmarts("[#7]1~[#6]~[#6]~[#7]~[#6]~1"),           # triazole-like
    Chem.MolFromSmarts("c1cc2ccccc2[nH]1"),                       # indole
    Chem.MolFromSmarts("[#7]1~[#6]~[#7]~[#6]~[#6]~1"),           # pyrimidine-like
    Chem.MolFromSmarts("c1cnc2ccccc2n1"),                         # quinazoline
    Chem.MolFromSmarts("[C;H2][C;H2][N]([CH3])[CH3]"),           # tertiary amine
]
CYP3A4_SMARTS = [s for s in CYP3A4_SMARTS if s is not None]


# ─── LIPINSKI RULE OF FIVE ──────────────────────────────────────────────
def calculate_lipinski(mol):
    """Calculate Lipinski Rule of Five properties."""
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "MW": round(mw, 2),
        "cLogP": round(logp, 2),
        "HBD": hbd,
        "HBA": hba,
        "Ro5_violations": violations,
        "Ro5_pass": violations <= 1
    }


# ─── VEBER RULES ────────────────────────────────────────────────────────
def calculate_veber(mol):
    """Calculate Veber rule properties (oral bioavailability predictors)."""
    if mol is None:
        return None
    tpsa = Descriptors.TPSA(mol)
    rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return {
        "TPSA": round(tpsa, 2),
        "RotBonds": rotbonds,
        "Veber_pass": tpsa <= 140 and rotbonds <= 10
    }


# ─── QED ─────────────────────────────────────────────────────────────────
def calculate_qed(mol):
    """Quantitative Estimate of Drug-likeness (0 to 1, higher = more drug-like)."""
    if mol is None:
        return 0.0
    try:
        return round(RDKitQED.qed(mol), 4)
    except Exception:
        return 0.0


# ─── SYNTHETIC ACCESSIBILITY SCORE ──────────────────────────────────────
def calculate_sas(mol):
    """Synthetic Accessibility Score (1=easy to 10=hard)."""
    if mol is None:
        return 10.0
    try:
        return round(sascorer.calculateScore(mol), 2)
    except Exception:
        return 5.0


# ─── CYP3A4 INHIBITION PREDICTION ──────────────────────────────────────
def predict_cyp3a4(mol):
    """SMARTS-based CYP3A4 inhibition risk. Returns risk level and score."""
    if mol is None:
        return {"risk": "Unknown", "score": 0.5}
    matches = sum(1 for smarts in CYP3A4_SMARTS if mol.HasSubstructMatch(smarts))
    # Also consider logP and MW
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    score = min(1.0, matches * 0.2 + (0.1 if logp > 3 else 0) + (0.1 if mw > 400 else 0))
    if score >= 0.6:
        risk = "High"
    elif score >= 0.3:
        risk = "Medium"
    else:
        risk = "Low"
    return {"risk": risk, "score": round(score, 3)}


# ─── hERG CARDIOTOXICITY PREDICTION ────────────────────────────────────
def predict_herg(mol):
    """hERG cardiotoxicity risk based on MW, logP, and charge heuristic."""
    if mol is None:
        return {"risk": "Unknown", "score": 0.5}
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    # Positive charge at physiological pH
    has_basic_n = mol.HasSubstructMatch(Chem.MolFromSmarts("[#7;+]")) or \
                  mol.HasSubstructMatch(Chem.MolFromSmarts("[NH2]")) or \
                  mol.HasSubstructMatch(Chem.MolFromSmarts("[NH;!$(NC=O)]"))
    score = 0.0
    if logp > 3.5:
        score += 0.3
    if mw > 350:
        score += 0.2
    if has_basic_n:
        score += 0.3
    if logp > 5:
        score += 0.2
    score = min(1.0, score)
    if score >= 0.6:
        risk = "High"
    elif score >= 0.3:
        risk = "Medium"
    else:
        risk = "Low"
    return {"risk": risk, "score": round(score, 3)}


# ─── BBB PENETRATION PREDICTION ────────────────────────────────────────
def predict_bbb(mol):
    """Blood-Brain Barrier penetration prediction heuristic."""
    if mol is None:
        return {"penetrant": False, "score": 0.0}
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    # Criteria: TPSA < 90, MW < 450, logP 1-5, HBD <= 3
    score = 0.0
    if tpsa < 90:
        score += 0.3
    elif tpsa < 120:
        score += 0.1
    if mw < 450:
        score += 0.25
    elif mw < 500:
        score += 0.1
    if 1 <= logp <= 5:
        score += 0.25
    if hbd <= 3:
        score += 0.2
    score = min(1.0, score)
    return {
        "penetrant": score >= 0.7,
        "score": round(score, 3),
        "label": "BBB+" if score >= 0.7 else "BBB-"
    }


# ─── ESOL SOLUBILITY (Delaney Model) ───────────────────────────────────
def predict_solubility(mol):
    """ESOL Predicted aqueous solubility using Delaney's equation.
    Returns logS and solubility class."""
    if mol is None:
        return {"logS": -5.0, "class": "Unknown"}
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ap = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    # Delaney equation: logS = 0.16 - 0.63*logP - 0.0062*MW + 0.066*RB - 0.74*AP_ratio
    n_heavy = mol.GetNumHeavyAtoms()
    ap_ratio = ap / max(n_heavy, 1)
    logS = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotbonds - 0.74 * ap_ratio
    logS = round(logS, 2)
    # Classify
    if logS >= -1:
        sol_class = "Highly Soluble"
    elif logS >= -2:
        sol_class = "Soluble"
    elif logS >= -4:
        sol_class = "Moderately Soluble"
    elif logS >= -6:
        sol_class = "Poorly Soluble"
    else:
        sol_class = "Insoluble"
    return {"logS": logS, "class": sol_class}


# ─── ORAL BIOAVAILABILITY SCORE ─────────────────────────────────────────
def predict_oral_bioavailability(mol):
    """Combined oral bioavailability score from multiple rules."""
    if mol is None:
        return {"score": 0.0, "label": "Poor"}
    lip = calculate_lipinski(mol)
    veb = calculate_veber(mol)
    logp = lip["cLogP"]
    score = 0.0
    # Lipinski compliance
    if lip["Ro5_pass"]:
        score += 0.35
    # Veber compliance
    if veb["Veber_pass"]:
        score += 0.25
    # Moderate logP
    if 0 <= logp <= 4:
        score += 0.2
    elif -1 <= logp <= 5:
        score += 0.1
    # Low MW bonus
    if lip["MW"] < 400:
        score += 0.1
    # Low rotatable bonds bonus
    if veb["RotBonds"] <= 7:
        score += 0.1
    score = min(1.0, round(score, 3))
    if score >= 0.7:
        label = "Good"
    elif score >= 0.4:
        label = "Moderate"
    else:
        label = "Poor"
    return {"score": score, "label": label}


# ─── PAINS FILTER ───────────────────────────────────────────────────────
def check_pains(mol):
    """Check for PAINS (Pan-Assay Interference Compounds) alerts."""
    if mol is None:
        return {"pass": False, "alerts": ["Invalid molecule"]}
    try:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)
        entry = catalog.GetFirstMatch(mol)
        if entry is not None:
            return {"pass": False, "alerts": [entry.GetDescription()]}
        return {"pass": True, "alerts": []}
    except Exception:
        return {"pass": True, "alerts": []}


# ─── FULL ADMET PROFILE ────────────────────────────────────────────────
def full_admet_profile(mol, smiles=None):
    """Compute complete ADMET profile for a molecule.
    Returns dict with all properties and scores."""
    if mol is None and smiles:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    lip = calculate_lipinski(mol)
    veb = calculate_veber(mol)
    qed = calculate_qed(mol)
    sas = calculate_sas(mol)
    cyp = predict_cyp3a4(mol)
    herg = predict_herg(mol)
    bbb = predict_bbb(mol)
    sol = predict_solubility(mol)
    oral = predict_oral_bioavailability(mol)
    pains = check_pains(mol)
    smi = smiles or Chem.MolToSmiles(mol)
    
    num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    num_rings = rdMolDescriptors.CalcNumRings(mol)

    profile = {
        "SMILES": smi,
        # Lipinski
        "MW": lip["MW"],
        "cLogP": lip["cLogP"],
        "HBD": lip["HBD"],
        "HBA": lip["HBA"],
        "Ro5_violations": lip["Ro5_violations"],
        "Ro5_pass": lip["Ro5_pass"],
        # Veber
        "TPSA": veb["TPSA"],
        "RotBonds": veb["RotBonds"],
        "Veber_pass": veb["Veber_pass"],
        # QED & SAS
        "QED": qed,
        "SAS": sas,
        # Ring info
        "AromaticRings": num_aromatic_rings,
        "NumRings": num_rings,
        # Predictions
        "CYP3A4_risk": cyp["risk"],
        "CYP3A4_score": cyp["score"],
        "hERG_risk": herg["risk"],
        "hERG_score": herg["score"],
        "BBB_penetrant": bbb["penetrant"],
        "BBB_score": bbb["score"],
        "BBB_label": bbb["label"],
        "Solubility_logS": sol["logS"],
        "Solubility_class": sol["class"],
        "OralBioavail_score": oral["score"],
        "OralBioavail_label": oral["label"],
        # PAINS
        "PAINS_pass": pains["pass"],
        "PAINS_alerts": pains["alerts"],
    }
    return profile


# ─── MPO MULTI-PARAMETER OPTIMIZATION ──────────────────────────────────
def calculate_mpo(profile):
    """Calculate Multi-Parameter Optimization score from ADMET profile.
    Returns score 0-1 (higher = better overall drug-likeness)."""
    if profile is None:
        return 0.0

    scores = {}
    # QED (already 0-1, higher better)
    scores["QED"] = profile.get("QED", 0.0)

    # SAS (1-10, lower better → normalize and invert)
    sas = profile.get("SAS", 5.0)
    scores["SAS"] = max(0, (10 - sas) / 9)

    # CYP3A4 (lower score = better → invert)
    scores["CYP3A4"] = 1.0 - profile.get("CYP3A4_score", 0.5)

    # hERG (lower score = better → invert)
    scores["hERG"] = 1.0 - profile.get("hERG_score", 0.5)

    # Solubility (logS, higher better, typical range -8 to 0)
    logS = profile.get("Solubility_logS", -5.0)
    scores["Solubility"] = max(0, min(1, (logS + 8) / 8))

    # BBB (0-1 score)
    scores["BBB"] = profile.get("BBB_score", 0.3)

    # Oral bioavailability (0-1 score)
    scores["OralBioavail"] = profile.get("OralBioavail_score", 0.3)

    # Lipinski compliance bonus
    scores["Lipinski"] = 1.0 if profile.get("Ro5_pass", False) else 0.3

    # Weighted sum
    weights = {
        "QED": 0.20,
        "SAS": 0.10,
        "CYP3A4": 0.10,
        "hERG": 0.15,
        "Solubility": 0.10,
        "BBB": 0.10,
        "OralBioavail": 0.15,
        "Lipinski": 0.10,
    }
    mpo = sum(scores[k] * weights[k] for k in weights)
    return round(mpo, 4)


# ─── FINAL RANKING SCORE ────────────────────────────────────────────────
def score_final(vina_score, mpo_score, classifier_prob):
    """Compute final ranking score.
    Formula: 0.5*(normalized_vina) + 0.3*(MPO) + 0.2*(classifier_prob)
    Vina scores are negative (more negative = better), so we normalize to 0-1."""
    # Normalize Vina: typical range -12 to 0, more negative = better
    # Map to 0-1 where 1 is best
    if vina_score is None:
        norm_vina = 0.3
    else:
        norm_vina = max(0, min(1, (-vina_score) / 12.0))

    final = 0.5 * norm_vina + 0.3 * mpo_score + 0.2 * classifier_prob
    return round(final, 4)


# ─── RADAR CHART GENERATION ─────────────────────────────────────────────
def generate_radar_chart(profile, title="ADMET Profile"):
    """Generate plotly radar chart for a single molecule's ADMET profile."""
    if profile is None:
        return go.Figure()

    categories = [
        "Drug-likeness\n(QED)",
        "Synth. Access.\n(inv SAS)",
        "CYP3A4 Safety\n(inv risk)",
        "hERG Safety\n(inv risk)",
        "Solubility",
        "BBB Penetration",
        "Oral Bioavail.",
        "Lipinski"
    ]

    sas_val = profile.get("SAS", 5.0)
    values = [
        profile.get("QED", 0),
        max(0, (10 - sas_val) / 9),
        1.0 - profile.get("CYP3A4_score", 0.5),
        1.0 - profile.get("hERG_score", 0.5),
        max(0, min(1, (profile.get("Solubility_logS", -5) + 8) / 8)),
        profile.get("BBB_score", 0.3),
        profile.get("OralBioavail_score", 0.3),
        1.0 if profile.get("Ro5_pass", False) else 0.3,
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.2)',
        line=dict(color='rgb(99, 110, 250)', width=2),
        name=title,
        hovertemplate='%{theta}: %{r:.2f}<extra></extra>'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["0.25", "0.50", "0.75", "1.00"],
                gridcolor='rgba(128,128,128,0.3)',
            ),
            angularaxis=dict(
                gridcolor='rgba(128,128,128,0.3)',
            )
        ),
        showlegend=False,
        title=dict(text=title, x=0.5, font=dict(size=14)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=80, r=80, t=60, b=40),
        height=400,
        width=450,
    )
    return fig


# ─── MULTI-MOLECULE COMPARISON CHART ───────────────────────────────────
def generate_comparison_chart(profiles, names=None):
    """Generate overlaid radar chart comparing multiple molecules."""
    if not profiles:
        return go.Figure()

    colors = [
        'rgb(99, 110, 250)', 'rgb(239, 85, 59)', 'rgb(0, 204, 150)',
        'rgb(171, 99, 250)', 'rgb(255, 161, 90)', 'rgb(25, 211, 243)',
        'rgb(255, 102, 146)', 'rgb(182, 232, 128)', 'rgb(255, 151, 255)',
        'rgb(254, 203, 82)',
    ]

    categories = ["QED", "Synth.", "CYP3A4", "hERG", "Solub.", "BBB", "Oral", "Lipinski"]

    fig = go.Figure()
    for i, profile in enumerate(profiles):
        name = names[i] if names and i < len(names) else f"Mol {i+1}"
        sas_val = profile.get("SAS", 5.0)
        values = [
            profile.get("QED", 0),
            max(0, (10 - sas_val) / 9),
            1.0 - profile.get("CYP3A4_score", 0.5),
            1.0 - profile.get("hERG_score", 0.5),
            max(0, min(1, (profile.get("Solubility_logS", -5) + 8) / 8)),
            profile.get("BBB_score", 0.3),
            profile.get("OralBioavail_score", 0.3),
            1.0 if profile.get("Ro5_pass", False) else 0.3,
        ]
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.1)'),
            line=dict(color=color, width=2),
            name=name,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(128,128,128,0.3)'),
            angularaxis=dict(gridcolor='rgba(128,128,128,0.3)'),
        ),
        showlegend=True,
        legend=dict(x=1.1, y=1.0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=80, r=120, t=40, b=40),
        height=450,
    )
    return fig


# ─── BATCH ADMET PROFILING ──────────────────────────────────────────────
def batch_admet_profiles(molecules, smiles_list=None):
    """Compute ADMET profiles for a list of molecules.
    Returns list of profile dicts with MPO scores added."""
    results = []
    for i, mol in enumerate(molecules):
        smi = smiles_list[i] if smiles_list and i < len(smiles_list) else None
        profile = full_admet_profile(mol, smiles=smi)
        if profile:
            profile["MPO"] = calculate_mpo(profile)
            results.append(profile)
    return results
