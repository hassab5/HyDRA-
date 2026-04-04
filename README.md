---
title: HyDRA Streamlit Version
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: streamlit
python_version: "3.10"
sdk_version: "1.38.0"
app_file: app.py
pinned: true
---

# 🧬 HyDRA — Hybrid Drug Discovery Research Accelerator

**Mohamed Sayed Hybrid Drug Design App**

A dual-pathway hybrid drug discovery pipeline combining **Fragment-Based Drug Design (FBDD)** and **Pharmacophore-Guided Database Screening** in a single Streamlit web application.

---

## 🚀 Features

### Dual-Pathway Architecture
- **Pathway 1 (FBDD):** Fragment library → classifier scoring → MW-dependent linking/merging → docking → ADMET → Top 20 leads
- **Pathway 2 (Database):** Multi-database screening (ZINC, ChEMBL, PubChem, Pharmit) → classifier filter → docking → ADMET → Top 20 leads

### Shared Intelligence
- **3D Pharmacophore Model:** Auto-generated from protein–ligand interaction geometry
- **ML Classifier:** RandomForest trained on actives vs property-matched decoys; shared by both pathways

### Fragment Evolution
- **50 chemical linkers** (amide, urea, sulfonamide, triazole, piperazine, etc.)
- **Multiple merging strategies** (MCS overlay, scaffold fusion, direct attachment)
- **MW-based decision:** combined MW < 300 → LINK, ≥ 300 → MERGE

### Comprehensive ADMET
- Lipinski Ro5, Veber rules, QED, SAS
- CYP3A4, hERG, BBB, solubility, oral bioavailability predictions
- MPO multi-parameter optimization scoring
- Interactive radar charts

### 8 Interactive Tabs
1. 🚀 **Launch** — File uploads, parameters, one-click demo
2. 🔬 **Pharmacophore** — Feature table, 3D viewer, classifier metrics
3. 🧬 **Fragment Results** — Top fragments, evolved molecules, ADMET
4. 🗄️ **Database Results** — Source distribution, top hits, ADMET
5. 📊 **Combined Ranking** — Unified leaderboard, parallel coordinates
6. 🔗 **Fragment Origins** — Provenance tracking for every fragment lead
7. 🧬 **3D Viewer** — Interactive py3Dmol with protein + pharmacophore + ligand
8. 📥 **Downloads** — CSV reports, SDF poses, JSON, HTML viewer, ZIP

---

## 🔧 Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Hugging Face Spaces
This app is designed for HF Spaces free tier (4GB RAM, CPU only).

```yaml
title: HyDRA Hybrid Drug Discovery
emoji: 🧬
colorFrom: indigo
colorTo: cyan
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: true
```

---

## 📁 Project Structure

```
mohamed-sayed-hybrid-drug-design/
├── app.py                          # Main Streamlit app (8 tabs)
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages (HF Spaces)
├── README.md                       # This file
├── modules/
│   ├── __init__.py
│   ├── pharmacophore.py            # Pharmacophore + ML classifier
│   ├── fragment_engine.py          # Fragment library + linking/merging
│   ├── screening.py                # Multi-database screening
│   ├── docking.py                  # AutoDock Vina wrapper + fallback
│   ├── admet.py                    # ADMET calculations + MPO
│   └── hybrid_orchestrator.py      # Pipeline coordinator
├── data/
│   └── demo_egfr_actives.txt       # EGFR demo SMILES
├── static/
│   └── logo.png                    # App logo
└── .streamlit/
    └── config.toml                 # Streamlit theming
```

---

## 📋 Input Formats

| Input | Format | Required |
|-------|--------|----------|
| Protein structure | `.pdb` | ✅ Yes |
| Active compounds | `.smi`, `.sdf`, `.txt` (one SMILES/line) | ⚠️ Required unless providing pharmacophore JSON |
| Fragment library | `.sdf` | Optional (built-in library available) |
| Pharmacophore | `.json` (Pharmit format) | Optional (auto-generated if not provided) |

---

## 🧪 Demo

Click **"🧪 Load EGFR Demo Data"** on the Launch tab to load pre-packaged EGFR inhibitor data and run the full pipeline with one click.

---

## 📊 Output Files

- `fragment_results.csv` — All Pathway 1 data
- `database_results.csv` — All Pathway 2 data
- `combined_leaderboard.csv` — Unified ranking
- `admet_full_report.csv` — Complete ADMET profiles
- `pharmacophore_model.json` — Pharmit-compatible model
- `fragment_origins.csv` — Provenance strings
- `docking_scores_all.csv` — All docking scores
- `pharmacophore_viewer.html` — Standalone 3D viewer
- SDF files with 5 docking poses per lead

---

## 🛠️ Technical Details

- **Memory:** Optimized for 4GB RAM with chunked processing
- **Docking:** AutoDock Vina (primary) with RDKit fallback scoring
- **ML:** scikit-learn RandomForest with 5-fold CV
- **3D:** py3Dmol for interactive molecular visualization
- **Charts:** Plotly for radar, scatter, parallel coordinates

---

## 👨‍🔬 Author

**Mohamed Sayed** — Computational Medicinal Chemistry

---

*For research use only. Not intended for clinical decision-making.*
