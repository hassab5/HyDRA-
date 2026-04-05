---
title: HyDRA Hybrid Drug Discovery
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
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

## 🔧 How to Run

### Option 1: Local Development (Without Docker)

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

> **Note:** Without system-level dependencies (libboost, swig), the Vina Python package won't install. The app will automatically fall back to RDKit heuristic scoring. This is fine for testing the UI and pipeline logic.

### Option 2: Local Development with Docker

```bash
# 1. Build the Docker image
docker build -t hydra-drug-discovery .

# 2. Run the container
docker run -p 7860:7860 hydra-drug-discovery

# 3. Open in browser
#    http://localhost:7860
```

> **Tip:** Use `-d` flag for detached mode: `docker run -d -p 7860:7860 hydra-drug-discovery`

### Option 3: Deploy on Hugging Face Spaces (Recommended)

This is the **recommended** approach for public deployment with full Vina docking support.

#### Step-by-step:

1. **Create a Hugging Face account** at https://huggingface.co/join (free)

2. **Install Git LFS** (for large files):
   ```bash
   git lfs install
   ```

3. **Create a new Space:**
   - Go to https://huggingface.co/new-space
   - Choose a name (e.g., `hydra-drug-discovery`)
   - Select **Docker** as the SDK
   - Set visibility to **Public** (free tier)
   - Click **Create Space**

4. **Clone and push your code:**
   ```bash
   # Clone the empty Space
   git clone https://huggingface.co/spaces/YOUR_USERNAME/hydra-drug-discovery
   cd hydra-drug-discovery

   # Copy all project files into this directory
   # (or initialize from your existing project)
   
   # Add all files
   git add .
   git commit -m "Initial deployment of HyDRA"
   git push
   ```

5. **Wait for build** (~5-10 minutes for first build)

6. **Access your app** at:
   ```
   https://YOUR_USERNAME-hydra-drug-discovery.hf.space
   ```

#### Alternative: Push from existing project
```bash
cd mohamed-sayed-hybrid-drug-design-v2

# Add HF Spaces as a remote
git init
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/hydra-drug-discovery

# Push
git add .
git commit -m "Deploy HyDRA to Hugging Face Spaces"
git push space main
```

---

## 📁 Project Structure

```
mohamed-sayed-hybrid-drug-design-v2/
├── app.py                          # Main Streamlit app (8 tabs)
├── Dockerfile                      # Docker configuration for HF Spaces
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages (HF Spaces legacy)
├── README.md                       # This file (+ HF Spaces metadata)
├── .dockerignore                   # Docker build exclusions
├── .gitignore                      # Git exclusions
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
    └── config.toml                 # Streamlit theming (local dev)
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
- **Docking:** AutoDock Vina Python API (primary) with RDKit fallback scoring
- **ML:** scikit-learn RandomForest with 5-fold CV
- **3D:** py3Dmol for interactive molecular visualization
- **Charts:** Plotly for radar, scatter, parallel coordinates
- **Deployment:** Docker-based for full system dependency support

---

## 🌐 Deployment Platforms Comparison

| Platform | Docker Support | Free Tier RAM | Vina Support | Verdict |
|----------|---------------|---------------|--------------|---------|
| **HF Spaces** | ✅ Docker SDK | 16 GB | ✅ Full | **Best choice** |
| Streamlit Cloud | ❌ No Docker | 1 GB | ❌ No system deps | Too limited |
| Render | ✅ Docker | 512 MB | ❌ Too little RAM | Not enough |
| Railway | ✅ Docker | 512 MB (trial) | ⚠️ Limited | Short trial |
| Fly.io | ✅ Docker | 256 MB | ❌ Too little RAM | Not enough |

---

## 👨‍🔬 Author

**Mohamed Sayed** — Computational Medicinal Chemistry

---

*For research use only. Not intended for clinical decision-making.*
