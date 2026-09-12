"""
Hybrid Drug Discovery Pipeline Orchestrator
=============================================
Coordinates the dual-pathway pipeline:
  1. Shared pharmacophore model generation
  2. Shared ML classifier training
  3. Pathway 1: Fragment-Based Drug Design (FBDD)
  4. Pathway 2: Pharmacophore-Guided Database Screening
  5. Combined ranking and output generation
"""

import io
import os
import csv
import json
import zipfile
import logging
import tempfile
from typing import List, Dict, Optional, Callable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
try:
    from rdkit.Chem import Draw
except ImportError:
    Draw = None

from modules.pharmacophore import (
    generate_pharmacophore, load_pharmit_json, export_pharmit_json,
    generate_decoys, train_classifier, score_molecules,
    generate_viewer_html, build_feature_vector, identify_binding_site
)
from modules.fragment_engine import (
    run_fragment_pathway, load_fragment_library
)
from modules.screening import (
    run_database_pathway, screen_databases
)
from modules.docking import (
    dock_batch, detect_binding_site, dock_molecule, generate_pose_sdf,
    post_docking_analysis
)
from modules.admet import (
    full_admet_profile, calculate_mpo, score_final,
    generate_radar_chart, batch_admet_profiles
)

logger = logging.getLogger(__name__)


class HybridPipeline:
    """Main orchestrator for the dual-pathway hybrid drug discovery pipeline."""
    
    def __init__(
        self,
        active_mols: List = None,
        active_smiles: List[str] = None,
        pdb_block: str = None,
        fragment_sdf_path: str = None,
        pharmit_json: dict = None,
        params: dict = None,
    ):
        self.active_mols = active_mols or []
        self.active_smiles = active_smiles or []
        self.pdb_block = pdb_block or ""
        self.fragment_sdf_path = fragment_sdf_path
        self.pharmit_json_input = pharmit_json
        self.params = params or {}
        
        # Pipeline state
        self.pharmacophore_features = None
        self.pharmit_json = None
        self.binding_residues = None
        self.binding_center = None
        self.classifier = None
        self.classifier_metrics = None
        self.classifier_threshold = 0.80
        
        # Results
        self.pathway1_results = None
        self.pathway2_results = None
        self.combined_results = None
        self.fragment_leads = None
        self.database_leads = None
        
        # Docking results
        self.fragment_docking = None
        self.database_docking = None
        
        # ADMET profiles
        self.fragment_admet = None
        self.database_admet = None
        
        # Final leads
        self.final_fragment_leads = None
        self.final_database_leads = None
        self.combined_leaderboard = None
        
        # Output directory
        self.output_dir = tempfile.mkdtemp()
        
        # Parameters
        self.exhaustiveness = self.params.get("exhaustiveness", 4)  # reduced: was 8
        self.box_size = tuple(self.params.get("box_size", [20.0, 20.0, 20.0]))
        self.num_modes = self.params.get("num_modes", 5)
        self.frag_threshold = self.params.get("frag_threshold", 0.70)
        self.evolved_threshold = self.params.get("evolved_threshold", 0.80)
        self.db_threshold = self.params.get("db_threshold", 0.80)
        self.data_dir = self.params.get("data_dir", "data")
        # Docking budget (per pathway): keep manageable on CPU-only HF Spaces
        self.max_dock_per_pathway = self.params.get("max_dock_per_pathway", 100)
    
    def _score_fn(self, mols):
        """Shared scoring function using trained classifier."""
        return score_molecules(mols, self.classifier, self.pharmacophore_features, self.binding_residues)
    
    # ═══════════════════════════════════════════════════════════════
    #  STEP 0: PHARMACOPHORE MODEL
    # ═══════════════════════════════════════════════════════════════
    
    def step0_pharmacophore(self, progress_callback=None):
        """Build or load pharmacophore model."""
        if progress_callback:
            progress_callback(0.0, "🔬 Building pharmacophore model...")
        
        if self.pharmit_json_input:
            # Load pre-built pharmacophore
            logger.info("Loading pre-built pharmacophore from JSON")
            self.pharmacophore_features = load_pharmit_json(self.pharmit_json_input)
            self.pharmit_json = self.pharmit_json_input
            # Still need binding site info for interaction fingerprints
            if self.pdb_block:
                self.binding_residues, self.binding_center = identify_binding_site(
                    self.pdb_block, self.active_mols
                )
                logger.info(f"Identified {len(self.binding_residues)} binding site residues for IFP")
            if progress_callback:
                progress_callback(1.0, "✅ Pharmacophore loaded from JSON")
        else:
            # Auto-generate from actives + PDB
            logger.info("Auto-generating pharmacophore from actives + PDB")
            self.pharmacophore_features, self.pharmit_json, \
                self.binding_residues, self.binding_center = generate_pharmacophore(
                    self.pdb_block, self.active_mols
                )
            if progress_callback:
                progress_callback(1.0, f"✅ Pharmacophore: {len(self.pharmacophore_features)} features")
        
        return self.pharmacophore_features
    
    # ═══════════════════════════════════════════════════════════════
    #  STEP 1: ML CLASSIFIER
    # ═══════════════════════════════════════════════════════════════
    
    def step1_classifier(self, progress_callback=None):
        """Train shared ML classifier."""
        if progress_callback:
            progress_callback(0.0, "🤖 Training ML classifier...")
        
        if not self.pharmacophore_features:
            self.step0_pharmacophore()
        
        # Generate decoys
        if progress_callback:
            progress_callback(0.2, "Generating property-matched decoys...")
        decoys = generate_decoys(self.active_mols, n_decoys=max(500, len(self.active_mols) * 50))
        
        # Train
        if progress_callback:
            progress_callback(0.5, "Training RandomForest classifier (5-fold CV)...")
        self.classifier, self.classifier_metrics, self.classifier_threshold = \
            train_classifier(self.active_mols, decoys, self.pharmacophore_features,
                           binding_residues=self.binding_residues)
        
        if progress_callback:
            if self.classifier_metrics:
                auc = self.classifier_metrics.get('roc_auc', 0)
                f1 = self.classifier_metrics.get('f1', 0)
                progress_callback(1.0, f"✅ Classifier trained: AUC={auc:.3f}, F1={f1:.3f}")
            else:
                progress_callback(1.0, "⚠️ Classifier training completed with limited data")
        
        return self.classifier, self.classifier_metrics
    
    # ═══════════════════════════════════════════════════════════════
    #  PATHWAY 1: FRAGMENT-BASED DRUG DESIGN
    # ═══════════════════════════════════════════════════════════════
    
    def pathway1_fbdd(self, progress_callback=None):
        """Run Fragment-Based Drug Design pathway."""
        if progress_callback:
            progress_callback(0.0, "🧬 Starting Fragment-Based Drug Design...")
        
        if self.classifier is None:
            self.step1_classifier()
        
        # Run fragment pathway
        self.pathway1_results = run_fragment_pathway(
            classifier=self.classifier,
            pharm_features=self.pharmacophore_features,
            score_fn=self._score_fn,
            fragment_sdf_path=self.fragment_sdf_path,
            data_dir=self.data_dir,
            frag_threshold=self.frag_threshold,
            evolved_threshold=self.evolved_threshold,
            top_fragments_n=100,     # was 200
            max_evolved=200,         # was 1000
            progress_callback=lambda s, m: progress_callback(s * 0.5, m) if progress_callback else None,
        )
        
        top_evolved = self.pathway1_results.get("top_evolved", [])
        logger.info(f"Fragment pathway produced {len(top_evolved)} candidate structures")
        
        # Safety net: if no evolved molecules, use top scored fragments directly
        if len(top_evolved) == 0:
            logger.warning("No evolved molecules from fragment pathway! Using top scored fragments.")
            top_frags = self.pathway1_results.get("top_fragments", [])
            for frag in top_frags[:50]:
                top_evolved.append({
                    "mol": frag["mol"],
                    "smiles": frag["smiles"],
                    "provenance": f"{frag['frag_id']} (direct fragment)",
                    "score": frag["score"],
                    "method": "fragment_direct",
                    "mw": frag.get("mw", 0),
                    "classifier_score": frag["score"],
                })
        
        # ── Pre-docking triage: Lipinski Ro5 + PAINS ──
        from rdkit.Chem import Descriptors as _D, rdMolDescriptors as _RD, FilterCatalog as _FC
        _pp = _FC.FilterCatalogParams()
        _pp.AddCatalog(_FC.FilterCatalogParams.FilterCatalogs.PAINS)
        _pains_cat = _FC.FilterCatalog(_pp)
        def _passes_frag_triage(m):
            if m is None:
                return False
            try:
                v = sum([_D.MolWt(m) > 550, _D.MolLogP(m) > 5.5,
                         _RD.CalcNumHBD(m) > 5, _RD.CalcNumHBA(m) > 10])
                return v == 0 and _pains_cat.GetFirstMatch(m) is None
            except Exception:
                return True

        _budget = self.max_dock_per_pathway
        _pool = [(e, e.get("mol")) for e in top_evolved if _passes_frag_triage(e.get("mol"))]
        _pool.sort(key=lambda x: x[0].get("score", 0), reverse=True)
        top_evolved_dock = [e for e, _ in _pool[:_budget]]
        if not top_evolved_dock:
            top_evolved_dock = sorted(top_evolved, key=lambda x: x.get("score", 0), reverse=True)[:_budget]

        # Dock evolved molecules (budget-capped)
        if progress_callback:
            progress_callback(0.5, f"⚗️ Docking {len(top_evolved_dock)} fragment-derived molecules (budget ≤{_budget})...")

        dock_mols   = [e["mol"]    for e in top_evolved_dock]
        dock_smiles = [e["smiles"] for e in top_evolved_dock]

        center, box = detect_binding_site(self.pdb_block)
        self.fragment_docking = dock_batch(
            dock_mols, dock_smiles, self.pdb_block,
            center=center, box_size=self.box_size,
            exhaustiveness=self.exhaustiveness,
            num_modes=self.num_modes,
            max_molecules=_budget,
            progress_callback=lambda c, t: progress_callback(
                0.5 + 0.3 * c / max(t, 1), f"Docking fragment {c}/{t}") if progress_callback else None,
        )

        docked_source = top_evolved_dock  # for downstream indexing
        
        # Get top 20 by docking score
        docked_with_info = []
        for i, dock_res in enumerate(self.fragment_docking):
            if i < len(docked_source):
                info = docked_source[i].copy()
                info["docking_score"]  = dock_res.get("best_score", 0.0)
                info["docking_poses"]  = dock_res.get("scores", [])
                info["docking_method"] = dock_res.get("method", "unknown")
                docked_with_info.append(info)
        
        docked_with_info.sort(key=lambda x: x.get("docking_score", 0.0))
        top_20_fragments = docked_with_info[:20]
        
        # ADMET profiling for top 20
        if progress_callback:
            progress_callback(0.85, "📊 Computing ADMET profiles for top 20 fragment leads...")
        
        for lead in top_20_fragments:
            mol = lead.get("mol")
            smi = lead.get("smiles")
            profile = full_admet_profile(mol, smiles=smi)
            if profile:
                mpo = calculate_mpo(profile)
                profile["MPO"] = mpo
                lead["admet"] = profile
                lead["mpo_score"] = mpo
                # Post-docking analysis (ProLIF + ODDT)
                pda = post_docking_analysis(mol, self.pdb_block, lead.get("docking_score"))
                lead["prolif"] = pda.get("prolif")
                lead["oddt"] = pda.get("oddt")
                lead["enhanced_docking_score"] = pda.get("enhanced_score", lead.get("docking_score", 0.0))
                # Final score with enhanced docking
                lead["final_score"] = score_final(
                    lead.get("docking_score"),
                    mpo,
                    lead.get("classifier_score", 0.5)
                )
                # Boost final score if ProLIF/ODDT provided extra info
                if lead["prolif"]:
                    lead["final_score"] = round(lead["final_score"] * 0.85 + pda["prolif"]["score"] * 0.15, 4)
                if lead["oddt"] and "rfscore" in lead["oddt"]:
                    norm_rf = max(0, min(1, lead["oddt"]["rfscore"] / 10.0))
                    lead["final_score"] = round(lead["final_score"] * 0.9 + norm_rf * 0.1, 4)
            else:
                lead["admet"] = {}
                lead["mpo_score"] = 0.0
                lead["final_score"] = 0.0
        
        # Sort by final score
        top_20_fragments.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        
        self.fragment_leads = top_20_fragments
        self.fragment_admet = [l.get("admet", {}) for l in top_20_fragments]
        
        if progress_callback:
            progress_callback(1.0, f"✅ Fragment pathway complete: {len(top_20_fragments)} leads")
        
        return self.fragment_leads
    
    # ═══════════════════════════════════════════════════════════════
    #  PATHWAY 2: DATABASE SCREENING
    # ═══════════════════════════════════════════════════════════════
    
    def pathway2_screening(self, progress_callback=None):
        """Run Pharmacophore-Guided Database Screening pathway."""
        if progress_callback:
            progress_callback(0.0, "🗄️ Starting Database Screening...")
        
        if self.classifier is None:
            self.step1_classifier()
        
        # Run database screening
        self.pathway2_results = run_database_pathway(
            classifier=self.classifier,
            pharm_features=self.pharmacophore_features,
            pharm_json=self.pharmit_json,
            score_fn=self._score_fn,
            active_smiles=self.active_smiles,
            classifier_threshold=self.db_threshold,
            top_n=100,   # was 500
            progress_callback=lambda s, m: progress_callback(s * 0.5, m) if progress_callback else None,
        )
        
        top_db_hits = self.pathway2_results.get("top_hits", [])
        logger.info(f"Database pathway: {len(top_db_hits)} candidates for docking")
        
        # Safety net: if no hits above threshold, progressively lower it
        if len(top_db_hits) == 0:
            all_hits = self.pathway2_results.get("all_hits", [])
            if all_hits:
                for fallback_thresh in [0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0]:
                    fallback = [h for h in all_hits if h.get("classifier_score", 0) >= fallback_thresh]
                    if len(fallback) >= 10:
                        logger.warning(
                            f"No DB hits above {self.db_threshold}. "
                            f"Lowered to {fallback_thresh} → {len(fallback)} hits."
                        )
                        fallback.sort(key=lambda x: x.get("classifier_score", 0), reverse=True)
                        top_db_hits = fallback[:100]   # cap at 100
                        break
                else:
                    logger.warning("Taking top 100 DB hits by classifier score regardless of threshold.")
                    all_hits.sort(key=lambda x: x.get("classifier_score", 0), reverse=True)
                    top_db_hits = all_hits[:100]

        # ── Pre-docking triage: Ro5 + PAINS for DB hits ──
        from rdkit.Chem import Descriptors as _D2, rdMolDescriptors as _RD2, FilterCatalog as _FC2
        _pp2 = _FC2.FilterCatalogParams()
        _pp2.AddCatalog(_FC2.FilterCatalogParams.FilterCatalogs.PAINS)
        _pains2 = _FC2.FilterCatalog(_pp2)
        def _triage_db(h):
            m = h.get("mol") or Chem.MolFromSmiles(h.get("smiles", ""))
            if m is None:
                return False
            try:
                v = sum([_D2.MolWt(m) > 550, _D2.MolLogP(m) > 5.5,
                         _RD2.CalcNumHBD(m) > 5, _RD2.CalcNumHBA(m) > 10])
                return v == 0 and _pains2.GetFirstMatch(m) is None
            except Exception:
                return True

        _budget_db = self.max_dock_per_pathway
        db_pool = sorted(
            [h for h in top_db_hits if _triage_db(h)],
            key=lambda x: x.get("classifier_score", 0), reverse=True
        )[:_budget_db]
        if not db_pool:
            db_pool = sorted(top_db_hits, key=lambda x: x.get("classifier_score", 0), reverse=True)[:_budget_db]

        # Dock database hits (budget-capped)
        if progress_callback:
            progress_callback(0.5, f"⚗️ Docking {len(db_pool)} database hits (budget ≤{_budget_db})...")

        dock_mols   = [h.get("mol") or Chem.MolFromSmiles(h["smiles"]) for h in db_pool]
        dock_smiles = [h.get("smiles", "") for h in db_pool]

        center, box = detect_binding_site(self.pdb_block)
        self.database_docking = dock_batch(
            dock_mols, dock_smiles, self.pdb_block,
            center=center, box_size=self.box_size,
            exhaustiveness=self.exhaustiveness,
            num_modes=self.num_modes,
            max_molecules=_budget_db,
            progress_callback=lambda c, t: progress_callback(
                0.5 + 0.3 * c / max(t, 1), f"Docking DB hit {c}/{t}") if progress_callback else None,
        )

        db_docked_source = db_pool  # for downstream indexing
        
        # Get top 20 by docking score
        docked_db = []
        for i, dock_res in enumerate(self.database_docking):
            if i < len(db_docked_source):
                info = db_docked_source[i].copy()
                info["docking_score"]  = dock_res.get("best_score", 0.0)
                info["docking_poses"]  = dock_res.get("scores", [])
                info["docking_method"] = dock_res.get("method", "unknown")
                docked_db.append(info)
        
        docked_db.sort(key=lambda x: x.get("docking_score", 0.0))
        top_20_db = docked_db[:20]
        
        # ADMET profiling
        if progress_callback:
            progress_callback(0.85, "📊 Computing ADMET profiles for top 20 database leads...")
        
        for lead in top_20_db:
            mol = lead.get("mol")
            smi = lead.get("smiles")
            if mol is None and smi:
                mol = Chem.MolFromSmiles(smi)
            profile = full_admet_profile(mol, smiles=smi)
            if profile:
                mpo = calculate_mpo(profile)
                profile["MPO"] = mpo
                lead["admet"] = profile
                lead["mpo_score"] = mpo
                # Post-docking analysis (ProLIF + ODDT)
                pda = post_docking_analysis(mol, self.pdb_block, lead.get("docking_score"))
                lead["prolif"] = pda.get("prolif")
                lead["oddt"] = pda.get("oddt")
                lead["enhanced_docking_score"] = pda.get("enhanced_score", lead.get("docking_score", 0.0))
                # Final score with enhanced docking
                lead["final_score"] = score_final(
                    lead.get("docking_score"),
                    mpo,
                    lead.get("classifier_score", 0.5)
                )
                # Boost final score if ProLIF/ODDT provided extra info
                if lead["prolif"]:
                    lead["final_score"] = round(lead["final_score"] * 0.85 + pda["prolif"]["score"] * 0.15, 4)
                if lead["oddt"] and "rfscore" in lead["oddt"]:
                    norm_rf = max(0, min(1, lead["oddt"]["rfscore"] / 10.0))
                    lead["final_score"] = round(lead["final_score"] * 0.9 + norm_rf * 0.1, 4)
            else:
                lead["admet"] = {}
                lead["mpo_score"] = 0.0
                lead["final_score"] = 0.0
        
        top_20_db.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        
        self.database_leads = top_20_db
        self.database_admet = [l.get("admet", {}) for l in top_20_db]
        
        if progress_callback:
            progress_callback(1.0, f"✅ Database pathway complete: {len(top_20_db)} leads")
        
        return self.database_leads
    
    # ═══════════════════════════════════════════════════════════════
    #  COMBINED RESULTS
    # ═══════════════════════════════════════════════════════════════
    
    def combine_results(self):
        """Merge results from both pathways into unified leaderboard."""
        combined = []
        
        # Fragment leads
        if self.fragment_leads:
            for i, lead in enumerate(self.fragment_leads):
                combined.append({
                    "rank": 0,  # Will be set after sorting
                    "pathway": "🧬 Fragment",
                    "pathway_label": "Fragment",
                    "smiles": lead.get("smiles", ""),
                    "docking_score": lead.get("docking_score", 0.0),
                    "classifier_score": lead.get("classifier_score", 0.0),
                    "mpo_score": lead.get("mpo_score", 0.0),
                    "final_score": lead.get("final_score", 0.0),
                    "provenance": lead.get("provenance", ""),
                    "method": lead.get("method", ""),
                    "source": "Fragment Evolution",
                    "docking_method": lead.get("docking_method", ""),
                    "admet": lead.get("admet", {}),
                    "mol": lead.get("mol"),
                })
        
        # Database leads
        if self.database_leads:
            for i, lead in enumerate(self.database_leads):
                combined.append({
                    "rank": 0,
                    "pathway": "🗄️ Database",
                    "pathway_label": "Database",
                    "smiles": lead.get("smiles", ""),
                    "docking_score": lead.get("docking_score", 0.0),
                    "classifier_score": lead.get("classifier_score", 0.0),
                    "mpo_score": lead.get("mpo_score", 0.0),
                    "final_score": lead.get("final_score", 0.0),
                    "provenance": lead.get("provenance", lead.get("id", "")),
                    "method": "database_screen",
                    "source": lead.get("source", ""),
                    "docking_method": lead.get("docking_method", ""),
                    "admet": lead.get("admet", {}),
                    "mol": lead.get("mol"),
                })
        
        # Sort by final score
        combined.sort(key=lambda x: x["final_score"], reverse=True)
        for i, entry in enumerate(combined):
            entry["rank"] = i + 1
        
        self.combined_leaderboard = combined
        return combined
    
    # ═══════════════════════════════════════════════════════════════
    #  FULL PIPELINE
    # ═══════════════════════════════════════════════════════════════
    
    def run_full_pipeline(self, progress_callback=None):
        """Run complete dual-pathway pipeline end-to-end."""
        
        def step_progress(overall_start, overall_end):
            def inner(frac, msg):
                if progress_callback:
                    overall = overall_start + frac * (overall_end - overall_start)
                    progress_callback(overall, msg)
            return inner
        
        # Step 0: Pharmacophore
        self.step0_pharmacophore(step_progress(0.0, 0.10))
        
        # Step 1: Classifier
        self.step1_classifier(step_progress(0.10, 0.20))
        
        # Pathway 1: Fragment-Based
        self.pathway1_fbdd(step_progress(0.20, 0.55))
        
        # Pathway 2: Database Screening
        self.pathway2_screening(step_progress(0.55, 0.90))
        
        # Combine
        if progress_callback:
            progress_callback(0.92, "📊 Combining results from both pathways...")
        self.combine_results()
        
        if progress_callback:
            n_total = len(self.combined_leaderboard) if self.combined_leaderboard else 0
            progress_callback(1.0, f"✅ Pipeline complete! {n_total} leads identified.")
        
        return self.combined_leaderboard
    
    # ═══════════════════════════════════════════════════════════════
    #  OUTPUT GENERATION
    # ═══════════════════════════════════════════════════════════════
    
    def _leads_to_dataframe(self, leads, pathway_name=""):
        """Convert lead list to DataFrame."""
        rows = []
        for lead in leads:
            admet = lead.get("admet", {})
            row = {
                "Rank": lead.get("rank", 0),
                "SMILES": lead.get("smiles", ""),
                "Pathway": pathway_name,
                "Docking_Score": lead.get("docking_score", 0.0),
                "Classifier_Score": lead.get("classifier_score", 0.0),
                "MPO_Score": lead.get("mpo_score", 0.0),
                "Final_Score": lead.get("final_score", 0.0),
                "MW": admet.get("MW", 0),
                "cLogP": admet.get("cLogP", 0),
                "TPSA": admet.get("TPSA", 0),
                "HBD": admet.get("HBD", 0),
                "HBA": admet.get("HBA", 0),
                "RotBonds": admet.get("RotBonds", 0),
                "QED": admet.get("QED", 0),
                "SAS": admet.get("SAS", 0),
                "CYP3A4_risk": admet.get("CYP3A4_risk", ""),
                "hERG_risk": admet.get("hERG_risk", ""),
                "BBB_label": admet.get("BBB_label", ""),
                "Solubility_class": admet.get("Solubility_class", ""),
                "Provenance": lead.get("provenance", ""),
                "Source": lead.get("source", ""),
                "Docking_Method": lead.get("docking_method", ""),
            }
            rows.append(row)
        return pd.DataFrame(rows)
    
    def export_fragment_results_csv(self):
        """Export fragment pathway results to CSV string."""
        if not self.fragment_leads:
            return ""
        df = self._leads_to_dataframe(self.fragment_leads, "Fragment")
        return df.to_csv(index=False)
    
    def export_database_results_csv(self):
        """Export database pathway results to CSV string."""
        if not self.database_leads:
            return ""
        df = self._leads_to_dataframe(self.database_leads, "Database")
        return df.to_csv(index=False)
    
    def export_combined_leaderboard_csv(self):
        """Export combined leaderboard to CSV string."""
        if not self.combined_leaderboard:
            return ""
        df = self._leads_to_dataframe(self.combined_leaderboard, "Combined")
        # Use actual pathway labels
        for i, entry in enumerate(self.combined_leaderboard):
            df.at[i, "Pathway"] = entry.get("pathway_label", "")
        return df.to_csv(index=False)
    
    def export_pharmacophore_json(self):
        """Get pharmacophore model as JSON string."""
        if self.pharmit_json:
            return json.dumps(self.pharmit_json, indent=2)
        return "{}"
    
    def export_fragment_origins_csv(self):
        """Export fragment provenance data."""
        if not self.fragment_leads:
            return ""
        rows = []
        for lead in self.fragment_leads:
            rows.append({
                "SMILES": lead.get("smiles", ""),
                "Provenance": lead.get("provenance", ""),
                "Method": lead.get("method", ""),
                "Docking_Score": lead.get("docking_score", 0.0),
                "Final_Score": lead.get("final_score", 0.0),
            })
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
    
    def export_docking_scores_csv(self):
        """Export all docking scores."""
        rows = []
        all_leads = (self.fragment_leads or []) + (self.database_leads or [])
        for lead in all_leads:
            for pose in lead.get("docking_poses", []):
                rows.append({
                    "SMILES": lead.get("smiles", ""),
                    "Pathway": lead.get("source", ""),
                    "Pose": pose.get("pose", 0),
                    "Score": pose.get("score", 0.0),
                    "RMSD_LB": pose.get("rmsd_lb", 0.0),
                    "RMSD_UB": pose.get("rmsd_ub", 0.0),
                })
        if rows:
            return pd.DataFrame(rows).to_csv(index=False)
        return ""
    
    def export_admet_full_csv(self):
        """Export full ADMET report for all leads."""
        all_admet = (self.fragment_admet or []) + (self.database_admet or [])
        if all_admet:
            df = pd.DataFrame(all_admet)
            return df.to_csv(index=False)
        return ""
    
    def export_viewer_html(self):
        """Generate standalone pharmacophore viewer HTML."""
        if self.pharmacophore_features and self.pdb_block:
            return generate_viewer_html(self.pdb_block, self.pharmacophore_features)
        return "<html><body><p>No pharmacophore data available.</p></body></html>"
    
    def create_zip(self):
        """Create ZIP archive containing all output files."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # CSV files
            csv_files = [
                ("fragment_results.csv", self.export_fragment_results_csv()),
                ("database_results.csv", self.export_database_results_csv()),
                ("combined_leaderboard.csv", self.export_combined_leaderboard_csv()),
                ("pharmacophore_model.json", self.export_pharmacophore_json()),
                ("fragment_origins.csv", self.export_fragment_origins_csv()),
                ("docking_scores_all.csv", self.export_docking_scores_csv()),
                ("admet_full_report.csv", self.export_admet_full_csv()),
            ]
            
            for name, content in csv_files:
                if content:
                    zf.writestr(name, content)
            
            # Viewer HTML
            viewer_html = self.export_viewer_html()
            zf.writestr("pharmacophore_viewer.html", viewer_html)
            
            # Generate SDF files for leads
            # Fragment leads
            if self.fragment_leads:
                for i, lead in enumerate(self.fragment_leads):
                    mol = lead.get("mol")
                    if mol:
                        try:
                            sdf_path = os.path.join(self.output_dir, f"fragment_lead_{i+1}.sdf")
                            poses = lead.get("docking_poses", [{"pose": 1, "score": 0}])
                            generate_pose_sdf(mol, poses, sdf_path)
                            if os.path.exists(sdf_path):
                                zf.write(sdf_path, f"fragment_leads/lead_{i+1}_poses.sdf")
                        except Exception as e:
                            logger.error(f"SDF generation failed for fragment lead {i}: {e}")
            
            # Database leads
            if self.database_leads:
                for i, lead in enumerate(self.database_leads):
                    mol = lead.get("mol")
                    if mol:
                        try:
                            sdf_path = os.path.join(self.output_dir, f"database_lead_{i+1}.sdf")
                            poses = lead.get("docking_poses", [{"pose": 1, "score": 0}])
                            generate_pose_sdf(mol, poses, sdf_path)
                            if os.path.exists(sdf_path):
                                zf.write(sdf_path, f"database_leads/lead_{i+1}_poses.sdf")
                        except Exception as e:
                            logger.error(f"SDF generation failed for database lead {i}: {e}")
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    # ═══════════════════════════════════════════════════════════════
    #  STATUS / SUMMARY
    # ═══════════════════════════════════════════════════════════════
    
    def get_summary(self):
        """Get pipeline run summary."""
        summary = {
            "pharmacophore": {
                "n_features": len(self.pharmacophore_features) if self.pharmacophore_features else 0,
                "source": "user_json" if self.pharmit_json_input else "auto_generated",
            },
            "classifier": self.classifier_metrics or {},
            "pathway1": {
                "n_fragments_library": self.pathway1_results.get("n_library", 0) if self.pathway1_results else 0,
                "n_top_fragments": self.pathway1_results.get("n_scored", 0) if self.pathway1_results else 0,
                "n_evolved": self.pathway1_results.get("n_evolved", 0) if self.pathway1_results else 0,
                "n_leads": len(self.fragment_leads) if self.fragment_leads else 0,
            },
            "pathway2": {
                "n_screened": self.pathway2_results.get("total_screened", 0) if self.pathway2_results else 0,
                "n_filtered": self.pathway2_results.get("total_filtered", 0) if self.pathway2_results else 0,
                "n_leads": len(self.database_leads) if self.database_leads else 0,
                "source_counts": self.pathway2_results.get("source_counts", {}) if self.pathway2_results else {},
            },
            "combined": {
                "n_total_leads": len(self.combined_leaderboard) if self.combined_leaderboard else 0,
            },
        }
        return summary


# ═══════════════════════════════════════════════════════════════
#  HELPER: PARSE INPUT FILES
# ═══════════════════════════════════════════════════════════════

def parse_actives_file(file_content, file_name):
    """Parse active compound file (.smi, .txt, or .sdf).
    Returns list of (mol, smiles) tuples."""
    actives = []
    
    if file_name.endswith('.sdf'):
        suppl = Chem.SDMolSupplier()
        suppl.SetData(file_content)
        for mol in suppl:
            if mol is not None:
                smi = Chem.MolToSmiles(mol)
                actives.append((mol, smi))
    else:
        # .txt or .smi — one SMILES per line
        for line in file_content.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Handle tab/space separated (SMILES\tname)
            parts = line.split()
            smi = parts[0]
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                actives.append((mol, smi))
    
    return actives


def parse_pdb_file(pdb_content):
    """Parse PDB file content."""
    if isinstance(pdb_content, bytes):
        return pdb_content.decode('utf-8')
    return pdb_content
