"""
Pharmacophore Model Generation & ML Classifier Module
======================================================
Handles:
  - PDB binding site analysis with ProDy
  - 3D conformer generation (ETKDGv3 + MMFF)
  - Interaction-aware pharmacophore feature extraction
  - Feature clustering and frequency filtering
  - Pharmit JSON import/export
  - Property-matched decoy generation
  - Combined fingerprint vector (3D pharmacophore FP + Morgan FP + RDKit descriptors)
  - RandomForest binary classifier training (shared by both pathways)
  - py3Dmol 3D visualization HTML
"""

import json
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
try:
    from rdkit.Chem import Draw
except ImportError:
    Draw = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, classification_report
)

logger = logging.getLogger(__name__)

# ─── PHARMACOPHORE FEATURE TYPES ────────────────────────────────────────
FEATURE_COLORS = {
    "HBA": "#FF4444",      # Red
    "HBD": "#4444FF",      # Blue
    "Aromatic": "#44FF44",  # Green
    "Hydrophobic": "#FFFF44",  # Yellow
    "PosIonizable": "#FF44FF", # Magenta
    "NegIonizable": "#44FFFF", # Cyan
}

FEATURE_SMARTS = {
    "HBA": [
        Chem.MolFromSmarts("[#7;!$([#7;H0](=O)~O)]"),  # N (not nitro)
        Chem.MolFromSmarts("[#8]"),                       # O
        Chem.MolFromSmarts("[F]"),                        # F
    ],
    "HBD": [
        Chem.MolFromSmarts("[#7;!H0]"),  # NH
        Chem.MolFromSmarts("[#8;!H0]"),  # OH
    ],
    "Aromatic": [
        Chem.MolFromSmarts("a1aaaa1"),     # 5-ring aromatic
        Chem.MolFromSmarts("a1aaaaa1"),    # 6-ring aromatic
    ],
    "Hydrophobic": [
        Chem.MolFromSmarts("[CH3]"),
        Chem.MolFromSmarts("[CH2]"),
        Chem.MolFromSmarts("[CH;!$(C=O)]"),
        Chem.MolFromSmarts("[c;!$(c~[#7,#8])]"),
        Chem.MolFromSmarts("[S;X2]"),
    ],
    "PosIonizable": [
        Chem.MolFromSmarts("[#7;+]"),
        Chem.MolFromSmarts("[NH2;!$(NC=O)]"),
        Chem.MolFromSmarts("[NH;!$(NC=O);!$(Nc)]([CH3])"),
        Chem.MolFromSmarts("c1cnc[nH]1"),  # imidazole
    ],
    "NegIonizable": [
        Chem.MolFromSmarts("[OX1;$(O~C~O)]"),  # carboxylate
        Chem.MolFromSmarts("[SX1;$(S~P)]"),     # phosphate/sulfate
        Chem.MolFromSmarts("[O;X1;$(O~S)]"),    # sulfonate
    ],
}
# Clean None entries
for k in FEATURE_SMARTS:
    FEATURE_SMARTS[k] = [s for s in FEATURE_SMARTS[k] if s is not None]

# Residues relevant for interaction matching
HBD_RESIDUES = {"SER", "THR", "TYR", "ASN", "GLN", "HIS", "TRP", "ARG", "LYS", "CYS"}
HBA_RESIDUES = {"SER", "THR", "TYR", "ASN", "GLN", "HIS", "ASP", "GLU", "CYS"}
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP", "MET"}
POS_CHARGE_RESIDUES = {"ASP", "GLU"}  # near these negative residues
NEG_CHARGE_RESIDUES = {"LYS", "ARG", "HIS"}  # near these positive residues

# 20 standard amino acids for binding site encoding
AMINO_ACIDS_20 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
]


# ═══════════════════════════════════════════════════════════════
#  PDB / BINDING SITE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def parse_pdb_residues(pdb_block: str):
    """Parse PDB to extract residue information and coordinates."""
    residues = {}
    atoms = []
    for line in pdb_block.strip().split('\n'):
        if line.startswith("ATOM"):
            try:
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21:22].strip()
                resnum = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                key = (chain, resnum, resname)
                if key not in residues:
                    residues[key] = {"name": resname, "chain": chain, "num": resnum,
                                     "atoms": [], "coords": []}
                residues[key]["atoms"].append(atom_name)
                residues[key]["coords"].append([x, y, z])
                atoms.append({"name": atom_name, "resname": resname, "chain": chain,
                              "resnum": resnum, "coord": [x, y, z]})
            except (ValueError, IndexError):
                continue
    return residues, atoms


def identify_binding_site(pdb_block: str, active_mols: list, distance=6.0):
    """Identify binding site residues within `distance` Å of any active molecule.
    Generates 3D conformers for actives and finds nearby protein residues."""
    residues, atoms = parse_pdb_residues(pdb_block)
    
    if not atoms:
        logger.warning("No ATOM records found in PDB")
        return [], np.array([0, 0, 0])
    
    # Get protein coordinates
    protein_coords = np.array([a["coord"] for a in atoms])
    
    # Try to get HETATM coordinates (existing ligand)
    hetatm_coords = []
    for line in pdb_block.strip().split('\n'):
        if line.startswith("HETATM"):
            resname = line[17:20].strip()
            if resname not in ("HOH", "WAT", "DOD"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    hetatm_coords.append([x, y, z])
                except (ValueError, IndexError):
                    pass
    
    reference_coords = []
    
    # Use HETATM ligand coordinates
    if hetatm_coords:
        reference_coords = hetatm_coords
    
    # Also generate coords from active molecules
    if active_mols:
        for mol in active_mols[:5]:  # Limit for speed
            if mol is None:
                continue
            try:
                mol3d = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                res = AllChem.EmbedMolecule(mol3d, params)
                if res == -1:
                    continue
                AllChem.MMFFOptimizeMolecule(mol3d, maxIters=200)
                conf = mol3d.GetConformer()
                for i in range(mol3d.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    reference_coords.append([pos.x, pos.y, pos.z])
            except Exception:
                continue
    
    if not reference_coords:
        # Fallback: use center of protein
        center = protein_coords.mean(axis=0)
        # Find residues near center
        binding_residues = []
        for key, res in residues.items():
            res_center = np.mean(res["coords"], axis=0)
            if np.linalg.norm(res_center - center) < 15.0:
                if res["name"] not in ("HOH", "WAT"):
                    binding_residues.append(res)
        return binding_residues[:30], center
    
    reference_coords = np.array(reference_coords)
    ref_center = reference_coords.mean(axis=0)
    
    # Find residues within distance of any reference coordinate
    binding_residues = []
    for key, res in residues.items():
        res_coords = np.array(res["coords"])
        for rc in reference_coords:
            dists = np.linalg.norm(res_coords - rc, axis=1)
            if np.min(dists) <= distance:
                if res["name"] not in ("HOH", "WAT"):
                    binding_residues.append(res)
                break
    
    return binding_residues, ref_center


# ═══════════════════════════════════════════════════════════════
#  CONFORMER GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_conformers(mol, num_conformers=20):
    """Generate 3D conformers using ETKDGv3 + MMFF optimization."""
    if mol is None:
        return None, []
    
    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 1
        params.pruneRmsThresh = 0.5
        
        cids = AllChem.EmbedMultipleConfs(mol3d, numConfs=num_conformers, params=params)
        
        if len(cids) == 0:
            # Fallback
            params2 = AllChem.ETKDG()
            params2.randomSeed = 42
            cids = AllChem.EmbedMultipleConfs(mol3d, numConfs=num_conformers, params=params2)
        
        if len(cids) == 0:
            return None, []
        
        # MMFF minimize each conformer
        for cid in cids:
            try:
                AllChem.MMFFOptimizeMolecule(mol3d, confId=int(cid), maxIters=200)
            except Exception:
                try:
                    AllChem.UFFOptimizeMolecule(mol3d, confId=int(cid), maxIters=200)
                except Exception:
                    pass
        
        return mol3d, list(cids)
    except Exception as e:
        logger.error(f"Conformer generation failed: {e}")
        return None, []


# ═══════════════════════════════════════════════════════════════
#  PHARMACOPHORE FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_features_from_conformer(mol3d, conf_id, binding_residues=None):
    """Extract pharmacophore features from a specific conformer,
    considering interaction geometry with binding site residues."""
    if mol3d is None:
        return []
    
    features = []
    conf = mol3d.GetConformer(conf_id)
    
    # Get binding residue info for interaction-aware extraction
    binding_resnames = set()
    if binding_residues:
        for res in binding_residues:
            binding_resnames.add(res["name"])
    
    for feat_type, smarts_list in FEATURE_SMARTS.items():
        for smarts in smarts_list:
            matches = mol3d.GetSubstructMatches(smarts)
            for match in matches:
                # Calculate center of matched atoms
                coords = []
                for idx in match:
                    pos = conf.GetAtomPosition(idx)
                    coords.append([pos.x, pos.y, pos.z])
                center = np.mean(coords, axis=0)
                
                # Apply interaction-aware filtering if binding site info available
                relevant = True
                if binding_residues and len(binding_resnames) > 0:
                    if feat_type == "HBA" and not binding_resnames.intersection(HBD_RESIDUES):
                        relevant = False  # No HBD residues nearby
                    elif feat_type == "HBD" and not binding_resnames.intersection(HBA_RESIDUES):
                        relevant = False
                    elif feat_type == "Aromatic" and not binding_resnames.intersection(AROMATIC_RESIDUES):
                        relevant = False
                    elif feat_type == "PosIonizable" and not binding_resnames.intersection(POS_CHARGE_RESIDUES):
                        relevant = False
                    elif feat_type == "NegIonizable" and not binding_resnames.intersection(NEG_CHARGE_RESIDUES):
                        relevant = False
                    # Hydrophobic always relevant
                
                if relevant:
                    features.append({
                        "type": feat_type,
                        "center": center.tolist(),
                        "radius": 1.5,
                        "atom_indices": list(match),
                    })
    
    return features


def extract_all_features(active_mols, binding_residues=None, num_conformers=20):
    """Extract pharmacophore features across all actives and their conformers."""
    all_features = []
    
    for mol_idx, mol in enumerate(active_mols):
        if mol is None:
            continue
        mol3d, cids = generate_conformers(mol, num_conformers)
        if mol3d is None:
            continue
        
        for cid in cids:
            features = extract_features_from_conformer(mol3d, int(cid), binding_residues)
            for feat in features:
                feat["mol_idx"] = mol_idx
                feat["conf_id"] = int(cid)
            all_features.extend(features)
    
    return all_features


# ═══════════════════════════════════════════════════════════════
#  FEATURE CLUSTERING
# ═══════════════════════════════════════════════════════════════

def cluster_features(features, rmsd_threshold=1.5):
    """Cluster pharmacophore features by type and spatial proximity.
    Features within rmsd_threshold Å are considered the same feature."""
    if not features:
        return []
    
    clustered = []
    
    # Group by feature type
    by_type = defaultdict(list)
    for feat in features:
        by_type[feat["type"]].append(feat)
    
    for feat_type, feats in by_type.items():
        if len(feats) == 1:
            clustered.append({
                "type": feat_type,
                "center": feats[0]["center"],
                "radius": 1.5,
                "count": 1,
                "frequency": 1.0,
            })
            continue
        
        # Cluster by spatial proximity
        coords = np.array([f["center"] for f in feats])
        
        if len(coords) < 2:
            clustered.append({
                "type": feat_type,
                "center": coords[0].tolist(),
                "radius": 1.5,
                "count": len(feats),
                "frequency": 1.0,
            })
            continue
        
        try:
            dists = pdist(coords)
            Z = linkage(dists, method='average')
            labels = fcluster(Z, t=rmsd_threshold, criterion='distance')
        except Exception:
            labels = np.ones(len(coords), dtype=int)
        
        for cluster_id in set(labels):
            mask = labels == cluster_id
            cluster_coords = coords[mask]
            center = cluster_coords.mean(axis=0)
            
            # Count unique molecules contributing to this cluster
            mol_indices = set()
            for idx in np.where(mask)[0]:
                mol_indices.add(feats[idx].get("mol_idx", 0))
            
            clustered.append({
                "type": feat_type,
                "center": center.tolist(),
                "radius": 1.5,
                "count": int(mask.sum()),
                "mol_count": len(mol_indices),
                "frequency": 0.0,  # Will be set in filter step
            })
    
    return clustered


def filter_by_frequency(clusters, num_actives, threshold=0.5):
    """Keep only feature clusters present in >threshold fraction of actives."""
    filtered = []
    for cluster in clusters:
        freq = cluster.get("mol_count", cluster["count"]) / max(num_actives, 1)
        cluster["frequency"] = round(freq, 3)
        if freq >= threshold or num_actives <= 2:  # Keep all if few actives
            filtered.append(cluster)
    return filtered


# ═══════════════════════════════════════════════════════════════
#  PHARMIT JSON IMPORT/EXPORT
# ═══════════════════════════════════════════════════════════════

def export_pharmit_json(features):
    """Export pharmacophore model to Pharmit-compatible JSON."""
    pharmit_features = []
    for i, feat in enumerate(features):
        pf = {
            "name": feat["type"],
            "enabled": True,
            "x": round(feat["center"][0], 3),
            "y": round(feat["center"][1], 3),
            "z": round(feat["center"][2], 3),
            "radius": feat.get("radius", 1.5),
            "svector": {"x": 0, "y": 0, "z": 0},  # No directionality for spheres
        }
        # Map type names to Pharmit types
        type_map = {
            "HBA": "HydrogenAcceptor",
            "HBD": "HydrogenDonor",
            "Aromatic": "Aromatic",
            "Hydrophobic": "Hydrophobic",
            "PosIonizable": "PositiveIon",
            "NegIonizable": "NegativeIon",
        }
        pf["name"] = type_map.get(feat["type"], feat["type"])
        pharmit_features.append(pf)
    
    model = {
        "points": pharmit_features,
        "smin": 0,
        "smax": 100,
    }
    return model


def load_pharmit_json(json_data):
    """Load pharmacophore from Pharmit JSON format."""
    if isinstance(json_data, str):
        model = json.loads(json_data)
    else:
        model = json_data
    
    features = []
    type_reverse = {
        "HydrogenAcceptor": "HBA",
        "HydrogenDonor": "HBD",
        "Aromatic": "Aromatic",
        "Hydrophobic": "Hydrophobic",
        "PositiveIon": "PosIonizable",
        "NegativeIon": "NegIonizable",
    }
    
    for pt in model.get("points", []):
        feat = {
            "type": type_reverse.get(pt.get("name", ""), pt.get("name", "Unknown")),
            "center": [pt.get("x", 0), pt.get("y", 0), pt.get("z", 0)],
            "radius": pt.get("radius", 1.5),
            "count": 1,
            "frequency": 1.0,
            "enabled": pt.get("enabled", True),
        }
        if feat["enabled"]:
            features.append(feat)
    
    return features


# ═══════════════════════════════════════════════════════════════
#  MULTI-FORMAT PHARMACOPHORE LOADERS
# ═══════════════════════════════════════════════════════════════

# Canonical type mapping used across all loaders
_CANONICAL_TYPES = {
    # Common names → internal types
    "hba": "HBA", "hydrogenacceptor": "HBA", "hydrogen_acceptor": "HBA",
    "acceptor": "HBA", "h-bond acceptor": "HBA", "a": "HBA",
    "hbd": "HBD", "hydrogendonor": "HBD", "hydrogen_donor": "HBD",
    "donor": "HBD", "h-bond donor": "HBD", "d": "HBD",
    "aromatic": "Aromatic", "ar": "Aromatic", "ring": "Aromatic",
    "pi": "Aromatic", "aro": "Aromatic",
    "hydrophobic": "Hydrophobic", "hyd": "Hydrophobic", "hp": "Hydrophobic",
    "lipo": "Hydrophobic", "lipophilic": "Hydrophobic",
    "posionizable": "PosIonizable", "positive": "PosIonizable",
    "positiveion": "PosIonizable", "pos": "PosIonizable", "pi+": "PosIonizable",
    "cation": "PosIonizable",
    "negionizable": "NegIonizable", "negative": "NegIonizable",
    "negativeion": "NegIonizable", "neg": "NegIonizable", "pi-": "NegIonizable",
    "anion": "NegIonizable",
    # MOE/Catalyst specific
    "don": "HBD", "acc": "HBA", "hydrophobe": "Hydrophobic",
    "cat": "PosIonizable", "ani": "NegIonizable",
}


def _normalize_feature_type(raw_type):
    """Normalize a raw feature type string to a canonical internal type."""
    key = raw_type.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    return _CANONICAL_TYPES.get(key, raw_type.strip())


def load_ph4_file(content):
    """Parse MOE .ph4 pharmacophore format.
    
    MOE .ph4 files typically contain feature definitions with type, coordinates,
    radius, optional expression, and optional weight/tolerance lines.
    
    Format examples:
        #moe:ph4que
        ACC  1.234  5.678  9.012  1.5
        DON  2.345  6.789  0.123  1.5
        HYD  3.456  7.890  1.234  2.0
        ARO  4.567  8.901  2.345  1.5
    """
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')
    
    features = []
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('!'):
            continue
        
        parts = line.split()
        if len(parts) < 4:
            continue
        
        # Try to parse: TYPE X Y Z [RADIUS]
        raw_type = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            radius = float(parts[4]) if len(parts) > 4 else 1.5
        except (ValueError, IndexError):
            continue
        
        feat_type = _normalize_feature_type(raw_type)
        features.append({
            "type": feat_type,
            "center": [x, y, z],
            "radius": radius,
            "count": 1,
            "frequency": 1.0,
        })
    
    logger.info(f"Loaded {len(features)} features from .ph4 file")
    return features


def load_lig_file(content):
    """Parse .lig pharmacophore/ligand definition files.
    
    .lig files can come from various tools and typically contain either:
    1) Feature definitions: TYPE X Y Z RADIUS
    2) Atom/coordinate definitions with pharmacophore annotations
    """
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')
    
    features = []
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith(';'):
            continue
        
        parts = line.split()
        
        # Try format: TYPE X Y Z [RADIUS]
        if len(parts) >= 4:
            raw_type = parts[0]
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                radius = float(parts[4]) if len(parts) > 4 else 1.5
                
                feat_type = _normalize_feature_type(raw_type)
                features.append({
                    "type": feat_type,
                    "center": [x, y, z],
                    "radius": radius,
                    "count": 1,
                    "frequency": 1.0,
                })
            except ValueError:
                continue
        
        # Try format: X Y Z TYPE [RADIUS] (alternative column order)
        if len(parts) >= 4 and len(features) == 0:
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                raw_type = parts[3]
                radius = float(parts[4]) if len(parts) > 4 else 1.5
                
                feat_type = _normalize_feature_type(raw_type)
                features.append({
                    "type": feat_type,
                    "center": [x, y, z],
                    "radius": radius,
                    "count": 1,
                    "frequency": 1.0,
                })
            except ValueError:
                continue
    
    logger.info(f"Loaded {len(features)} features from .lig file")
    return features


def load_pml_pharmacophore(content):
    """Parse PyMOL .pml script for pharmacophore pseudoatom definitions.
    
    PyMOL pharmacophore files typically use pseudoatom commands:
        pseudoatom HBA_1, pos=[1.23, 4.56, 7.89], vdw=1.5, color=red
        pseudoatom HBD_1, pos=[2.34, 5.67, 8.90], vdw=1.5, color=blue
    """
    import re
    
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')
    
    features = []
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Match pseudoatom definitions
        if 'pseudoatom' in line.lower() or 'sphere' in line.lower():
            # Extract name
            name_match = re.search(r'pseudoatom\s+(\w+)', line, re.IGNORECASE)
            name = name_match.group(1) if name_match else ""
            
            # Extract position
            pos_match = re.search(r'pos\s*=\s*\[([^\]]+)\]', line)
            if not pos_match:
                pos_match = re.search(r'pos\s*=\s*\(([^\)]+)\)', line)
            
            if pos_match:
                try:
                    coords = [float(c.strip()) for c in pos_match.group(1).split(',')]
                    if len(coords) >= 3:
                        # Extract radius
                        vdw_match = re.search(r'vdw\s*=\s*([\d.]+)', line)
                        radius = float(vdw_match.group(1)) if vdw_match else 1.5
                        
                        # Determine type from name
                        feat_type = _normalize_feature_type(name.split('_')[0])
                        
                        features.append({
                            "type": feat_type,
                            "center": coords[:3],
                            "radius": radius,
                            "count": 1,
                            "frequency": 1.0,
                        })
                except ValueError:
                    continue
        
        # Also try CGO sphere definitions
        elif 'SPHERE' in line.upper():
            # CGO format: SPHERE, x, y, z, radius
            parts = line.replace(',', ' ').split()
            try:
                idx = next(i for i, p in enumerate(parts) if 'SPHERE' in p.upper())
                if idx + 4 < len(parts):
                    x = float(parts[idx + 1])
                    y = float(parts[idx + 2])
                    z = float(parts[idx + 3])
                    radius = float(parts[idx + 4])
                    features.append({
                        "type": "Hydrophobic",  # Default type for unnamed spheres
                        "center": [x, y, z],
                        "radius": radius,
                        "count": 1,
                        "frequency": 1.0,
                    })
            except (ValueError, StopIteration):
                continue
    
    logger.info(f"Loaded {len(features)} features from .pml file")
    return features


def load_mol2_pharmacophore(content):
    """Parse .mol2 (Tripos/SYBYL) file for pharmacophore-relevant atoms.
    
    Extracts pharmacophore features from atom types and coordinates in the
    @<TRIPOS>ATOM section.
    """
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')
    
    features = []
    in_atom_section = False
    
    # Mol2 atom type → pharmacophore feature mapping
    type_map = {
        "N.ar": "HBA", "N.am": "HBD", "N.3": "HBD", "N.2": "HBA",
        "N.pl3": "HBD", "N.4": "PosIonizable", "N.1": "HBA",
        "O.3": "HBA", "O.2": "HBA", "O.co2": "NegIonizable",
        "S.3": "Hydrophobic", "S.2": "Hydrophobic",
        "C.ar": "Aromatic", "C.3": "Hydrophobic",
    }
    
    for line in content.strip().split('\n'):
        line = line.strip()
        
        if line.startswith('@<TRIPOS>ATOM'):
            in_atom_section = True
            continue
        elif line.startswith('@<TRIPOS>'):
            in_atom_section = False
            continue
        
        if in_atom_section and line:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    atom_type = parts[5]
                    
                    feat_type = type_map.get(atom_type)
                    if feat_type:
                        features.append({
                            "type": feat_type,
                            "center": [x, y, z],
                            "radius": 1.5,
                            "count": 1,
                            "frequency": 1.0,
                        })
                except (ValueError, IndexError):
                    continue
    
    # Cluster nearby features of the same type to reduce redundancy
    if features:
        features = _cluster_loaded_features(features)
    
    logger.info(f"Loaded {len(features)} features from .mol2 file")
    return features


def load_xyz_pharmacophore(content):
    """Parse .xyz coordinate file with pharmacophore annotations.
    
    Expected format:
        N_FEATURES
        Comment line (may contain metadata)
        TYPE  X  Y  Z  [RADIUS]
        ...
    
    Also handles simple X Y Z TYPE format.
    """
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')
    
    features = []
    lines = content.strip().split('\n')
    
    # Skip header lines (first 1-2 lines may be count and comment)
    start = 0
    if lines and lines[0].strip().isdigit():
        start = 2  # Skip count + comment
    elif lines and len(lines) > 1:
        try:
            float(lines[0].split()[0])
            start = 0  # No header
        except (ValueError, IndexError):
            start = 1  # Skip comment header
    
    for line in lines[start:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) < 4:
            continue
        
        # Try TYPE X Y Z [RADIUS]
        try:
            raw_type = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            radius = float(parts[4]) if len(parts) > 4 else 1.5
            
            feat_type = _normalize_feature_type(raw_type)
            features.append({
                "type": feat_type,
                "center": [x, y, z],
                "radius": radius,
                "count": 1,
                "frequency": 1.0,
            })
            continue
        except ValueError:
            pass
        
        # Try X Y Z TYPE [RADIUS]
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            raw_type = parts[3]
            radius = float(parts[4]) if len(parts) > 4 else 1.5
            
            feat_type = _normalize_feature_type(raw_type)
            features.append({
                "type": feat_type,
                "center": [x, y, z],
                "radius": radius,
                "count": 1,
                "frequency": 1.0,
            })
        except ValueError:
            continue
    
    logger.info(f"Loaded {len(features)} features from .xyz file")
    return features


def load_csv_pharmacophore(content, delimiter=None):
    """Parse CSV/TSV/TXT tabular pharmacophore definitions.
    
    Expects columns: type, x, y, z, [radius]
    Header row is auto-detected and supported.
    Separators: comma, tab, semicolon, or whitespace (auto-detected).
    """
    import csv as csv_mod
    
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')
    
    features = []
    lines = content.strip().split('\n')
    
    if not lines:
        return features
    
    # Auto-detect delimiter
    if delimiter is None:
        first_data = lines[1] if len(lines) > 1 else lines[0]
        if '\t' in first_data:
            delimiter = '\t'
        elif ',' in first_data:
            delimiter = ','
        elif ';' in first_data:
            delimiter = ';'
        else:
            delimiter = None  # whitespace
    
    # Detect header
    first_line = lines[0].strip()
    has_header = False
    header_lower = first_line.lower()
    if any(kw in header_lower for kw in ['type', 'feature', 'x', 'y', 'z', 'radius', 'name']):
        has_header = True
    
    # Parse column mapping from header
    type_col = 0
    x_col = 1
    y_col = 2
    z_col = 3
    r_col = 4
    
    if has_header:
        if delimiter:
            headers = [h.strip().lower() for h in first_line.split(delimiter)]
        else:
            headers = first_line.lower().split()
        
        for i, h in enumerate(headers):
            if h in ('type', 'feature', 'name', 'feature_type', 'feat_type'):
                type_col = i
            elif h == 'x':
                x_col = i
            elif h == 'y':
                y_col = i
            elif h == 'z':
                z_col = i
            elif h in ('radius', 'r', 'tolerance', 'tol'):
                r_col = i
    
    data_lines = lines[1:] if has_header else lines
    
    for line in data_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if delimiter:
            parts = line.split(delimiter)
        else:
            parts = line.split()
        
        parts = [p.strip() for p in parts]
        if len(parts) < 4:
            continue
        
        try:
            raw_type = parts[type_col]
            x = float(parts[x_col])
            y = float(parts[y_col])
            z = float(parts[z_col])
            radius = float(parts[r_col]) if r_col < len(parts) else 1.5
        except (ValueError, IndexError):
            continue
        
        feat_type = _normalize_feature_type(raw_type)
        features.append({
            "type": feat_type,
            "center": [x, y, z],
            "radius": radius,
            "count": 1,
            "frequency": 1.0,
        })
    
    logger.info(f"Loaded {len(features)} features from CSV/TXT file")
    return features


def load_sdf_pharmacophore(content):
    """Extract pharmacophore features from an SDF file.
    
    Analyzes the molecule(s) and extracts pharmacophoric features
    based on functional groups and 3D coordinates (if available).
    """
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')
    
    features = []
    
    try:
        suppl = Chem.SDMolSupplier()
        suppl.SetData(content)
        
        all_raw_features = []
        for mol in suppl:
            if mol is None:
                continue
            
            # Generate 3D coordinates if not present
            try:
                if mol.GetNumConformers() == 0:
                    mol = Chem.AddHs(mol)
                    params = AllChem.ETKDGv3()
                    params.randomSeed = 42
                    AllChem.EmbedMolecule(mol, params)
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                
                if mol.GetNumConformers() > 0:
                    raw_feats = extract_features_from_conformer(mol, 0)
                    all_raw_features.extend(raw_feats)
            except Exception:
                continue
        
        if all_raw_features:
            features = _cluster_loaded_features(all_raw_features)
    except Exception as e:
        logger.error(f"SDF pharmacophore extraction failed: {e}")
    
    logger.info(f"Loaded {len(features)} features from .sdf file")
    return features


def _cluster_loaded_features(features, distance_threshold=2.0):
    """Cluster loaded features by type and proximity to remove duplicates."""
    from collections import defaultdict
    
    by_type = defaultdict(list)
    for f in features:
        by_type[f["type"]].append(f)
    
    clustered = []
    for feat_type, feats in by_type.items():
        if len(feats) == 1:
            clustered.append(feats[0])
            continue
        
        # Simple greedy clustering
        remaining = list(feats)
        while remaining:
            ref = remaining.pop(0)
            rc = np.array(ref["center"])
            cluster_members = [ref]
            
            still_remaining = []
            for f in remaining:
                fc = np.array(f["center"])
                if np.linalg.norm(rc - fc) <= distance_threshold:
                    cluster_members.append(f)
                else:
                    still_remaining.append(f)
            remaining = still_remaining
            
            # Average coordinates of cluster
            avg_center = np.mean([np.array(m["center"]) for m in cluster_members], axis=0)
            clustered.append({
                "type": feat_type,
                "center": avg_center.tolist(),
                "radius": ref.get("radius", 1.5),
                "count": len(cluster_members),
                "frequency": 1.0,
            })
    
    return clustered


def load_pharmacophore_any_format(content, filename):
    """Universal pharmacophore loader — dispatches to the correct parser based on file extension.
    
    Args:
        content: raw file content (bytes or str)
        filename: original filename with extension
    
    Returns:
        list of pharmacophore feature dicts, or empty list on failure
    """
    ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    
    try:
        if ext == 'json':
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            json_data = json.loads(content)
            return load_pharmit_json(json_data)
        
        elif ext == 'ph4':
            return load_ph4_file(content)
        
        elif ext == 'lig':
            return load_lig_file(content)
        
        elif ext == 'pml':
            return load_pml_pharmacophore(content)
        
        elif ext == 'mol2':
            return load_mol2_pharmacophore(content)
        
        elif ext == 'xyz':
            return load_xyz_pharmacophore(content)
        
        elif ext in ('csv', 'tsv'):
            return load_csv_pharmacophore(content)
        
        elif ext == 'txt':
            # Try CSV first, then fall back to .lig-style
            features = load_csv_pharmacophore(content)
            if not features:
                features = load_lig_file(content)
            return features
        
        elif ext == 'sdf':
            return load_sdf_pharmacophore(content)
        
        else:
            # Try JSON first, then CSV, then .lig as fallbacks
            logger.warning(f"Unknown pharmacophore format '{ext}', trying auto-detect...")
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            
            # Try JSON
            try:
                json_data = json.loads(content)
                features = load_pharmit_json(json_data)
                if features:
                    return features
            except (json.JSONDecodeError, Exception):
                pass
            
            # Try CSV
            features = load_csv_pharmacophore(content)
            if features:
                return features
            
            # Try .lig / .ph4 format
            features = load_lig_file(content)
            if features:
                return features
            
            logger.error(f"Could not parse pharmacophore from format: {ext}")
            return []
    
    except Exception as e:
        logger.error(f"Failed to load pharmacophore from {filename}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
#  FULL PHARMACOPHORE GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_pharmacophore(pdb_block: str, active_mols: list, num_conformers=20,
                           cluster_threshold=1.5, frequency_threshold=0.5):
    """Complete pharmacophore generation pipeline.
    
    Returns:
        features: list of pharmacophore feature dicts
        pharmit_json: Pharmit-compatible JSON dict
        binding_residues: list of binding site residue dicts
        binding_center: numpy array of binding site center
    """
    logger.info(f"Generating pharmacophore from {len(active_mols)} actives")
    
    # Step 1: Identify binding site
    binding_residues, binding_center = identify_binding_site(pdb_block, active_mols)
    logger.info(f"Found {len(binding_residues)} binding site residues")
    
    # Step 2: Extract features from all actives × conformers
    all_features = extract_all_features(active_mols, binding_residues, num_conformers)
    logger.info(f"Extracted {len(all_features)} raw features")
    
    # Step 3: Cluster features
    clustered = cluster_features(all_features, cluster_threshold)
    logger.info(f"Clustered into {len(clustered)} feature groups")
    
    # Step 4: Filter by frequency
    filtered = filter_by_frequency(clustered, len(active_mols), frequency_threshold)
    logger.info(f"After frequency filter: {len(filtered)} features")
    
    # Step 5: Export Pharmit JSON
    pharmit_json = export_pharmit_json(filtered)
    
    return filtered, pharmit_json, binding_residues, binding_center


# ═══════════════════════════════════════════════════════════════
#  DECOY GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_decoys(active_mols, n_decoys=500):
    """Generate property-matched decoys (same MW ±25, logP ±1, different scaffold).
    Uses scaffold hopping via random SMILES mutations."""
    if not active_mols:
        return []
    
    # Get property ranges from actives
    props = []
    for mol in active_mols:
        if mol is None:
            continue
        props.append({
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
        })
    
    if not props:
        return []
    
    avg_mw = np.mean([p["mw"] for p in props])
    avg_logp = np.mean([p["logp"] for p in props])
    
    # Generate decoys from a diverse set of scaffolds
    decoy_scaffolds = [
        # Drug-like scaffolds different from typical kinase/GPCR targets
        "C1CCCCC1", "C1CCNCC1", "C1CCOCC1", "c1ccccc1",
        "C1CCCC1", "C1CCC(=O)NC1", "C1CC2CCCCC2C1",
        "c1ccncc1", "c1ccc2[nH]ccc2c1", "c1ccc2ncccc2c1",
        "C1CC2(CCCC2)CC1", "C1CNC(=O)NC1=O", "C1CCSC1",
        "c1csc(-c2ccccn2)n1", "c1cc(-c2ccco2)ccn1",
        "C1CC(=O)N(C1)c1ccccc1", "c1ccc(-c2cn[nH]c2)cc1",
        "C1CCN(C1)c1ccccn1", "c1ccc2c(c1)CCO2",
        "C1CC2(CCCC2)OC1", "c1ccc(-n2ccnc2)cc1",
        "c1ccc2c(c1)OCO2", "C1CCC(CC1)NC(=O)c1ccccc1",
        "c1ccc(-c2ccccn2)cc1", "c1ccc2[nH]ncc2c1",
    ]
    
    decoys = []
    # Common substituents for decoration
    substituents = [
        "C", "CC", "CCC", "C(C)C", "F", "Cl", "OC", "N", "NC",
        "C(=O)N", "C(=O)O", "C(F)(F)F", "OCC", "NCC", "c1ccccc1",
        "C1CCNCC1", "C1CCOCC1", "C#N", "S(=O)(=O)N", "C(=O)NC",
    ]
    
    import random
    random.seed(42)
    
    attempts = 0
    max_attempts = n_decoys * 20
    
    seen_inchikeys = set()
    # Add active InChIKeys to avoid
    for mol in active_mols:
        if mol:
            try:
                ik = Chem.inchi.MolToInchi(mol)
                seen_inchikeys.add(ik)
            except Exception:
                pass
    
    while len(decoys) < n_decoys and attempts < max_attempts:
        attempts += 1
        
        scaffold = random.choice(decoy_scaffolds)
        mol = Chem.MolFromSmiles(scaffold)
        if mol is None:
            continue
        
        # Add 1-3 random substituents
        smi = scaffold
        n_subs = random.randint(1, 3)
        for _ in range(n_subs):
            sub = random.choice(substituents)
            # Simple concatenation with random position
            connector = random.choice(["", "C", "CC", "N", "O"])
            new_smi = f"{smi}({connector}{sub})"
            test_mol = Chem.MolFromSmiles(new_smi)
            if test_mol is not None:
                smi = new_smi
                mol = test_mol
        
        if mol is None:
            continue
        
        # Check property matching
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        if abs(mw - avg_mw) > 50 or abs(logp - avg_logp) > 2:
            continue
        
        # Check uniqueness
        try:
            ik = Chem.MolToSmiles(mol)
            if ik in seen_inchikeys:
                continue
            seen_inchikeys.add(ik)
        except Exception:
            pass
        
        decoys.append(mol)
    
    logger.info(f"Generated {len(decoys)} decoys from {attempts} attempts")
    return decoys


# ═══════════════════════════════════════════════════════════════
#  FINGERPRINT VECTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def compute_pharmacophore_fingerprint(mol, features, n_bins=15, bin_size=1.0):
    """Compute 3D pharmacophore fingerprint.
    Distance bins between all feature pairs (0-15Å, bin size 1Å)."""
    if mol is None or not features:
        return np.zeros(n_bins * 6)  # 6 feature types
    
    # Get molecule features
    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        res = AllChem.EmbedMolecule(mol3d, params)
        if res == -1:
            return np.zeros(n_bins * 6)
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=100)
    except Exception:
        return np.zeros(n_bins * 6)
    
    mol_features = extract_features_from_conformer(mol3d, 0)
    
    if not mol_features:
        return np.zeros(n_bins * 6)
    
    # Build distance histogram between mol features and pharmacophore features
    fp = np.zeros(n_bins * 6)
    type_idx = {"HBA": 0, "HBD": 1, "Aromatic": 2, "Hydrophobic": 3,
                "PosIonizable": 4, "NegIonizable": 5}
    
    for mf in mol_features:
        ti = type_idx.get(mf["type"], 0)
        mc = np.array(mf["center"])
        for pf in features:
            pc = np.array(pf["center"])
            dist = np.linalg.norm(mc - pc)
            bin_idx = min(int(dist / bin_size), n_bins - 1)
            fp[ti * n_bins + bin_idx] += 1
    
    # Normalize
    max_val = fp.max()
    if max_val > 0:
        fp = fp / max_val
    
    return fp


def compute_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """Compute Morgan circular fingerprint."""
    if mol is None:
        return np.zeros(n_bits)
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(n_bits)


def compute_descriptor_vector(mol):
    """Compute RDKit descriptor vector: MW, logP, TPSA, HBD, HBA, RotBonds, AromaticRings."""
    if mol is None:
        return np.zeros(7)
    try:
        return np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
        ])
    except Exception:
        return np.zeros(7)


def compute_pharmacophore_match_score(mol, features):
    """Fraction of pharmacophore features matched by molecule."""
    if mol is None or not features:
        return 0.0
    
    matched = 0
    for feat in features:
        feat_type = feat["type"]
        smarts_list = FEATURE_SMARTS.get(feat_type, [])
        for smarts in smarts_list:
            if mol.HasSubstructMatch(smarts):
                matched += 1
                break
    
    return matched / max(len(features), 1)


# ═══════════════════════════════════════════════════════════════
#  PROTEIN ACTIVE SITE DESCRIPTOR
# ═══════════════════════════════════════════════════════════════

def compute_binding_site_descriptor(binding_residues):
    """Compute 20D binding site descriptor from amino acid composition.

    Encodes the protein active site as a normalized frequency vector
    of the 20 standard amino acids, giving the classifier protein context
    to learn residue-specific interaction patterns.
    """
    if not binding_residues:
        return np.zeros(20)

    counts = np.zeros(20)
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS_20)}

    for res in binding_residues:
        resname = res.get("name", "")
        idx = aa_to_idx.get(resname)
        if idx is not None:
            counts[idx] += 1

    total = counts.sum()
    if total > 0:
        counts = counts / total

    return counts


# ═══════════════════════════════════════════════════════════════
#  PROTEIN-LIGAND INTERACTION COMPATIBILITY FINGERPRINT
# ═══════════════════════════════════════════════════════════════

def compute_interaction_fingerprint(mol, binding_residues):
    """Compute protein-ligand interaction compatibility fingerprint (30D).

    For each of 6 pharmacophore interaction types, encode 5 features
    describing the compatibility between ligand features and binding site:
      1. n_complementary_residues (normalized)
      2. n_ligand_features (normalized, capped at 10)
      3. interaction_potential = min(res, lig) / max(res, lig, 1)
      4. has_both = 1 if both sides present, else 0
      5. balance = 1 - |res - lig| / max(res + lig, 1)

    Total: 6 types x 5 = 30 dimensions.
    """
    n_types = 6
    n_measures = 5
    fp = np.zeros(n_types * n_measures)

    if mol is None or not binding_residues:
        return fp

    # Complementary residue sets for each ligand feature type
    interaction_pairs = [
        ("HBD", HBA_RESIDUES),
        ("HBA", HBD_RESIDUES),
        ("Aromatic", AROMATIC_RESIDUES),
        ("Hydrophobic", HYDROPHOBIC_RESIDUES),
        ("PosIonizable", POS_CHARGE_RESIDUES),
        ("NegIonizable", NEG_CHARGE_RESIDUES),
    ]

    residue_counts = []
    for _, res_set in interaction_pairs:
        count = sum(1 for r in binding_residues if r.get("name") in res_set)
        residue_counts.append(count)

    max_res = max(residue_counts) if residue_counts else 1

    ligand_counts = []
    for lig_type, _ in interaction_pairs:
        count = 0
        smarts_list = FEATURE_SMARTS.get(lig_type, [])
        for smarts in smarts_list:
            try:
                matches = mol.GetSubstructMatches(smarts)
                count += len(matches)
            except Exception:
                pass
        ligand_counts.append(count)

    for i in range(n_types):
        n_res = residue_counts[i]
        n_lig = ligand_counts[i]
        base = i * n_measures
        fp[base + 0] = n_res / max(max_res, 1)
        fp[base + 1] = min(n_lig, 10) / 10.0
        fp[base + 2] = min(n_res, n_lig) / max(max(n_res, n_lig), 1)
        fp[base + 3] = 1.0 if (n_res > 0 and n_lig > 0) else 0.0
        fp[base + 4] = 1.0 - abs(n_res - n_lig) / max(n_res + n_lig, 1)

    return fp


# ═══════════════════════════════════════════════════════════════
#  3D SHAPE DESCRIPTORS (USR - Ultrafast Shape Recognition)
# ═══════════════════════════════════════════════════════════════

def compute_3d_shape_descriptors(mol):
    """Compute USR (Ultrafast Shape Recognition) shape descriptors (12D).

    Encodes 3D molecular shape using 4 reference points:
      1. ctd - centroid of all atoms
      2. cst - atom closest to centroid
      3. fct - atom farthest from centroid
      4. ftf - atom farthest from fct

    For each reference, 3 distribution moments: mean, variance, skewness.
    Total: 4 x 3 = 12 dimensions.
    """
    if mol is None:
        return np.zeros(12)

    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        res = AllChem.EmbedMolecule(mol3d, params)
        if res == -1:
            return np.zeros(12)
        try:
            AllChem.MMFFOptimizeMolecule(mol3d, maxIters=100)
        except Exception:
            pass

        conf = mol3d.GetConformer()
        coords = []
        for i in range(mol3d.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        coords = np.array(coords)

        if len(coords) < 3:
            return np.zeros(12)

        # 4 reference points
        ctd = coords.mean(axis=0)
        dists_ctd = np.linalg.norm(coords - ctd, axis=1)
        cst = coords[np.argmin(dists_ctd)]
        fct = coords[np.argmax(dists_ctd)]
        dists_fct = np.linalg.norm(coords - fct, axis=1)
        ftf = coords[np.argmax(dists_fct)]

        descriptors = []
        for ref in [ctd, cst, fct, ftf]:
            dists = np.linalg.norm(coords - ref, axis=1)
            mean_d = np.mean(dists)
            var_d = np.var(dists)
            std_d = np.std(dists)
            skew_d = np.mean(((dists - mean_d) / std_d) ** 3) if std_d > 1e-8 else 0.0
            descriptors.extend([mean_d, var_d, skew_d])

        return np.array(descriptors)
    except Exception:
        return np.zeros(12)


# ═══════════════════════════════════════════════════════════════
#  3D PHARMACOPHORE FINGERPRINT (pmapper)
# ═══════════════════════════════════════════════════════════════

def compute_pmapper_fingerprint(mol, n_bits=1024):
    """Compute 3D pharmacophore fingerprint using pmapper (1024-bit).

    Generates hashed 3D pharmacophore triplet fingerprints that capture
    the spatial arrangement of pharmacophoric features as triangles.
    Falls back to zeros if pmapper is not installed.

    pmapper's get_fp() returns a set of activated bit indices.
    We map these into a fixed-length bit vector.

    Returns: numpy array of shape (n_bits,)
    """
    if mol is None:
        return np.zeros(n_bits)

    try:
        from pmapper.pharmacophore import Pharmacophore as PMPharmacophore
    except ImportError:
        return np.zeros(n_bits)

    try:
        # Generate 3D conformer if needed
        mol3d = Chem.AddHs(mol)
        if mol3d.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            res = AllChem.EmbedMolecule(mol3d, params)
            if res == -1:
                return np.zeros(n_bits)
            try:
                AllChem.MMFFOptimizeMolecule(mol3d, maxIters=100)
            except Exception:
                pass

        # Create pharmacophore from mol
        p = PMPharmacophore()
        p.load_from_mol(mol3d)

        fp = np.zeros(n_bits)

        # get_fp returns a set of activated bit indices
        # Use feature triplets (min=2, max=3) for triangles
        activated_bits = p.get_fp(min_features=2, max_features=3)

        if activated_bits is not None:
            if isinstance(activated_bits, set):
                # Map bit indices into our fixed-length vector using modulo
                for bit_idx in activated_bits:
                    fp[bit_idx % n_bits] = 1
            elif isinstance(activated_bits, dict):
                for bit_idx in activated_bits:
                    if isinstance(bit_idx, int):
                        fp[bit_idx % n_bits] = 1
            elif hasattr(activated_bits, '__iter__'):
                for bit_idx in activated_bits:
                    if isinstance(bit_idx, (int, np.integer)):
                        fp[int(bit_idx) % n_bits] = 1
            return fp

        # Fallback: use signature hashing
        try:
            signatures = p.get_signature_md5(min_features=2, max_features=3)
            if signatures:
                for sig in signatures:
                    sig_str = str(sig)
                    h = int(hashlib.md5(sig_str.encode()).hexdigest(), 16)
                    fp[h % n_bits] = 1
                return fp
        except Exception:
            pass

        return fp
    except Exception as e:
        logger.debug("pmapper FP failed: %s" % str(e))
        return np.zeros(n_bits)


# ═══════════════════════════════════════════════════════════════
#  COMBINED FEATURE VECTOR
# ═══════════════════════════════════════════════════════════════

def build_feature_vector(mol, pharmacophore_features, binding_residues=None):
    """Build combined fingerprint vector for ML classifier.

    Concatenates 8 feature blocks:
      1. 3D Pharmacophore FP (90D)  - distance bins mol<->pharmacophore
      2. Morgan FP (2048D)          - ECFP4 circular fingerprint
      3. RDKit Descriptors (7D)     - MW, logP, TPSA, HBD, HBA, RotB, ArRings
      4. Pharmacophore Match (1D)   - fraction of features matched
      5. Binding Site Desc (20D)    - amino acid composition of active site
      6. Interaction Compat (30D)   - ligand<->protein interaction potential
      7. USR Shape (12D)            - 3D molecular shape moments
      8. pmapper 3D Pharm FP (1024D)- hashed 3D pharmacophore triplets

    Total: 3232 dimensions.
    """
    pharm_fp = compute_pharmacophore_fingerprint(mol, pharmacophore_features)
    morgan_fp = compute_morgan_fingerprint(mol)
    desc = compute_descriptor_vector(mol)
    match_score = np.array([compute_pharmacophore_match_score(mol, pharmacophore_features)])

    # Protein-ligand interaction features
    site_desc = compute_binding_site_descriptor(binding_residues)
    ifp = compute_interaction_fingerprint(mol, binding_residues)
    usr = compute_3d_shape_descriptors(mol)

    # pmapper 3D pharmacophore fingerprint
    pmapper_fp = compute_pmapper_fingerprint(mol)

    return np.concatenate([pharm_fp, morgan_fp, desc, match_score, site_desc, ifp, usr, pmapper_fp])


# ═══════════════════════════════════════════════════════════════
#  ML CLASSIFIER
# ═══════════════════════════════════════════════════════════════

def train_classifier(active_mols, decoy_mols, pharmacophore_features, binding_residues=None, n_splits=5):
    """Train RandomForest binary classifier.
    
    Returns:
        model: trained sklearn Pipeline (scaler + RF)
        metrics: dict of performance metrics
        threshold: classification threshold (0.80)
    """
    logger.info(f"Training classifier: {len(active_mols)} actives, {len(decoy_mols)} decoys")
    
    # Build feature matrices
    X_pos = []
    for mol in active_mols:
        if mol is not None:
            X_pos.append(build_feature_vector(mol, pharmacophore_features, binding_residues))
    
    X_neg = []
    for mol in decoy_mols:
        if mol is not None:
            X_neg.append(build_feature_vector(mol, pharmacophore_features, binding_residues))
    
    if not X_pos or not X_neg:
        logger.error("Not enough data to train classifier")
        return None, {}, 0.80
    
    X = np.array(X_pos + X_neg)
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    
    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Build pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=min(n_splits, min(len(X_pos), len(X_neg))),
                         shuffle=True, random_state=42)
    
    try:
        cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
        cv_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        cv_acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    except Exception as e:
        logger.warning(f"CV failed, training on full data: {e}")
        cv_f1 = [0.0]
        cv_auc = [0.0]
        cv_acc = [0.0]
    
    # Fit on full data
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "roc_auc": round(roc_auc_score(y, y_proba), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "cv_f1_mean": round(np.mean(cv_f1), 4),
        "cv_f1_std": round(np.std(cv_f1), 4),
        "cv_auc_mean": round(np.mean(cv_auc), 4),
        "cv_acc_mean": round(np.mean(cv_acc), 4),
        "n_actives": len(X_pos),
        "n_decoys": len(X_neg),
        "n_features": X.shape[1],
    }
    
    logger.info(f"Classifier trained: F1={metrics['f1']}, AUC={metrics['roc_auc']}")
    
    return model, metrics, 0.80


def score_molecules(mols, model, pharmacophore_features, binding_residues=None):
    """Score a list of molecules with the trained classifier.
    Returns list of probability scores."""
    if model is None or not mols:
        return [0.0] * len(mols)
    
    scores = []
    for mol in mols:
        if mol is None:
            scores.append(0.0)
            continue
        try:
            fv = build_feature_vector(mol, pharmacophore_features, binding_residues)
            fv = np.nan_to_num(fv.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
            prob = model.predict_proba(fv)[0][1]
            scores.append(round(float(prob), 4))
        except Exception:
            scores.append(0.0)
    
    return scores


# ═══════════════════════════════════════════════════════════════
#  3D VISUALIZATION (py3Dmol)
# ═══════════════════════════════════════════════════════════════

def generate_viewer_html(pdb_block, features, title="Pharmacophore Model"):
    """Generate interactive py3Dmol HTML showing protein + pharmacophore features."""
    
    # Escape PDB for JavaScript
    pdb_escaped = pdb_block.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    
    spheres_js = ""
    for feat in features:
        color = FEATURE_COLORS.get(feat["type"], "#FFFFFF")
        x, y, z = feat["center"]
        r = feat.get("radius", 1.5)
        label = feat["type"]
        spheres_js += f"""
        viewer.addSphere({{
            center: {{x: {x:.3f}, y: {y:.3f}, z: {z:.3f}}},
            radius: {r},
            color: '{color}',
            opacity: 0.6,
        }});
        viewer.addLabel("{label}", {{
            position: {{x: {x:.3f}, y: {y + r + 0.5:.3f}, z: {z:.3f}}},
            fontSize: 10,
            fontColor: '{color}',
            backgroundColor: 'rgba(0,0,0,0.5)',
            backgroundOpacity: 0.5,
        }});
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{ margin: 0; background: #1a1a2e; }}
        #viewer {{ width: 100%; height: 100vh; position: relative; }}
        .legend {{
            position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.7);
            padding: 10px; border-radius: 8px; color: white; font-family: Arial;
            font-size: 12px; z-index: 100;
        }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
    </style>
    </head>
    <body>
    <div id="viewer"></div>
    <div class="legend">
        <strong>{title}</strong><br>
        <div class="legend-item"><div class="legend-dot" style="background:#FF4444"></div>H-Bond Acceptor</div>
        <div class="legend-item"><div class="legend-dot" style="background:#4444FF"></div>H-Bond Donor</div>
        <div class="legend-item"><div class="legend-dot" style="background:#44FF44"></div>Aromatic</div>
        <div class="legend-item"><div class="legend-dot" style="background:#FFFF44"></div>Hydrophobic</div>
        <div class="legend-item"><div class="legend-dot" style="background:#FF44FF"></div>Pos. Ionizable</div>
        <div class="legend-item"><div class="legend-dot" style="background:#44FFFF"></div>Neg. Ionizable</div>
    </div>
    <script>
        var viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "0x1a1a2e"}});
        var pdb = `{pdb_escaped}`;
        viewer.addModel(pdb, "pdb");
        viewer.setStyle({{}}, {{cartoon: {{color: "spectrum", opacity: 0.8}}}});
        viewer.addSurface($3Dmol.SurfaceType.VDW, {{
            opacity: 0.15, color: "white"
        }});
        {spheres_js}
        viewer.zoomTo();
        viewer.render();
    </script>
    </body>
    </html>
    """
    return html


def generate_viewer_component(pdb_block, features, selected_mol_sdf=None, height=500):
    """Generate py3Dmol viewer code for Streamlit embedding via stmol or html component."""
    
    pdb_escaped = pdb_block.replace("'", "\\'").replace("\n", "\\n")
    
    spheres_js = ""
    for feat in features:
        color = FEATURE_COLORS.get(feat["type"], "#FFFFFF")
        x, y, z = feat["center"]
        r = feat.get("radius", 1.5)
        spheres_js += f"""
        viewer.addSphere({{center:{{x:{x:.2f},y:{y:.2f},z:{z:.2f}}},radius:{r},color:'{color}',opacity:0.5}});"""
    
    mol_js = ""
    if selected_mol_sdf:
        sdf_escaped = selected_mol_sdf.replace("'", "\\'").replace("\n", "\\n")
        mol_js = f"""
        viewer.addModel('{sdf_escaped}', 'sdf');
        viewer.setStyle({{'model': -1}}, {{'stick': {{'colorscheme': 'greenCarbon'}}}});
        """
    
    view_code = f"""
    var viewer = $3Dmol.createViewer($('#viewer-{hash(str(features))[:8] if features else "default"}'), 
        {{backgroundColor: '0x1a1a2e'}});
    viewer.addModel('{pdb_escaped}', 'pdb');
    viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum', opacity: 0.7}}}});
    {spheres_js}
    {mol_js}
    viewer.zoomTo();
    viewer.render();
    """
    return view_code
