"""
Molecular Docking Module — AutoDock Vina (Python API Primary)
=============================================================
Handles:
  - Receptor/ligand preparation (PDB → PDBQT)
  - Binding site auto-detection from PDB coordinates
  - AutoDock Vina docking via Python API (primary) or CLI (secondary)
  - Multi-pose output parsing (5 poses per molecule)
  - Fallback scoring using RDKit shape/pharmacophore ONLY when Vina unavailable
  - Batch docking with chunked processing for memory efficiency

Priority Order:
  1. Python Vina API  (from vina import Vina)
  2. Vina CLI binary   (subprocess)
  3. Fallback scoring   (RDKit heuristic — last resort)
"""

import os
import re
import shutil
import subprocess
import tempfile
import logging
from typing import List, Dict, Optional, Tuple, Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from rdkit.Chem import rdMolAlign

logger = logging.getLogger(__name__)

# ─── CHECK VINA AVAILABILITY AT MODULE LOAD ────────────────────────────
_VINA_PYTHON_AVAILABLE = False
try:
    from vina import Vina as _VinaClass
    _VINA_PYTHON_AVAILABLE = True
    logger.info("✅ Python Vina API available — real docking enabled")
except ImportError:
    logger.warning(
        "⚠️ Python Vina not installed. Install with: "
        "pip install vina (requires libboost-all-dev, swig). "
        "Falling back to heuristic scoring."
    )

_MEEKO_AVAILABLE = False
try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    _MEEKO_AVAILABLE = True
    logger.info("✅ Meeko available for PDBQT ligand preparation")
except ImportError:
    logger.warning("⚠️ Meeko not installed — ligand prep will use simple converter")


def find_vina_binary():
    """Locate the AutoDock Vina binary on the system."""
    # Check Python API first (preferred)
    if _VINA_PYTHON_AVAILABLE:
        return "python_vina"
    # Then check CLI binaries
    candidates = [
        "vina",
        "vina_1.2.5",
        "/usr/bin/vina",
        "/usr/local/bin/vina",
        shutil.which("vina"),
    ]
    for path in candidates:
        if path and shutil.which(path):
            return shutil.which(path)
    return None


def check_vina_available():
    """Check if AutoDock Vina is available (Python API or CLI)."""
    return _VINA_PYTHON_AVAILABLE or find_vina_binary() is not None


# ─── BINDING SITE DETECTION ─────────────────────────────────────────────
def detect_binding_site(pdb_block: str, active_mols=None, distance_cutoff=6.0):
    """Auto-detect binding site center and dimensions from PDB.
    
    Strategy:
    1) If HETATM ligand present → use its centroid
    2) If active molecules provided → find residues within distance_cutoff
    3) Fallback → geometric center of all atoms
    
    Returns: (center_x, center_y, center_z), (size_x, size_y, size_z)
    """
    lines = pdb_block.strip().split('\n')
    
    # Try to find HETATM ligand (not water)
    hetatm_coords = []
    all_coords = []
    for line in lines:
        if line.startswith("HETATM"):
            resname = line[17:20].strip()
            if resname not in ("HOH", "WAT", "DOD", "SOL"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    hetatm_coords.append([x, y, z])
                except (ValueError, IndexError):
                    pass
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                all_coords.append([x, y, z])
            except (ValueError, IndexError):
                pass

    if hetatm_coords:
        coords = np.array(hetatm_coords)
        center = coords.mean(axis=0)
        spread = coords.max(axis=0) - coords.min(axis=0)
        box_size = np.maximum(spread + 10, 20)  # At least 20 Å per dimension
    elif all_coords:
        coords = np.array(all_coords)
        center = coords.mean(axis=0)
        box_size = np.array([20.0, 20.0, 20.0])
    else:
        center = np.array([0.0, 0.0, 0.0])
        box_size = np.array([20.0, 20.0, 20.0])

    # Cap box size to reasonable limits
    box_size = np.minimum(box_size, 30.0)
    
    return tuple(center.round(3)), tuple(box_size.round(1))


# ─── LIGAND PDBQT PREPARATION (IN-MEMORY) ───────────────────────────────
def _prepare_ligand_3d(mol):
    """Add hydrogens and generate 3D coordinates for a molecule.
    Returns the 3D mol or None on failure."""
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            # Fallback to simpler embedding
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if result == -1:
                return None
        # Minimize with MMFF
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
        return mol
    except Exception as e:
        logger.debug(f"3D embedding failed: {e}")
        return None


def prepare_ligand_pdbqt_string(mol) -> Optional[str]:
    """Prepare ligand PDBQT as an in-memory string using meeko.
    
    This avoids writing temporary files for ligand preparation,
    which is critical for batch docking performance.
    
    Returns: PDBQT string or None on failure.
    """
    mol3d = _prepare_ligand_3d(mol)
    if mol3d is None:
        return None

    # Method 1: Meeko (preferred — proper torsion tree, charges)
    if _MEEKO_AVAILABLE:
        try:
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol3d)
            setup = mol_setups[0]
            pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setup)
            if pdbqt_string and len(pdbqt_string.strip()) > 0:
                return pdbqt_string
        except Exception as e:
            logger.debug(f"Meeko ligand prep failed: {e}")

    # Method 2: Simple PDB → PDBQT conversion
    try:
        pdb_block = Chem.MolToPDBBlock(mol3d)
        if not pdb_block:
            return None
        pdbqt_lines = ["ROOT"]
        for line in pdb_block.split('\n'):
            if line.startswith("HETATM") or line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                element = atom_name[0]
                charge_map = {
                    "C": 0.1, "N": -0.3, "O": -0.4, "S": -0.2,
                    "H": 0.1, "F": -0.2, "P": 0.2,
                }
                charge = charge_map.get(element, 0.0)
                atom_type_map = {
                    "C": "C", "N": "NA", "O": "OA", "S": "SA",
                    "H": "HD", "F": "F", "P": "P",
                }
                atom_type = atom_type_map.get(element, "C")
                pdbqt_line = f"{line[:54].rstrip():<54s}  1.00  0.00    {charge:+6.3f} {atom_type:>2s}"
                pdbqt_lines.append(pdbqt_line)
        pdbqt_lines.append("ENDROOT\nTORSDOF 0\n")
        return '\n'.join(pdbqt_lines)
    except Exception as e:
        logger.debug(f"Simple ligand PDBQT conversion failed: {e}")
        return None


def prepare_ligand(mol, output_dir: str, mol_id: str = "ligand") -> Optional[str]:
    """Prepare ligand PDBQT file on disk.
    Used by CLI docking path. For Python API, use prepare_ligand_pdbqt_string().
    """
    pdbqt_string = prepare_ligand_pdbqt_string(mol)
    if pdbqt_string is None:
        return None
    pdbqt_path = os.path.join(output_dir, f"{mol_id}.pdbqt")
    with open(pdbqt_path, 'w') as f:
        f.write(pdbqt_string)
    return pdbqt_path


# ─── RECEPTOR PREPARATION ───────────────────────────────────────────────
def prepare_receptor(pdb_block: str, output_dir: str) -> str:
    """Prepare receptor PDBQT from PDB block.
    Tries multiple methods: obabel, prepare_receptor, or simplified conversion."""
    pdb_path = os.path.join(output_dir, "receptor.pdb")
    pdbqt_path = os.path.join(output_dir, "receptor.pdbqt")
    
    # Write clean PDB (ATOM lines only, no HETATM ligands)
    clean_lines = []
    for line in pdb_block.strip().split('\n'):
        if line.startswith("ATOM"):
            clean_lines.append(line)
        elif line.startswith("TER") or line.startswith("END"):
            clean_lines.append(line)
    
    with open(pdb_path, 'w') as f:
        f.write('\n'.join(clean_lines) + '\n')
    
    # Method 1: Try obabel
    obabel = shutil.which("obabel")
    if obabel:
        try:
            result = subprocess.run(
                [obabel, pdb_path, "-O", pdbqt_path, "-xr", "-xn", "-xp"],
                capture_output=True, text=True, timeout=60
            )
            if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
                return pdbqt_path
        except Exception as e:
            logger.warning(f"obabel receptor prep failed: {e}")
    
    # Method 2: Try prepare_receptor from ADFR suite
    prep_receptor = shutil.which("prepare_receptor")
    if prep_receptor:
        try:
            result = subprocess.run(
                [prep_receptor, "-r", pdb_path, "-o", pdbqt_path],
                capture_output=True, text=True, timeout=120
            )
            if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
                return pdbqt_path
        except Exception as e:
            logger.warning(f"prepare_receptor failed: {e}")
    
    # Method 3: Simple PDB → PDBQT conversion (add charges heuristically)
    pdbqt_lines = []
    for line in clean_lines:
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
            # Assign Gasteiger-like partial charges based on element
            charge_map = {"C": 0.0, "N": -0.35, "O": -0.40, "S": -0.20, "H": 0.15}
            charge = charge_map.get(element.upper(), 0.0)
            atom_type = element.upper()
            if atom_type == "C":
                atom_type = "C"
            elif atom_type == "N":
                atom_type = "NA" if "OD" in atom_name or "OE" in atom_name else "N"
            elif atom_type == "O":
                atom_type = "OA"
            elif atom_type == "S":
                atom_type = "SA"
            pdbqt_line = f"{line[:54]}  1.00  0.00    {charge:+6.3f} {atom_type:>2s}"
            pdbqt_lines.append(pdbqt_line)
        elif line.startswith("TER") or line.startswith("END"):
            pdbqt_lines.append(line)
    
    with open(pdbqt_path, 'w') as f:
        f.write('\n'.join(pdbqt_lines) + '\n')
    
    return pdbqt_path


# ─── PYTHON VINA DOCKING (PRIMARY PATH) ─────────────────────────────────
def _dock_molecule_python_vina(
    mol,
    receptor_pdbqt_path: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
    num_modes: int = 5,
    energy_range: float = 3.0,
) -> Optional[Dict]:
    """Dock using the Python Vina API (from vina import Vina).
    
    This is the PRIMARY docking method. Uses in-memory PDBQT strings
    for ligands (via meeko) to avoid excessive disk I/O.
    """
    if not _VINA_PYTHON_AVAILABLE:
        return None

    try:
        from vina import Vina
        v = Vina(sf_name='vina')
        v.set_receptor(receptor_pdbqt_path)
        v.compute_vina_maps(center=list(center), box_size=list(box_size))

        # Prepare ligand PDBQT string (in-memory, no temp files)
        pdbqt_string = prepare_ligand_pdbqt_string(mol)
        if not pdbqt_string:
            logger.warning("Failed to prepare ligand PDBQT string")
            return None

        # Try in-memory ligand loading first
        try:
            v.set_ligand_from_string(pdbqt_string)
        except Exception:
            # Fallback: write to temp file
            with tempfile.NamedTemporaryFile(
                suffix='.pdbqt', mode='w', delete=False
            ) as f:
                f.write(pdbqt_string)
                tmp_lig = f.name
            try:
                v.set_ligand_from_file(tmp_lig)
            finally:
                try:
                    os.unlink(tmp_lig)
                except OSError:
                    pass

        # Run docking
        v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)

        # Extract energies
        energies = v.energies()
        scores = []
        for i, energy in enumerate(energies):
            scores.append({
                "pose": i + 1,
                "score": round(float(energy[0]), 2),
                "rmsd_lb": round(float(energy[1]), 2) if len(energy) > 1 else 0.0,
                "rmsd_ub": round(float(energy[2]), 2) if len(energy) > 2 else 0.0,
            })

        # Write poses to temp file for downstream use
        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, "docked.pdbqt")
        try:
            v.write_poses(output_path, n_poses=num_modes)
        except Exception:
            pass

        return {
            "scores": scores,
            "output_file": output_path,
            "best_score": scores[0]["score"] if scores else None,
            "num_poses": len(scores),
            "method": "vina_python",
        }
    except Exception as e:
        logger.error(f"Python Vina docking failed: {e}")
        return None


# ─── VINA CLI DOCKING (SECONDARY PATH) ──────────────────────────────────
def dock_molecule_vina_cli(
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
    num_modes: int = 5,
    energy_range: float = 3.0,
    output_dir: str = None,
) -> Optional[Dict]:
    """Dock a single molecule using Vina CLI (secondary path)."""
    vina_bin = find_vina_binary()
    if not vina_bin or vina_bin == "python_vina":
        return None
    
    if output_dir is None:
        output_dir = os.path.dirname(ligand_pdbqt)
    
    output_pdbqt = os.path.join(output_dir, "docked_out.pdbqt")
    
    cmd = [
        vina_bin,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--energy_range", str(energy_range),
        "--out", output_pdbqt,
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            logger.warning(f"Vina error: {result.stderr[:500]}")
            return None
        
        # Parse output
        scores = parse_vina_output(output_pdbqt)
        return {
            "scores": scores,
            "output_file": output_pdbqt,
            "best_score": scores[0]["score"] if scores else None,
            "num_poses": len(scores),
            "method": "vina_cli",
        }
    except subprocess.TimeoutExpired:
        logger.warning("Vina timed out")
        return None
    except Exception as e:
        logger.error(f"Vina docking failed: {e}")
        return None


# ─── PARSE VINA OUTPUT ──────────────────────────────────────────────────
def parse_vina_output(pdbqt_path: str) -> List[Dict]:
    """Parse docked PDBQT output file to extract scores and poses."""
    scores = []
    if not os.path.exists(pdbqt_path):
        return scores
    
    with open(pdbqt_path, 'r') as f:
        content = f.read()
    
    # Parse MODEL/REMARK lines
    models = content.split("MODEL")
    for model_block in models[1:]:  # Skip header
        lines = model_block.strip().split('\n')
        pose_num = None
        score = None
        rmsd_lb = 0.0
        rmsd_ub = 0.0
        for line in lines:
            if line.strip().startswith("MODEL") or (not pose_num and line.strip().isdigit()):
                try:
                    pose_num = int(line.strip().split()[-1])
                except (ValueError, IndexError):
                    pose_num = len(scores) + 1
            if "REMARK VINA RESULT:" in line:
                parts = line.split()
                try:
                    idx = parts.index("RESULT:") + 1
                    score = float(parts[idx])
                    if idx + 1 < len(parts):
                        rmsd_lb = float(parts[idx + 1])
                    if idx + 2 < len(parts):
                        rmsd_ub = float(parts[idx + 2])
                except (ValueError, IndexError):
                    pass
        
        if score is not None:
            scores.append({
                "pose": pose_num or (len(scores) + 1),
                "score": round(score, 2),
                "rmsd_lb": round(rmsd_lb, 2),
                "rmsd_ub": round(rmsd_ub, 2),
            })
    
    return scores


# ─── FALLBACK SCORING (LAST RESORT) ────────────────────────────────────
def fallback_score(mol, reference_mols=None, pharmacophore_features=None):
    """Estimate a 'docking-like' score when Vina is unavailable.
    Uses RDKit molecular descriptors + shape complementarity heuristic.
    
    ⚠️ This is NOT real docking. Scores are heuristic approximations.
    Returns a simulated Vina-like score (negative, more negative = better).
    """
    if mol is None:
        return 0.0
    
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = Descriptors.TPSA(mol)
        rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        
        # Empirical scoring function (approximates Vina-like behavior)
        # More HBA/HBD → more H-bonds → better score
        hbond_score = -0.3 * min(hba, 8) - 0.4 * min(hbd, 5)
        
        # Hydrophobic contribution
        hydrophobic_score = -0.15 * min(max(logp, 0), 5)
        
        # Aromatic stacking
        aromatic_score = -0.2 * min(n_rings, 4)
        
        # Size penalty (too big or too small)
        if 300 <= mw <= 500:
            size_score = -0.5
        elif 200 <= mw < 300 or 500 < mw <= 600:
            size_score = -0.2
        else:
            size_score = 0.0
        
        # Flexibility penalty
        flex_penalty = 0.05 * max(rotbonds - 5, 0)
        
        # TPSA contribution
        tpsa_score = -0.01 * min(tpsa, 140)
        
        # Add some controlled randomness (±0.5)
        import hashlib
        smi = Chem.MolToSmiles(mol)
        hash_val = int(hashlib.md5(smi.encode()).hexdigest()[:8], 16)
        noise = (hash_val % 100 - 50) / 100.0  # -0.5 to 0.5
        
        total = hbond_score + hydrophobic_score + aromatic_score + \
                size_score - flex_penalty + tpsa_score + noise - 3.0
        
        # Clamp to realistic Vina range (-12 to -2)
        total = max(-12.0, min(-2.0, total))
        
        return round(total, 2)
    except Exception:
        return -5.0


def generate_fallback_poses(mol, num_poses=5):
    """Generate multiple conformer 'poses' as fallback when Vina unavailable."""
    if mol is None:
        return []
    
    poses = []
    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 1
        cids = AllChem.EmbedMultipleConfs(mol3d, numConfs=num_poses, params=params)
        
        for i, cid in enumerate(cids):
            try:
                AllChem.MMFFOptimizeMolecule(mol3d, confId=cid, maxIters=100)
            except Exception:
                pass
            
            # Compute score for each conformer
            base_score = fallback_score(mol)
            pose_score = base_score + (i * 0.3)  # Each successive pose slightly worse
            
            poses.append({
                "pose": i + 1,
                "score": round(pose_score, 2),
                "rmsd_lb": round(i * 0.5, 2),
                "rmsd_ub": round(i * 0.8, 2),
                "conformer_id": int(cid),
                "mol": mol3d,
            })
        
        return poses
    except Exception:
        return [{
            "pose": 1,
            "score": fallback_score(mol),
            "rmsd_lb": 0.0,
            "rmsd_ub": 0.0,
        }]


# ─── UNIFIED DOCKING INTERFACE ──────────────────────────────────────────
def dock_molecule(
    mol,
    pdb_block: str,
    center: Tuple[float, float, float] = None,
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
    num_modes: int = 5,
    energy_range: float = 3.0,
    output_dir: str = None,
) -> Dict:
    """Unified docking interface.
    
    Priority:
      1. Python Vina API  → method="vina_python"
      2. Vina CLI binary  → method="vina_cli"
      3. Fallback scoring → method="fallback_scoring"
    
    Returns dict with: scores, best_score, num_poses, method
    """
    if mol is None:
        return {"scores": [], "best_score": 0.0, "num_poses": 0, "method": "failed"}
    
    if center is None:
        center, box_size = detect_binding_site(pdb_block)
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    # ── PATH 1: Python Vina API (PRIMARY) ──
    if _VINA_PYTHON_AVAILABLE:
        try:
            receptor_path = prepare_receptor(pdb_block, output_dir)
            result = _dock_molecule_python_vina(
                mol, receptor_path, center, box_size,
                exhaustiveness, num_modes, energy_range
            )
            if result and result.get("scores"):
                logger.debug(
                    f"Python Vina docked: best={result['best_score']} kcal/mol"
                )
                return result
        except Exception as e:
            logger.warning(f"Python Vina failed: {e}")

    # ── PATH 2: Vina CLI binary (SECONDARY) ──
    vina_bin = find_vina_binary()
    if vina_bin and vina_bin != "python_vina":
        try:
            receptor_path = prepare_receptor(pdb_block, output_dir)
            lig_path = prepare_ligand(mol, output_dir, "lig")
            if receptor_path and lig_path:
                result = dock_molecule_vina_cli(
                    lig_path, receptor_path, center, box_size,
                    exhaustiveness, num_modes, energy_range, output_dir
                )
                if result and result["scores"]:
                    return result
        except Exception as e:
            logger.warning(f"Vina CLI failed: {e}")

    # ── PATH 3: Fallback scoring (LAST RESORT) ──
    logger.info("⚠️ Using fallback scoring (Vina not available)")
    poses = generate_fallback_poses(mol, num_modes)
    scores = [{"pose": p["pose"], "score": p["score"],
               "rmsd_lb": p["rmsd_lb"], "rmsd_ub": p["rmsd_ub"]} for p in poses]
    
    return {
        "scores": scores,
        "best_score": scores[0]["score"] if scores else 0.0,
        "num_poses": len(scores),
        "method": "fallback_scoring",
    }


# ─── BATCH DOCKING ──────────────────────────────────────────────────────
def dock_batch(
    molecules: List,
    smiles_list: List[str],
    pdb_block: str,
    center: Tuple[float, float, float] = None,
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
    num_modes: int = 5,
    max_molecules: int = 500,
    progress_callback: Callable = None,
) -> List[Dict]:
    """Dock a batch of molecules. Returns list of results with scores.
    
    Optimization: Receptor is prepared ONCE and reused for all molecules.
    With Python Vina API, ligands are prepared as in-memory PDBQT strings
    to avoid excessive disk I/O.
    """
    if center is None:
        center, box_size = detect_binding_site(pdb_block)
    
    results = []
    n = min(len(molecules), max_molecules)

    # ── Prepare receptor ONCE for the entire batch ──
    shared_receptor_dir = tempfile.mkdtemp()
    shared_receptor_path = None
    vina_instance = None

    if _VINA_PYTHON_AVAILABLE:
        try:
            shared_receptor_path = prepare_receptor(pdb_block, shared_receptor_dir)
            # Pre-compute Vina maps once (expensive step)
            from vina import Vina
            vina_instance = Vina(sf_name='vina')
            vina_instance.set_receptor(shared_receptor_path)
            vina_instance.compute_vina_maps(
                center=list(center), box_size=list(box_size)
            )
            logger.info(
                f"Vina maps computed for batch docking: "
                f"center={center}, box={box_size}"
            )
        except Exception as e:
            logger.warning(f"Failed to pre-compute Vina maps: {e}")
            vina_instance = None

    for i in range(n):
        mol = molecules[i]
        smi = (
            smiles_list[i]
            if i < len(smiles_list)
            else Chem.MolToSmiles(mol) if mol else "Unknown"
        )
        
        try:
            # ── Fast path: reuse pre-computed Vina maps ──
            if vina_instance is not None and mol is not None:
                try:
                    pdbqt_string = prepare_ligand_pdbqt_string(mol)
                    if pdbqt_string:
                        # Try in-memory first
                        try:
                            vina_instance.set_ligand_from_string(pdbqt_string)
                        except Exception:
                            with tempfile.NamedTemporaryFile(
                                suffix='.pdbqt', mode='w', delete=False
                            ) as f:
                                f.write(pdbqt_string)
                                tmp_lig = f.name
                            try:
                                vina_instance.set_ligand_from_file(tmp_lig)
                            finally:
                                try:
                                    os.unlink(tmp_lig)
                                except OSError:
                                    pass

                        vina_instance.dock(
                            exhaustiveness=exhaustiveness,
                            n_poses=num_modes
                        )
                        energies = vina_instance.energies()
                        scores = []
                        for j, energy in enumerate(energies):
                            scores.append({
                                "pose": j + 1,
                                "score": round(float(energy[0]), 2),
                                "rmsd_lb": (
                                    round(float(energy[1]), 2)
                                    if len(energy) > 1 else 0.0
                                ),
                                "rmsd_ub": (
                                    round(float(energy[2]), 2)
                                    if len(energy) > 2 else 0.0
                                ),
                            })
                        
                        dock_result = {
                            "scores": scores,
                            "best_score": (
                                scores[0]["score"] if scores else 0.0
                            ),
                            "num_poses": len(scores),
                            "method": "vina_python",
                            "smiles": smi,
                            "mol_index": i,
                        }
                        results.append(dock_result)

                        if progress_callback:
                            progress_callback(i + 1, n)
                        continue
                except Exception as e:
                    logger.debug(
                        f"Batch Vina failed for mol {i}, "
                        f"falling through: {e}"
                    )

            # ── Slow path: per-molecule dock_molecule() ──
            with tempfile.TemporaryDirectory() as tmpdir:
                dock_result = dock_molecule(
                    mol, pdb_block, center, box_size,
                    exhaustiveness, num_modes, output_dir=tmpdir
                )
                dock_result["smiles"] = smi
                dock_result["mol_index"] = i
                results.append(dock_result)

        except Exception as e:
            logger.error(f"Docking failed for mol {i}: {e}")
            results.append({
                "scores": [],
                "best_score": 0.0,
                "num_poses": 0,
                "method": "failed",
                "smiles": smi,
                "mol_index": i,
            })
        
        if progress_callback:
            progress_callback(i + 1, n)

    # Cleanup shared receptor dir
    try:
        shutil.rmtree(shared_receptor_dir, ignore_errors=True)
    except Exception:
        pass
    
    # Sort by best score (more negative = better)
    results.sort(key=lambda x: x.get("best_score", 0.0))
    return results


# ─── SDF OUTPUT ──────────────────────────────────────────────────────────
def generate_pose_sdf(mol, scores, output_path):
    """Generate SDF file for a molecule's docking poses."""
    if mol is None:
        return False
    try:
        writer = Chem.SDWriter(output_path)
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cids = AllChem.EmbedMultipleConfs(mol3d, numConfs=len(scores), params=params)
        
        for i, score_info in enumerate(scores):
            conf_id = int(cids[i]) if i < len(cids) else 0
            mol_copy = Chem.RWMol(mol3d)
            mol_copy.SetProp("Pose", str(score_info.get("pose", i + 1)))
            mol_copy.SetProp("Score", str(score_info.get("score", 0.0)))
            mol_copy.SetProp("RMSD_LB", str(score_info.get("rmsd_lb", 0.0)))
            mol_copy.SetProp("RMSD_UB", str(score_info.get("rmsd_ub", 0.0)))
            writer.write(mol_copy, confId=conf_id)
        
        writer.close()
        return True
    except Exception as e:
        logger.error(f"SDF generation failed: {e}")
        return False


# ─── ProLIF INTERACTION FINGERPRINT ─────────────────────────────────────
def compute_prolif_interactions(mol, pdb_block, docked_pdbqt_path=None):
    """Compute protein-ligand interaction fingerprint using ProLIF.
    
    Identifies specific per-residue interactions:
    H-bonds, hydrophobic, pi-stacking, salt bridges, etc.
    
    Returns:
        dict with 'interactions' (list), 'count' (int), 'score' (float 0-1),
        or None if ProLIF is not available.
    """
    try:
        import prolif as plf
        import MDAnalysis as mda
    except ImportError:
        logger.debug("ProLIF not available, skipping PLIF")
        return None
    
    prot_pdb = None
    lig_sdf = None
    
    try:
        # Write PDB to temp file for MDAnalysis
        with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as f:
            for line in pdb_block.strip().split('\n'):
                if line.startswith("ATOM") or line.startswith("TER") or line.startswith("END"):
                    f.write(line + '\n')
            prot_pdb = f.name
        
        # Prepare ligand with 3D coords and explicit hydrogens
        mol3d = Chem.AddHs(mol)
        if mol3d.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol3d, params) == -1:
                return None
            try:
                AllChem.MMFFOptimizeMolecule(mol3d, maxIters=100)
            except Exception:
                pass
        
        # Convert ligand to ProLIF Molecule directly from RDKit
        lig_mol = plf.Molecule.from_rdkit(mol3d)
        
        # Load protein via MDAnalysis and convert to ProLIF Molecule
        prot_u = mda.Universe(prot_pdb)
        prot_sel = prot_u.select_atoms("protein")
        prot_mol = plf.Molecule.from_mda(prot_sel)
        
        # Run ProLIF fingerprint
        fp_gen = plf.Fingerprint()
        fp_gen.run_from_iterable([lig_mol], prot_mol)
        
        # Extract interactions from DataFrame
        ifp_df = fp_gen.to_dataframe()
        interactions = []
        total_count = 0
        
        for col in ifp_df.columns:
            try:
                if ifp_df[col].any():
                    if isinstance(col, tuple):
                        if len(col) >= 3:
                            residue = str(col[1])
                            interaction_type = str(col[2])
                        elif len(col) == 2:
                            residue = str(col[0])
                            interaction_type = str(col[1])
                        else:
                            residue = str(col[0])
                            interaction_type = "Unknown"
                    else:
                        residue = str(col)
                        interaction_type = "Unknown"
                    
                    interactions.append({
                        "residue": residue,
                        "type": interaction_type,
                        "present": True,
                    })
                    total_count += 1
            except Exception:
                continue
        
        # Score: normalize by expected max interactions (cap at 15)
        interaction_score = min(1.0, total_count / 15.0)
        
        return {
            "interactions": interactions,
            "count": total_count,
            "score": round(interaction_score, 4),
            "method": "prolif",
        }
    except Exception as e:
        logger.debug("ProLIF computation failed: %s" % str(e))
        return None
    finally:
        # Cleanup temp files
        for path in [prot_pdb, lig_sdf]:
            if path:
                try:
                    os.unlink(path)
                except Exception:
                    pass


# ─── ODDT RF-SCORE RE-SCORING ───────────────────────────────────────────
def oddt_rescore(mol, pdb_block):
    """Re-score a molecule using ODDT's Random Forest scoring function.
    
    RF-Score is a machine-learning-based scoring function trained on
    protein-ligand complexes from PDBbind. It predicts binding affinity.
    
    Returns:
        dict with 'rfscore' (float), 'nnscore' (float), or None if unavailable.
    """
    try:
        import oddt
        from oddt.toolkits import rdk as oddt_rdk
        oddt.toolkit = oddt_rdk
    except (ImportError, Exception):
        logger.debug("ODDT not available or toolkit init failed")
        return None
    
    try:
        # Write PDB to temp file
        with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as f:
            for line in pdb_block.strip().split('\n'):
                if line.startswith("ATOM") or line.startswith("TER") or line.startswith("END"):
                    f.write(line + '\n')
            prot_pdb = f.name
        
        # Load receptor
        rec = next(oddt_rdk.readfile('pdb', prot_pdb))
        rec.protein = True
        
        # Generate ligand SDF
        mol3d = Chem.AddHs(mol)
        if mol3d.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol3d, params) == -1:
                return None
            try:
                AllChem.MMFFOptimizeMolecule(mol3d, maxIters=100)
            except Exception:
                pass
        
        with tempfile.NamedTemporaryFile(suffix='.sdf', mode='w', delete=False) as f:
            writer = Chem.SDWriter(f.name)
            writer.write(mol3d)
            writer.close()
            lig_sdf = f.name
        
        lig = next(oddt_rdk.readfile('sdf', lig_sdf))
        
        scores = {}
        
        # Try RF-Score
        try:
            from oddt.scoring.functions import RFScore
            rf = RFScore.load()
            rf_score = rf.predict_ligand(lig)
            if hasattr(rf_score, '__len__'):
                scores["rfscore"] = round(float(rf_score[0]), 3)
            else:
                scores["rfscore"] = round(float(rf_score), 3)
        except Exception as e:
            logger.debug("RF-Score failed: %s" % str(e))
        
        # Try NNScore
        try:
            from oddt.scoring.functions import NNScore
            nn = NNScore.load()
            nn_score = nn.predict_ligand(lig)
            if hasattr(nn_score, '__len__'):
                scores["nnscore"] = round(float(nn_score[0]), 3)
            else:
                scores["nnscore"] = round(float(nn_score), 3)
        except Exception as e:
            logger.debug("NNScore failed: %s" % str(e))
        
        # Cleanup
        try:
            os.unlink(prot_pdb)
            os.unlink(lig_sdf)
        except Exception:
            pass
        
        if scores:
            scores["method"] = "oddt"
            return scores
        return None
    except Exception as e:
        logger.debug("ODDT re-scoring failed: %s" % str(e))
        return None


# ─── ENHANCED POST-DOCKING ANALYSIS ─────────────────────────────────────
def post_docking_analysis(mol, pdb_block, vina_score=None):
    """Run all post-docking analyses: ProLIF + ODDT.
    
    Returns dict with all available scores. Gracefully handles
    cases where ProLIF or ODDT are not installed.
    """
    result = {
        "vina_score": vina_score,
        "prolif": None,
        "oddt": None,
        "enhanced_score": vina_score or 0.0,
    }
    
    # Try ProLIF
    prolif_result = compute_prolif_interactions(mol, pdb_block)
    if prolif_result:
        result["prolif"] = prolif_result
    
    # Try ODDT
    oddt_result = oddt_rescore(mol, pdb_block)
    if oddt_result:
        result["oddt"] = oddt_result
    
    # Compute enhanced score combining all available scores
    scores_available = []
    weights = []
    
    # Vina score (normalized: -12 to 0 -> 0 to 1)
    if vina_score is not None:
        norm_vina = max(0, min(1, (-vina_score) / 12.0))
        scores_available.append(norm_vina)
        weights.append(0.5)
    
    # ProLIF interaction score
    if prolif_result:
        scores_available.append(prolif_result["score"])
        weights.append(0.3)
    
    # ODDT RF-Score (normalize: typical range 3-10 pKd)
    if oddt_result and "rfscore" in oddt_result:
        norm_rf = max(0, min(1, oddt_result["rfscore"] / 10.0))
        scores_available.append(norm_rf)
        weights.append(0.2)
    
    if scores_available:
        total_weight = sum(weights)
        result["enhanced_score"] = round(
            sum(s * w for s, w in zip(scores_available, weights)) / total_weight, 4
        )
    
    return result
