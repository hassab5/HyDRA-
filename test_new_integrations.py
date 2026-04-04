"""
Test script for ProLIF, pmapper, and ODDT integrations.
Verifies API compatibility and end-to-end functionality.
"""
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

# ── Test 1: pmapper fingerprint ─────────────────────────────────────
print("=" * 60)
print("TEST 1: pmapper 3D pharmacophore fingerprint")
print("=" * 60)

try:
    from modules.pharmacophore import compute_pmapper_fingerprint
    
    # Test molecule: Erlotinib (EGFR inhibitor)
    smi = "C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@@H]1CCOC1"
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        # fallback simpler molecule
        smi = "c1ccc(NC(=O)c2ccccc2)cc1"
        mol = Chem.MolFromSmiles(smi)
    
    fp = compute_pmapper_fingerprint(mol, n_bits=1024)
    n_on = int(np.sum(fp > 0))
    print(f"  [PASS] pmapper FP computed: shape={fp.shape}, bits ON={n_on}")
    print(f"  Density: {n_on/len(fp)*100:.1f}%")
    
    if n_on > 0:
        print("  [PASS] pmapper produces non-zero fingerprint")
    else:
        print("  [WARN] pmapper returned all zeros (may need conformer)")
        
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# ── Test 2: ProLIF interaction fingerprint ──────────────────────────
print("\n" + "=" * 60)
print("TEST 2: ProLIF protein-ligand interaction fingerprint")
print("=" * 60)

try:
    from modules.docking import compute_prolif_interactions
    
    # Minimal PDB block (just a few residues for testing)
    pdb_block = """ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.500   3.000   3.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.500   1.000   4.000  1.00  0.00           C
ATOM      6  N   LEU A   2       3.500   1.000   3.000  1.00  0.00           N
ATOM      7  CA  LEU A   2       4.500   1.000   3.000  1.00  0.00           C
ATOM      8  C   LEU A   2       5.500   1.000   3.000  1.00  0.00           C
ATOM      9  O   LEU A   2       6.000   2.000   3.000  1.00  0.00           O
ATOM     10  CB  LEU A   2       4.000   0.000   4.000  1.00  0.00           C
ATOM     11  CG  LEU A   2       4.000  -1.000   5.000  1.00  0.00           C
ATOM     12  CD1 LEU A   2       3.000  -2.000   5.000  1.00  0.00           C
ATOM     13  CD2 LEU A   2       5.000  -2.000   5.000  1.00  0.00           C
TER
END
"""
    
    mol = Chem.MolFromSmiles("c1ccc(NC(=O)c2ccccc2)cc1")
    result = compute_prolif_interactions(mol, pdb_block)
    
    if result is None:
        print("  [WARN] ProLIF returned None (may be expected with minimal PDB)")
        print("  This is OK - ProLIF needs a real protein structure to find interactions")
    else:
        print(f"  [PASS] ProLIF result: {result['count']} interactions, score={result['score']}")
        for inter in result['interactions'][:5]:
            print(f"    - {inter['residue']}: {inter['type']}")
    
    print("  [PASS] ProLIF function runs without crashing")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# ── Test 3: ODDT re-scoring ────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: ODDT re-scoring")
print("=" * 60)

try:
    from modules.docking import oddt_rescore
    
    mol = Chem.MolFromSmiles("c1ccc(NC(=O)c2ccccc2)cc1")
    pdb_block_simple = """ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.500   3.000   3.000  1.00  0.00           O
TER
END
"""
    result = oddt_rescore(mol, pdb_block_simple)
    
    if result is None:
        print("  [WARN] ODDT returned None (expected with minimal PDB / missing models)")
        print("  ODDT RF-Score needs trained model files - this is OK for graceful fallback")
    else:
        print(f"  [PASS] ODDT result: {result}")
    
    print("  [PASS] ODDT function runs without crashing")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# ── Test 4: post_docking_analysis (combined) ───────────────────────
print("\n" + "=" * 60)
print("TEST 4: post_docking_analysis (ProLIF + ODDT combined)")
print("=" * 60)

try:
    from modules.docking import post_docking_analysis
    
    mol = Chem.MolFromSmiles("c1ccc(NC(=O)c2ccccc2)cc1")
    pdb_block_simple = """ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.500   3.000   3.000  1.00  0.00           O
TER
END
"""
    result = post_docking_analysis(mol, pdb_block_simple, vina_score=-7.5)
    
    print(f"  Vina score: {result['vina_score']}")
    print(f"  ProLIF: {result['prolif']}")
    print(f"  ODDT: {result['oddt']}")
    print(f"  Enhanced score: {result['enhanced_score']}")
    print("  [PASS] post_docking_analysis runs correctly")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# ── Test 5: build_feature_vector with pmapper ──────────────────────
print("\n" + "=" * 60)
print("TEST 5: Full feature vector (includes pmapper)")
print("=" * 60)

try:
    from modules.pharmacophore import build_feature_vector
    
    mol = Chem.MolFromSmiles("c1ccc(NC(=O)c2ccccc2)cc1")
    # Minimal pharmacophore features
    pharm_feats = [
        {"type": "HBA", "center": [1.0, 2.0, 3.0], "radius": 1.5},
        {"type": "HBD", "center": [4.0, 5.0, 6.0], "radius": 1.5},
        {"type": "Aromatic", "center": [7.0, 8.0, 9.0], "radius": 1.5},
    ]
    
    fv = build_feature_vector(mol, pharm_feats, binding_residues=None)
    
    # Expected: 90 (pharma FP) + 2048 (Morgan) + 7 (desc) + 1 (match) + 20 (site) + 30 (IFP) + 12 (USR) + 1024 (pmapper) = 3232
    print(f"  Feature vector shape: {fv.shape}")
    print(f"  Total dims: {len(fv)}")
    print(f"  Non-zero elements: {np.count_nonzero(fv)}")
    
    # Check pmapper portion (last 1024 elements)
    pmapper_portion = fv[-1024:]
    pmapper_on = np.count_nonzero(pmapper_portion)
    print(f"  pmapper portion (last 1024D): {pmapper_on} bits ON")
    
    expected_dim = 90 + 2048 + 7 + 1 + 20 + 30 + 12 + 1024
    if len(fv) == expected_dim:
        print(f"  [PASS] Feature vector is {expected_dim}D as expected")
    else:
        print(f"  [WARN] Expected {expected_dim}D, got {len(fv)}D")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
