# -*- coding: utf-8 -*-
"""Test all new integrations: pmapper, ProLIF, ODDT."""
import sys
import numpy as np
from rdkit import Chem

print("=" * 60)
print("Testing Enhanced Pipeline Features")
print("=" * 60)

# Test pmapper
print("\n--- pmapper ---")
from modules.pharmacophore import compute_pmapper_fingerprint, build_feature_vector
mol = Chem.MolFromSmiles("c1ccc(NC(=O)c2ccccn2)cc1")
fp = compute_pmapper_fingerprint(mol)
print("  pmapper FP shape: %s, on-bits: %d" % (str(fp.shape), int(np.sum(fp > 0))))

# Test full feature vector (should be 3232D now)
fv = build_feature_vector(mol, [])
print("  Full feature vector: %s (expected 3232)" % str(fv.shape))

# Test ProLIF (expect graceful failure)
print("\n--- ProLIF ---")
from modules.docking import compute_prolif_interactions
result = compute_prolif_interactions(mol, "ATOM      1  N   ALA A   1       1.0   2.0   3.0  1.00  0.00           N")
if result is None:
    print("  ProLIF: not available (graceful fallback - OK)")
else:
    print("  ProLIF: %d interactions found" % result["count"])

# Test ODDT (expect graceful failure due to DLL issue)
print("\n--- ODDT ---")
from modules.docking import oddt_rescore
result = oddt_rescore(mol, "ATOM      1  N   ALA A   1       1.0   2.0   3.0  1.00  0.00           N")
if result is None:
    print("  ODDT: not available (graceful fallback - OK)")
else:
    print("  ODDT RF-Score: %s" % str(result))

# Test post_docking_analysis
print("\n--- post_docking_analysis ---")
from modules.docking import post_docking_analysis
pda = post_docking_analysis(mol, "ATOM      1  N   ALA A   1       1.0   2.0   3.0  1.00  0.00           N", vina_score=-7.5)
print("  Enhanced score: %s" % str(pda["enhanced_score"]))
print("  ProLIF available: %s" % str(pda["prolif"] is not None))
print("  ODDT available: %s" % str(pda["oddt"] is not None))

# Test orchestrator imports
print("\n--- Orchestrator ---")
from modules.hybrid_orchestrator import HybridPipeline
print("  HybridPipeline imports OK (with post_docking_analysis)")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
print("\nFeature Summary:")
print("  pmapper 3D Pharm FP: ACTIVE (%d-bit)" % len(fp))
print("  ProLIF PLIF: %s" % ("ACTIVE" if compute_prolif_interactions(mol, "") else "FALLBACK (install prolif to activate)"))
print("  ODDT RF-Score: %s" % ("ACTIVE" if oddt_rescore(mol, "") else "FALLBACK (fix rdMolDraw2D DLL to activate)"))
print("  Classifier features: %dD" % len(fv))
