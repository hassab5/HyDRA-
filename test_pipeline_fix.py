"""Test that the pipeline now generates leads (no more zero results)."""
import sys, os, logging
logging.basicConfig(level=logging.WARNING)

from rdkit import Chem
from modules.hybrid_orchestrator import HybridPipeline

EGFR_DEMO_SMILES = [
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
    "COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1",
    "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
    "c1ccc(Nc2ncnc3ccccc23)cc1",
]

EGFR_DEMO_PDB = """ATOM      1  N   MET A  696      18.168  54.537  36.088  1.00 41.78           N
ATOM      2  CA  MET A  696      18.399  53.816  34.827  1.00 41.03           C
ATOM     20  N   LEU A  718      25.898  51.312  30.088  1.00 29.80           N
ATOM     30  N   VAL A  726      29.128  48.013  25.800  1.00 23.71           N
ATOM     40  N   ALA A  743      31.250  44.188  22.395  1.00 18.43           N
ATOM     50  N   LYS A  745      33.680  42.003  23.775  1.00 20.74           N
ATOM     60  N   THR A  790      30.851  42.810  18.085  1.00 18.75           N
ATOM     70  N   MET A  793      32.621  38.723  19.105  1.00 17.63           N
ATOM     80  N   LEU A  788      28.512  44.150  17.655  1.00 19.56           N
ATOM     90  N   ASP A  855      27.115  38.965  23.200  1.00 20.95           N
HETATM  200  C1  ERL A  900      28.500  42.100  20.500  1.00 15.00           C
HETATM  201  C2  ERL A  900      29.200  41.300  21.200  1.00 15.00           C
HETATM  202  N1  ERL A  900      30.100  40.800  20.400  1.00 15.00           N
END
"""

mols = [Chem.MolFromSmiles(s) for s in EGFR_DEMO_SMILES if Chem.MolFromSmiles(s)]
smiles_list = [Chem.MolToSmiles(m) for m in mols]
print("Active molecules:", len(mols))

pipeline = HybridPipeline(
    active_mols=mols,
    active_smiles=smiles_list,
    pdb_block=EGFR_DEMO_PDB,
    params={
        "exhaustiveness": 4,
        "box_size": [20.0, 20.0, 20.0],
        "num_modes": 2,
        "frag_threshold": 0.70,
        "evolved_threshold": 0.80,
        "db_threshold": 0.80,
        "data_dir": os.path.join(os.path.dirname(__file__), "data"),
    }
)

def progress(frac, msg):
    # Only print key milestones
    if frac <= 0.01 or frac >= 0.99 or "complete" in msg.lower():
        safe_msg = msg.encode('ascii', 'replace').decode('ascii')
        print("  [{:.0%}] {}".format(frac, safe_msg))

print("Running pipeline...")
pipeline.run_full_pipeline(progress_callback=progress)

s = pipeline.get_summary()
print()
print("=== RESULTS ===")
print("Pharmacophore features:", s["pharmacophore"]["n_features"])
print("Fragment leads:", s["pathway1"]["n_leads"])
print("Database leads:", s["pathway2"]["n_leads"])
print("Combined leads:", s["combined"]["n_total_leads"])

if pipeline.fragment_leads:
    print()
    print("Top 3 fragment leads:")
    for i, l in enumerate(pipeline.fragment_leads[:3]):
        score = l.get("final_score", 0)
        smi = l["smiles"][:60]
        print("  {}. Score={:.4f} SMILES={}".format(i+1, score, smi))

if pipeline.database_leads:
    print()
    print("Top 3 database leads:")
    for i, l in enumerate(pipeline.database_leads[:3]):
        score = l.get("final_score", 0)
        smi = l["smiles"][:60]
        print("  {}. Score={:.4f} SMILES={}".format(i+1, score, smi))

print()
total = s["combined"]["n_total_leads"]
if total > 0:
    print("SUCCESS! {} total leads generated.".format(total))
else:
    print("FAILED - no leads generated")
