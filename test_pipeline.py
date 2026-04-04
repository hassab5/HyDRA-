import logging
import sys
from rdkit import Chem

from modules.hybrid_orchestrator import HybridPipeline, parse_actives_file

logging.basicConfig(level=logging.INFO)

def main():
    print("Testing pipeline locally...")
    with open("data/demo_egfr_actives.txt", "r") as f:
        content = f.read()
    
    actives = parse_actives_file(content, "demo_egfr_actives.txt")
    active_mols = [x[0] for x in actives]
    active_smiles = [x[1] for x in actives]
    
    print(f"Loaded {len(active_mols)} actives")
    
    pipeline = HybridPipeline(
        active_mols=active_mols,
        active_smiles=active_smiles,
        pdb_block="HETATM    1  C1  LIG A   1      -6.222   5.738  17.551  1.00 13.55           C  ",
        params={"exhaustiveness": 1, "frag_threshold": 0.5, "evolved_threshold": 0.5, "db_threshold": 0.5}
    )
    
    def cb(progress, msg):
        safe_msg = msg.encode('ascii', 'ignore').decode('ascii')
        print(f"[{progress:.2f}] {safe_msg}")
        
    leads = pipeline.run_full_pipeline(progress_callback=cb)
    print(f"Final leads: {len(leads)}")
    for l in leads[:5]:
        print(f" - {l['smiles']} (Score: {l['final_score']}) from {l['pathway']}")

if __name__ == "__main__":
    main()
