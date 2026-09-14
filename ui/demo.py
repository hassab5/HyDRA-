"""
ui/demo.py — Built-in EGFR demo data
======================================
Provides sample active compounds and a minimal PDB structure
so users can test HyDRA without uploading their own files.
"""

from rdkit import Chem

# ── 8 known EGFR inhibitor SMILES ─────────────────────────────
EGFR_SMILES = [
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
    "COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1",
    "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
    "Nc1ccc(-c2cc3c(Nc4cccc(Cl)c4)ncnc3[nH]2)cc1",
    "COc1cc(Nc2ncnc3cc(OC)c(OC)cc23)ccc1NC(=O)C=C",
    "c1ccc(Nc2ncnc3ccccc23)cc1",
]

# ── Minimal PDB for the EGFR kinase binding site ─────────────
EGFR_PDB = """\
HEADER    TRANSFERASE                             01-JUL-03   1M17
ATOM      1  N   MET A  696      18.168  54.537  36.088  1.00 41.78           N
ATOM      2  CA  MET A  696      18.399  53.816  34.827  1.00 41.03           C
ATOM      3  C   MET A  696      19.886  53.747  34.555  1.00 39.52           C
ATOM     20  N   LEU A  718      25.898  51.312  30.088  1.00 29.80           N
ATOM     21  CA  LEU A  718      26.645  50.225  29.490  1.00 28.18           C
ATOM     30  N   VAL A  726      29.128  48.013  25.800  1.00 23.71           N
ATOM     40  N   ALA A  743      31.250  44.188  22.395  1.00 18.43           N
ATOM     50  N   LYS A  745      33.680  42.003  23.775  1.00 20.74           N
ATOM     60  N   THR A  790      30.851  42.810  18.085  1.00 18.75           N
ATOM     70  N   MET A  793      32.621  38.723  19.105  1.00 17.63           N
ATOM     80  N   LEU A  788      28.512  44.150  17.655  1.00 19.56           N
ATOM     90  N   ASP A  855      27.115  38.965  23.200  1.00 20.95           N
ATOM    100  N   PHE A  856      24.831  38.230  21.881  1.00 20.42           N
ATOM    110  N   GLU A  762      36.221  44.862  20.042  1.00 21.78           N
ATOM    120  N   ARG A  841      22.985  40.352  25.610  1.00 26.51           N
ATOM    130  N   TRP A  880      24.112  35.525  27.180  1.00 20.06           N
ATOM    140  N   TYR A  869      27.812  33.245  24.388  1.00 18.43           N
ATOM    150  N   HIS A  781      26.050  47.218  15.330  1.00 21.22           N
HETATM  200  C1  ERL A  900      28.500  42.100  20.500  1.00 15.00           C
HETATM  207  O2  ERL A  900      31.500  40.500  18.500  1.00 15.00           O
END
"""


def load_demo():
    """
    Parse the demo SMILES into RDKit mol objects and return
    ``(mols, smiles_list, pdb_block)``.
    """
    mols, smiles = [], []
    for smi in EGFR_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            smiles.append(smi)
    return mols, smiles, EGFR_PDB
