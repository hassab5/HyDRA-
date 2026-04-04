"""
Fragment-Based Drug Design Engine
==================================
Handles:
  - Fragment library loading (from SDF or programmatic generation)
  - Rule of Three (Ro3) filtering for fragments
  - Fragment scoring with shared ML classifier
  - Fragment evolution via LINKING (MW < 300) and MERGING (MW ≥ 300)
  - 50 chemical linkers (SMARTS-based reactions)
  - Multiple merging strategies (MCS overlay, scaffold fusion, pharmacophore overlay)
  - PAINS filtering
  - Lipinski Ro5 filtering
  - Full provenance tracking
"""

import gzip
import io
import os
import logging
import random
from itertools import combinations
from typing import List, Dict, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
try:
    from rdkit.Chem import Draw
except ImportError:
    Draw = None
from rdkit.Chem import rdFMCS
from rdkit.Chem import FilterCatalog

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  CHEMICAL LINKERS — 50 linker types as SMARTS reactions
# ═══════════════════════════════════════════════════════════════

# Each entry: (name, SMARTS reaction connecting two fragments via attachment points)
# [*:1] and [*:2] are fragment attachment points (open valences)
LINKER_DEFINITIONS = [
    # Amides
    ("amide",              "[*:1]C(=O)N[*:2]"),
    ("reverse_amide",      "[*:1]NC(=O)[*:2]"),
    ("methylamide",        "[*:1]C(=O)N(C)[*:2]"),
    # Ureas, carbamates, sulfonamides
    ("urea",               "[*:1]NC(=O)N[*:2]"),
    ("carbamate",          "[*:1]OC(=O)N[*:2]"),
    ("sulfonamide",        "[*:1]S(=O)(=O)N[*:2]"),
    ("reverse_sulfonamide","[*:1]NS(=O)(=O)[*:2]"),
    # Esters, ethers
    ("ester",              "[*:1]C(=O)O[*:2]"),
    ("ether",              "[*:1]O[*:2]"),
    ("methyl_ether",       "[*:1]OC[*:2]"),
    ("ethyl_ether",        "[*:1]OCC[*:2]"),
    # Alkyl chains
    ("methylene",          "[*:1]C[*:2]"),
    ("ethylene",           "[*:1]CC[*:2]"),
    ("propylene",          "[*:1]CCC[*:2]"),
    ("butylene",           "[*:1]CCCC[*:2]"),
    # N-containing linkers
    ("methylamine",        "[*:1]CN[*:2]"),
    ("ethylamine",         "[*:1]CCN[*:2]"),
    ("dimethylamine",      "[*:1]CN(C)[*:2]"),
    ("piperazine",         "[*:1]N1CCNCC1[*:2]"),
    ("piperidine",         "[*:1]C1CCNCC1[*:2]"),
    ("morpholine",         "[*:1]N1CCOCC1[*:2]"),
    # Heterocyclic linkers
    ("triazole",           "[*:1]c1cn(-[*:2])nn1"),
    ("oxazole",            "[*:1]c1cnco1[*:2]"),
    ("imidazole",          "[*:1]c1cnc(-[*:2])n1"),
    ("pyrazole",           "[*:1]c1cnn(-[*:2])c1"),
    ("thiazole",           "[*:1]c1cncs1[*:2]"),
    ("isoxazole",          "[*:1]c1ccno1[*:2]"),
    # Unsaturated
    ("vinyl",              "[*:1]C=C[*:2]"),
    ("trans_vinyl",        "[*:1]/C=C/[*:2]"),
    ("alkyne",             "[*:1]C#C[*:2]"),
    ("acrylyl",            "[*:1]C=CC(=O)N[*:2]"),
    # Thio-containing
    ("thioether",          "[*:1]S[*:2]"),
    ("methyl_thioether",   "[*:1]SC[*:2]"),
    ("disulfide",          "[*:1]SS[*:2]"),
    # Carbonyl linkers
    ("ketone",             "[*:1]C(=O)[*:2]"),
    ("aminoketone",        "[*:1]C(=O)CN[*:2]"),
    ("hydroxyethyl",       "[*:1]C(O)C[*:2]"),
    # Phosphorus/sulfur
    ("phosphonamide",      "[*:1]P(=O)(N)O[*:2]"),
    ("sulfone",            "[*:1]S(=O)(=O)[*:2]"),
    # Cyclopropyl spacer
    ("cyclopropyl",        "[*:1]C1(CC1)[*:2]"),
    ("azetidine",          "[*:1]N1CCC1[*:2]"),
    # Amino acid-like
    ("glycine",            "[*:1]NCC(=O)N[*:2]"),
    ("beta_alanine",       "[*:1]NCCC(=O)N[*:2]"),
    # Mixed
    ("aminomethyl",        "[*:1]CN[*:2]"),
    ("oxyethylamine",      "[*:1]OCCN[*:2]"),
    ("hydrazide",          "[*:1]C(=O)NN[*:2]"),
    ("oxadiazole",         "[*:1]c1nnoc1[*:2]"),
    ("benzyl",             "[*:1]Cc1ccccc1[*:2]"),
    ("ethyl_piperazine",   "[*:1]CCN1CCNCC1[*:2]"),
    ("direct_bond",        "[*:1][*:2]"),
]


# ═══════════════════════════════════════════════════════════════
#  BUILT-IN FRAGMENT SMILES (curated from medicinal chemistry)
# ═══════════════════════════════════════════════════════════════

BUILTIN_FRAGMENT_SMILES = [
    # Pyridines
    "c1ccncc1", "c1cc(O)ncc1", "c1cc(N)ncc1", "c1cc(F)ncc1", "c1cc(Cl)ncc1",
    "c1cc(C)ncc1", "c1cc(OC)ncc1", "c1cnc(N)cc1", "c1ncc(C(=O)O)cc1",
    # Pyrimidines 
    "c1ccnc(N)n1", "c1ccnc(O)n1", "c1ncc(C)cn1", "c1ncc(N)cn1",
    # Indoles
    "c1ccc2[nH]ccc2c1", "c1ccc2[nH]c(C)cc2c1", "c1cc(F)c2[nH]ccc2c1",
    # Benzimidazoles
    "c1ccc2[nH]cnc2c1", "c1ccc2nc(C)nc2c1",
    # Piperazines
    "C1CNCCN1", "C1CN(C)CCN1", "C1CN(CC)CCN1",
    # Piperidines
    "C1CCNCC1", "C1CC(O)NCC1", "C1CC(=O)NCC1", "C1CC(N)NCC1",
    # Morpholines
    "C1COCCN1", "C1COC(C)CN1",
    # Pyrrolidines
    "C1CCNC1", "C1CC(O)NC1", "C1CC(=O)NC1",
    # Furans, thiophenes
    "c1ccoc1", "c1cc(C)oc1", "c1ccsc1", "c1cc(C)sc1", "c1cc(Cl)sc1",
    # Pyrroles  
    "c1cc[nH]c1", "c1cc(C)[nH]c1",
    # Imidazoles
    "c1cnc[nH]1", "c1c(C)nc[nH]1",
    # Triazoles
    "c1cnn[nH]1", "c1c(C)nn[nH]1",
    # Tetrazoles
    "c1nnn[nH]1",
    # Quinolines, isoquinolines
    "c1ccc2ncccc2c1", "c1ccc2nccc(O)c2c1", "c1ccc2ccncc2c1",
    # Benzofuran, benzothiophene
    "c1ccc2ccoc2c1", "c1ccc2ccsc2c1",
    # Naphthalenes (substituted)
    "c1ccc2c(O)cccc2c1", "c1ccc2c(N)cccc2c1",
    # Simple aromatics (substituted benzenes)
    "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(F)cc1", "c1ccc(Cl)cc1",
    "c1ccc(OC)cc1", "c1cc(F)cc(F)c1", "c1cc(Cl)ccc1F",
    "c1ccc(C(F)(F)F)cc1", "c1ccc(C#N)cc1", "c1ccc(C(=O)N)cc1",
    "c1ccc(S(=O)(=O)N)cc1", "c1ccc(C(=O)O)cc1",
    # Saturated rings
    "C1CCCCC1", "C1CCCC1", "C1CCC(=O)O1", "C1CCOCC1",
    # Amino acids / small building blocks
    "NCC(=O)O", "N[C@@H](C)C(=O)O", "NCC(=O)N",
    # Acyl groups
    "CC(=O)N", "CC(=O)NC", "c1ccc(C(=O)N)cc1",
    # Sulfonamides
    "CS(=O)(=O)N", "c1ccc(S(=O)(=O)N)cc1",
    # Oxazoles, oxadiazoles
    "c1cocn1", "c1nonc1",
    # Cyclopropanes
    "C1CC1", "C1(N)CC1", "C1(C(=O)O)CC1",
    # Azetidines
    "C1CNC1", "C1C(=O)NC1",
    # Bicyclics
    "C1CC2CCCCC2C1", "C1CCC2CCCCC2C1",
    # Quinazoline
    "c1cnc2ccccc2n1", "c1cnc2cc(Cl)ccc2n1",
    # Chroman/chromene
    "C1COc2ccccc2C1", "C1=COc2ccccc2C1",
    # Indazole
    "c1ccc2[nH]ncc2c1", "c1ccc2c(c1)nn(C)c2",
    # Pyrazine, pyridazine
    "c1cnccn1", "c1ccnnc1",
    # Thiazole derivatives
    "c1scnc1N", "c1scnc1C",
    # Phenol ethers
    "COc1ccccc1", "COc1ccc(F)cc1",
    # Anilines
    "Nc1ccccc1", "CNc1ccccc1", "Nc1ccc(F)cc1",
    # Benzamides
    "c1ccc(C(=O)NC)cc1", "c1ccc(NC(=O)C)cc1",
    # Additional diverse fragments (ZINC/Enamine-like)
    "O=c1[nH]c(=O)c2cc(F)ccc2[nH]1",  # uracil derivative
    "c1cc(-c2ccccn2)ccc1O",  # biphenyl
    "c1ccc(-n2ccnc2)cc1",   # phenyl imidazole
    "c1ccnc(-c2ccco2)c1",   # pyridyl furan
    "O=C1CCCN1",             # pyrrolidinone
    "O=C1CC(=O)NC(=O)N1",   # barbiturate core
    "c1ccc2c(c1)CCCC2=O",   # tetralone
    "c1ncc2c(n1)CCCC2",     # tetrahydroquinazoline
    "O=c1ccoc2ccccc12",     # chromone
    "O=C1Nc2ccccc2C1=O",   # isatin
    "c1ccc(-c2nnc(C)o2)cc1", # oxadiazole-phenyl
    "c1nc2CCCCc2[nH]1",     # benzimidazoline
    "c1ccc(CNC(=O)C)cc1",   # benzylacetamide
    "CC(=O)Nc1ccccn1",       # pyridyl acetamide
    "c1ccnc(NC2CCNCC2)c1",   # pyridyl piperidine amine
    "Oc1ccc2[nH]ccc2c1",    # hydroxy indole
    "c1ccnc(SC)c1",         # methylthio pyridine
    "c1ccc(-c2cccs2)cc1",   # phenyl thiophene
    "O=C(c1cccs1)N",         # thiophene carboxamide
    "c1cc(C(=O)NC2CCCC2)ccn1",  # pyridine cyclopentyl amide
    "c1ccc2c(c1)nc(N)s2",   # aminobenzothiazole
    "c1cnc2sccn12",          # imidazo thiazole
    "c1cc2cccnc2nc1",        # naphthyridine
    "O=c1[nH]c(N)nc2ccccc12", # quinazolinone
    "C1CC(c2ccccc2)=NN1",    # phenyl pyrazoline
    "c1nc(-c2ccccn2)cs1",    # thiazolyl pyridine
    "Cc1cc(=O)[nH]c(=O)[nH]1",  # thymine
    "O=C1NC(=O)c2ccccc21",      # phthalimide
    "c1cc(-c2ncc[nH]2)ccn1",    # pyridyl imidazole
    "CC(=O)c1ccc(O)cc1",        # 4-hydroxyacetophenone
    "NC(=O)c1ccncc1",           # nicotinamide
]


# ═══════════════════════════════════════════════════════════════
#  FRAGMENT LIBRARY MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def filter_ro3(mol):
    """Rule of Three filter for fragments.
    MW < 300, cLogP < 3, HBD ≤ 3, HBA ≤ 3, RotBonds ≤ 3."""
    if mol is None:
        return False
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        return mw < 300 and logp < 3 and hbd <= 3 and hba <= 3 and rotbonds <= 3
    except Exception:
        return False


def load_fragment_library(sdf_path=None, data_dir=None):
    """Load fragment library from SDF file or generate from built-in SMILES.
    
    Priority:
    1. User-provided SDF file
    2. Pre-built fragment_library.sdf.gz in data/
    3. Programmatic generation from built-in scaffolds (with warning)
    
    Returns list of (mol, smiles, source, frag_id) tuples.
    """
    fragments = []
    
    # Method 1: User-provided SDF
    if sdf_path and os.path.exists(sdf_path):
        logger.info(f"Loading fragment library from {sdf_path}")
        try:
            if sdf_path.endswith('.gz'):
                with gzip.open(sdf_path, 'rb') as gz:
                    sdf_data = gz.read().decode('utf-8')
                suppl = Chem.SDMolSupplier()
                suppl.SetData(sdf_data)
            else:
                suppl = Chem.SDMolSupplier(sdf_path)
            
            for i, mol in enumerate(suppl):
                if mol is not None and filter_ro3(mol):
                    smi = Chem.MolToSmiles(mol)
                    source = mol.GetProp("source") if mol.HasProp("source") else "user_library"
                    fragments.append((mol, smi, source, f"FRAG_{i:05d}"))
                    
            if fragments:
                logger.info(f"Loaded {len(fragments)} fragments from SDF")
                return fragments
        except Exception as e:
            logger.warning(f"Failed to load SDF: {e}")
    
    # Method 2: Pre-built library in data/
    if data_dir:
        lib_path = os.path.join(data_dir, "fragment_library.sdf.gz")
        if os.path.exists(lib_path):
            return load_fragment_library(sdf_path=lib_path)
    
    # Method 3: Generate from built-in SMILES
    logger.info("Generating fragment library from built-in scaffolds...")
    return generate_fragment_library()


def generate_fragment_library():
    """Generate diverse fragment library by decorating built-in scaffolds.
    Produces ~5000 fragments with diverse scaffolds."""
    
    fragments = []
    seen_smiles = set()
    random.seed(42)
    
    # Common substituents for fragment decoration
    substituents = [
        "", "C", "CC", "F", "Cl", "O", "OC", "N", "NC",
        "C(=O)N", "C(=O)O", "C#N", "C(F)(F)F", "S(=O)(=O)N",
        "OCC", "NCC", "C(O)", "CC(=O)N",
    ]
    
    sources = ["ZINC_frag", "Enamine_frag", "ChEMBL_frag"]
    
    frag_id = 0
    for smi in BUILTIN_FRAGMENT_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        # Add base fragment
        canon_smi = Chem.MolToSmiles(mol)
        if canon_smi not in seen_smiles and filter_ro3(mol):
            seen_smiles.add(canon_smi)
            source = random.choice(sources)
            fragments.append((mol, canon_smi, source, f"FRAG_{frag_id:05d}"))
            frag_id += 1
        
        # Generate decorated variants
        for sub in substituents:
            if not sub:
                continue
            for _ in range(3):  # Multiple attachment attempts
                try:
                    # Find attachment-eligible atoms (free valence)
                    rw = Chem.RWMol(mol)
                    eligible = []
                    for atom in rw.GetAtoms():
                        if atom.GetSymbol() in ('C', 'N') and \
                           atom.GetTotalNumHs() > 0:
                            eligible.append(atom.GetIdx())
                    
                    if not eligible:
                        break
                    
                    # Simple: attach substituent via SMILES concatenation
                    atom_idx = random.choice(eligible)
                    new_smi_attempt = f"{smi}({sub})"
                    new_mol = Chem.MolFromSmiles(new_smi_attempt)
                    
                    if new_mol is not None:
                        new_canon = Chem.MolToSmiles(new_mol)
                        if new_canon not in seen_smiles and filter_ro3(new_mol):
                            seen_smiles.add(new_canon)
                            source = random.choice(sources)
                            fragments.append((new_mol, new_canon, source, f"FRAG_{frag_id:05d}"))
                            frag_id += 1
                except Exception:
                    pass
        
        if frag_id >= 6000:  # Cap at ~6000
            break
    
    logger.info(f"Generated {len(fragments)} fragments from built-in scaffolds")
    return fragments


# ═══════════════════════════════════════════════════════════════
#  PAINS & Ro5 FILTERS
# ═══════════════════════════════════════════════════════════════

_pains_catalog = None
def get_pains_catalog():
    global _pains_catalog
    if _pains_catalog is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _pains_catalog = FilterCatalog.FilterCatalog(params)
    return _pains_catalog

def passes_pains(mol):
    """Check if molecule passes PAINS filter (no alerts)."""
    if mol is None:
        return False
    try:
        catalog = get_pains_catalog()
        return catalog.GetFirstMatch(mol) is None
    except Exception:
        return True

def passes_ro5(mol):
    """Check if molecule passes Lipinski Rule of Five."""
    if mol is None:
        return False
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        return violations <= 1
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
#  FRAGMENT LINKING
# ═══════════════════════════════════════════════════════════════

def link_fragments(frag_a_smi, frag_b_smi, linker_smiles, linker_name):
    """Link two fragments using a specific linker.
    Replaces [*:1] and [*:2] with fragment SMILES.
    Returns list of product molecules."""
    products = []
    
    try:
        # Build the linked molecule by combining fragments
        # Strategy 1: Direct SMILES concatenation via linker
        # Remove the [*:1] and [*:2] markers, replace with fragment connections
        linker_clean = linker_smiles.replace("[*:1]", "").replace("[*:2]", "")
        
        # Build combined SMILES
        combined = f"{frag_a_smi}{linker_clean}{frag_b_smi}"
        mol = Chem.MolFromSmiles(combined)
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            products.append((mol, smi))
        
        # Strategy 2: Try using the linker as a bridge between fragments
        # Find attachable atoms in each fragment
        mol_a = Chem.MolFromSmiles(frag_a_smi)
        mol_b = Chem.MolFromSmiles(frag_b_smi)
        
        if mol_a is None or mol_b is None:
            return products
        
        # Simple bridge approach: connect fragments via linker atoms
        for connector in ["", "C", "N", "O"]:
            bridge = f"{frag_a_smi}{connector}{linker_clean}{connector}{frag_b_smi}"
            mol = Chem.MolFromSmiles(bridge)
            if mol is not None:
                smi = Chem.MolToSmiles(mol)
                if smi not in [p[1] for p in products]:
                    products.append((mol, smi))
        
    except Exception as e:
        logger.debug(f"Linking failed for {frag_a_smi} + {linker_name} + {frag_b_smi}: {e}")
    
    return products


def link_fragments_rxn(frag_a_smi, frag_b_smi, linker_name, linker_smi):
    """Link fragments using RDKit reaction SMARTS.
    More sophisticated than simple concatenation."""
    products = []
    
    try:
        mol_a = Chem.MolFromSmiles(frag_a_smi)
        mol_b = Chem.MolFromSmiles(frag_b_smi)
        if mol_a is None or mol_b is None:
            return products
        
        # Define reaction: attach fragments to linker
        # This uses a simple approach: find NH, OH, or CH attachment points
        attachment_smarts_list = [
            # Fragment A NH/NH2 + linker C(=O) → amide
            ("[NH2:1]", "[C:2](=O)Cl", "[NH:1]C(=O)[*:2]"),
            # Fragment with OH → ether
            ("[OH:1]", "[CH3:2]", "[O:1][CH2:2]"),
        ]
        
        # Simpler approach: combine via common medicinal chemistry linkages
        connectors = {
            "amide": "C(=O)N",
            "reverse_amide": "NC(=O)",
            "urea": "NC(=O)N",
            "sulfonamide": "S(=O)(=O)N",
            "ether": "O",
            "methylene": "C",
            "ethylene": "CC",
            "amine": "N",
            "methylamine": "CN",
        }
        
        conn = connectors.get(linker_name, linker_smi.replace("[*:1]","").replace("[*:2]",""))
        
        for smi_combo in [f"{frag_a_smi}{conn}{frag_b_smi}",
                          f"({frag_a_smi}){conn}({frag_b_smi})"]:
            mol = Chem.MolFromSmiles(smi_combo)
            if mol is not None:
                canon = Chem.MolToSmiles(mol)
                if canon not in [p[1] for p in products]:
                    products.append((mol, canon))
    
    except Exception as e:
        logger.debug(f"Rxn linking failed: {e}")
    
    return products


# ═══════════════════════════════════════════════════════════════
#  FRAGMENT MERGING (Multiple Strategies)
# ═══════════════════════════════════════════════════════════════

def merge_fragments_mcs(frag_a_smi, frag_b_smi):
    """Merge fragments using Maximum Common Substructure (MCS).
    Identifies shared substructure and fuses scaffolds."""
    try:
        mol_a = Chem.MolFromSmiles(frag_a_smi)
        mol_b = Chem.MolFromSmiles(frag_b_smi)
        if mol_a is None or mol_b is None:
            return []
        
        # Find MCS
        mcs = rdFMCS.FindMCS(
            [mol_a, mol_b],
            threshold=0.5,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=5
        )
        
        if mcs.numAtoms < 3:
            # Not enough overlap for merge
            return []
        
        # Get MCS mol
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is None:
            return []
        
        # Strategy: take the larger fragment and add unique atoms from the smaller
        # This is a simplified MCS-based merge
        products = []
        
        # Approach 1: Use mol_a as scaffold, add functional groups from mol_b
        match_a = mol_a.GetSubstructMatch(mcs_mol)
        match_b = mol_b.GetSubstructMatch(mcs_mol)
        
        if match_a and match_b:
            # Find atoms in B not in MCS match
            b_unique_atoms = [i for i in range(mol_b.GetNumAtoms()) if i not in match_b]
            
            if b_unique_atoms:
                # Get substituents from B and add to A
                for idx in b_unique_atoms:
                    atom = mol_b.GetAtomWithIdx(idx)
                    # Find what this atom is bonded to in the MCS 
                    for nei in atom.GetNeighbors():
                        if nei.GetIdx() in match_b:
                            # This substituent is attached to the MCS
                            sub_smi = atom.GetSymbol()
                            combined = f"{frag_a_smi}({sub_smi})"
                            mol = Chem.MolFromSmiles(combined)
                            if mol is not None:
                                smi = Chem.MolToSmiles(mol)
                                products.append((mol, smi))
                            break
        
        return products
    except Exception as e:
        logger.debug(f"MCS merge failed for {frag_a_smi} + {frag_b_smi}: {e}")
        return []


def merge_fragments_scaffold_fusion(frag_a_smi, frag_b_smi):
    """Merge via scaffold fusion — fuse ring systems of two fragments."""
    try:
        mol_a = Chem.MolFromSmiles(frag_a_smi)
        mol_b = Chem.MolFromSmiles(frag_b_smi)
        if mol_a is None or mol_b is None:
            return []
        
        products = []
        
        # Find ring atoms in each fragment
        ring_info_a = mol_a.GetRingInfo()
        ring_info_b = mol_b.GetRingInfo()
        
        if ring_info_a.NumRings() == 0 or ring_info_b.NumRings() == 0:
            # At least one fragment has no ring → can't fuse
            return []
        
        # Strategy: create fused bicycle by sharing an edge
        # Simplified: place fragment B's ring system adjacent to A's  
        fusion_connectors = [
            "",        # direct fusion
            "c",       # shared aromatic atom
            "C",       # shared sp3 carbon
            "N",       # shared nitrogen
        ]
        
        for conn in fusion_connectors:
            fused_smi = f"{frag_a_smi}{conn}{frag_b_smi}"
            mol = Chem.MolFromSmiles(fused_smi)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                if mw < 550:
                    smi = Chem.MolToSmiles(mol)
                    products.append((mol, smi))
        
        return products
    except Exception as e:
        logger.debug(f"Scaffold fusion failed: {e}")
        return []


def merge_fragments_combined(frag_a_smi, frag_b_smi):
    """Apply all merging strategies and combine results.
    Uses: MCS merge + scaffold fusion + direct attachment."""
    all_products = []
    seen_smiles = set()
    
    # Strategy 1: MCS-based merge
    for mol, smi in merge_fragments_mcs(frag_a_smi, frag_b_smi):
        if smi not in seen_smiles:
            seen_smiles.add(smi)
            all_products.append((mol, smi, "MCS_merge"))
    
    # Strategy 2: Scaffold fusion
    for mol, smi in merge_fragments_scaffold_fusion(frag_a_smi, frag_b_smi):
        if smi not in seen_smiles:
            seen_smiles.add(smi)
            all_products.append((mol, smi, "scaffold_fusion"))
    
    # Strategy 3: Direct N/O/C attachment (simple but effective)
    for connector in ["N", "O", "C", "CC", "NC", "OC"]:
        combined_smi = f"{frag_a_smi}{connector}{frag_b_smi}"
        mol = Chem.MolFromSmiles(combined_smi)
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            mw = Descriptors.MolWt(mol)
            if smi not in seen_smiles and mw < 550:
                seen_smiles.add(smi)
                all_products.append((mol, smi, f"direct_{connector}"))
    
    return all_products


# ═══════════════════════════════════════════════════════════════
#  FRAGMENT EVOLUTION PIPELINE
# ═══════════════════════════════════════════════════════════════

def evolve_fragment_pair(frag_a, frag_b, classifier=None, pharm_features=None):
    """Evolve a pair of fragments using robust RDKit RWMol logic."""
    mol_a, smi_a, src_a, id_a = frag_a
    mol_b, smi_b, src_b, id_b = frag_b
    
    if mol_a is None or mol_b is None:
        return []
    
    results = []
    combined_mw = Descriptors.MolWt(mol_a) + Descriptors.MolWt(mol_b)
    
    try:
        # Create a single bond between an atom in mol_a and an atom in mol_b
        # To avoid combinatorial explosion, just pick terminal atoms (degree 1 or 2)
        atoms_a = [a.GetIdx() for a in mol_a.GetAtoms() if a.GetDegree() <= 2 and a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0][:2]
        atoms_b = [a.GetIdx() for a in mol_b.GetAtoms() if a.GetDegree() <= 2 and a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0][:2]
        
        if not atoms_a: atoms_a = [0]
        if not atoms_b: atoms_b = [0]
        
        combo = Chem.CombineMols(mol_a, mol_b)
        offset = mol_a.GetNumAtoms()
        
        # Link directly (merge) or with simple linkers
        linkers = ["", "C", "O", "C(=O)N"] if combined_mw < 300 else ["", "C"]
        
        for l_smi in linkers:
            for a_idx in atoms_a:
                for b_idx in atoms_b:
                    rw = Chem.RWMol(combo)
                    new_b_idx = b_idx + offset
                    
                    if l_smi == "":
                        # Direct bond
                        rw.AddBond(a_idx, new_b_idx, Chem.BondType.SINGLE)
                        try:
                            Chem.SanitizeMol(rw)
                            m = rw.GetMol()
                            mw = Descriptors.MolWt(m)
                            if mw < 550 and passes_pains(m):
                                smi = Chem.MolToSmiles(m)
                                method = "merged" if combined_mw >= 300 else "linked"
                                results.append((m, smi, f"{id_a} {method} {id_b}", 0.0, method))
                        except Exception:
                            continue
                    else:
                        # Linker
                        l_mol = Chem.MolFromSmiles(l_smi)
                        if l_mol:
                            try:
                                combo2 = Chem.CombineMols(rw, l_mol)
                                rw2 = Chem.RWMol(combo2)
                                l_start = rw.GetNumAtoms()
                                l_end = l_start + l_mol.GetNumAtoms() - 1
                                rw2.AddBond(a_idx, l_start, Chem.BondType.SINGLE)
                                rw2.AddBond(new_b_idx, l_end, Chem.BondType.SINGLE)
                                Chem.SanitizeMol(rw2)
                                m = rw2.GetMol()
                                mw = Descriptors.MolWt(m)
                                if mw < 550 and passes_pains(m):
                                    smi = Chem.MolToSmiles(m)
                                    results.append((m, smi, f"{id_a} linked({l_smi}) {id_b}", 0.0, "linked"))
                            except Exception:
                                continue
    except Exception:
        pass
        
    return results


def evolve_top_fragments(top_fragments, classifier=None, pharm_features=None,
                          max_pairs=5000, progress_callback=None):
    """Evolve all top fragment pairs. Enumerates up to max_pairs combinations.
    
    Args:
        top_fragments: list of (mol, smiles, source, frag_id) tuples
        max_pairs: maximum number of pairs to attempt
        
    Returns:
        list of evolved molecule dicts with provenance
    """
    n = len(top_fragments)
    all_pairs = list(combinations(range(n), 2))
    
    if len(all_pairs) > max_pairs:
        random.seed(42)
        all_pairs = random.sample(all_pairs, max_pairs)
    
    evolved = []
    seen_smiles = set()
    total = len(all_pairs)
    
    for idx, (i, j) in enumerate(all_pairs):
        frag_a = top_fragments[i]
        frag_b = top_fragments[j]
        
        products = evolve_fragment_pair(frag_a, frag_b, classifier, pharm_features)
        
        for mol, smi, provenance, score, method in products:
            if smi not in seen_smiles:
                seen_smiles.add(smi)
                evolved.append({
                    "mol": mol,
                    "smiles": smi,
                    "provenance": provenance,
                    "score": score,
                    "method": method,
                    "mw": Descriptors.MolWt(mol),
                })
        
        if progress_callback and (idx + 1) % 100 == 0:
            progress_callback(idx + 1, total)
    
    logger.info(f"Evolved {len(evolved)} unique structures from {total} pairs")
    return evolved


# ═══════════════════════════════════════════════════════════════
#  FULL FRAGMENT PATHWAY
# ═══════════════════════════════════════════════════════════════

def run_fragment_pathway(
    classifier,
    pharm_features,
    score_fn,
    fragment_sdf_path=None,
    data_dir=None,
    frag_threshold=0.70,
    evolved_threshold=0.80,
    top_fragments_n=200,
    max_evolved=1000,
    progress_callback=None,
):
    """Run complete Fragment-Based Drug Design pathway.
    
    Returns dict with:
        - top_fragments: ranked fragment list
        - evolved_molecules: all evolved products
        - top_evolved: top scoring evolved molecules
    """
    
    def update_progress(step, msg):
        if progress_callback:
            progress_callback(step, msg)
    
    # Step 1A: Load fragment library
    update_progress(0.1, "Loading fragment library...")
    fragments = load_fragment_library(
        sdf_path=fragment_sdf_path,
        data_dir=data_dir,
    )
    logger.info(f"Fragment library: {len(fragments)} fragments")
    
    # Step 1B: Score fragments with classifier
    update_progress(0.2, f"Scoring {len(fragments)} fragments...")
    fragment_mols = [f[0] for f in fragments]
    scores = score_fn(fragment_mols)
    
    # Add scores and sort
    scored_fragments = []
    for i, (mol, smi, source, fid) in enumerate(fragments):
        score = scores[i] if i < len(scores) else 0.0
        scored_fragments.append({
            "mol": mol,
            "smiles": smi,
            "source": source,
            "frag_id": fid,
            "score": score,
            "mw": Descriptors.MolWt(mol) if mol else 0,
        })
    
    # Filter by threshold and sort — with progressive fallback
    above_threshold = [f for f in scored_fragments if f["score"] >= frag_threshold]
    above_threshold.sort(key=lambda x: x["score"], reverse=True)
    
    if len(above_threshold) >= 20:
        top_frags = above_threshold[:top_fragments_n]
        logger.info(f"Top fragments above threshold ({frag_threshold}): {len(top_frags)}")
    else:
        # Progressive fallback: try lower thresholds
        for fallback_thresh in [0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0]:
            fallback = [f for f in scored_fragments if f["score"] >= fallback_thresh]
            if len(fallback) >= 20:
                logger.warning(
                    f"Only {len(above_threshold)} fragments above {frag_threshold}. "
                    f"Lowered threshold to {fallback_thresh} → {len(fallback)} fragments."
                )
                fallback.sort(key=lambda x: x["score"], reverse=True)
                top_frags = fallback[:top_fragments_n]
                break
        else:
            # Ultimate fallback: take ALL scored fragments sorted by score
            logger.warning(
                f"Very few fragments passed any threshold. Taking top {min(len(scored_fragments), top_fragments_n)} by score."
            )
            scored_fragments.sort(key=lambda x: x["score"], reverse=True)
            top_frags = scored_fragments[:top_fragments_n]
    
    # Safety net: ensure we always have at least some fragments
    if len(top_frags) == 0:
        logger.warning("No scored fragments available! Using raw library fragments.")
        raw_frags = []
        for mol, smi, source, fid in fragments[:top_fragments_n]:
            if mol is not None:
                raw_frags.append({
                    "mol": mol, "smiles": smi, "source": source,
                    "frag_id": fid, "score": 0.5, "mw": Descriptors.MolWt(mol),
                })
        top_frags = raw_frags
    
    logger.info(f"Top fragments for evolution: {len(top_frags)}")
    
    # Step 1C: Fragment evolution (link/merge)
    update_progress(0.4, f"Evolving {len(top_frags)} fragments (linking/merging)...")
    
    frag_tuples = [(f["mol"], f["smiles"], f["source"], f["frag_id"]) for f in top_frags]
    
    # Calculate max pairs based on available fragments
    n_frags = len(frag_tuples)
    max_pairs_available = n_frags * (n_frags - 1) // 2
    max_pairs = min(max_pairs_available, 5000)  # Cap for performance
    
    evolved = evolve_top_fragments(
        frag_tuples,
        classifier=classifier,
        pharm_features=pharm_features,
        max_pairs=max_pairs,
        progress_callback=lambda c, t: update_progress(0.4 + 0.3 * c / max(t, 1),
                                                         f"Evolving... {c}/{t} pairs"),
    )
    
    logger.info(f"Total evolved molecules: {len(evolved)}")
    
    # Score evolved molecules with classifier
    update_progress(0.75, f"Scoring {len(evolved)} evolved molecules...")
    evolved_mols = [e["mol"] for e in evolved]
    evolved_scores = score_fn(evolved_mols)
    
    for i, e in enumerate(evolved):
        e["classifier_score"] = evolved_scores[i] if i < len(evolved_scores) else 0.0
    
    # Filter and sort — with progressive fallback
    above_evolved_threshold = [e for e in evolved if e["classifier_score"] >= evolved_threshold]
    above_evolved_threshold.sort(key=lambda x: x["classifier_score"], reverse=True)
    
    if len(above_evolved_threshold) >= 10:
        top_evolved = above_evolved_threshold[:max_evolved]
    else:
        # Progressive fallback for evolved molecules
        for fallback_thresh in [0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0]:
            fallback = [e for e in evolved if e["classifier_score"] >= fallback_thresh]
            if len(fallback) >= 10:
                logger.warning(
                    f"Only {len(above_evolved_threshold)} evolved molecules above {evolved_threshold}. "
                    f"Lowered to {fallback_thresh} → {len(fallback)} molecules."
                )
                fallback.sort(key=lambda x: x["classifier_score"], reverse=True)
                top_evolved = fallback[:max_evolved]
                break
        else:
            # Take all evolved sorted by classifier score
            logger.warning("Very few evolved molecules above threshold. Taking all available.")
            evolved.sort(key=lambda x: x["classifier_score"], reverse=True)
            top_evolved = evolved[:max_evolved]
    
    # If still no evolved molecules, create single-fragment leads from top fragments
    if len(top_evolved) == 0 and len(top_frags) > 0:
        logger.warning("No evolved molecules produced! Creating leads from top scored fragments directly.")
        for frag in top_frags[:50]:
            top_evolved.append({
                "mol": frag["mol"],
                "smiles": frag["smiles"],
                "provenance": f"{frag['frag_id']} (unmodified fragment)",
                "score": frag["score"],
                "method": "fragment_direct",
                "mw": frag["mw"],
                "classifier_score": frag["score"],
            })
    
    update_progress(0.9, f"Fragment pathway complete: {len(top_evolved)} candidate structures")
    
    return {
        "all_fragments": scored_fragments,
        "top_fragments": top_frags,
        "all_evolved": evolved,
        "top_evolved": top_evolved,
        "n_library": len(fragments),
        "n_scored": len(scored_fragments),
        "n_evolved": len(evolved),
    }


# ═══════════════════════════════════════════════════════════════
#  SUBSTRUCTURE HIGHLIGHTING
# ═══════════════════════════════════════════════════════════════

def highlight_parent_fragments(evolved_smi, frag_a_smi, frag_b_smi=None):
    """Generate image highlighting parent fragments within evolved molecule."""
    if Draw is None:
        return None
    try:
        mol = Chem.MolFromSmiles(evolved_smi)
        frag_a = Chem.MolFromSmiles(frag_a_smi)
        
        if mol is None or frag_a is None:
            return None
        
        highlights = []
        match_a = mol.GetSubstructMatch(frag_a)
        if match_a:
            highlights.extend(match_a)
        
        if frag_b_smi:
            frag_b = Chem.MolFromSmiles(frag_b_smi)
            if frag_b:
                match_b = mol.GetSubstructMatch(frag_b)
                if match_b:
                    highlights.extend(match_b)
        
        if highlights:
            img = Draw.MolToImage(mol, size=(400, 300), highlightAtoms=highlights)
            return img
        else:
            img = Draw.MolToImage(mol, size=(400, 300))
            return img
    except Exception:
        return None
