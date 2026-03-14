"""
Binding Affinity Model Training Script

This script trains a ligand–protein binding affinity prediction model using
the PDBbind refined dataset.

Main steps:
1. Ligand feature extraction from molecular structures (RDKit)
2. Protein pocket residue feature extraction from PDB structures
3. Construction of ligand–pocket interaction representations
4. Training of a bilinear interaction model with attention-based residue pooling

Model details:
- Ligand input: 11 physicochemical features
- Protein pocket input: 37-dimensional residue features
  (amino acid encoding + physicochemical properties + geometric features)

Training dataset:
- PDBbind refined set

Evaluation dataset:
- CASF-2016 benchmark set

Output:
- Trained model checkpoint for ligand–protein binding affinity prediction

"""

import pandas as pd
import numpy as np
import csv
import os
from pathlib import Path
import random
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem import Descriptors, rdMolDescriptors as rdmd
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from Bio.PDB import PDBParser
from Bio.PDB import NeighborSearch

import scipy.stats as st


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_casf2016_ids(path):
    with open(path, "r") as f:
        return set(line.strip().lower() for line in f if line.strip())

casf_ids = load_casf2016_ids("path/to/casf2016_ids.txt")
print(f"Number of CASF ID : {len(casf_ids)}")

refined_root = "Database/CASF-2016/coreset/"
train_ids = [d for d in os.listdir(refined_root) if os.path.isdir(os.path.join(refined_root, d))]

valid_ids = []

for pid in train_ids:
    sdf = os.path.join(refined_root, pid, f"{pid}_ligand.sdf")
    pocket = os.path.join(refined_root, pid, f"{pid}_pocket.pdb")
    if os.path.exists(sdf) and os.path.exists(pocket):
        valid_ids.append(pid)

print(f"Number of valid training samples: {len(valid_ids)}")

### Data preprocessing ###

refined_root_p = Path("Database/CASF-2016/coreset")
# valid_ids: e.g.) ['1abc','2def', ...]

# Feature definitions and computation
FEATURE_KEYS = [
    'HBD', 'HBA', 'RotBonds', 'AromRings',
    'Heteroatoms', 'TPSA', 'Csp3', 'FormalCharge',
    'MolWt', 'LabuteASA', 'DonorAcceptorRatio'
]

def compute_features(mol):
    # Assume the molecule has already been sanitized
    features = {
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'AromRings': Descriptors.NumAromaticRings(mol),
        'Heteroatoms': rdmd.CalcNumHeteroatoms(mol), 
        'TPSA': Descriptors.TPSA(mol),
        'Csp3': Descriptors.FractionCSP3(mol),
        'FormalCharge': sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        'MolWt': Descriptors.MolWt(mol),
        'LabuteASA': Descriptors.LabuteASA(mol),
    }
    features['DonorAcceptorRatio'] = features['HBD'] / (features['HBA'] + 1e-5)
    return torch.tensor([features[k] for k in FEATURE_KEYS], dtype=torch.float32)

# Assume the molecule has already been sanitized
def safe_load_molecule(sdf_path: Path, mol2_path: Path):
    # 1) Attempt loading from SDF (standard case)
    if sdf_path.exists():
        try:
            # Most files contain a single molecule
            mol = Chem.MolFromMolFile(str(sdf_path), sanitize=True, removeHs=False)
            if mol is not None:
                return mol
        except Exception as e:
            print(f"[SDF sanitize failed] {sdf_path.name}: {e}")

        # 2) If SDF has structural issues, load with sanitize=False and sanitize manually
        try:
            suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
            if mol is not None:
                Chem.SanitizeMol(mol)
                return mol
        except Exception as e:
            print(f"[SDF manual sanitize failed] {sdf_path.name}: {e}")

    # 3) Fallback: try loading from MOL2
    if mol2_path.exists():
        try:
            mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
            if mol is not None:
                return mol
        except Exception as e:
            print(f"[MOL2 sanitize failed] {mol2_path.name}: {e}")

        try:
            mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
            if mol is not None:
                Chem.SanitizeMol(mol)
                return mol
        except Exception as e:
            print(f"[MOL2 manual sanitize failed] {mol2_path.name}: {e}")

    return None

## Execution ##

## 1. Ligand feature extraction
ligand_raw_map = {}
missing_files = 0
load_fail = 0
feat_fail = 0

for pid in tqdm(valid_ids, desc="Ligand RAW features"):
    lig_dir = refined_root_p / pid
    sdf_path = lig_dir / f"{pid}_ligand.sdf"
    mol2_path = lig_dir / f"{pid}_ligand.mol2"

    if not sdf_path.exists() and not mol2_path.exists():
        missing_files += 1
        if missing_files <= 10:
            print(f"[Missing file] {pid}: {sdf_path.name} / {mol2_path.name}")
        continue

    mol = safe_load_molecule(sdf_path, mol2_path)
    if mol is None:
        load_fail += 1
        if load_fail <= 10:
            print(f"[Molecule loading failed] {pid}")
        continue

    try:
        f = compute_features(mol)  # [11]
        ligand_raw_map[pid] = f.detach().clone().float()
    except Exception as e:
        feat_fail += 1
        if feat_fail <= 10:
            print(f"[Feature computation failed] {pid}: {e}")

print(f"\nLigand raw feature extraction completed: {len(ligand_raw_map)} / All: {len(valid_ids)}")
print(f"Missing file: {missing_files}, Loading failed: {load_fail}, Feature computation failed: {feat_fail}")

failed_ligand = [pid for pid in valid_ids if pid not in ligand_raw_map]
if failed_ligand:
    print("Failed samples (partial list):", failed_ligand[:10])


## 2. Protein cavity feature extraction
AA_CODES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
            'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
            'T', 'W', 'Y', 'V']

# Amino acid physicochemical properties
# Features include:
# Hydrophobicity (Kyte–Doolittle scale)
# Number of hydrogen bond donors / acceptors
# Molecular weight
# Isoelectric point (pI)
# Van der Waals volume
# Polarity
# Flexibility
# Side-chain pKa
# Aromatic / aliphatic indicator
# Net charge at physiological pH
# Relative solvent accessibility

AA_PROPERTIES = {
    'A': [ 1.8, 0, 0, 89.1, 6.01, 67, 0, 1, -1, 0, 1,  0, 45],
    'R': [-4.5, 1, 3, 174.2,10.76,148, 1, 0, 12.5, 0, 1, +1, 95],
    'N': [-3.5, 1, 1, 132.1, 5.41, 96, 1, 1, -1, 0, 1,  0, 90],
    'D': [-3.5, 0, 2, 133.1, 2.77, 91, 1, 1,  3.9, 0, 1, -1, 95],
    'C': [ 2.5, 0, 1, 121.2, 5.07, 86, 0, 0,  8.3, 0, 1,  0, 20],
    'Q': [-3.5, 1, 1, 146.1, 5.65,114, 1, 1, -1, 0, 1,  0, 90],
    'E': [-3.5, 0, 2, 147.1, 3.22,109, 1, 1,  4.3, 0, 1, -1, 95],
    'G': [-0.4, 0, 0,  75.1, 5.97, 48, 0, 1, -1, 0, 1,  0, 85],
    'H': [-3.2, 1, 1, 155.2, 7.59,118, 1, 1,  6.0, 1, 0, +1, 95],
    'I': [ 4.5, 0, 0, 131.2, 6.05,124, 0, 0, -1, 0, 1,  0, 20],
    'L': [ 3.8, 0, 0, 131.2, 6.01,124, 0, 0, -1, 0, 1,  0, 20],
    'K': [-3.9, 1, 2, 146.2, 9.74,135, 1, 1, 10.5, 0, 1, +1, 95],
    'M': [ 1.9, 0, 0, 149.2, 5.74,124, 0, 1, -1, 0, 1,  0, 50],
    'F': [ 2.8, 0, 0, 165.2, 5.48,135, 0, 0, -1, 1, 0,  0, 20],
    'P': [-1.6, 0, 0, 115.1, 6.30, 90, 0, 0, -1, 0, 1,  0, 50],
    'S': [-0.8, 1, 1, 105.1, 5.68, 73, 1, 1, 13.6, 0, 1,  0, 85],
    'T': [-0.7, 1, 1, 119.1, 5.60, 93, 1, 1, 13.6, 0, 1,  0, 85],
    'W': [-0.9, 1, 1, 204.2, 5.89,163, 1, 0, -1, 1, 0,  0, 20],
    'Y': [-1.3, 1, 1, 181.2, 5.66,141, 1, 0, 10.1, 1, 0,  0, 50],
    'V': [ 4.2, 0, 0, 117.1, 6.00,105, 0, 0, -1, 0, 1,  0, 25],
}

three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

refined_root = "Database/CASF-2016/coreset/"
refined_root_p = Path(refined_root)
parser = PDBParser(QUIET=True)

def get_first_chain_id(pdb_path):
    structure = parser.get_structure("pocket", pdb_path)
    for model in structure:
        for chain in model:
            return chain.id
    return None

def get_residue_feature(residue):
    resname_3 = residue.get_resname().strip()
    if resname_3 not in three_to_one: return None
    aa = three_to_one[resname_3]
    if aa not in AA_PROPERTIES or aa not in AA_CODES: return None
    onehot = [int(aa == x) for x in AA_CODES]
    props = AA_PROPERTIES[aa]
    return torch.tensor(onehot + props, dtype=torch.float32)

def extract_residue_ids_from_pocket(pdb_path):
    structure = parser.get_structure("pocket", pdb_path)
    ids = []
    for model in structure:
        for chain in model:
            for res in chain:
                hetflag, resseq, icode = res.get_id()  # (' ', 42, ' ')
                if hetflag.strip():      # Skip HETATM entries (ligands, water, etc.)
                    continue
                if res.get_resname().strip() not in three_to_one:
                    continue
                ids.append(resseq)
    # Ensure deterministic ordering
    return sorted(set(ids))
    
parser = PDBParser(QUIET=True)

cavity_raw_map = {}
cavity_meta_map = {}   # pid -> [(chain, resseq, icode, resname3), ...]
failed_cavity = []

for pid in tqdm(valid_ids, desc="Cavity RAW features"):
    pocket_path = refined_root_p / pid / f"{pid}_pocket.pdb"
    try:
        structure = parser.get_structure("pocket", str(pocket_path))
        model = structure[0]

        feats, metas = [], []

        for chain in model:
            chain_id = chain.id
            for res in chain:
                hetflag, resseq, icode = res.get_id()  # (' ', 42, ' ')
                if hetflag.strip():      
                    continue
                resname3 = res.get_resname().strip()
                if resname3 not in three_to_one:
                    continue

                feat = get_residue_feature(res)         # [33]
                if feat is None:
                    continue

                feats.append(feat)
                metas.append((chain_id, int(resseq), (icode or '').strip(), resname3))

        if not feats:
            raise ValueError("No valid cavity residues found.")

        cavity_raw_map[pid]  = torch.stack(feats).float()  # [T, 33]
        cavity_meta_map[pid] = metas                       

    except Exception as e:
        failed_cavity.append((pid, str(e)))
        print(f"Cavity extraction failed for  {pid}: {e}")

print(f"Cavity feature extraction completed: {len(cavity_raw_map)} / Failed {len(failed_cavity)}")
print(f"Cavity metadata stored: {len(cavity_meta_map)} (e.g.: {next(iter(cavity_meta_map.values()))[:3]})")


## 3. Model input preprocessing

parser = PDBParser(QUIET=True)

def _meta_to_key(meta):
    """
    Normalize cavity_meta_map entries to the format (chain, resseq, icode).

    Supported formats:
      (chain, resseq, resname)        -> (chain, resseq, '')
      (chain, resseq, icode, resname) -> (chain, resseq, icode)
    """
    if len(meta) >= 4:
        chain, resseq, icode, _ = meta[:4]
        return (str(chain), int(resseq), (icode or '').strip())
    elif len(meta) == 3:
        chain, resseq, _ = meta
        return (str(chain), int(resseq), '')
    else:  # fallback for unexpected formats
        chain = str(meta[0]); resseq = int(meta[1])
        icode = meta[2] if len(meta) > 2 else ''
        return (chain, resseq, (icode or '').strip())

def get_ca_xyz_by_meta(pdb_path, metas):
    """
    metas: cavity_meta_map[pid] (contains chain/residue/icode information)

    Returns:
        torch.float32 tensor of shape [T,3] containing residue coordinates
    """
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]

    # (chain, resseq, icode) -> coordinate dictionary
    ca_dict = {}
    fallback = {}
    for chain in model:
        for res in chain:
            het, resseq, icode = res.get_id()
            if het.strip():
                continue
            key = (chain.id, int(resseq), (icode or '').strip())
            if "CA" in res:
                ca_dict[key] = res["CA"].get_coord().astype(np.float32)
            else:
                # If CA atom is missing, estimate coordinate from backbone atoms
                coords = []
                for an in ("N", "C", "O", "CB"):
                    if an in res:
                        coords.append(res[an].get_coord())
                if coords:
                    fallback[key] = np.mean(np.vstack(coords), axis=0).astype(np.float32)
                else:
                    # Final fallback: use the first available atom coordinate
                    atoms = list(res.get_atoms())
                    if atoms:
                        fallback[key] = atoms[0].get_coord().astype(np.float32)

    out = []
    for meta in metas:
        key = _meta_to_key(meta)                     
        xyz = ca_dict.get(key)
        if xyz is None:
            # try matching without insertion code
            key_ni = (key[0], key[1], '')
            xyz = ca_dict.get(key_ni)
        if xyz is None:
            xyz = fallback.get(key) or fallback.get((key[0], key[1], ''))
        if xyz is None:
            xyz = np.zeros(3, dtype=np.float32)
        out.append(xyz)

    arr = np.asarray(out, dtype=np.float32)          # [T,3]
    return torch.from_numpy(arr)

def ligand_centroid_from_sdf_or_mol2(lig_path):
    """
    Compute ligand centroid from SDF or MOL2 coordinates.
    Returns a zero vector if extraction fails.
    """
    mol = None
    p = str(lig_path)
    try:
        if p.lower().endswith(".sdf"):
            suppl = Chem.SDMolSupplier(p, sanitize=False, removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        else:
            mol = Chem.MolFromMol2File(p, sanitize=False, removeHs=False)
    except Exception:
        mol = None
    if mol is None or mol.GetNumConformers() == 0:
        return torch.zeros(3, dtype=torch.float32)
    conf = mol.GetConformer(0)
    coords = [[conf.GetAtomPosition(i).x,
               conf.GetAtomPosition(i).y,
               conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())]
    if not coords:
        return torch.zeros(3, dtype=torch.float32)
    center = np.mean(np.asarray(coords, dtype=np.float32), axis=0)
    return torch.from_numpy(center)

# Coordinate generation
C_xyz_map = {}     # pid -> residue coordinates [T,3]
L_center_map = {}  # pid -> ligand centroid [3]

refined_root_p = Path("Database/CASF-2016/coreset")

for pid in cavity_meta_map.keys():  # metas are retrieved from cavity_meta_map
    pocket_pdb = refined_root_p / pid / f"{pid}_pocket.pdb"
    sdf_path   = refined_root_p / pid / f"{pid}_ligand.sdf"
    mol2_path  = refined_root_p / pid / f"{pid}_ligand.mol2"
    lig_path   = sdf_path if sdf_path.exists() else mol2_path

    try:
        metas = cavity_meta_map[pid]                              
        C_xyz = get_ca_xyz_by_meta(str(pocket_pdb), metas)        # [T,3]
        L_ctr = ligand_centroid_from_sdf_or_mol2(lig_path)        # [3]
        # Safety check for length mismatch (rare but handled defensively)
        T = len(metas)
        if C_xyz.shape[0] != T:
            if C_xyz.shape[0] > T:
                C_xyz = C_xyz[:T]
            else:
                pad = torch.zeros((T - C_xyz.shape[0], 3), dtype=torch.float32)
                C_xyz = torch.cat([C_xyz, pad], dim=0)
        C_xyz_map[pid] = C_xyz
        L_center_map[pid] = L_ctr
    except Exception as e:
        print(f"[coord warn] {pid}: {e}")
        T = len(cavity_meta_map[pid])
        C_xyz_map[pid] = torch.zeros((T,3), dtype=torch.float32)
        L_center_map[pid] = torch.zeros(3, dtype=torch.float32)


# feature augmentation
def build_cavity_aug_map(cavity_raw_map, C_xyz_map, L_center_map, eps=1e-6):
    """
    Construct augmented cavity features.

    For each pocket residue, geometric features relative to the ligand
    centroid are added to the physicochemical residue features.

    Input
    -----
    cavity_raw_map : dict
        pid -> residue feature tensor [T,33]

    C_xyz_map : dict
        pid -> residue coordinates [T,3]

    L_center_map : dict
        pid -> ligand centroid coordinates [3]

    Returns
    -------
    cavity_aug_map : dict
        pid -> augmented cavity features [T,37]

    Added geometric features
    ------------------------
    dx, dy, dz : residue position relative to ligand centroid
    r          : Euclidean distance to ligand centroid
    """
    cavity_aug_map = {}
    for pid, C_raw in cavity_raw_map.items():
        xyz = C_xyz_map[pid].float()          # [T,3]
        ctr = L_center_map[pid].float()       # [3]
        dxyz = xyz - ctr.unsqueeze(0)         # [T,3]
        r = torch.linalg.norm(dxyz, dim=1, keepdim=True).clamp_min(eps)  # [T,1]
        C_aug = torch.cat([C_raw, dxyz, r], dim=1)  # [T, 37]
        cavity_aug_map[pid] = C_aug
    return cavity_aug_map

# Build augmented cavity feature map
cavity_aug_map = build_cavity_aug_map(cavity_raw_map, C_xyz_map, L_center_map)
print("Example cavity_aug_map shape:", next(iter(cavity_aug_map.values())).shape)  # [T,37] 기대



### Model ###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bilinear affinity model (same architecture used during training)
class BilinearAffinityZ_MLP_LR(nn.Module):
    def __init__(self, lig_dim, cav_dim, proj_dim=64, rank=10, 
                 mu_L=None, sigma_L=None, mu_C=None, sigma_C=None):
        super().__init__()
        # z-score normalization statistics (stored as fixed buffers)
        self.register_buffer("mu_L", mu_L.clone().float())
        self.register_buffer("sigma_L", sigma_L.clone().float().clamp_min(1e-6))
        self.register_buffer("mu_C", mu_C.clone().float())
        self.register_buffer("sigma_C", sigma_C.clone().float().clamp_min(1e-6))

        # ligand and cavity encoders
        self.lig_proj = nn.Sequential(
            nn.Linear(lig_dim, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim), nn.Dropout(0.05)
        )
        self.cav_proj = nn.Sequential(
            nn.Linear(cav_dim, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim), nn.Dropout(0.05)
        )

        # low-rank bilinear interaction: W = U V^T
        self.U = nn.Parameter(torch.empty(proj_dim, rank))
        self.V = nn.Parameter(torch.empty(proj_dim, rank))
        nn.init.xavier_uniform_(self.U); nn.init.xavier_uniform_(self.V)

        # main effects
        self.a = nn.Parameter(torch.zeros(proj_dim))
        self.b = nn.Parameter(torch.zeros(proj_dim))

        # bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, L_raw, C_raw, lengths):
        """
        L_raw : [B, 11]        ligand raw features
        C_raw : [B, T, 37]     cavity features including geometric terms
        lengths : [B]          number of valid residues per pocket
        """
        
        B, T, D = C_raw.shape

        # per-feature z-score
        Lz = (L_raw - self.mu_L) / self.sigma_L              # [B,11]
        Cz_per_res = (C_raw - self.mu_C) / self.sigma_C      # [B,T,37]

        # projection
        Lh = self.lig_proj(Lz)                                # [B,H]
        Ch_full = self.cav_proj(Cz_per_res.reshape(-1, D)).view(B, T, -1)  # [B,T,H]

        # attention score s_t
        W = self.U @ self.V.T                                 # [H,H]
        LW = Lh @ W                                           # [B,H]
        s = (LW.unsqueeze(1) * Ch_full).sum(-1) + (self.b * Ch_full).sum(-1)  # [B,T]

        # valid mask
        idxs = torch.arange(T, device=C_raw.device).unsqueeze(0).expand(B, T)
        mask = idxs < lengths.unsqueeze(1)

        # attention weights α & weighted pooling
        alpha = torch.softmax(s.masked_fill(~mask, -1e9), dim=1)             # [B,T]
        Ch = (alpha.unsqueeze(2) * Ch_full).sum(1)                            # [B,H]

        # final prediction: bilinear + main + bias
        bilinear = (LW * Ch).sum(1)                                           # [B]
        main = (Lh * self.a).sum(1) + (Ch * self.b).sum(1)                    # [B]
        return bilinear + main + self.bias


# Attention-based binding classifier (initialized from bilinear seed model)
class AttnBind(nn.Module):
    def __init__(self, bilinear_seed, tau=1.0):
        super().__init__()
        import copy
        # copy normalization statistics
        self.register_buffer("mu_L", bilinear_seed.mu_L.clone())
        self.register_buffer("sigma_L", bilinear_seed.sigma_L.clone())
        self.register_buffer("mu_C", bilinear_seed.mu_C.clone())
        self.register_buffer("sigma_C", bilinear_seed.sigma_C.clone())

        # copy projection layers and interaction parameters
        self.lig_proj = copy.deepcopy(bilinear_seed.lig_proj)
        self.cav_proj = copy.deepcopy(bilinear_seed.cav_proj)
        self.U = nn.Parameter(bilinear_seed.U.detach().clone())
        self.V = nn.Parameter(bilinear_seed.V.detach().clone())
        self.a = nn.Parameter(bilinear_seed.a.detach().clone())
        self.b = nn.Parameter(bilinear_seed.b.detach().clone())
        self.bias = nn.Parameter(bilinear_seed.bias.detach().clone())

        # automatically infer hidden dimension
        H = self.U.size(0)
        self.proj_dim = H
        self.tau = tau

        # automatically infer hidden dimension
        self.head = nn.Sequential(
            nn.Linear(2*H, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        # optional attention prior correction
        self.delta = nn.Sequential(
            nn.Linear(2*H, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, L_raw, C_raw, lengths, return_alpha=False):
        B, T, D = C_raw.shape
        # z-score (per-residue)
        Lz = (L_raw - self.mu_L) / self.sigma_L             # [B,11]
        Cz_full = (C_raw - self.mu_C) / self.sigma_C        # [B,T,D]

        # proj
        Lh = self.lig_proj(Lz)                              # [B,H]
        Ch_full = self.cav_proj(Cz_full.reshape(-1, D)).view(B, T, -1)  # [B,T,H]

        # prior α from WD
        W  = self.U @ self.V.T                              # [H,H]
        LW = Lh @ W                                         # [B,H]
        s_core = (LW.unsqueeze(1) * Ch_full).sum(-1)        # [B,T]
        s_cav  = (self.b * Ch_full).sum(-1)                 # [B,T]
        s_lig  = (self.a * Lh).sum(-1, keepdim=True)        # [B,1]
        s = s_core + s_cav + s_lig / lengths.clamp(min=1).unsqueeze(1).to(L_raw.dtype)

        # mask/alpha
        idx = torch.arange(T, device=C_raw.device).unsqueeze(0).expand(B, T)
        mask = (idx < lengths.unsqueeze(1))
        alpha_prior = torch.softmax(s.masked_fill(~mask, -1e9) / self.tau, dim=1)

        feat = torch.cat([Lh.unsqueeze(1).expand(-1, T, -1), Ch_full], dim=-1)     # [B,T,2H]
        delta = self.delta(feat).squeeze(-1).masked_fill(~mask, 0.0)               # [B,T]
        alpha = torch.softmax((s + delta).masked_fill(~mask, -1e9) / self.tau, dim=1)

        # context + logit
        context = torch.bmm(alpha.unsqueeze(1), Ch_full).squeeze(1)                # [B,H]
        logit = self.head(torch.cat([Lh, context], dim=-1)).squeeze(1) + self.bias # [B]
        return (logit, alpha, alpha_prior) if return_alpha else logit


## Model load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WD(affinity) model
# The architecture and hyperparameters must match those used during training
def load_wd_model(ckpt_path, proj_dim=64, rank=10):
    ck = torch.load(ckpt_path, map_location="cpu")
    mu_L, sigma_L = ck["mu_L"], ck["sigma_L"]
    mu_C, sigma_C = ck["mu_C"], ck["sigma_C"]

    wd = BilinearAffinityZ_MLP_LR(
        lig_dim=11, cav_dim=37, proj_dim=proj_dim, rank=rank,
        mu_L=mu_L, sigma_L=sigma_L, mu_C=mu_C, sigma_C=sigma_C
    )
    wd.load_state_dict(ck["state_dict"], strict=True)
    wd.to(device).eval()
    return wd, ck["y_mean"], ck["y_std"]

# AttnBind classification model
# The checkpoint is assumed to contain the trained state_dict and optionally normalization statistics
def load_attn_model(ckpt_path, seed_ckpt_path=None, proj_dim=64, rank=10, tau=1.0):
    attn_ck = torch.load(ckpt_path, map_location="cpu")

    if seed_ckpt_path is not None:
        # Copy normalization statistics from the WD seed model
        # (ensures consistency with the training setup)
        seed_ck = torch.load(seed_ckpt_path, map_location="cpu")
        seed_model = BilinearAffinityZ_MLP_LR(
            lig_dim=11, cav_dim=37, proj_dim=proj_dim, rank=rank,
            mu_L=seed_ck["mu_L"], sigma_L=seed_ck["sigma_L"],
            mu_C=seed_ck["mu_C"], sigma_C=seed_ck["sigma_C"]
        )
        seed_model.load_state_dict(seed_ck["state_dict"], strict=True)
        model = AttnBind(seed_model, tau=tau)   # 통계/프로젝션/저랭크 가중치 시드 복제
    else:
        # Initialize AttnBind using parameters from the WD model
        model = AttnBind_from_stats(
            mu_L=attn_ck["mu_L"], sigma_L=attn_ck["sigma_L"],
            mu_C=attn_ck["mu_C"], sigma_C=attn_ck["sigma_C"],
            proj_dim=proj_dim, rank=rank, tau=tau
        )

    model.load_state_dict(attn_ck["state_dict"], strict=False)
    model.to(device).eval()
    return model

# Load
wd_ckpt   = "bilinear_mise2_best.pt"        # WD affinity regression model
attn_ckpt = "attnbind_cls_best.pt"          # AttnBind classification model

seed_model, y_mean, y_std = load_wd_model(wd_ckpt, proj_dim=64, rank=10)
attn = load_attn_model(attn_ckpt, seed_ckpt_path=wd_ckpt, proj_dim=64, rank=10, tau=1.0)


## Inference using existing feature maps
pid = "4ty7"  # Change to any sample you want to inspect
print("[PID]", pid)

# 11D ligand, 37D cavity
L_raw = ligand_raw_map[pid]               # [11]
C_raw = cavity_aug_map[pid]               # [T,37]  ← 중요: 이미 만들어둔 37D 맵
T = C_raw.size(0)

L_b   = L_raw.unsqueeze(0).to(device)     # [1,11]
C_b   = C_raw.unsqueeze(0).to(device)     # [1,T,37]
len_b = torch.tensor([T], dtype=torch.long, device=device)

# Defensive checks for dimensional consistency
assert C_b.ndim == 3 and C_b.size(-1) == 37, f"C_b shape={C_b.shape}"
assert getattr(attn, "mu_C").numel() == 37, "attn.mu_C dim mismatch"
assert getattr(seed_model, "mu_C").numel() == 37, "wd.mu_C dim mismatch"

# Inspect z-score distribution (debugging purpose)
with torch.no_grad():
    Lz = (L_b - attn.mu_L) / attn.sigma_L
    Cz = (C_b - attn.mu_C) / attn.sigma_C
print("Lz mean/std:", float(Lz.mean()), float(Lz.std()))
print("Cz mean/std:", float(Cz.mean()), float(Cz.std()))

# WD affinity regression prediction
seed_model.eval()
with torch.no_grad():
    pred_aff = seed_model(L_b, C_b, len_b)
print(f"[WD predicted affinity] {float(pred_aff):.4f}")

# AttnBind classification prediction
attn.eval()
with torch.no_grad():
    logit, alpha, alpha_prior = attn(L_b, C_b, len_b, return_alpha=True)
    prob = torch.sigmoid(logit).item()
print(f"[Attn predicted binding prob] {prob:.4f}")
print("alpha sum:", float(alpha[0,:T].sum()), "max:", float(alpha[0,:T].max()), "min:", float(alpha[0,:T].min()))

# Top-k residues ranked by attention weights
topk = min(5, T)
idxs = alpha[0,:T].detach().cpu().numpy().argsort()[-topk:][::-1]
for i in idxs:
    chain, resseq, icode, res3 = cavity_meta_map[pid][int(i)]
    icode_disp = icode if icode else ' '
    print(f"rank {i}: {res3} {chain}{resseq}{icode_disp} (alpha={alpha[0,i]:.4f})")

