"""
### Protein Pocket Representation

Protein binding pockets were represented using motif-based residue selection.

For each target protein structure, key pocket residues were identified based on a conserved binding motif (e.g., GLY–ASP–ASN–GLU). These residues were used to define the functional binding cavity.

Each residue was encoded using a 35-dimensional feature vector including:

- Amino acid identity (one-hot encoding)
- Hydrophobicity (Kyte–Doolittle scale)
- Side-chain pKa
- Aromatic and aliphatic flags
- Estimated charge at pH 7
- Residue mass and van der Waals volume
- Atomic polarity fraction
- B-factor
- Solvent accessible surface area (SASA)
- Relative spatial coordinates within the cavity

*35-dimensional : 20 (AA one-hot) + 5 physicochemical properties + 10 structural features

The residue features were projected into a 128-dimensional embedding space using a feed-forward neural layer.

"""

import pandas as pd
import numpy as np
import csv
import os
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import math, numpy as np, torch
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from Bio.PDB import PDBParser
from Bio.PDB import NeighborSearch

### Protein embedding ###

from Bio.PDB import PDBParser

pdb_path = "path/to/target/protein.pdb"
parser = PDBParser(QUIET=True)
structure = parser.get_structure("prot", pdb_path)

target_motif = ["GLY", "ASP", "ASN", "GLU"] #binding pocket motif

for chain in structure[0]:
    residues = list(chain.get_residues())
    for i in range(len(residues) - 3):
        window = residues[i:i+4]
        names = [res.get_resname() for res in window]
        if names == target_motif:
            print(f"Found motif in chain {chain.id}")
            for j, res in enumerate(window):
                print(f"  {j+1}: {res.get_resname()} - Chain: {chain.id}, PDB residue number: {res.get_id()[1]}")



# Standard biochemical constants used for residue feature computation
# Includes Kyte–Doolittle hydrophobicity, side-chain pKa values,
# maximum solvent-accessible surface area (Max ASA),
# and residue aromatic/aliphatic classifications.

KYTE_DOOLITTLE = {
    'A': 1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C': 2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
    'H':-3.2,'I': 4.5,'L': 3.8,'K':-3.9,'M': 1.9,'F': 2.8,'P':-1.6,'S':-0.8,
    'T':-0.7,'W':-0.9,'Y':-1.3,'V': 4.2
}
SIDECHAIN_PKA = {'D':3.9,'E':4.3,'H':6.0,'C':8.3,'Y':10.1,'K':10.5,'R':12.5}
AROMATIC = set(['F','Y','W','H'])
ALIPHATIC = set(['A','V','I','L','M'])

# Maximum solvent accessible surface area values (Å^2)
# Commonly used reference values reported in Tien et al. and Miller et al.
MAX_ASA = {
    'A':129,'R':274,'N':195,'D':193,'C':167,'Q':225,'E':223,'G':104,'H':224,'I':197,
    'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174
}

# Atomic masses and van der Waals radii 
# Used for approximate residue mass and volume estimation
# (spherical approximation; overlap between atoms is ignored)
ATOMIC_MASS = {"H":1.008,"C":12.011,"N":14.007,"O":15.999,"S":32.06,"P":30.974,
               "F":18.998,"Cl":35.45,"Br":79.904,"I":126.90,"Se":78.971}
VDW_RADIUS  = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"S":1.80,"P":1.80,"F":1.47,
               "Cl":1.75,"Br":1.85,"I":1.98,"Se":1.90}

AA3_TO_1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
            'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
            'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
AA_CODES = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']


# Estimate side-chain charge at physiological pH (~7)
# using the Henderson–Hasselbalch approximation
def charge_at_pH7(aa, pH=7.0):
    if aa not in SIDECHAIN_PKA:
        return 0.0
    pKa = SIDECHAIN_PKA[aa]
    if aa in {'D','E','C','Y'}:  # 산성
        frac_deprot = 1.0 / (1.0 + 10**(pKa - pH))  # 음전하 비율
        return -frac_deprot
    elif aa in {'K','R','H'}:    # 염기성
        frac_prot   = 1.0 / (1.0 + 10**(pH - pKa))  # 양전하 비율
        return +frac_prot
    return 0.0

# Structure-based helper functions for residue feature computation
def residue_atom_array(residue, include_h=False):
    coords, elems, bvals = [], [], []
    for atom in residue.get_atoms():
        if not include_h and atom.element == 'H':
            continue
        coords.append(atom.get_coord())
        e = atom.element if atom.element in ATOMIC_MASS else atom.element.capitalize()
        elems.append(e)
        bvals.append(atom.get_bfactor())
    if not coords:
        return None, None, None
    return np.vstack(coords), elems, np.array(bvals, dtype=float)

def centroid(residue):
    arr, _, _ = residue_atom_array(residue, include_h=False)
    return None if arr is None else arr.mean(axis=0)

def mass(residue):
    _, elems, _ = residue_atom_array(residue, include_h=True)
    if elems is None: return 0.0
    return float(sum(ATOMIC_MASS.get(e, 0.0) for e in elems))

def vdw_volume(residue):
    _, elems, _ = residue_atom_array(residue, include_h=False)
    if elems is None: return 0.0
    vol = 0.0
    for e in elems:
        r = VDW_RADIUS.get(e)
        if r: vol += (4.0/3.0)*math.pi*(r**3)
    return float(vol)

def polarity_fraction(residue):
    _, elems, _ = residue_atom_array(residue, include_h=False)
    if not elems: return 0.0
    pol = sum(1 for e in elems if e in ('N','O','S'))
    return float(pol)/float(len(elems))

def bfactor_mean(residue):
    _, _, b = residue_atom_array(residue, include_h=False)
    if b is None or len(b)==0: return 0.0
    return float(np.nanmean(b))

# Compute residue-level solvent accessible surface area (SASA)
# using the Shrake–Rupley algorithm implemented in Biopython
def sasa_residue_map_biopython(structure):
    from Bio.PDB.SASA import ShrakeRupley
    sr = ShrakeRupley()                  # 기본: probe_radius=1.4Å, n_points=100
    sr.compute(structure, level='R')     # residue 레벨로 계산하여 res.xtra['EXP_ACC'] 채움
    sasa_map = {}
    for ch in structure[0]:
        for res in ch:
            if res.get_id()[0] != ' ':   # HETATM 제외
                continue
            key = (ch.id, res.get_id())  # (chain_id, (' ', seqnum, icode))
            sasa_map[key] = float(res.xtra.get('EXP_ACC', 0.0))
    return sasa_map


# Assemble the residue-level feature vector combining physicochemical descriptors and structural features
#    one-hot(20)
#    + [hydrophobicity, sidechain_pKa_or0, aromatic(0/1), aliphatic(0/1), charge_pH7]
#    + [mass, vdw_volume, polarity_frac, bfactor_mean, SASA, RSA, rel_x, rel_y, rel_z, dist]

def residue_feature_vector(residue, cavity_center, pH=7.0, structure=None, precomputed_sasa=None):
    res3 = residue.get_resname().strip()
    aa   = AA3_TO_1.get(res3)
    if aa is None or aa not in AA_CODES:
        return None

    onehot = [1 if aa==x else 0 for x in AA_CODES]
    hydrop = KYTE_DOOLITTLE[aa]
    pka_sc = float(SIDECHAIN_PKA.get(aa, 0.0))
    arom   = 1.0 if aa in AROMATIC else 0.0
    aliph  = 1.0 if aa in ALIPHATIC else 0.0
    ch7    = charge_at_pH7(aa, pH=pH)

    ctr = centroid(residue)
    if ctr is None:
        return None
    rel = ctr - cavity_center
    dist = float(np.linalg.norm(rel))
    m    = mass(residue)
    vdwv = vdw_volume(residue)
    pfrac= polarity_fraction(residue)
    bavg = bfactor_mean(residue)

    sasa = 0.0
    if precomputed_sasa is not None and structure is not None:
        key = (residue.get_parent().id, residue.get_id())
        sasa = float(precomputed_sasa.get(key, 0.0))

    rsa = 0.0
    if aa in MAX_ASA and MAX_ASA[aa] > 0:
        rsa = min(max(sasa / float(MAX_ASA[aa]), 0.0), 1.0)

    feats = (
        onehot
        + [hydrop, pka_sc, arom, aliph, ch7]
        + [m, vdwv, pfrac, bavg, sasa, rsa, float(rel[0]), float(rel[1]), float(rel[2]), dist]
    )
    return torch.tensor(feats, dtype=torch.float32)


# Build cavity feature matrix.
# The cavity center is defined as the mean centroid of selected residues.
def build_cavity_features(pdb_path, chain_id, residue_numbers, pH=7.0, use_sasa=True):
    parser   = PDBParser(QUIET=True)
    structure= parser.get_structure("prot", pdb_path)
    model    = structure[0]
    if chain_id not in [ch.id for ch in model]:
        raise KeyError(f"There is no {chain_id} in PDB.")
    chain    = model[chain_id]

    picked, centers = [], []
    for res in chain:
        hetflag, seqnum, icode = res.get_id()
        if hetflag != ' ':
            continue
        if seqnum in residue_numbers:
            c = centroid(res)
            if c is not None:
                picked.append(res); centers.append(c)
    if not picked:
        raise ValueError("There is no residue where you choose residue_numbers.")

    cavity_center = np.vstack(centers).mean(axis=0)
    picked.sort(key=lambda r: r.get_id()[1])

    # calculate SASA using Biopython
    sasa_map = sasa_residue_map_biopython(structure) if use_sasa else None

    feats = []
    for res in picked:
        v = residue_feature_vector(
            res, cavity_center, pH=pH, structure=structure, precomputed_sasa=sasa_map
        )
        if v is not None:
            feats.append(v)

    C = torch.stack(feats, dim=0)  # [p, 35]
    return C, picked, cavity_center

C_raw_target, residues, ctr = build_cavity_features(
    pdb_path="/home/yejin/Downloads/9cgi.pdb",
    chain_id="A",
    residue_numbers=[831,832,833,834], # Specify residue numbers that define the target binding pocket motif
    pH=7.0,
    use_sasa=True  # Set to True to compute SASA using the Shrake–Rupley algorithm
)

residue_numbers=[831,832,833,834] # Specify residue numbers that define the target binding pocket motif

C=C_raw_target

import os, json, torch
from datetime import datetime

def save_cavity_cache(pdb_id, chain_id, residue_numbers, C, residues, cavity_center,
                      out_dir="./cache_cavity", use_sasa=False, schema_version="cavity35_v1"):
    """
    C: torch.Tensor [p, d]
    residues: Bio.PDB residue list
    cavity_center: np.array(3,)
    """
    os.makedirs(out_dir, exist_ok=True)

    # file name: <PDBID>_<CHAIN>_<minRes>-<maxRes>_<schema>.pt
    tag = f"{min(residue_numbers)}-{max(residue_numbers)}" if residue_numbers else "custom"
    fname = f"{pdb_id}_{chain_id}_{tag}_{schema_version}.pt"
    fpath = os.path.join(out_dir, fname)
    tmp_path = fpath + ".tmp"

    # Residue metadata used to verify correct pocket matching when loading cache
    res_meta = []
    for r in residues:
        hetflag, seqnum, icode = r.get_id()
        res_meta.append({
            "resname3": r.get_resname().strip(),
            "seqnum": int(seqnum),
            "icode": icode.strip() if isinstance(icode, str) else str(icode),
            "hetflag": hetflag.strip() if isinstance(hetflag, str) else str(hetflag),
            "chain": r.get_parent().id
        })

    payload = {
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "residue_numbers": list(map(int, residue_numbers)),
        "schema_version": schema_version,      
        "use_sasa": bool(use_sasa),
        "shape": tuple(C.shape),
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "cavity_center": cavity_center.astype(float),  
        "residues": res_meta,
        "C": C.cpu(),  # GPU 텐서일 수 있으니 CPU로 저장
    }

    torch.save(payload, tmp_path)
    os.replace(tmp_path, fpath)

    return fpath

cache_path = save_cavity_cache(
    pdb_id="9cgi",
    chain_id="A",
    residue_numbers=[831,832,833,834],
    C=C,
    residues=residues,
    cavity_center=ctr,
    out_dir="path/to/save/cavity_cache",
    use_sasa=True,
    schema_version="cavity35_v1"
)
print("Saved cavity cache →", cache_path)

class ProteinEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(0.1),
        )
    def forward(self, x):
        return self.net(x)

prot_embedder = ProteinEmbedding(input_dim=C.shape[1], embed_dim=128)

C_embed = prot_embedder(C)   # [n_res, 128]
print("C:", C.shape, "C_embed:", C_embed.shape)
