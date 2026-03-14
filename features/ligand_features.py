import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors as rdmd

FEATURE_KEYS = [
    'HBD','HBA','RotBonds','AromRings',
    'Heteroatoms','TPSA','Csp3','FormalCharge',
    'MolWt','LabuteASA','DonorAcceptorRatio'
]

def compute_features(mol):
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

def safe_load_molecule(sdf_path, mol2_path):

    if sdf_path.exists():
        try:
            mol = Chem.MolFromMolFile(str(sdf_path), sanitize=True, removeHs=False)
            if mol is not None:
                return mol
        except:
            pass

    if mol2_path.exists():
        try:
            mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
            if mol is not None:
                return mol
        except:
            pass

    return None
