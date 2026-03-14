import torch

AA_CODES = ['A','R','N','D','C','Q','E','G',
            'H','I','L','K','M','F','P','S',
            'T','W','Y','V']

AA_PROPERTIES = {
    'A':[1.8,0,0,89.1,6.01,67,0,1,-1,0,1,0,45],
    'R':[-4.5,1,3,174.2,10.76,148,1,0,12.5,0,1,1,95],
    'N':[-3.5,1,1,132.1,5.41,96,1,1,-1,0,1,0,90],
    'D':[-3.5,0,2,133.1,2.77,91,1,1,3.9,0,1,-1,95],
    'C':[2.5,0,1,121.2,5.07,86,0,0,8.3,0,1,0,20],
}

three_to_one = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D',
    'CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
    'HIS':'H','ILE':'I','LEU':'L','LYS':'K',
    'MET':'M','PHE':'F','PRO':'P','SER':'S',
    'THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

def get_residue_feature(residue):

    resname = residue.get_resname().strip()

    if resname not in three_to_one:
        return None

    aa = three_to_one[resname]

    if aa not in AA_PROPERTIES:
        return None

    onehot = [int(aa == x) for x in AA_CODES]
    props = AA_PROPERTIES[aa]

    return torch.tensor(onehot + props, dtype=torch.float32)
