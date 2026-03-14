import torch

def build_cavity_aug_map(cavity_raw_map, C_xyz_map, L_center_map):

    cavity_aug_map = {}

    for pid, C_raw in cavity_raw_map.items():

        xyz = C_xyz_map[pid]
        ctr = L_center_map[pid]

        dxyz = xyz - ctr.unsqueeze(0)

        r = torch.linalg.norm(dxyz, dim=1, keepdim=True)

        C_aug = torch.cat([C_raw, dxyz, r], dim=1)

        cavity_aug_map[pid] = C_aug

    return cavity_aug_map
