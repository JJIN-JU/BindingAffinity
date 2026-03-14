import torch
import pandas as pd

# --- 라벨 세팅 ---
ligand_labels = [
    'HBD','HBA','RotBonds','AromRings','Heteroatoms','TPSA',
    'Csp3','FormalCharge','MolWt','LabuteASA','DonorAcceptorRatio'
]

AA_CODES = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
prop_labels_13 = [
    "Hydrophobicity","numHBD","numHBA","MolecularWeight","pI",
    "VdW_Volume","Polarity","Flexibility","SideChain_pKa",
    "Aromatic","Aliphatic","Charge","SolventAccessibility"
]
cavity_labels_33 = [f"AA_{a}" for a in AA_CODES] + prop_labels_13
cavity_labels_37 = cavity_labels_33 + ["dX","dY","dZ","r"]

def local_weff_11xD(attn_model, L_raw, C_raw):
    """
    L_raw: [11]
    C_raw: [T,33] 또는 [T,37]  (torch.Tensor)
    반환: (Weff[11,D], contrib_df[11 x D])
    """
    device = next(attn_model.parameters()).device
    attn_model.eval()

    T, D = C_raw.shape
    # 라벨 자동 선택
    if D == 33:
        cav_labels = cavity_labels_33
    elif D == 37:
        cav_labels = cavity_labels_37
    else:
        raise ValueError(f"지원하지 않는 cavity dim D={D} (예상: 33 또는 37)")

    # 평균 캐비티
    Cbar = C_raw.mean(0)                          # [D]

    # z-score
    Lz = (L_raw - attn_model.mu_L.cpu()) / attn_model.sigma_L.cpu()
    Cz = (Cbar  - attn_model.mu_C.cpu()[:D]) / attn_model.sigma_C.cpu()[:D]  # 버퍼가 37이면 33일 때 앞부분만 씀
    Lz = Lz.to(device); Cz = Cz.to(device)

    # 자코비안 계산 준비
    L_raw_ = L_raw.clone().detach().to(device).requires_grad_(True).unsqueeze(0)  # [1,11]
    Cbar_  = Cbar.clone().detach().to(device).requires_grad_(True).unsqueeze(0)   # [1,D]

    # proj 통과
    Lh = attn_model.lig_proj((L_raw_ - attn_model.mu_L)/attn_model.sigma_L)             # [1,H]
    # cav_proj의 in_features는 33 또는 37로 학습됨 → 현재 Cbar_의 D와 일치해야 함
    # (모델이 37D로 학습됐고 입력이 33D면, 여길 통과시키기 전에 37D로 패딩하거나
    #  모델도 33D 버전으로 써야 함. 보통은 C_raw를 37D로 넣는 게 안전)
    Ch = attn_model.cav_proj((Cbar_ - attn_model.mu_C[:D]) / attn_model.sigma_C[:D])     # [1,H]

    H = Lh.size(-1)

    # jac(Lh wrt L_raw): [H,11]
    JL = torch.zeros(H, L_raw_.size(-1), device=device)
    for h in range(H):
        grad = torch.autograd.grad(Lh[0, h], L_raw_, retain_graph=True)[0]  # [1,11]
        JL[h] = grad[0]

    # jac(Ch wrt Cbar): [H,D]
    JC = torch.zeros(H, Cbar_.size(-1), device=device)
    for h in range(H):
        grad = torch.autograd.grad(Ch[0, h], Cbar_, retain_graph=True)[0]   # [1,D]
        JC[h] = grad[0]

    # 프로젝션 공간 W = U V^T
    W_h = attn_model.U @ attn_model.V.T       # [H,H]

    # 입력공간 유효 W: J_L^T · W_h · J_C   → [11,D]
    Weff = JL.T @ W_h @ JC

    # 샘플 기여 행렬 (선형근사)
    contrib = (Lz[:, None] * Weff * Cz[None, :]).detach().cpu()

    df = pd.DataFrame(contrib.numpy(), index=ligand_labels, columns=cav_labels)
    return Weff.detach().cpu(), df

pid = "4ty7"

# 37D로 학습된 모델이라면:
C_raw = cavity_aug_map[pid]   # [T,37]  (dX,dY,dZ,r 포함)
# 33D 모델이라면:
# C_raw = cavity_raw_map[pid] # [T,33]

L_raw = ligand_raw_map[pid]   # [11]

Weff, contrib_df = local_weff_11xD(attn, L_raw, C_raw)
contrib_df.to_csv(f"2_{pid}_contrib_heatmap_11x{C_raw.size(1)}.csv")
print("saved:", f"{pid}_contrib_heatmap_11x{C_raw.size(1)}.csv")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ligand_labels = [
    'HBD','HBA','RotBonds','AromRings','Heteroatoms','TPSA',
    'Csp3','FormalCharge','MolWt','LabuteASA','DonorAcceptorRatio'
]

prop_labels = [
    "Hydrophobicity","numHBD","numHBA","MolecularWeight","pI",
    "VdW_Volume","Polarity","Flexibility","SideChain_pKa",
    "Aromatic","Aliphatic","Charge","SolventAccessibility"
]

def save_prop_only_heatmap(contrib_11x13, lig_labels, prop_labels,
                           out_png=None, vmin=-5, vmax=5, annotate=True):
    arr = np.asarray(contrib_11x13)  # shape [11,13]
    plt.figure(figsize=(16,6))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)  # 0을 흰색
    ax = sns.heatmap(
        arr, annot=annotate, fmt=".1f",
        cmap="bwr", norm=norm,
        cbar_kws={"label": "Contribution"},
        linewidths=0.6, linecolor="lightgray",
        xticklabels=prop_labels, yticklabels=lig_labels
    )
    ax.set_title("Ligand vs. Protein-property contribution (sample-specific)", fontsize=16)
    plt.xticks(rotation=60, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    if out_png: plt.savefig(out_png, dpi=200)
    plt.show()

# CSV 로드하고 prop 컬럼만 자르기 (열 20~32가 prop이라고 가정)
df = pd.read_csv("2_4ty7_contrib_heatmap_11x37.csv", index_col=0)
prop_start_idx = 20
arr_prop = df.iloc[:, prop_start_idx:].values  # [11,13]

# ★ 라벨은 CSV에서 가져오지 말고, 우리가 정의한 걸 그대로 사용
save_prop_only_heatmap(
    arr_prop,
    lig_labels=ligand_labels,
    prop_labels=prop_labels,
    out_png="1g2k_prop_only_heatmap_annot.png",
    vmin=-5, vmax=5,
    annotate=True
)

import csv
import torch

# seed_model = BilinearAffinityZ_MLP_LR(...).load_state_dict(...).eval().to(device)
# ligand_raw_map, cavity_raw_map, (선택) y_true_dict 준비되었다고 가정
out_csv = "casf_pred_affinity.csv"
rows = []
seed_model.eval()

with torch.no_grad():
    for pid in sorted(set(ligand_raw_map) & set(cavity_raw_map)):
        L_raw = ligand_raw_map[pid].unsqueeze(0).to(device)   # [1,11]
        C_raw = cavity_raw_map[pid].unsqueeze(0).to(device)   # [1,T,33]
        lengths = torch.tensor([C_raw.size(1)], device=device)
        y_pred = seed_model(L_raw, C_raw, lengths).item()
        rows.append({"pid": pid, "pred_affinity": y_pred})

# 저장
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["pid","pred_affinity"])
    writer.writeheader()
    writer.writerows(rows)

print(f"saved: {out_csv}  (n={len(rows)})")

