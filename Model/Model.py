import pandas as pd
import numpy as np
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from Bio.PDB import PDBParser
from Bio.PDB import NeighborSearch


class AffinityPredictor_before_WD(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim))  # [128, 128]
        self.bias = nn.Parameter(torch.zeros(1))      # scalar

    def forward(self, L_embed, C_embed):
        """
        L_embed: [B, d] or [d]
        C_embed: [n_res, d]
        """
        if L_embed.dim() == 1:
            L_embed = L_embed.unsqueeze(0)  # [1, d]

        # W @ C.T → [d, n_res]
        WC = torch.matmul(self.W, C_embed.T)  # [d, n_res]
        # L @ WC → [B, n_res]
        interaction_scores = torch.matmul(L_embed, WC)  # [B, n_res]

        # 평균 pooling → binding affinity
        S = interaction_scores.mean(dim=1) + self.bias  # [B]
        return S

# --------------------------
# using weighted model
# --------------------------
class EnhancedAffinity(nn.Module):
    def __init__(
        self,
        lig_dim=1039, cav_dim=35,
        hid=128, heads=4,
        lig15_dim=15, ecfp_dim=1024,
        dropout=0.1,
        mu_L15=None, sigma_L15=None,   # ← 15D만
        mu_C=None, sigma_C=None
    ):
        super().__init__()
        # --- z-score ---
        self.register_buffer("mu_L15", mu_L15.clone().float())
        self.register_buffer("sigma_L15", sigma_L15.clone().float().clamp_min(1e-6))
        self.register_buffer("mu_C", mu_C.clone().float())
        self.register_buffer("sigma_C", sigma_C.clone().float().clamp_min(1e-6))

        # --- Ligand towers ---
        self.lig15 = nn.Sequential(
            nn.Linear(lig15_dim, hid),
            nn.ReLU(), nn.LayerNorm(hid), nn.Dropout(dropout),
        )
        self.ligfp = nn.Sequential(
            nn.Linear(ecfp_dim, hid),
            nn.ReLU(), nn.LayerNorm(hid), nn.Dropout(dropout),
        )
        self.lig_fuse = nn.Sequential(
            nn.Linear(2*hid, hid),
            nn.ReLU(), nn.LayerNorm(hid), nn.Dropout(dropout),
        )

        # --- Cavity tower (per-residue → self-attn) ---
        self.cav_proj = nn.Sequential(
            nn.Linear(cav_dim, hid),
            nn.ReLU(), nn.LayerNorm(hid), nn.Dropout(dropout),
        )
        self.self_attn = nn.MultiheadAttention(embed_dim=hid, num_heads=heads, batch_first=True, dropout=dropout)
        self.cav_ff = nn.Sequential(  # transformer-style FFN
            nn.Linear(hid, 4*hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4*hid, hid), nn.LayerNorm(hid),
        )

        # --- Bilinear for WD term ---
        rank = 32
        self.U = nn.Parameter(torch.empty(hid, rank))
        self.V = nn.Parameter(torch.empty(hid, rank))
        nn.init.xavier_uniform_(self.U); nn.init.xavier_uniform_(self.V)

        # main effects
        self.a = nn.Parameter(torch.zeros(hid))
        self.b = nn.Parameter(torch.zeros(hid))
        self.bias = nn.Parameter(torch.zeros(1))

        # --- Interaction head (concat) ---
        # concat: [Lh, Ch, Lh*Ch, |Lh-Ch|, bilinear_scalar]
        self.head = nn.Sequential(
            nn.Linear(4*hid + 1, 2*hid),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2*hid, hid//2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid//2, 1)
        )

        self.res_drop_p = 0.15

    def forward(self, L_raw, C_raw, lengths):
        """
        L_raw: [B, 1039] = [15D props | 1024D ECFP]
        C_raw: [B, T, 35]
        """
        B, T, D = C_raw.shape

        # split
        L15 = L_raw[:, :15]      
        Lfp = L_raw[:, 15:]      # ECFP (0/1 or sparse float)

        # 15D만 z-score
        L15z = (L15 - self.mu_L15) / self.sigma_L15
        # cavity (all feature)
        Cz   = (C_raw - self.mu_C) / self.sigma_C

        # ligand towers
        L15h = self.lig15(L15z)
        Lfph = self.ligfp(Lfp)
        Lh   = self.lig_fuse(torch.cat([L15h, Lfph], dim=1))  # [B,hid]

        # cavity contextualization
        Ch0 = self.cav_proj(Cz.reshape(-1, D)).view(B, T, -1)

        idxs = torch.arange(T, device=C_raw.device).unsqueeze(0).expand(B, T)
        key_padding_mask = ~(idxs < lengths.unsqueeze(1))  # True=mask

        Ch_ctx, _ = self.self_attn(Ch0, Ch0, Ch0, key_padding_mask=key_padding_mask)
        Ch_ctx = Ch0 + Ch_ctx
        Ch_ctx = Ch_ctx + self.cav_ff(Ch_ctx)

        W  = self.U @ self.V.T
        LW = Lh @ W
        s  = (LW.unsqueeze(1) * Ch_ctx).sum(-1) + (self.b * Ch_ctx).sum(-1)

        if self.training and self.res_drop_p > 0:
            B, T = key_padding_mask.shape
            keep = torch.rand(B, T, device=key_padding_mask.device) > self.res_drop_p
            keep = keep & (~key_padding_mask)
            # at least 1
            first_valid = torch.zeros_like(keep); first_valid[:,0] = True
            keep = keep | first_valid
            key_padding_mask = key_padding_mask | (~keep)

        s = s.masked_fill(key_padding_mask, -1e9)
        alpha = torch.softmax(s, dim=1)
        Ch = (alpha.unsqueeze(2) * Ch_ctx).sum(1)

        bilinear = (LW * Ch).sum(1, keepdim=True)
        main = (Lh * self.a).sum(1, keepdim=True) + (Ch * self.b).sum(1, keepdim=True) + self.bias

        inter = torch.cat([Lh, Ch, Lh*Ch, torch.abs(Lh-Ch), bilinear], dim=1)
        out = self.head(inter) + main
        return out.view(-1)

  import torch, numpy as np, pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# === Load Checkpoint ===
ck = torch.load("enhanced_affinity_cosine_warmup_best2.pt", map_location="cpu")

mu_L15, sigma_L15 = ck["mu_L15"], ck["sigma_L15"]
mu_C,   sigma_C   = ck["mu_C"],   ck["sigma_C"]
y_mean, y_std     = ck["y_mean"], ck["y_std"]

# === Model generation/load/moving device ===
aff_model = EnhancedAffinity(
    lig_dim=1039,           # 15D + ECFP1024
    cav_dim=35, hid=128, heads=4, dropout=0.1,
    mu_L15=mu_L15, sigma_L15=sigma_L15,
    mu_C=mu_C,   sigma_C=sigma_C
)
aff_model.load_state_dict(ck["state_dict"], strict=True)
aff_model.to(device).eval()

# fix buffer device
for name, buf in aff_model.named_buffers():
    setattr(aff_model, name, buf.to(device))

# === moving input tensor device ===
assert X_L.shape[1] in (1039, 2063), f"lig_dim mismatch: got {X_L.shape[1]}"
X_L = X_L.to(device)                   # [N, 1039] (또는 2063)
C_raw_target = C_raw_target.to(device) # [T, 35]
T = C_raw_target.size(0)

# name length checking
assert len(ligand_names) == X_L.size(0), "names length != X_L N"

# === inference batch ===
batch_size = 512
preds = []

with torch.no_grad():
    for s in range(0, X_L.size(0), batch_size):
        e = min(s + batch_size, X_L.size(0))
        Lb = X_L[s:e]  # [B, 1039]
        Cb = C_raw_target.unsqueeze(0).repeat(e - s, 1, 1)  # [B, T, 35]
        lengths = torch.full((e - s,), T, device=device, dtype=torch.long)

        y_hat = aff_model(Lb, Cb, lengths)     
        y_pred = y_hat * y_std + y_mean        
        preds.append(y_pred.detach().cpu())   
        
preds_t = torch.cat(preds, dim=0)   # [N] (CPU)
print("preds_t shape:", preds_t.shape)

# === Save result to CSV ===
df = pd.DataFrame({
    "Ligand": ligand_names,
    "Predicted_affinity": preds_t.numpy()
})
out_csv = "/home/yejin/affinity/drugbank_predictions.csv"
df.to_csv(out_csv, index=False)
print(f"Compelte save: {out_csv}")

# 정렬본
df_sorted = df.sort_values("Predicted_affinity", ascending=False).reset_index(drop=True)
out_csv_sorted = "/home/yejin/affinity/drugbank_predictions_sorted.csv"
df_sorted.to_csv(out_csv_sorted, index=False)
print(f"Compelte save sorted: {out_csv_sorted}")

# === Printing Top-K ===
topk = 50
order = torch.argsort(preds_t, descending=True)
print("\nTop-{} predictions:".format(topk))
for i in order[:topk].tolist():
    print(f"{ligand_names[i]}\tscore={preds_t[i].item():.4f}")

class AttnBindV2(nn.Module):
    def __init__(self, enhanced_seed, tau=0.5):
        super().__init__()
        import copy

        self.register_buffer("mu_L15",   enhanced_seed.mu_L15.clone())
        self.register_buffer("sigma_L15",enhanced_seed.sigma_L15.clone())
        self.register_buffer("mu_C",     enhanced_seed.mu_C.clone())
        self.register_buffer("sigma_C",  enhanced_seed.sigma_C.clone())


        self.lig15   = copy.deepcopy(enhanced_seed.lig15)
        self.ligfp   = copy.deepcopy(enhanced_seed.ligfp)
        self.lig_fuse= copy.deepcopy(enhanced_seed.lig_fuse)
        self.cav_proj= copy.deepcopy(enhanced_seed.cav_proj)
        self.self_attn = copy.deepcopy(enhanced_seed.self_attn)
        self.cav_ff    = copy.deepcopy(enhanced_seed.cav_ff)


        self.U = nn.Parameter(enhanced_seed.U.detach().clone())
        self.V = nn.Parameter(enhanced_seed.V.detach().clone())
        self.a = nn.Parameter(enhanced_seed.a.detach().clone())
        self.b = nn.Parameter(enhanced_seed.b.detach().clone())
        self.bias = nn.Parameter(enhanced_seed.bias.detach().clone())

        hid = self.U.size(0)
        self.tau = tau


        self.head = nn.Sequential(
            nn.Linear(2*hid, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.delta = nn.Sequential(
            nn.Linear(2*hid, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, L_raw, C_raw, lengths, return_alpha=False):
        B, T, D = C_raw.shape  # D=35

        # ---- ligand (15D z-score + ECFP) ----
        L15 = L_raw[:, :15]
        Lfp = L_raw[:, 15:]
        L15z = (L15 - self.mu_L15) / self.sigma_L15
        L15h = self.lig15(L15z)
        Lfph = self.ligfp(Lfp)
        Lh   = self.lig_fuse(torch.cat([L15h, Lfph], dim=1))     # [B,hid]

        # ---- cavity (z-score per-residue → self-attn) ----
        Cz = (C_raw - self.mu_C) / self.sigma_C
        Ch0 = self.cav_proj(Cz.reshape(-1, D)).view(B, T, -1)
        idx = torch.arange(T, device=C_raw.device).unsqueeze(0).expand(B, T)
        key_padding_mask = ~(idx < lengths.unsqueeze(1))  # True=pad
        Ch_ctx, _ = self.self_attn(Ch0, Ch0, Ch0, key_padding_mask=key_padding_mask)
        Ch_ctx = Ch0 + Ch_ctx
        Ch_ctx = Ch_ctx + self.cav_ff(Ch_ctx)

        # ---- WD score s_t (prior) ----
        W  = self.U @ self.V.T
        LW = Lh @ W
        s_core = (LW.unsqueeze(1) * Ch_ctx).sum(-1)
        s_cav  = (self.b * Ch_ctx).sum(-1)
        s_lig  = (self.a * Lh).sum(-1, keepdim=True)
        s = s_core + s_cav + s_lig/lengths.clamp(min=1).unsqueeze(1).to(L_raw.dtype)

        # ---- alpha with learnable delta ----
        feat = torch.cat([Lh.unsqueeze(1).expand(-1, T, -1), Ch_ctx], dim=-1)  # [B,T,2hid]
        delta = self.delta(feat).squeeze(-1).masked_fill(key_padding_mask, 0.0)
        logits = (s + delta).masked_fill(key_padding_mask, -1e9) / self.tau
        alpha = torch.softmax(logits, dim=1)

        # ---- context → bag logit ----
        context = torch.bmm(alpha.unsqueeze(1), Ch_ctx).squeeze(1)
        logit = self.head(torch.cat([Lh, context], dim=-1)).squeeze(1) + self.bias  # [B]

        return (logit, alpha) if return_alpha else logit

)
