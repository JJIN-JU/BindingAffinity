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
# 가중치 학습 모델 사용
# --------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # --- z-score 통계 버퍼 (15D만) ---
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
        # 결합 후 얇게 정리
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

        # --- Bilinear (저랭크) for WD term ---
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
        L15 = L_raw[:, :15]      # 연속 특성
        Lfp = L_raw[:, 15:]      # ECFP (0/1 또는 sparse float)

        # 15D만 z-score
        L15z = (L15 - self.mu_L15) / self.sigma_L15
        # 캐비티는 기존대로 전 특징 정규화
        Cz   = (C_raw - self.mu_C) / self.sigma_C

        # ligand towers
        L15h = self.lig15(L15z)
        Lfph = self.ligfp(Lfp)
        Lh   = self.lig_fuse(torch.cat([L15h, Lfph], dim=1))  # [B,hid]

        # cavity contextualization (기존과 동일)
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
            # 최소 1개는 유지
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

# === 0) 체크포인트 로드 ===
ck = torch.load("enhanced_affinity_cosine_warmup_best2.pt", map_location="cpu")

mu_L15, sigma_L15 = ck["mu_L15"], ck["sigma_L15"]
mu_C,   sigma_C   = ck["mu_C"],   ck["sigma_C"]
y_mean, y_std     = ck["y_mean"], ck["y_std"]

# === 1) 모델 생성/로드/디바이스 이동 ===
aff_model = EnhancedAffinity(
    lig_dim=1039,           # 15D + ECFP1024
    cav_dim=35, hid=128, heads=4, dropout=0.1,
    mu_L15=mu_L15, sigma_L15=sigma_L15,
    mu_C=mu_C,   sigma_C=sigma_C
)
aff_model.load_state_dict(ck["state_dict"], strict=True)
aff_model.to(device).eval()

# (안전망) 버퍼들도 디바이스 고정
for name, buf in aff_model.named_buffers():
    setattr(aff_model, name, buf.to(device))

# === 2) 입력 텐서 디바이스 이동 ===
assert X_L.shape[1] in (1039, 2063), f"lig_dim mismatch: got {X_L.shape[1]}"
X_L = X_L.to(device)                   # [N, 1039] (또는 2063)
C_raw_target = C_raw_target.to(device) # [T, 35]
T = C_raw_target.size(0)

# 이름 길이 체크
assert len(ligand_names) == X_L.size(0), "names length != X_L N"

# === 3) 배치 추론 ===
batch_size = 512
preds = []

with torch.no_grad():
    for s in range(0, X_L.size(0), batch_size):
        e = min(s + batch_size, X_L.size(0))
        Lb = X_L[s:e]  # [B, 1039]
        # C는 모든 배치 공통이므로 repeat
        Cb = C_raw_target.unsqueeze(0).repeat(e - s, 1, 1)  # [B, T, 35]
        lengths = torch.full((e - s,), T, device=device, dtype=torch.long)

        y_hat = aff_model(Lb, Cb, lengths)      # 정규화 공간
        y_pred = y_hat * y_std + y_mean         # 복원
        preds.append(y_pred.detach().cpu())     # CPU로 모으기

# 리스트[Tensors] -> 하나의 1D 텐서
preds_t = torch.cat(preds, dim=0)   # [N] (CPU)
print("preds_t shape:", preds_t.shape)

# === 4) CSV 저장 (원본 순서 & 정렬본 둘 다) ===
df = pd.DataFrame({
    "Ligand": ligand_names,
    "Predicted_affinity": preds_t.numpy()
})
out_csv = "/home/yejin/affinity/drugbank_predictions.csv"
df.to_csv(out_csv, index=False)
print(f"✅ 저장 완료: {out_csv}")

# 정렬본
df_sorted = df.sort_values("Predicted_affinity", ascending=False).reset_index(drop=True)
out_csv_sorted = "/home/yejin/affinity/drugbank_predictions_sorted.csv"
df_sorted.to_csv(out_csv_sorted, index=False)
print(f"✅ 정렬 저장 완료: {out_csv_sorted}")

# === 5) Top-K 출력 (토치 정렬 사용) ===
topk = 50
order = torch.argsort(preds_t, descending=True)
print("\nTop-{} predictions:".format(topk))
for i in order[:topk].tolist():
    print(f"{ligand_names[i]}\tscore={preds_t[i].item():.4f}")

class AttnBindV2(nn.Module):
    def __init__(self, enhanced_seed, tau=0.5):
        super().__init__()
        import copy
        # 통계
        self.register_buffer("mu_L15",   enhanced_seed.mu_L15.clone())
        self.register_buffer("sigma_L15",enhanced_seed.sigma_L15.clone())
        self.register_buffer("mu_C",     enhanced_seed.mu_C.clone())
        self.register_buffer("sigma_C",  enhanced_seed.sigma_C.clone())

        # 리간드 타워 / 캐비티 타워 복제
        self.lig15   = copy.deepcopy(enhanced_seed.lig15)
        self.ligfp   = copy.deepcopy(enhanced_seed.ligfp)
        self.lig_fuse= copy.deepcopy(enhanced_seed.lig_fuse)
        self.cav_proj= copy.deepcopy(enhanced_seed.cav_proj)
        self.self_attn = copy.deepcopy(enhanced_seed.self_attn)
        self.cav_ff    = copy.deepcopy(enhanced_seed.cav_ff)

        # WD 파라미터 복제
        self.U = nn.Parameter(enhanced_seed.U.detach().clone())
        self.V = nn.Parameter(enhanced_seed.V.detach().clone())
        self.a = nn.Parameter(enhanced_seed.a.detach().clone())
        self.b = nn.Parameter(enhanced_seed.b.detach().clone())
        self.bias = nn.Parameter(enhanced_seed.bias.detach().clone())

        hid = self.U.size(0)
        self.tau = tau

        # 분류 헤드/델타
        self.head = nn.Sequential(
            nn.Linear(2*hid, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.delta = nn.Sequential(
            nn.Linear(2*hid, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, L_raw, C_raw, lengths, return_alpha=False):
        B, T, D = C_raw.shape  # D=35

        # ---- ligand (15D z-score + ECFP 그대로) ----
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
        
import torch

@torch.no_grad()
def compute_wd_scores(model, Lb, Cb, lengths=None):
    """
    모델 내부 모듈을 그대로 사용해 논문식 Weighted-Dot 원시 점수(S_raw)를 계산.
    return: S_raw [B] (리간드당 스칼라)
    필요 모듈/통계: model.mu_L15, model.sigma_L15, model.mu_C, model.sigma_C,
                   model.lig15, model.ligfp, model.lig_fuse,
                   model.cav_proj, model.self_attn, model.cav_ff, model.U, model.V
    """
    device = next(model.parameters()).device
    Lb = Lb.to(device); Cb = Cb.to(device)
    if lengths is not None: lengths = lengths.to(device)

    # --- ligand 임베딩 ---
    L15, Lfp = Lb[:, :15], Lb[:, 15:]                      # [B,15], [B,1024] 등
    L15z = (L15 - model.mu_L15) / model.sigma_L15
    L15h = model.lig15(L15z)                                # [B,h]
    Lfph = model.ligfp(Lfp)                                 # [B,h]
    Lh   = model.lig_fuse(torch.cat([L15h, Lfph], dim=1))   # [B,h]

    # --- cavity 컨텍스트 ---
    B, T, D = Cb.shape
    Cz   = (Cb - model.mu_C) / model.sigma_C
    Ch0  = model.cav_proj(Cz.reshape(-1, D)).view(B, T, -1)     # [B,T,h]

    if lengths is None:
        kpm = torch.zeros((B, T), dtype=torch.bool, device=device)   # pad 없음
    else:
        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        kpm = ~(idx < lengths.unsqueeze(1))  # True = pad

    Ch_ctx, _ = model.self_attn(Ch0, Ch0, Ch0, key_padding_mask=kpm)
    Ch_ctx = Ch0 + Ch_ctx
    Ch_ctx = Ch_ctx + model.cav_ff(Ch_ctx)                     # [B,T,h]

    # --- Weighted Dot (논문식) ---
    W  = model.U @ model.V.t()                                 # [h,h]
    LW = torch.matmul(Lh, W)                                    # [B,h]
    s_core = torch.einsum('bh,bth->bt', LW, Ch_ctx)             # [B,T]

    valid = ~kpm
    denom = valid.float().sum(1).clamp_min(1)
    S_raw = (s_core.masked_fill(~valid, 0.0).sum(1) / denom)    # [B]
    return S_raw
# --- EnhancedAffinity 체크포인트 로드 ---
ck = torch.load("enhanced_affinity_cosine_warmup_best2.pt", map_location="cpu")
y_mean, y_std     = ck["y_mean"], ck["y_std"]
mu_L15, sigma_L15 = ck["mu_L15"], ck["sigma_L15"]
mu_C,  sigma_C    = ck["mu_C"],   ck["sigma_C"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enh = EnhancedAffinity(
    lig_dim=1039, cav_dim=35, hid=128, heads=4, dropout=0.1,
    mu_L15=mu_L15, sigma_L15=sigma_L15,
    mu_C=mu_C, sigma_C=sigma_C
).to(device)
enh.load_state_dict(ck["state_dict"], strict=True)

# --- 분류 모델 정의 & 로드 ---
# --- 기존 그대로 ---
clf = AttnBindV2(enh).to(device)
ckb = torch.load("attnbind_bin_best.pt", map_location="cpu")
clf.load_state_dict(ckb["state_dict"])
clf.eval()

from joblib import load as joblib_load
platt = joblib_load("cal_platt.joblib")
iso   = joblib_load("cal_isotonic.joblib")

# --- 추론 ---
logits_list = []
wd_list     = []      # (NEW)
bs = 512

with torch.no_grad():
    for s in range(0, X_L.size(0), bs):
        e = min(s+bs, X_L.size(0))
        Lb = X_L[s:e].to(device)                                  # [B,1039]
        Cb = C_raw_target.unsqueeze(0).repeat(e-s,1,1).to(device) # [B,T,35]
        lengths = torch.full((e-s,), T, device=device, dtype=torch.long)

        # 기존 로짓
        logit = clf(Lb, Cb, lengths)              # [B]
        logits_list.append(logit.detach().cpu())

        # (NEW) WD 원시 점수
        S_raw = compute_wd_scores(clf, Lb, Cb, lengths)  # [B]
        wd_list.append(S_raw.detach().cpu())

# numpy 변환
import numpy as np, pandas as pd, torch
logits = torch.cat(logits_list).numpy().reshape(-1)
wd_raw = torch.cat(wd_list).numpy().reshape(-1)

# 기존 보정 확률
p_raw   = 1.0/(1.0 + np.exp(-logits))
p_platt = platt.predict_proba(logits.reshape(-1,1))[:,1]
p_iso   = iso.predict(p_raw)
p_final = p_platt

# (NEW) 논문식: WD → min–max → cutoff/랭킹
wd_min, wd_max = wd_raw.min(), wd_raw.max()
wd_norm = (wd_raw - wd_min) / (wd_max - wd_min + 1e-8)
wd_label = (wd_norm >= 0.5).astype(int)

# (옵션) 논문 스킨 범위로 매핑
CENTER, WIDTH = 0.5033, 0.0068
wd_mapped = CENTER + (WIDTH/2) - wd_norm * WIDTH

# 저장
df_prob = pd.DataFrame({
    "Ligand": ligand_names,
    "P_raw": p_raw, "P_platt": p_platt, "P_iso": p_iso, "P_final": p_final
})
df_prob.to_csv("/home/yejin/affinity/drugbank_classification_probs_calibrated.csv", index=False)

df_wd = pd.DataFrame({
    "Ligand": ligand_names,
    "WD_raw": wd_raw, "WD_norm": wd_norm, "Label_bin": wd_label, "Score_mapped": wd_mapped
}).sort_values("WD_norm", ascending=False).reset_index(drop=True)
df_wd.insert(0, "Rank", np.arange(1, len(df_wd)+1))
df_wd.to_csv("/home/yejin/affinity/wd_minmax_rank.csv", index=False)
df_wd.head(500).to_csv("/home/yejin/affinity/wd_minmax_top500.csv", index=False)

print("saved: calibrated probs + WD(min-max) rank + top500")
import torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FEATURE_KEYS = [
    'HBD','HBA','RotBonds','AromRings',
    'Heteroatoms','TPSA','Csp3','FormalCharge',
    'MolWt','LabuteASA','LogP','DonorAcceptorRatio',
    'AcidicEstimate','BasicEstimate','HydrophobicEstimate'
]

@torch.no_grad()
def heatmap_residue_by_ligfeat_multiC(
    X_L, C_raw_target, i,
    mu_L15, sigma_L15,
    cavity_feat_idx_map,              # 예: {"HBD_pot":0,"HBA_pot":1,"Hydrophobicity":2,"Charge":3,...}
    cavity_axes=None,                 # None이면 map 전부 사용, 아니면 ["Hydrophobicity","Charge",...]
    weights=None,                     # 축별 가중치 리스트(길이 = len(cavity_axes)); None이면 동일가중
    cav_norm="z",                     # "z" | "minmax" | None  (축별 정규화 방식)
    out_png="residue_x_ligfeat_multiC.png",
    out_csv="residue_x_ligfeat_multiC.csv",
    ligand_name="ligand",
    residue_numbers=None              # 예: [831,832,...]  (y축 라벨용)
):
    # CPU로 통일
    X = X_L.detach().cpu().float()
    C = C_raw_target.detach().cpu().float()
    T = C.size(0)

    # 사용할 cavity 축들
    if cavity_axes is None:
        cavity_axes = list(cavity_feat_idx_map.keys())
    idxs = [cavity_feat_idx_map[n] for n in cavity_axes]

    # 선택 축 행렬 [T, K]
    C_sel = torch.stack([C[:, j] for j in idxs], dim=1)  # [T, K]

    # 축별 정규화
    if cav_norm == "z":
        mu = C_sel.mean(dim=0, keepdim=True)
        sd = C_sel.std(dim=0, keepdim=True) + 1e-8
        Cn = (C_sel - mu) / sd
    elif cav_norm == "minmax":
        mn = C_sel.min(dim=0, keepdim=True).values
        mx = C_sel.max(dim=0, keepdim=True).values
        Cn = (C_sel - mn) / (mx - mn + 1e-8)
    else:
        Cn = C_sel

    # 가중치
    K = Cn.size(1)
    if weights is None:
        w = torch.ones(K) / K
    else:
        w = torch.as_tensor(weights, dtype=torch.float32)
        w = w / (w.sum() + 1e-8)
    # residue별 합성 스칼라 v_comb[r]
    v_comb = (Cn * w) .sum(dim=1)   # [T]

    # 리간드 15D z-score
    mu15 = torch.as_tensor(mu_L15[:15]).float()
    sg15 = torch.as_tensor(sigma_L15[:15]).float()
    lig15_z = (X[i, :15] - mu15) / (sg15 + 1e-8)        # [15]

    # 히트맵 행렬
    M = torch.outer(v_comb, lig15_z).numpy()            # [T, 15]

    # CSV 저장
    df = pd.DataFrame(M, columns=FEATURE_KEYS)
    if residue_numbers is not None:
        df.insert(0, "residue", [f"Res{n}" for n in residue_numbers])
    else:
        df.insert(0, "residue_idx", np.arange(T))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 그림
    H, W = M.shape
    vmin, vmax = np.nanmin(M), np.nanmax(M)
    mid = (vmin + vmax) / 2.0

    fig, ax = plt.subplots(figsize=(10, max(3, T/6)))
    im = ax.imshow(M, aspect='auto', vmin=vmin, vmax=vmax, cmap="viridis")
    cbar = plt.colorbar(im, ax=ax, label="Interaction score")

    # tick 라벨
    ax.set_xticks(range(W)); ax.set_xticklabels(FEATURE_KEYS, rotation=45, ha='right')
    if residue_numbers is not None:
        ax.set_yticks(range(T)); ax.set_yticklabels([f"Res{n}" for n in residue_numbers])
    else:
        ax.set_yticks(range(T)); ax.set_yticklabels(range(T))

    # --- grid (경계선) ---
    ax.set_xticks(np.arange(-.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    # --- 각 셀에 숫자 표시 ---
    for r in range(H):
        for c in range(W):
            val = M[r, c]
            txt_color = "white" if val > mid else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color)

    used = ", ".join(cavity_axes)
    ax.set_xlabel("Ligand Features"); ax.set_ylabel("Residues")
    ax.set_title(f"Residue × LigandFeatures — {ligand_name}\nCavity axes: {used}  |  norm={cav_norm}")

    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

    return out_png, out_csv
# 35D(raw) -> 8D(summary) 변환
import torch

AA_CODES = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

_DONOR_W = torch.tensor([  # HBD 잠재력(잔기 가중치)
    0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,1,1,1,1,0
], dtype=torch.float32)

_ACCEPT_W = torch.tensor([  # HBA 잠재력(잔기 가중치)
    0,0,1,1,1,1,1,0,1,0,0,0,0,0,0,1,1,0,1,0
], dtype=torch.float32)

def build_cavity_summary_axes(C_raw35: torch.Tensor):
    """
    C_raw35: [T,35] = one-hot20 + 15 scalars
    return:  C_axes [T,8], cav_idx dict
    """
    C = C_raw35.detach().cpu().float()
    assert C.dim()==2 and C.size(1) >= 35, "C_raw_target shape must be [T,35]"

    onehot = C[:, :20]      # [T,20]

    HYDROP = C[:, 20]
    AROM   = C[:, 22]
    CHG7   = C[:, 24]
    BFACT  = C[:, 28]
    RSA    = C[:, 30]
    DIST   = C[:, 34]

    # --- summary axes ---
    HBD = (onehot * _DONOR_W).sum(dim=1)     # donor potential
    HBA = (onehot * _ACCEPT_W).sum(dim=1)    # acceptor potential
    Hydrophobicity = HYDROP
    Charge         = CHG7
    Aromatic       = AROM
    Flexibility    = BFACT
    RSAnorm        = RSA
    Distance       = DIST

    C_axes = torch.stack(
        [HBD, HBA, Hydrophobicity, Charge, Aromatic, Flexibility, RSAnorm, Distance],
        dim=1
    )  # [T,8]

    cav_idx = {
        "HBD":0, "HBA":1, "Hydrophobicity":2, "Charge":3,
        "Aromatic":4, "Flexibility":5, "SASA/RSA":6, "Distance":7
    }
    return C_axes, cav_idx
C_axes, cav_idx = build_cavity_summary_axes(C_raw_target)

i = next(k for k,s in enumerate(ligand_names) if "Remdesivir" in str(s))

png, csv = heatmap_residue_by_ligfeat_multiC(
    X_L, C_axes, i, mu_L15, sigma_L15,
    cavity_feat_idx_map=cav_idx,
    cavity_axes=["HBD","HBA","Hydrophobicity","Charge","Aromatic","Flexibility","SASA/RSA","Distance"],
    cav_norm="z",
    out_png=f"/home/yejin/affinity/residue_x_ligfeat_ALL_{ligand_names[i]}.png",
    out_csv=f"/home/yejin/affinity/residue_x_ligfeat_ALL_{ligand_names[i]}.csv",
    ligand_name=ligand_names[i],
    residue_numbers=residue_numbers
)
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from pathlib import Path

FEATURE_KEYS = [
    'HBD','HBA','RotBonds','AromRings',
    'Heteroatoms','TPSA','Csp3','FormalCharge',
    'MolWt','LabuteASA','LogP','DonorAcceptorRatio',
    'AcidicEstimate','BasicEstimate','HydrophobicEstimate'
]

def _lig15_z(X_L, i, mu_L15, sigma_L15):
    mu15 = torch.as_tensor(mu_L15[:15]).float()
    sg15 = torch.as_tensor(sigma_L15[:15]).float()
    return ((X_L[i,:15].detach().cpu().float() - mu15) / (sg15 + 1e-8)).numpy()

def heatmap_single_with_reslabels(
        C_axes,                       # [T, K] = build_cavity_summary_axes 결과 (요약 8축)
        cav_idx,                      # dict: {"HBD":0,"HBA":1,...}
        X_L, ligand_names, i,         # 리간드 선택 인덱스
        mu_L15, sigma_L15,
        residue_numbers=None,         # [831,832,833,834] 같은 번호
        residue_names=None,           # ["Lys","Asp","Gly","Tyr"] 같은 이름(옵션)
        cavity_axes=("HBD","HBA","Hydrophobicity","Charge","Aromatic","Flexibility","SASA/RSA","Distance"),
        cav_norm="z",
        out_png="residue_x_ligfeat_single.png",
        out_csv="residue_x_ligfeat_single.csv",
        annotate=True,
        vmin=-0.5, vmax=0.5
    ):
    # cavity 합성 스칼라 (선택 축 평균; 축별 정규화 옵션)
    idxs = [cav_idx[a] for a in cavity_axes]
    C_sel = torch.stack([C_axes[:,j] for j in idxs], dim=1).cpu().float()  # [T,K]
    if cav_norm == "z":
        mu = C_sel.mean(0, keepdim=True); sd = C_sel.std(0, keepdim=True)+1e-8
        Cn = (C_sel - mu) / sd
    elif cav_norm == "minmax":
        mn = C_sel.min(0, keepdim=True).values; mx = C_sel.max(0, keepdim=True).values
        Cn = (C_sel - mn) / (mx - mn + 1e-8)
    else:
        Cn = C_sel
    v_comb = Cn.mean(dim=1).numpy()  # [T]

    # 리간드 15D z-score
    lig15 = _lig15_z(X_L, i, mu_L15, sigma_L15)  # [15]
    M = np.outer(v_comb, lig15)                  # [T,15]

    # 라벨 구성
    if residue_names is not None and residue_numbers is not None:
        ylabels = [f"{residue_names[k]}{residue_numbers[k]}" for k in range(len(residue_numbers))]
    elif residue_numbers is not None:
        ylabels = [f"Res{n}" for n in residue_numbers]
    else:
        ylabels = [f"Res{i}" for i in range(M.shape[0])]

    # 저장(CSV)
    df = pd.DataFrame(M, columns=FEATURE_KEYS)
    df.insert(0, "residue", ylabels)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 그림
    H,W = M.shape
    if vmin is None: vmin = np.nanmin(M)
    if vmax is None: vmax = np.nanmax(M)
    mid = (vmin+vmax)/2
    fig, ax = plt.subplots(figsize=(12, max(3, H/5)))
    im = ax.imshow(M, aspect='auto', vmin=vmin, vmax=vmax, cmap="viridis")
    cbar = plt.colorbar(im, ax=ax, label="Interaction score")

    ax.set_xticks(range(W)); ax.set_xticklabels(FEATURE_KEYS, rotation=45, ha='right')
    ax.set_yticks(range(H)); ax.set_yticklabels(ylabels)

    # grid + 숫자
    ax.set_xticks(np.arange(-.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for r in range(H):
            for c in range(W):
                val = M[r,c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=("white" if val>mid else "black"))

    used = ", ".join(cavity_axes)
    ax.set_xlabel("Ligand Features"); ax.set_ylabel("Residues")
    ax.set_title(f"Residue × LigandFeatures — {ligand_names[i]}\nCavity axes: {used}  |  norm={cav_norm}")
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
    return out_png, out_csv
      
# A) 단일 히트맵 (잔기 이름까지)
png_single, csv_single = heatmap_single_with_reslabels(
    C_axes, cav_idx, X_L, ligand_names,
    i = next(i for i,s in enumerate(ligand_names) if "Remdesivir".lower() in str(s).lower()),
    mu_L15=mu_L15, sigma_L15=sigma_L15,
    residue_numbers=[831,832,833,834],
    residue_names=["Gly","Asp","Asn","Glu"],   # 실제 이름에 맞게 넣으세요
    out_png="/home/yejin/affinity/imap_Ramdesivir_labeled.png",
    out_csv="/home/yejin/affinity/imap_Ramdesivir_labeled.csv"
)
