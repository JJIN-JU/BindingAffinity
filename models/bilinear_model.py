import torch
import torch.nn as nn

class BilinearAffinityZ_MLP_LR(nn.Module):

    def __init__(self, lig_dim, cav_dim, proj_dim=64, rank=10):

        super().__init__()

        self.lig_proj = nn.Sequential(
            nn.Linear(lig_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim)
        )

        self.cav_proj = nn.Sequential(
            nn.Linear(cav_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim)
        )

        self.U = nn.Parameter(torch.empty(proj_dim, rank))
        self.V = nn.Parameter(torch.empty(proj_dim, rank))

        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, L, C):

        Lh = self.lig_proj(L)
        Ch = self.cav_proj(C)

        W = self.U @ self.V.T

        LW = Lh @ W

        bilinear = (LW * Ch).sum(1)

        return bilinear + self.bias
