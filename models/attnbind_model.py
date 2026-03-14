import torch
import torch.nn as nn

class AttnBind(nn.Module):

    def __init__(self, lig_dim, cav_dim, hidden=128):

        super().__init__()

        self.lig_proj = nn.Linear(lig_dim, hidden)
        self.cav_proj = nn.Linear(cav_dim, hidden)

        self.attn = nn.Linear(hidden, 1)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, L, C):

        Lh = self.lig_proj(L)
        Ch = self.cav_proj(C)

        scores = self.attn(Ch).squeeze(-1)

        alpha = torch.softmax(scores, dim=1)

        context = torch.bmm(alpha.unsqueeze(1), Ch).squeeze(1)

        x = torch.cat([Lh, context], dim=1)

        return self.head(x)
