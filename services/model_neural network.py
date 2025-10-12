# src/model_nn.py
import torch.nn as nn

class MultiModalRegressor(nn.Module):
    def __init__(self, text_dim=768, img_dim=512, tab_dim=0, hidden=512, dropout=0.2):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden//2), nn.ReLU(), nn.BatchNorm1d(hidden//2))
        self.img_proj  = nn.Sequential(nn.Linear(img_dim, hidden//2), nn.ReLU(), nn.BatchNorm1d(hidden//2))
        self.tab_proj  = None
        if tab_dim>0:
            self.tab_proj = nn.Sequential(nn.Linear(tab_dim, hidden//2), nn.ReLU(), nn.BatchNorm1d(hidden//2))
        fusion_dim = (hidden//2) * (2 + (1 if tab_dim>0 else 0))
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden//2),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, t, i, tab=None):
        t = self.text_proj(t)
        i = self.img_proj(i)
        if self.tab_proj is not None and tab is not None and tab.numel()>0:
            tabp = self.tab_proj(tab)
            x = torch.cat([t, i, tabp], dim=1)
        else:
            x = torch.cat([t, i], dim=1)
        out = self.fusion(x).squeeze(1)
        return out
