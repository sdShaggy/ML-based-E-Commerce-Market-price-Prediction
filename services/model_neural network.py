import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
class MultiModalPricePredictorV2(nn.Module):
    def __init__(self, text_dim=3072, image_dim=512, hidden_dim=512, dropout=0.3):
        super().__init__()
        # TEXT branch deeper
        self.text_fc1 = nn.Linear(text_dim, hidden_dim*2)
        self.text_bn1 = nn.BatchNorm1d(hidden_dim*2)
        self.text_fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.text_bn2 = nn.BatchNorm1d(hidden_dim)
        self.text_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.text_bn3 = nn.BatchNorm1d(hidden_dim)

        # IMAGE branch same as before
        self.image_fc1 = nn.Linear(image_dim, hidden_dim//2)
        self.image_bn1 = nn.BatchNorm1d(hidden_dim//2)
        self.image_fc2 = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.image_bn2 = nn.BatchNorm1d(hidden_dim//2)

        # FUSION
        self.fusion_fc1 = nn.Linear(hidden_dim + hidden_dim//2, hidden_dim)
        self.fusion_bn1 = nn.BatchNorm1d(hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fusion_fc3 = nn.Linear(hidden_dim//2, 1)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, text_emb, image_emb):
        # TEXT
        x_text = self.act(self.text_bn1(self.text_fc1(text_emb)))
        x_text = self.dropout(x_text)
        x_text = self.act(self.text_bn2(self.text_fc2(x_text)))
        x_text = self.dropout(x_text)
        x_text = self.act(self.text_bn3(self.text_fc3(x_text)))

        # IMAGE
        x_image = self.act(self.image_bn1(self.image_fc1(image_emb)))
        x_image = self.dropout(x_image)
        x_image = self.act(self.image_bn2(self.image_fc2(x_image)))

        # FUSION with residual connection
        x = torch.cat([x_text, x_image], dim=1)
        x_res = x  # residual
        x = self.act(self.fusion_bn1(self.fusion_fc1(x))) + x_res[:, :x.size(1)]  # add only matching dims
        x = self.dropout(x)
        x = self.act(self.fusion_fc2(x))
        price = self.fusion_fc3(x)
        return price.squeeze(1)
