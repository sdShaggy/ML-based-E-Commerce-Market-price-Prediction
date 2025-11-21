import torch
import torch.nn as nn
from torch.utils.data import Dataset

class PriceDataset(Dataset):

    def __init__(self, text_embeddings, image_embeddings, prices):

        self.text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        self.image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
        self.prices = torch.tensor(prices, dtype=torch.float32)

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, idx):
        return self.text_embeddings[idx], self.image_embeddings[idx], self.prices[idx]

class MultiModalPricePredictor(nn.Module):
    def __init__(self, text_dim=3072, image_dim=512, hidden_dim=512, dropout=0.3):
        super(MultiModalPricePredictor, self).__init__()

        #TEXT
        self.text_fc1 = nn.Linear(text_dim, hidden_dim*2)
        self.text_bn1 = nn.BatchNorm1d(hidden_dim*2)
        self.text_fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.text_bn2 = nn.BatchNorm1d(hidden_dim)

        #IMAGE
        self.image_fc1 = nn.Linear(image_dim, hidden_dim//2)
        self.image_bn1 = nn.BatchNorm1d(hidden_dim//2)
        self.image_fc2 = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.image_bn2 = nn.BatchNorm1d(hidden_dim//2)

        #MULTIMODAL FUSION
        self.fusion_fc1 = nn.Linear(hidden_dim + hidden_dim//2, hidden_dim)
        self.fusion_bn1 = nn.BatchNorm1d(hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fusion_fc3 = nn.Linear(hidden_dim//2, 1)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, text_emb, image_emb):
        #TEXT
        x_text = self.activation(self.text_bn1(self.text_fc1(text_emb)))
        x_text = self.dropout(x_text)
        x_text = self.activation(self.text_bn2(self.text_fc2(x_text)))

        #IMAGE
        x_image = self.activation(self.image_bn1(self.image_fc1(image_emb)))
        x_image = self.dropout(x_image)
        x_image = self.activation(self.image_bn2(self.image_fc2(x_image)))

        #MULTIMODAL FUSION
        x = torch.cat([x_text, x_image], dim=1)
        x = self.activation(self.fusion_bn1(self.fusion_fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.fusion_fc2(x))
        price = self.fusion_fc3(x)

        return price.squeeze(1)
