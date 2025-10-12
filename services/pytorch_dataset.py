# src/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, text_emb_file, img_emb_file, indices=None, targets=None, tabular=None):
        self.text = np.load(text_emb_file)  # Load text embeddings
        self.img  = np.load(img_emb_file)   # Load image embeddings
        self.indices = indices if indices is not None else np.arange(len(self.text))
        self.targets = targets
        self.tab = tabular  # Optional structured data like IPQ or brand

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        t = torch.tensor(self.text[idx]).float()
        im = torch.tensor(self.img[idx]).float()
        tab = torch.tensor(self.tab[idx]).float() if self.tab is not None else torch.empty(0)
        
        if self.targets is not None:
            y = torch.tensor(self.targets[idx]).float()
            return t, im, tab, y
        
        return t, im, tab


# ==============================
# ✅ Define File Paths Here
# ==============================

text_emb_file = "output/text_embs_train.npy"            # Text Embeddings
img_emb_file  = "output/train_image_embeddings_pca256.npy"  # Image Embeddings

# ==============================
# ✅ Example Usage (You can remove later if importing)
# ==============================

if __name__ == "__main__":
    dataset = EmbeddingDataset(text_emb_file, img_emb_file)

    print("Dataset length:", len(dataset))
    sample = dataset[0]
    print("Sample Shapes:")
    print("Text Embedding:", sample[0].shape)
    print("Image Embedding:", sample[1].shape)
