# check_files.py
import numpy as np, pandas as pd, os, scipy.sparse as sps

train = pd.read_csv("dataset/train.csv")
print("train rows:", len(train))

# Check for existing artifacts - TFIDF, image emb, text emb
for f in ["artifacts/X_text_train.npz","artifacts/train_image_embeddings.npy","artifacts/text_embs_train.npy"]:
    print(f, "exists:", os.path.exists(f))
try:
    if os.path.exists("artifacts/X_text_train.npz"):
        X = sps.load_npz("artifacts/X_text_train.npz")
        print("TFIDF shape:", X.shape)
    if os.path.exists("artifacts/train_image_embeddings.npy"):
        img = np.load("artifacts/train_image_embeddings.npy")
        print("image emb shape:", img.shape)
    if os.path.exists("artifacts/text_embs_train.npy"):
        te = np.load("artifacts/text_embs_train.npy")
        print("text emb shape:", te.shape)
except Exception as e:
    print("Error checking files:", e)
