
import numpy as np
import os

text_file = "ML-based-E-Commerce-Market-price-Prediction/outputs/text_embs_train.npy"   # ensure exists
img_file  = "ML-based-E-Commerce-Market-price-Prediction/outputs/train_image_embeddings_pca256.npy"  # or train_image_embeddings.npy

text = np.load(text_file)
img  = np.load(img_file)
print("text shape", text.shape, "img shape", img.shape)
X = np.hstack([text, img])
np.save("ML-based-E-Commerce-Market-price-Prediction/outputs/X_train_for_model.npy", X)
print("Saved artifacts/X_train_for_model.npy shape", X.shape)
