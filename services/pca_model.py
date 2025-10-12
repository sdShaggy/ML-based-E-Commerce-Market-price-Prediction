from sklearn.decomposition import PCA
import numpy as np

# ==========================
# 1. Load Embeddings
# ==========================
train_img = np.load('output/train_image_embeddings.npy')
test_img = np.load('output/test_image_embeddings.npy')

# ==========================
# 2. Apply PCA
# ==========================
pca = PCA(n_components=256, random_state=42)

train_img_pca = pca.fit_transform(train_img)   # Fit on train
test_img_pca = pca.transform(test_img)         # Transform test

# ==========================
# 3. Save Reduced Embeddings
# ==========================
np.save('output/train_image_embeddings_pca256.npy', train_img_pca)
np.save('output/test_image_embeddings_pca256.npy', test_img_pca)

print("✅ PCA reduction complete! Saved to output/")
