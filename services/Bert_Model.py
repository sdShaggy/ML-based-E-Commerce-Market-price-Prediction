# Colab cell: text embeddings
from sentence_transformers import SentenceTransformer
import pandas as pd, numpy as np
model = SentenceTransformer('all-mpnet-base-v2')   # good quality

train = pd.read_csv('dataset/train.csv')
test  = pd.read_csv('dataset/test.csv')

texts_train = train['catalog_content'].fillna('').astype(str).tolist()
texts_test  = test['catalog_content'].fillna('').astype(str).tolist()

emb_train = model.encode(texts_train, batch_size=64, show_progress_bar=True)
emb_test  = model.encode(texts_test,  batch_size=64, show_progress_bar=True)

np.save('outputs/text_embs_train.npy', emb_train)
np.save('outputs/text_embs_test.npy', emb_test)
print("Saved text embeddings", emb_train.shape, emb_test.shape)
