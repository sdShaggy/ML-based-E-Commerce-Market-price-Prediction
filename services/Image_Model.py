import os
import torch
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import clip  # OpenAI CLIP package

#CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#LOAD DATA
train = pd.read_csv("Base_Model/data/train.csv")
test = pd.read_csv("Base_Model/data/test.csv")
print(f"Train shape: {train.shape}")
print(train.head())

#LOAD MODEL
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.to(device)
model.eval()

#IMAGE PROCESSING

def load_image(url):
    """
    Download image from URL and preprocess using CLIP's transform
    """
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = preprocess(img)  # CLIP preprocessing
        return img
    except Exception as e:
        print(f"Failed to load {url[:60]}... -> {e}")
        return torch.zeros(3, 224, 224)

#EMBEDDING
def extract_clip_embedding(img_tensor):
    """
    Extract 512-dimensional CLIP image embedding
    """
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze().cpu().numpy()


def process_image_column(df, column_name="image_link"):
    embeddings = []
    for url in tqdm(df[column_name].tolist(), desc="Extracting CLIP Embeddings"):
        img = load_image(url)
        emb = extract_clip_embedding(img)
        embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)

#RUN AND SAVE
train_image_embeddings = process_image_column(train)
test_image_embeddings = process_image_column(test)

os.makedirs("outputs", exist_ok=True)
np.save(os.path.join("Base_Model/outputs/train_image_embeddings_clip.npy"), train_image_embeddings)
np.save(os.path.join("Base_Model/outputs/test_image_embeddings_clip.npy"), test_image_embeddings)
print("Saved CLIP embeddings as train_image_embeddings_clip.npy")
print("Embedding shape:", train_image_embeddings.shape)

