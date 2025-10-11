import os
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# LOAD DATA
train = pd.read_csv("dataset/train.csv")

print(f"Train shape: {train.shape}")
print(train.head())


# IMAGE TRANSFORM & MODEL
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # remove final FC
resnet.to(device)
resnet.eval()


# FUNCTIONS TOOLS
def load_image(url):
    """Download and preprocess image"""
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = transform(img)
        return img
    except:
        # Return zero tensor if failed
        print("failed")
        return torch.zeros(3, 224, 224)

def extract_embedding(img_tensor):
    """Extract 2048-d ResNet50 embedding"""
    img_tensor = img_tensor.unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        embedding = resnet(img_tensor)
    return embedding.squeeze().cpu().numpy()  # shape: [2048]

def process_image_column(df, column_name="image_link"):
    embeddings = []
    for url in tqdm(df[column_name].tolist()):
        img = load_image(url)
        emb = extract_embedding(img)
        embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)


# EXTRACT IMAGE EMBEDDINGS
train_image_embeddings = process_image_column(train)

# SAVE EMBEDDINGS
np.save(os.path.join("outputs/train_image_embeddings.npy"), train_image_embeddings)

