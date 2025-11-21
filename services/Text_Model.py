import os
import re
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, get_context

#CPU OPTIMIZATION
num_threads = max(4, cpu_count() - 2)
torch.set_num_threads(num_threads)
device = "cpu"
print(f"Using device: {device} ({num_threads} threads)")

#LOAD DATA
df = pd.read_csv("Base_Model/data/test.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

#PREPROCESSING (REGEX CLEANER)
def extract_sections(text):
    if not isinstance(text, str):
        return {"item_name": "", "bullets": "", "description": "", "value_unit": ""}

    text = re.sub(r"[^\x00-\x7F]+", " ", text).strip()  # remove weird Unicode chars

    sections = {}
    sections["item_name"] = re.search(r"Item Name:\s*(.*)", text)
    sections["bullets"] = " ".join(re.findall(r"Bullet Point\s*\d*:\s*(.*)", text))
    sections["description"] = re.search(r"Product Description:\s*(.*)", text)
    val = re.search(r"Value:\s*([\d\.]+)", text)
    unit = re.search(r"Unit:\s*([A-Za-z]+)", text)
    val_unit = f"{val.group(1)} {unit.group(1)}" if val and unit else ""

    return {
        "item_name": sections["item_name"].group(1).strip() if sections["item_name"] else "",
        "bullets": sections["bullets"].strip(),
        "description": sections["description"].group(1).strip() if sections["description"] else "",
        "value_unit": val_unit,
    }

#INITIALISE MODEL
def init_worker(model_name):
    global model
    model = SentenceTransformer(model_name, device="cpu")
    model.eval()

#EMBEDDING
def process_row(text):
    """Encodes one row of catalog text."""
    sections = extract_sections(text)

    name_emb = model.encode(sections["item_name"], convert_to_tensor=True)
    bullet_emb = model.encode(sections["bullets"], convert_to_tensor=True)
    desc_emb = model.encode(sections["description"], convert_to_tensor=True)
    val_emb = model.encode(sections["value_unit"], convert_to_tensor=True)

    combined = torch.cat([name_emb, bullet_emb, desc_emb, val_emb])
    return combined.cpu().numpy()

#MULTIPROCESSING
def create_embeddings_parallel(df, column="catalog_content", num_workers=None, model_name="all-mpnet-base-v2"):
    texts = df[column].fillna("").tolist()
    num_workers = num_workers or max(1, cpu_count() - 2)

    with get_context("spawn").Pool(processes=num_workers, initializer=init_worker, initargs=(model_name,)) as pool:
        results = list(tqdm(pool.imap(process_row, texts, chunksize=16), total=len(texts), desc="Embedding rows"))

    return np.array(results, dtype=np.float32)

#RUN AND SAVE
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    model_name = "all-mpnet-base-v2"

    X_text = create_embeddings_parallel(df, num_workers=8, model_name=model_name)
    np.save("outputs/X_test_text_structured.npy", X_text)

    print("Saved embeddings:", X_text.shape)
