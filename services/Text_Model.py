import string
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

#LOAD DATA
train = pd.read_csv("dataset/train.csv")

print(f"Train shape: {train.shape}")
print(train.head())

#BASIC CLEANING FUNCTION
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)                  # remove URLs
    text = re.sub(r"\d+", " ", text)                     # remove numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train["clean_text"] = train["catalog_content"].apply(clean_text)

#TF-IDF FEATURE EXTRACTION
vectorizer = TfidfVectorizer(
    max_features=8000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_text_train = vectorizer.fit_transform(train["clean_text"])

print("TF-IDF features generated:")
print("X_text_train:", X_text_train.shape)

#SAVE VERCTORIZATION MODEL AND EMBEDDINGS
import pickle

# Save TF-IDF vectorizer
with open('models/vectorizer.pkl', "wb") as f:
    pickle.dump(vectorizer, f)

# Save text embeddings (sparse matrices)
save_npz('outputs/X_text_train.npz', X_text_train)
