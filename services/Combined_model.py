import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, load_npz
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb

#LOAD DATA
train = pd.read_csv("dataset/train.csv")

#LOAD EMBEDDINGS
X_text_train = load_npz("outputs/X_text_train.npz")
train_image_embeddings = np.load("outputs/train_image_embeddings.npy")
X_image_train_sparse = csr_matrix(train_image_embeddings)

print("TF-IDF features loaded:")
print("X_text_train:", X_text_train.shape)

print("Image embeddings loaded:")
print("train_image_embeddings:", X_image_train_sparse.shape)

#COMBINE TEXT AND IMAGE FEATURES
X = hstack([X_text_train, X_image_train_sparse]).astype(np.float32)
y = train["price"].values

#TRAIN-TEST SPLIT
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


#TRAIN LIGHTGBM MODEL
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": 12,
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "verbose": -1
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data]
)

#EVALUATION(SMAPE)
def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

val_preds = model.predict(X_val)
val_preds[val_preds < 0] = 0
print(f"Validation SMAPE: {smape(y_val, val_preds):.3f}%")

#SAVE MODEL
model.save_model("models/lgb_combined_model.txt")
