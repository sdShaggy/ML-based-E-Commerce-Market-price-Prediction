import os, numpy as np, pandas as pd, random, torch
from sklearn.model_selection import StratifiedKFold
from pytorch_dataset import EmbeddingDataset
from model_neural_network import MultiModalRegressor
from torch.utils.data import DataLoader

# ✅ Auto-detect device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
# ---------------------------
# Normalization function
# ---------------------------
def normalize_embeddings(arr):
    return (arr - arr.mean(axis=0)) / (arr.std(axis=0)+1e-6)

# ---------------------------
# Training function
# ---------------------------
def train():
    # === paths & settings ===
    text_file = 'ML-based-E-Commerce-Market-price-Prediction/outputs/text_embs_train.npy'
    img_file  = 'ML-based-E-Commerce-Market-price-Prediction/outputs/train_image_embeddings_pca256.npy'
    train_csv = 'ML-based-E-Commerce-Market-price-Prediction/dataset/train.csv'
    models_dir = 'ML-based-E-Commerce-Market-price-Prediction/models'
    os.makedirs(models_dir, exist_ok=True)

    set_seed(42)

    # ---------------------------
    # Load data
    # ---------------------------
    df = pd.read_csv(train_csv)
    y = df['price'].values
    y_log = np.log1p(y)
    bins = pd.qcut(df['price'], q=10, labels=False, duplicates='drop')

    # ---------------------------
    # Normalize embeddings
    # ---------------------------
    text_embs = np.load(text_file)
    img_embs = np.load(img_file)
    text_embs = normalize_embeddings(text_embs)
    img_embs  = normalize_embeddings(img_embs)
    np.save("ML-based-E-Commerce-Market-price-Prediction/outputs/text_embs_train.npy", text_embs)
    np.save("ML-based-E-Commerce-Market-price-Prediction/outputs/train_image_embeddings_pca256.npy", img_embs)

    # ---------------------------
    # Stratified K-Fold
    # ---------------------------
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(df))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(df, bins)):
        print(f"FOLD {fold}")

        train_ds = EmbeddingDataset("ML-based-E-Commerce-Market-price-Prediction/outputs/text_embs_train.npy",
                                    "ML-based-E-Commerce-Market-price-Prediction/outputs/train_image_embeddings_pca256.npy",
                                    indices=tr_idx, targets=y_log)
        val_ds = EmbeddingDataset("ML-based-E-Commerce-Market-price-Prediction/outputs/text_embs_train.npy",
                                  "ML-based-E-Commerce-Market-price-Prediction/outputs/train_image_embeddings_pca256.npy",
                                  indices=val_idx, targets=y_log)

        tr_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

        model = MultiModalRegressor(text_dim=text_embs.shape[1], img_dim=img_embs.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        best_smape = 1e9
        best_path = f"{models_dir}/nn_fold_{fold}.pt"

        for epoch in range(1, 51):  # Increased epochs for better convergence
            model.train()
            for t, i, tab, yb in tr_loader:
                t, i, yb = t.to(device), i.to(device), yb.to(device)
                opt.zero_grad()
                out = model(t, i)
                loss = torch.nn.functional.mse_loss(out, yb)  # MSE on log prices
                loss.backward()
                opt.step()

            # ---------------------------
            # Validation
            # ---------------------------
            model.eval()
            preds_val, truths = [], []
            with torch.no_grad():
                for t, i, tab, yb in val_loader:
                    t, i = t.to(device), i.to(device)
                    out = model(t, i)
                    preds_val.append(out.cpu().numpy())
                    truths.append(yb.numpy())

            preds_val = np.concatenate(preds_val)
            truths = np.concatenate(truths)
            preds_price = np.expm1(preds_val)
            true_price = np.expm1(truths)
            denom = (np.abs(preds_price) + np.abs(true_price))/2
            denom[denom==0] = 1e-6
            smape = np.mean(np.abs(preds_price - true_price)/denom)*100
            print(f" fold {fold} epoch {epoch} smape {smape:.4f}")

            # save best
            if smape < best_smape:
                best_smape = smape
                torch.save(model.state_dict(), best_path)

            scheduler.step(smape)  # reduce LR if validation smape plateaus

        # ---------------------------
        # Load best and save OOF
        # ---------------------------
        model.load_state_dict(torch.load(best_path))
        model.eval()
        val_preds = []
        with torch.no_grad():
            for t, i, tab, yb in val_loader:
                t, i = t.to(device), i.to(device)
                out = model(t, i)
                val_preds.append(out.cpu().numpy())
        oof[val_idx] = np.concatenate(val_preds)

    # Save log-space OOF
    np.save('ML-based-E-Commerce-Market-price-Prediction/outputs/oof_nn_preds_tuned.npy', oof)
    print("Saved tuned OOF NN preds. OOF shape:", oof.shape)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    train()
