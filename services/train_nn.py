import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import logging
from Combined_nn_create import PriceDataset, MultiModalPricePredictor

#LOGGING
logging.basicConfig(
    filename="/outputs/training.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

#SMAPE FUNCTION
def smape(y_true, y_pred, epsilon=1e-6):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon))

#SMAPE LOSS(FOR NN)
class SMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(SMAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        return torch.mean(
            2 * torch.abs(y_pred - y_true) / (torch.abs(y_pred) + torch.abs(y_true) + self.epsilon)
        )

#SINGLE FOLD CALCULATION
def train_fold(model, train_loader, val_loader, fold_num, lr=1e-4, epochs=50, patience=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_smape = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        #Train
        model.train()
        train_losses, train_preds, train_targets = [], [], []
        for text_emb, image_emb, prices in train_loader:
            optimizer.zero_grad()
            text_emb, image_emb, prices = text_emb.to(device), image_emb.to(device), prices.to(device)
            outputs = model(text_emb, image_emb)
            loss = criterion(outputs, prices)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(prices.cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        train_smape_score = smape(np.expm1(np.concatenate(train_targets)), np.expm1(np.concatenate(train_preds)))

        #Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for text_emb, image_emb, prices in val_loader:
                text_emb, image_emb, prices = text_emb.to(device), image_emb.to(device), prices.to(device)
                outputs = model(text_emb, image_emb)
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(prices.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        # Safety check for NaN/Inf
        if np.any(np.isnan(val_preds)) or np.any(np.isnan(val_targets)):
            print(f"NaN detected in validation fold {fold_num}, skipping epoch {epoch+1}")
            continue
        if np.any(np.isinf(val_preds)) or np.any(np.isinf(val_targets)):
            print(f"Inf detected in validation fold {fold_num}, skipping epoch {epoch+1}")
            continue

        val_smape_score = smape(np.expm1(val_targets), np.expm1(val_preds))

        log_msg = (f"Fold {fold_num} | Epoch {epoch+1}/{epochs} | "
                   f"Train Loss: {avg_train_loss:.4f} | Train SMAPE: {train_smape_score:.2f} | "
                   f"Val SMAPE: {val_smape_score:.2f}")
        print(log_msg)
        logging.info(log_msg)


        if val_smape_score < best_val_smape:
            best_val_smape = val_smape_score
            model_path = f"best_model_fold{fold_num}.pth"
            full_path = os.path.abspath(model_path)
            torch.save(model.state_dict(), full_path)
            print(f"Fold {fold_num} best model saved at: {full_path}")
            logging.info(f"Fold {fold_num} best model saved at: {full_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} for fold {fold_num}")
                logging.info(f"Early stopping at epoch {epoch+1} for fold {fold_num}")
                break

    return best_val_smape

#K-FOLDING
def train_kfold(text_embeddings, image_embeddings, prices, n_splits=5, batch_size=32, epochs=50):
    dataset = PriceDataset(text_embeddings, image_embeddings, prices)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_smape = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        logging.info(f"\n===== Fold {fold+1} =====")
        print(f"\n===== Fold {fold+1} =====")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

        model = MultiModalPricePredictor(text_dim=1536, image_dim=512)
        best_smape = train_fold(model, train_loader, val_loader, fold_num=fold+1,
                                lr=1e-4, epochs=epochs, patience=7)
        fold_smape.append(best_smape)
        logging.info(f"Fold {fold+1} Best SMAPE: {best_smape:.2f}")
        print(f"Fold {fold+1} Best SMAPE: {best_smape:.2f}")

    avg_smape = np.mean(fold_smape)
    logging.info(f"\n===== Average SMAPE across {n_splits} folds: {avg_smape:.2f} =====")
    print(f"\n===== Average SMAPE across {n_splits} folds: {avg_smape:.2f} =====")
    return avg_smape

#MAIN
if __name__ == "__main__":
    text_embeddings = np.load("/outputs/X_text_structured.npy").astype(np.float32)
    image_embeddings = np.load("/outputs/train_image_embeddings_clip.npy").astype(np.float32)

    train_df = pd.read_csv("/dataset/train.csv")
    prices_raw = train_df["price"].values.astype(np.float32)
    prices_log = np.log1p(prices_raw)

    avg_smape = train_kfold(text_embeddings, image_embeddings, prices_log,
                            n_splits=5, batch_size=32, epochs=50)
    print(f"Final Average SMAPE: {avg_smape:.2f}")

