import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model_neural network import MultiModalPricePredictor  # your model file

#Dataset for test
class TestDataset(Dataset):
    def __init__(self, text_embeddings, image_embeddings):
        self.text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        self.image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        return self.text_embeddings[idx], self.image_embeddings[idx]

#Evaluation Function
def evaluate_models(model_paths, text_embeddings, image_embeddings, batch_size=64, output_path="output.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TestDataset(text_embeddings, image_embeddings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_fold_preds = []

    for model_path in model_paths:
        print(f"Loading model weights from {model_path}")
        model = MultiModalPricePredictor(text_dim=3072, image_dim=512)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        fold_preds = []

        with torch.no_grad():
            for text_emb, image_emb in loader:
                text_emb, image_emb = text_emb.to(device), image_emb.to(device)
                outputs = model(text_emb, image_emb)
                fold_preds.append(outputs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds)
        all_fold_preds.append(fold_preds)
        print(f"Finished predictions for model: {model_path}")

    #Ensemble all folds
    preds = np.mean(np.vstack(all_fold_preds), axis=0)
    preds = np.expm1(preds)  # reverse log1p transform


    preds = np.clip(preds, 0, np.percentile(preds, 99.9))

    #Prepare submission
    test_df = pd.read_csv("/dataset/test.csv")
    assert len(test_df) == len(preds), "Test CSV and embeddings length mismatch!"

    submission = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "price": preds
    })

    output_path = os.path.abspath(output_path)
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(submission.head())

    np.save("/outputs/test_predictions.npy", preds)
    print("Saved raw predictions to test_predictions.npy")

#Main Execution
if __name__ == "__main__":
    # Load embeddings
    text_embeddings = np.load("/outputs/X_test_text_structured.npy").astype(np.float32)
    image_embeddings = np.load("/outputs/test_image_embeddings_clip.npy").astype(np.float32)

    # All trained model paths (from folds)
    model_paths = [
        "/models/best_model_fold1.pth",
        "/models/best_model_fold2.pth",
        "/models/best_model_fold3.pth",
        "/models/best_model_fold4.pth",
        "/models/best_model_fold5.pth"
    ]

    evaluate_models(model_paths, text_embeddings, image_embeddings,
                    batch_size=64,
                    output_path="/outputs/test_out.csv")


