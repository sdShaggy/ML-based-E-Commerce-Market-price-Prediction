# Colab cell for inference
import numpy as np, pandas as pd, torch, os
from services.model_neural_network import MultiModalRegressor
from services.pytorch_dataset import EmbeddingDataset
from torch.utils.data import DataLoader

text_test = 'outputs/text_embs_test.npy'
img_test  =  'outputs/test_image_embeddings.npy'
test_df = pd.read_csv('dataset/test.csv')
ds = EmbeddingDataset(text_test, img_test, indices=None)
loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)

n_splits=5
preds_sum = None
for fold in range(n_splits):
    model = MultiModalRegressor(text_dim=np.load(text_test).shape[1], img_dim=np.load(img_test).shape[1]).cuda()
    model.load_state_dict(torch.load(f"/models/nn_fold_{fold}.pt"))
    model.eval()
    fold_preds = []
    with torch.no_grad():
        for t,i,tab in loader:
            t,i = t.cuda(), i.cuda()
            out = model(t,i)
            fold_preds.append(out.cpu().numpy())
    fold_preds = np.concatenate(fold_preds)
    if preds_sum is None:
        preds_sum = fold_preds
    else:
        preds_sum += fold_preds
preds_avg = preds_sum / n_splits
pred_prices = np.expm1(preds_avg)
np.save('outputs/nn_test_preds.npy', pred_prices)
print("Saved nn test preds", pred_prices.shape)
