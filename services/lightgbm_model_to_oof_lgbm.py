## in Colab or local - produce lgb test predictions
#import lightgbm as lgb, numpy as np, scipy.sparse as sps, pandas as pd
#bst = lgb.Booster(model_file='models/lgb_combined_model.txt')
#X_test = sps.load_npz('/X_text_test.npz')   # adjust path
#pred_log = bst.predict(X_test)   # if bst was trained on log; else it's price
## if log1p used for training:
#pred_price = np.expm1(pred_log)
#np.save('outputs/lgb_test_preds.npy', pred_price)
## also compute OOF lgb preds if you have them; otherwise run LGB training to get oof.
#
# lgb_oof.py
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import os

os.makedirs("models", exist_ok=True)

train = pd.read_csv("ML-based-E-Commerce-Market-price-Prediction/dataset/train.csv")
y = train['price'].values
y_log = np.log1p(y)

# load features
X = np.load("ML-based-E-Commerce-Market-price-Prediction/outputs/X_train_for_model.npy")  # dense numpy
print("X shape:", X.shape)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
oof = np.zeros(len(train))

params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "seed": 42
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    print("LGB fold", fold)
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y_log[tr_idx], y_log[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)

    # ✅ Updated: Use callbacks instead of deprecated early_stopping_rounds / verbose_eval
    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=30)  # Logs every 100 rounds
        ]
    )

    oof[val_idx] = bst.predict(X_val, num_iteration=bst.best_iteration)
    bst.save_model(f"models/lgb_fold_{fold}.txt")

# Save OOF
oof_price = np.expm1(oof)
np.save("ML-based-E-Commerce-Market-price-Prediction/outputs/oof_lgb_price.npy", oof_price)
print("Saved oof_lgb_price.npy")

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100

smape_score = smape(y, oof_price)
print(f"✅ SMAPE Score on Train OOF Predictions: {smape_score:.4f}%")
