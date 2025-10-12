# stack_and_eval_oof.py
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Load
df = pd.read_csv("ML-based-E-Commerce-Market-price-Prediction/dataset/train.csv")
y = df['price'].values

oof_lgb_price = np.load("ML-based-E-Commerce-Market-price-Prediction/outputs/oof_lgb_price.npy")   # shape (n,)
oof_nn_price  = np.load("ML-based-E-Commerce-Market-price-Prediction/outputs/oof_nn_preds.npy")    # shape (n,)

# stack features
X_meta = np.vstack([oof_lgb_price, oof_nn_price]).T

# train meta on train OOFs
meta = Ridge(alpha=1.0)
meta.fit(X_meta, y)

# OOF meta predictions
oof_meta = meta.predict(X_meta)
oof_meta = np.clip(oof_meta, 0.01, None)

# SMAPE calculation
def smape(a,b):
    denom = (np.abs(a)+np.abs(b))/2
    denom[denom==0]=1e-6
    return (np.abs(a-b)/denom).mean()*100

print("LGB OOF SMAPE:", smape(y, oof_lgb_price))
print("NN  OOF SMAPE:", smape(y, oof_nn_price))
print("STACK OOF SMAPE:", smape(y, oof_meta))

# Save final OOFs if needed
#np.save("artifacts/oof_meta_price.npy", oof_meta)
