import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df_full = pd.read_csv("../merged_data/full_gen8.csv")
df_full["An"] = df_full["An"].astype(int)

df_full = df_full.sort_values("An").reset_index(drop=True)

# target
df_full["price_per_sqm"] = df_full["price_per_sqm"].astype(float)

#Cream lag-uri
df_full["price_lag1"] = df_full["price_per_sqm"].shift(1)
df_full["price_lag2"] = df_full["price_per_sqm"].shift(2)

# stergem primii ani fara lag complet
df_full = df_full.dropna(subset=["price_lag1", "price_lag2"]).reset_index(drop=True)

# feature-uri
feature_cols = [
    "index_cost_mat",
    "urban_incr_proc",
    "rata_infl",
    "pib_dolj_mld_ron",
    "price_lag1",
    "price_lag2",
]

X = df_full[feature_cols]
y = df_full["price_per_sqm"]

#Train/Test split (dupa An)
train_mask = (df_full["An"] >= 2012) & (df_full["An"] <= 2022)
test_mask  = (df_full["An"] >= 2023) & (df_full["An"] <= 2025)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
years_test = df_full.loc[test_mask, "An"].values

gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gbr.fit(X_train, y_train)
y_pred_ml = gbr.predict(X_test)

df_eval_ml = pd.DataFrame({
    "An": years_test,
    "price_real": y_test.values,
    "price_pred": y_pred_ml,
})

df_eval_ml["abs_error"] = (df_eval_ml["price_pred"] - df_eval_ml["price_real"]).abs()
df_eval_ml["pct_error"] = df_eval_ml["abs_error"] / df_eval_ml["price_real"] * 100.0

MAE  = df_eval_ml["abs_error"].mean()
RMSE = np.sqrt(mean_squared_error(df_eval_ml["price_real"], df_eval_ml["price_pred"]))
MAPE = df_eval_ml["pct_error"].mean()

print("\n=== Gradient Boosting (lags + features) ===")
print(df_eval_ml)
print("\nMAE  =", f"{MAE:.2f} RON/sqm")
print("RMSE =", f"{RMSE:.2f} RON/sqm")
print("MAPE =", f"{MAPE:.2f} %")

df_eval_ml.to_csv("GBR_eval_2012_2020_vs_2021_2025.csv", index=False)
print("\nSaved: GBR_eval_2012_2020_vs_2021_2025.csv")