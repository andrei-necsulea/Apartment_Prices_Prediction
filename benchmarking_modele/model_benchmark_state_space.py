import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

df_full = pd.read_csv("../merged_data/full_gen8.csv")

df_full["An"] = df_full["An"].astype(int)

if "price_per_sqm" not in df_full.columns:
    raise ValueError("full_gen8.csv must contain a 'price_per_sqm' column.")

y = df_full["price_per_sqm"]

exog_cols = ["index_cost_mat", "urban_incr_proc", "rata_infl", "pib_dolj_mld_ron"]
for c in exog_cols:
    if c not in df_full.columns:
        raise ValueError(f"Column '{c}' is missing from full_gen8.csv.")
X = df_full[exog_cols]

#Train: 2012–2022, Test: 2023–2025
train_mask = (df_full["An"] >= 2012) & (df_full["An"] <= 2022)
test_mask  = (df_full["An"] >= 2023) & (df_full["An"] <= 2025)

y_train, y_test = y[train_mask], y[test_mask]
X_train, X_test = X[train_mask], X[test_mask]

#Sanity check
if len(y_test) == 0:
    raise ValueError("No rows found for test period 2023–2025 in full_gen8.csv.")

#Define and fit State Space model
model = UnobservedComponents(
    endog=y_train,
    level="local linear trend",
    autoregressive=1,
    exog=X_train,
    stochastic_level=True,
    stochastic_trend=True,
    mle_regression=1,
)

res = model.fit(method="powell", disp=False)
print(res.summary())

#Forecast for 2023–2025
steps = len(y_test)
fc_res = res.get_forecast(steps=steps, exog=X_test)
y_pred = fc_res.predicted_mean

#Evaluation: MAE, RMSE, MAPE + table year/real/pred
years_test = df_full.loc[test_mask, "An"].values

df_eval = pd.DataFrame({
    "An": years_test,
    "price_real": y_test.values,
    "price_pred": y_pred.values,
})

df_eval["abs_error"] = (df_eval["price_pred"] - df_eval["price_real"]).abs()
df_eval["pct_error"] = df_eval["abs_error"] / df_eval["price_real"] * 100.0

MAE = df_eval["abs_error"].mean()
RMSE = np.sqrt(((df_eval["price_pred"] - df_eval["price_real"])**2).mean())
MAPE = df_eval["pct_error"].mean()

print("\n=== Evaluation on 2023–2025 ===")
print(df_eval)
print("\nMAE  =", f"{MAE:.2f} RON/sqm")
print("RMSE =", f"{RMSE:.2f} RON/sqm")
print("MAPE =", f"{MAPE:.2f} %")

df_eval.to_csv("model_eval_2012_2022_vs_2023_2025.csv", index=False)
print("\nSaved: model_eval_2012_2022_vs_2023_2025.csv")