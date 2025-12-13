import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

df_full = pd.read_csv("../merged_data/full_gen8.csv")
df_full["An"] = df_full["An"].astype(int)
df_full = df_full.sort_values("An").reset_index(drop=True)

if "price_per_sqm" not in df_full.columns:
    raise ValueError("full_gen8.csv must contain a 'price_per_sqm' column.")

df_full["price_per_sqm"] = df_full["price_per_sqm"].astype(float)

#regressors for State Space
exog_cols = ["index_cost_mat", "urban_incr_proc", "rata_infl", "pib_dolj_mld_ron"]
for c in exog_cols:
    if c not in df_full.columns:
        raise ValueError(f"Column '{c}' is missing from full_gen8.csv.")
X_all = df_full[exog_cols]

# lag features for GBR
df_full["price_lag1"] = df_full["price_per_sqm"].shift(1)
df_full["price_lag2"] = df_full["price_per_sqm"].shift(2)

#Train / test masks
train_mask = (df_full["An"] >= 2012) & (df_full["An"] <= 2022)
test_mask  = (df_full["An"] >= 2023) & (df_full["An"] <= 2025)

y_all = df_full["price_per_sqm"]
y_train_base = y_all[train_mask]
X_train_base = X_all[train_mask]
X_test_base  = X_all[test_mask]
y_test_real  = y_all[test_mask]
years_test   = df_full.loc[test_mask, "An"].values

if len(y_test_real) == 0:
    raise ValueError("No rows found for test period 2023–2025 in full_gen8.csv.")

#Baseline State Space model (trend + AR(1))
model = UnobservedComponents(
    endog=y_train_base,
    level="local linear trend",
    autoregressive=1,
    exog=X_train_base,
    stochastic_level=True,
    stochastic_trend=True,
)

res_base = model.fit(method="powell", disp=False)
print(res_base.summary())

#baseline forecast for 2023–2025
steps = len(y_test_real)
fc_base = res_base.get_forecast(steps=steps, exog=X_test_base)
y_pred_base = fc_base.predicted_mean.values  # baseline predictions

#Residuals on train (for GBR)
fitted_train = res_base.fittedvalues  # same index as y_train_base
df_train_base = df_full.loc[train_mask].copy()
df_train_base["base_fitted"] = fitted_train.values
df_train_base["residual"] = df_train_base["price_per_sqm"] - df_train_base["base_fitted"]

#Prepare features for GBR (train on residuals)
feature_cols_ml = [
    "index_cost_mat",
    "urban_incr_proc",
    "rata_infl",
    "pib_dolj_mld_ron",
    "price_lag1",
    "price_lag2",
]

#train rows where we have full lags
ml_train_mask = train_mask & df_full["price_lag2"].notna()
df_ml_train = df_full.loc[ml_train_mask].copy()

#attach residuals (by year)
df_ml_train = df_ml_train.merge(
    df_train_base[["An", "residual"]],
    on="An",
    how="inner",
)

X_ml_train = df_ml_train[feature_cols_ml]
y_ml_train = df_ml_train["residual"]

#test features for 2023–2025 (with lags)
ml_test_mask = test_mask & df_full["price_lag2"].notna()
df_ml_test = df_full.loc[ml_test_mask].copy()
X_ml_test = df_ml_test[feature_cols_ml]

#sanity check
if len(df_ml_test) != steps:
    print("Warning: ml_test rows != test steps, check lags alignment.")
    print("Years in df_ml_test:", df_ml_test["An"].tolist())
    print("Years in test_mask:", years_test.tolist())


#Fit GBR on residuals
gbr_resid = GradientBoostingRegressor(
    n_estimators=900,
    learning_rate=0.05,
    max_depth=2,
    subsample=0.4,
    min_samples_split=2,
    min_samples_leaf=1,
)

gbr_resid.fit(X_ml_train, y_ml_train)
resid_pred = gbr_resid.predict(X_ml_test)

#GBR simplu pe pret direct (nu pe reziduuri)
#train pe aceiași ani (2012–2022) unde avem lag-uri complete
y_ml_train_price = df_ml_train["price_per_sqm"].values

gbr_direct = GradientBoostingRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.7,
        max_features="sqrt",
        loss="absolute_error",
        random_state=42
)

gbr_direct.fit(X_ml_train, y_ml_train_price)
gbr_pred = gbr_direct.predict(X_ml_test)

#LightGBM pe pret direct
lgbm = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
lgbm.fit(X_ml_train, y_ml_train_price)
lgbm_pred = lgbm.predict(X_ml_test)

#XGBoost pe pret direct
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=2,
    subsample=0.4,
    objective="reg:squarederror",
)
xgb.fit(X_ml_train, y_ml_train_price)
xgb_pred = xgb.predict(X_ml_test)

cat = CatBoostRegressor(
    depth=6,
    iterations=900,
    learning_rate=0.05,
    loss_function="MAE",
    verbose=0,
    random_state=42
)

cat.fit(X_ml_train, y_ml_train_price)
cat_pred = cat.predict(X_ml_test)

#fit GBR pe reziduuri
gbr_resid = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
gbr_resid.fit(X_ml_train, y_ml_train)
resid_pred = gbr_resid.predict(X_ml_test)

from sklearn.ensemble import ExtraTreesRegressor

et = ExtraTreesRegressor(
    n_estimators=1000,
    max_depth=10,
)
et.fit(X_ml_train, y_ml_train_price)
et_pred = et.predict(X_ml_test)

from ngboost import NGBRegressor
from ngboost.distns import Normal

# NGBoost pe pret direct
ngb = NGBRegressor(
    Dist=Normal,
    n_estimators=1500,
    learning_rate=0.03,
    minibatch_frac=1.0,
    random_state=42,
    verbose=False
)

ngb.fit(X_ml_train, y_ml_train_price)
ngb_pred = ngb.predict(X_ml_test)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# Kernel: constant * RBF + noise
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
         + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3))

gpr = Pipeline([
    ("scaler", StandardScaler()),
    ("gpr", GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=42
    ))
])

gpr.fit(X_ml_train, y_ml_train_price)

# Predict mean + std (uncertainty)
gpr_pred, gpr_std = gpr.predict(X_ml_test, return_std=True)

#Hybrid predictions = baseline + residual_correction
df_hybrid = df_ml_test[["An"]].copy()
df_hybrid["price_real"]  = y_test_real.values
df_hybrid["base_pred"]   = y_pred_base
df_hybrid["gbr_pred"]    = gbr_pred
df_hybrid["lgbm_pred"]   = lgbm_pred
df_hybrid["xgb_pred"]    = xgb_pred
df_hybrid["cat_pred"] = cat_pred
df_hybrid["et_pred"] = et_pred
df_hybrid["ngb_pred"] = ngb_pred
df_hybrid["resid_pred"]  = resid_pred
df_hybrid["hybrid_pred"] = df_hybrid["base_pred"] + df_hybrid["resid_pred"]

df_hybrid["gpr_pred"] = gpr_pred
df_hybrid["gpr_std"] = gpr_std
df_hybrid["gpr_low_95"]  = df_hybrid["gpr_pred"] - 1.96 * df_hybrid["gpr_std"]
df_hybrid["gpr_high_95"] = df_hybrid["gpr_pred"] + 1.96 * df_hybrid["gpr_std"]

#Metrics: baseline vs GBR direct vs LGBM vs XGB vs hybrid
def compute_metrics(y_true, y_hat):
    abs_err = np.abs(y_hat - y_true)
    mae  = abs_err.mean()
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    mape = (abs_err / y_true * 100.0).mean()
    return mae, rmse, mape

mae_base, rmse_base, mape_base = compute_metrics(
    df_hybrid["price_real"].values,
    df_hybrid["base_pred"].values,
)
mae_gbr, rmse_gbr, mape_gbr = compute_metrics(
    df_hybrid["price_real"].values,
    df_hybrid["gbr_pred"].values,
)
mae_lgbm, rmse_lgbm, mape_lgbm = compute_metrics(
    df_hybrid["price_real"].values,
    df_hybrid["lgbm_pred"].values,
)
mae_xgb, rmse_xgb, mape_xgb = compute_metrics(
    df_hybrid["price_real"].values,
    df_hybrid["xgb_pred"].values,
)
mae_cat, rmse_cat, mape_cat = compute_metrics(
    df_hybrid["price_real"].values,
    df_hybrid["cat_pred"].values,
)

mae_hyb, rmse_hyb, mape_hyb = compute_metrics(
    df_hybrid["price_real"].values,
    df_hybrid["hybrid_pred"].values,
)
mae_et, rmse_et, mape_et = compute_metrics(
    df_hybrid["price_real"].values,
    df_hybrid["et_pred"].values,
)
mae_ngb, rmse_ngb, mape_ngb = compute_metrics(
    df_hybrid["price_real"].values,
    df_hybrid["ngb_pred"].values,
)
mae_gpr, rmse_gpr, mape_gpr = compute_metrics(
    df_hybrid["price_real"].values,
    df_hybrid["gpr_pred"].values,
)

print("\n=== Baseline State Space (2012–2022 -> 2023–2025) ===")
print(df_hybrid[["An", "price_real", "base_pred"]])
print(f"\nMAE  (base)  = {mae_base:.2f} RON/sqm")
print(f"RMSE (base)  = {rmse_base:.2f} RON/sqm")
print(f"MAPE (base)  = {mape_base:.2f} %")

print("\n=== GBR simplu (direct pe price_per_sqm) ===")
print(df_hybrid[["An", "price_real", "gbr_pred"]])
print(f"\nMAE  (GBR)   = {mae_gbr:.2f} RON/sqm")
print(f"RMSE (GBR)   = {rmse_gbr:.2f} RON/sqm")
print(f"MAPE (GBR)   = {mape_gbr:.2f} %")

print("\n=== LightGBM (direct pe price_per_sqm) ===")
print(df_hybrid[["An", "price_real", "lgbm_pred"]])
print(f"\nMAE  (LGBM)  = {mae_lgbm:.2f} RON/sqm")
print(f"RMSE (LGBM)  = {rmse_lgbm:.2f} RON/sqm")
print(f"MAPE (LGBM)  = {mape_lgbm:.2f} %")

print("\n=== XGBoost (direct pe price_per_sqm) ===")
print(df_hybrid[["An", "price_real", "xgb_pred"]])
print(f"\nMAE  (XGB)   = {mae_xgb:.2f} RON/sqm")
print(f"RMSE (XGB)   = {rmse_xgb:.2f} RON/sqm")
print(f"MAPE (XGB)   = {mape_xgb:.2f} %")

print("\n=== Hybrid (State Space + GBR residuals) ===")
print(df_hybrid[["An", "price_real", "hybrid_pred"]])
print(f"\nMAE  (hybrid) = {mae_hyb:.2f} RON/sqm")
print(f"RMSE (hybrid) = {rmse_hyb:.2f} RON/sqm")
print(f"MAPE (hybrid) = {mape_hyb:.2f} %")

print("\n=== CatBoost (direct pe price_per_sqm) ===")
print(df_hybrid[["An", "price_real", "cat_pred"]])
print(f"\nMAE  (hybrid) = {mae_cat:.2f} RON/sqm")
print(f"RMSE (hybrid) = {rmse_cat:.2f} RON/sqm")
print(f"MAPE (hybrid) = {mape_cat:.2f} %")

print("\n=== ExtraTrees (direct pe price_per_sqm) ===")
print(df_hybrid[["An", "price_real", "et_pred"]])
print(f"\nMAE  (hybrid) = {mae_et:.2f} RON/sqm")
print(f"RMSE (hybrid) = {rmse_et:.2f} RON/sqm")
print(f"MAPE (hybrid) = {mape_et:.2f} %")

print("\n=== NGBoost (direct pe price_per_sqm) ===")
print(df_hybrid[["An", "price_real", "ngb_pred"]])
print(f"\nMAE  (NGB)   = {mae_ngb:.2f} RON/sqm")
print(f"RMSE (NGB)   = {rmse_ngb:.2f} RON/sqm")
print(f"MAPE (NGB)   = {mape_ngb:.2f} %")

print("\n=== Gaussian Process (direct pe price_per_sqm) ===")
print(df_hybrid[["An", "price_real", "gpr_pred"]])
print(f"\nMAE  (GPR)   = {mae_gpr:.2f} RON/sqm")
print(f"RMSE (GPR)   = {rmse_gpr:.2f} RON/sqm")
print(f"MAPE (GPR)   = {mape_gpr:.2f} %")

print("\n=== GPR 95% confidence intervals ===")
print(df_hybrid[["An", "price_real", "gpr_low_95", "gpr_high_95"]])

df_hybrid.to_csv("hybrid_2012_2022_vs_2023_2025.csv", index=False)
print("\nSaved: hybrid_2012_2022_vs_2023_2025.csv")