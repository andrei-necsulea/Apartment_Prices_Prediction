import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

prices_file = next(ROOT.rglob("craiova_apartment_prices_1900_2025_yearly_mean_gen8.csv"))
features_file = next(ROOT.rglob("new_features.csv"))

df_prices = pd.read_csv(prices_file)
df_features = pd.read_csv(features_file)

df_merged = df_features.merge(
    df_prices[["An", "price_per_sqm"]],
    on="An",
    how="left"
)

output_path = prices_file.parent / "full_gen8.csv"
df_merged.to_csv(output_path, index=False)

print("Saved : ", output_path)