import pandas as pd

df = pd.read_csv("craiova_apartment_prices_1900_2025_gen8.csv")

df["An"] = pd.to_datetime(df["date"], format="mixed").dt.year

dup_counts = df.groupby("An").size()
print(dup_counts[dup_counts > 1])

df_yearly = (
    df.groupby("An", as_index=False)["price_per_sqm"]
      .mean()
)

df_yearly["price_per_sqm"] = df_yearly["price_per_sqm"].round(2)

df_yearly.to_csv("craiova_apartment_prices_1900_2025_yearly_mean_gen8.csv", index=False)