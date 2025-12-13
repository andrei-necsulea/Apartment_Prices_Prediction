import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = "ins_indici_cost_mat_construct_clean.csv"
OUTPUT_CSV = "index_cost_mat_1900_2025.csv"

df_raw = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values("date")

df_raw["indice_to_2021"] = pd.to_numeric(df_raw["indice_to_2021"], errors="coerce")
df_raw = df_raw.set_index("date").sort_index()

#seria anuala(media pe an)
df_annual = df_raw["indice_to_2021"].resample("AS").mean()
df_annual = df_annual.to_frame(name="indice_to_2021")

#parametri pentru backfill
MIN_1900 = 25.0

first_year = int(df_annual.index.year.min())
first_val = df_annual.loc[df_annual.index.year == first_year, "indice_to_2021"].iloc[0]

print(f"[INFO] Primul an INS: {first_year}, valoare: {first_val:.2f}")

#ani pentru backfill: 1900 - first_year-1
years_past = np.arange(1900, first_year)
n_past = len(years_past)

#curba exponentiala 1900 - first_year-1
#index(year) = MIN_1900 * (first_val / MIN_1900)^((year-1900)/(first_year-1900))
ratio = first_val / MIN_1900
powers = (years_past - 1900) / (first_year - 1900)
values_past = MIN_1900 * (ratio ** powers)

df_past = pd.DataFrame({
    "year": years_past,
    "index_cost_mat": values_past
}).set_index("year")

#perioada cu valori reale INS (first_year - last_year)
df_obs = df_annual.copy()
df_obs["year"] = df_obs.index.year
df_obs = df_obs.set_index("year")
df_obs = df_obs.rename(columns={"indice_to_2021": "index_cost_mat"})

last_year = int(df_obs.index.max())

#perioada dupa ultimul an INS (last_year+1 - 2025) – tinem constant sau usoara extrapolare
years_future = np.arange(last_year + 1, 2026)
values_future = np.full_like(years_future, fill_value=df_obs.loc[last_year, "index_cost_mat"], dtype=float)

df_future = pd.DataFrame({
    "year": years_future,
    "index_cost_mat": values_future
}).set_index("year")

#combinam totul 1900–2025
df_all = pd.concat([df_past, df_obs, df_future])
df_all = df_all.sort_index()

#punem data de 1 ianuarie din fiecare an
df_all["date"] = pd.to_datetime(df_all.index.astype(str) + "-01-01")
df_all['An'] = df_all["date"].dt.year
df_all = df_all[["An", "date", "index_cost_mat"]].reset_index(drop=True)

df_all_year_only = df_all.copy()
df_all_year_only.drop(["date"], axis=1, inplace=True)
df_all_year_only['index_cost_mat'] = df_all_year_only["index_cost_mat"].round(3)

df_all_year_only.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")

print("\nPrimii 15 ani (1900–1914):")
print(df_all.head(15))

print("\nUltimii 15 ani:")
print(df_all.tail(15))

plt.figure(figsize=(12, 5))
plt.plot(df_all["date"], df_all["index_cost_mat"],
         label="Model 1900–2025", color="steelblue")

plt.scatter(df_annual.index, df_annual["indice_to_2021"],
            color="red", label="INS anual observat")

plt.title("Indice cost materiale constructii - 2021 = 100% – backfill 1900–2025")
plt.xlabel("Anul")
plt.ylabel("Indice (2021 = 100%)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()