import pandas as pd

df_1900_1970 = pd.read_csv("inflation_1900_1970_model.csv")
df_1971_2024 = pd.read_csv("ins_INFL_1971_2024.csv")

df_1900_1970['An'] = df_1900_1970['An'].astype(int)
df_1971_2024['An'] = df_1971_2024['An'].astype(int)

infl_2025 = df_1971_2024['rata_infl'].tail(3).mean().round(2)


row_2025 = pd.DataFrame({
    "An": [2025],
    "rata_infl": [infl_2025]
})

df_1971_2025 = pd.concat([df_1971_2024[['An', 'rata_infl']], row_2025], ignore_index=True)

df_1900_2025 = pd.concat([df_1900_1970, df_1971_2025], ignore_index=True)

df_1900_2025.to_csv("inflation_1900_2025.csv", index=False)
print("Saved : inflation_1900_2025.csv")