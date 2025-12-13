import pandas as pd

df_1900_1992 = pd.read_csv("urban_population_dolj_1900_1992.csv")
df_1993_2025 = pd.read_csv("ins_total_populatie_urban_dolj_1993_2025.csv")

df_1900_2025 = pd.concat([df_1900_1992, df_1993_2025], ignore_index=True)

df_1900_2025.to_csv("urban_population_increase_dolj_1900_2025.csv", index=False)
print("Saved : urban_population_increase_dolj_1900_2025.csv")