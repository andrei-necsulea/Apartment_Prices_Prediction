import numpy as np
import pandas as pd

df = pd.read_csv("ins_total_populatie_urban_dolj_1993_2025.csv")

df["An"] = df["An"].astype(int)
df["crestere_procente"] = df["crestere_procente"].astype(float)

val_1993 = df.loc[df["An"] == 1993, "crestere_procente"].iloc[0]

#definim ancora pentru 1900 (40% din valoarea din 1993)
val_1900 = 0.4 * val_1993

#construim cadru 1900–1993 (includem 1993 ca sa avem a doua ancora)
df_hist = pd.DataFrame({"An": np.arange(1900, 1994)})
df_hist["crestere_procente"] = np.nan

df_hist.loc[df_hist["An"] == 1900, "crestere_procente"] = val_1900
df_hist.loc[df_hist["An"] == 1993, "crestere_procente"] = val_1993

#interpolare log-lineara intre 1900 si 1993
log_vals = np.log(df_hist["crestere_procente"])
log_vals = log_vals.interpolate()
df_hist["crestere_procente"] = np.exp(log_vals).round(4)

#pastram doar 1900–1992
df_1900_1992 = df_hist[df_hist["An"] < 1993].copy()

df_1900_1992.to_csv("urban_population_dolj_1900_1992.csv", index=False)

print("Saved: urban_population_dolj_1900_1992.csv")