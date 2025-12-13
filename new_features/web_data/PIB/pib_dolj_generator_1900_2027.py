import numpy as np
import pandas as pd

#2010–2014
df_2010_2014 = pd.read_csv("pib_dolj_2010_2014.csv")

#2019–2025(date + prognoze CNP)
df_2019_2025 = pd.read_csv("pib_dolj_2019_2025.csv")

#construim skeleton pentru 1900–2025
years_all = np.arange(1900, 2026)
series = pd.Series(index=years_all, dtype="float64")#PIB mld. lei, Dolj

for _, row in pd.concat([df_2010_2014, df_2019_2025]).iterrows():
    series.loc[int(row["An"])] = float(row["PIB"])

#definim ancorele istorice(procente din PIB 2010)

#valoarea reala pentru 2010
gdp_2010 = df_2010_2014.loc[df_2010_2014["An"] == 2010, "PIB"].iloc[0]

#procente aproximative fata de PIB 2010
#economie mult mai mică in 1900, crestere treptats, crestere mare dupa 1990
anchor_perc = {
    1900: 0.03,#3% din PIB 2010
    1914: 0.05,
    1918: 0.04,#scadere în WW1
    1938: 0.08,
    1945: 0.06,#distrugeri WW2
    1950: 0.07,
    1970: 0.15,
    1990: 0.35,
    2000: 0.55,
    2010: 1.00,#exact PIB 2010
}

anchors = {year: perc * gdp_2010 for year, perc in anchor_perc.items()}

#suprascriem in serie aceste ancore
for y, v in anchors.items():
    series.loc[y] = v

#asiguram si ancorele la capete(deja avem 1900 si 2027)
#2014 si 2019 sunt observate si ajuta la completarea zonei 2015–2018
#2025 vine din fisier ne bazam pe el ca ultima ancora prognozata


#interpolare log-liniara intre toate punctele cunoscute
known_years = series.dropna().index.to_numpy()

for i in range(len(known_years) - 1):
    y0 = known_years[i]
    y1 = known_years[i + 1]
    v0 = series.loc[y0]
    v1 = series.loc[y1]

    gap_years = np.arange(y0 + 1, y1)
    if len(gap_years) == 0:
        continue

    #interpolam in log ca sa obtinem o evolutie de tip crestere sau exponențiala
    log_v0, log_v1 = np.log(v0), np.log(v1)
    logs = np.linspace(log_v0, log_v1, len(gap_years) + 2)[1:-1]
    series.loc[gap_years] = np.exp(logs)

df_out = pd.DataFrame({
    "An": years_all,
    "pib_dolj_mld_lei": series.values
})

df_out.to_csv("pib_dolj_1900_2025.csv", index=False, float_format="%.3f")

print("Saved: pib_dolj_1900_2025.csv")