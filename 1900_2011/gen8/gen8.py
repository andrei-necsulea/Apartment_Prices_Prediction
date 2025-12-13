import numpy as np
import pandas as pd
from datetime import date
import statistics

def generate_historical_series(seed: int | None = 42) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    prices = []

    #salarii_lunare_1900 = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300]
    #salariu_mediu_lunar_1900 = statistics.mean(salarii_lunare_1900)

    def adjust_salary_to_modern():
        salariu_mediu_2025 = 5387
        period = 2025-1900
        g = 0.02
        modern_salary = salariu_mediu_2025 / ((1.0 + g) ** period)
        return modern_salary

    salariu_mediu_lunar_1900 = adjust_salary_to_modern()

    suprafata_medie_apartament = 50
    cati_ani_de_plata = 2

    current_price = (salariu_mediu_lunar_1900 * 12 * cati_ani_de_plata)/suprafata_medie_apartament

    print(current_price)

    def add(y, p):
        prices.append({
            "date": date(y, 1, 1),
            "price_per_sqm": float(p)
        })

    # 1900-1913 (crestere lenta)
    for year in range(1900, 1914):
        current_price = current_price * 1.005 + np.random.normal(0, 0.2)
        add(year, current_price)

    # 1914-1918 (WW1)
    for year in range(1914, 1919):
        current_price = current_price * 0.99 + np.random.normal(0, 0.4)
        add(year, current_price)

    # 1919-1938 (revenire lenta)
    for year in range(1919, 1939):
        current_price = current_price * 1.005 + np.random.normal(0, 0.19)
        add(year, current_price)

    # 1939-1945 (WW2)
    for year in range(1939, 1946):
        current_price = current_price * 0.98 + np.random.normal(0, 1.2)
        add(year, current_price)

    # 1946-1989 (comunism, stabilitate)
    for year in range(1946, 1990):
        current_price = current_price * 1.015 + np.random.normal(0, 1)
        add(year, current_price)

    # 1990-1999 (tranzitie, volatilitate mare)
    for year in range(1990, 2000):
        current_price = current_price * (1 + np.random.uniform(0.00, 0.02)) + np.random.normal(0, 15)
        add(year, current_price)

    # 2000-2007 (boom imobiliar)
    for year in range(2000, 2008):
        current_price = current_price * (1 + np.random.uniform(0.10, 0.15)) + np.random.normal(0, 10)
        add(year, current_price)

    # 2008-2010 (criza imobiliara)
    for year in range(2008, 2011):
        current_price = current_price * (1 + np.random.uniform(-0.15, -0.10)) + np.random.normal(0, 5)
        add(year, current_price)

    # 2011 (stabilizare usoara)
    current_price = current_price * (1 + np.random.uniform(0.02, 0.04)) + np.random.normal(0, 3)
    add(2011, current_price)

    df = pd.DataFrame(prices)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# 2011 - 75% din pretul din 2012

REAL_CSV = "../../2012_2025/craiova_apartment_prices_2012_2025.csv"

df_real = pd.read_csv(REAL_CSV, parse_dates=["date"])
first_real_price = df_real.iloc[0]["price_per_sqm"]

#Target 2011 = 65% din 2012-01
TARGET_2011 = first_real_price * 0.65

df_hist = generate_historical_series(seed=42)
price_2011_raw = df_hist.loc[df_hist["date"].dt.year == 2011, "price_per_sqm"].iloc[0]

scale_factor = TARGET_2011 / price_2011_raw

print("Pret 2011 brut:", price_2011_raw)
print("Pret 2012 real:", first_real_price)
print("Tintim 2011 la:", TARGET_2011)
print("Factor scalare:", scale_factor)

df_hist["price_per_sqm"] *= scale_factor

print("Pret 2011 dupa scalare:",
      df_hist.loc[df_hist["date"].dt.year == 2011, "price_per_sqm"].iloc[0])

df_hist.to_csv("craiova_apartment_prices_1900_2011_gen8.csv", index=False)
print("\nAm salvat:", "craiova_apartment_prices_1900_2011_gen8.csv")

def generate_chart():
    import matplotlib.pyplot as plt

    df = pd.read_csv("craiova_apartment_prices_1900_2011_gen8.csv", parse_dates=["date"])

    plt.figure(figsize=(14, 6))

    plt.plot(df["date"], df["price_per_sqm"], color="steelblue", linewidth=2)

    plt.title(
        "Evolutia pretului mediu pe metrul patrat (RON/sqm)\nCraiova, 1900–2011",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Anul", fontsize=13)
    plt.ylabel("Pret (RON/sqm)", fontsize=13)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig("craiova_prices_1900_2011_lei.png", dpi=200)
    plt.show()

    print("Saved: craiova_prices_1900_2011_gen8.png")

generate_chart()

print(df_hist.head(10))
print(df_hist.tail(15))

print(df_hist.loc[df_hist["date"] == "1900-01-01"])
print(df_hist.loc[df_hist["date"] == "2011-01-01"])