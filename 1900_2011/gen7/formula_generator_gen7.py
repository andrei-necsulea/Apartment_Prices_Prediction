import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

SEED = 42
TARGET_RATIO_2011 = 0.65      # 2011 = 65% din pretul din 2012-01
BASE_YEAR_PRICE = 250.0       # baza interna

GROWTH_PRE_WW1   = 0.006      # 1900-1913
GROWTH_INTERWAR  = 0.006      # 1919-1938
GROWTH_COMMUNISM = 0.010      # 1946-1989 (mai lent decat 1.5%)
BOOM_MIN, BOOM_MAX = 0.07, 0.12   # 2000-2007
CRISIS_MIN, CRISIS_MAX = -0.15, -0.10


def generate_historical_series(seed: int | None = SEED) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    prices = []
    current_price = BASE_YEAR_PRICE

    def add(y, p):
        prices.append({
            "date": date(y, 1, 1),
            "price_per_sqm": float(p)
        })

    # 1900-1913 (crestere lenta)
    for year in range(1900, 1914):
        current_price = current_price * (1 + GROWTH_PRE_WW1) + np.random.normal(0, 0.2)
        add(year, current_price)

    # 1914-1918 (WW1)
    for year in range(1914, 1919):
        current_price = current_price * 0.99 + np.random.normal(0, 0.4)
        add(year, current_price)

    # 1919-1938 (revenire lenta)
    for year in range(1919, 1939):
        current_price = current_price * (1 + GROWTH_INTERWAR) + np.random.normal(0, 0.2)
        add(year, current_price)

    # 1939-1945 (WW2)
    for year in range(1939, 1946):
        current_price = current_price * 0.98 + np.random.normal(0, 1.2)
        add(year, current_price)

    # 1946-1989 (comunism, mai lent)
    for year in range(1946, 1990):
        current_price = current_price * (1 + GROWTH_COMMUNISM) + np.random.normal(0, 1.0)
        add(year, current_price)

    # 1990-1999 (tranzitie, volatilitate)
    for year in range(1990, 2000):
        current_price = current_price * (1 + np.random.uniform(-0.02, 0.04)) + np.random.normal(0, 15)
        add(year, current_price)

    # 2000-2007 (boom mai domol)
    for year in range(2000, 2008):
        current_price = current_price * (1 + np.random.uniform(BOOM_MIN, BOOM_MAX)) + np.random.normal(0, 10)
        add(year, current_price)

    # 2008-2010 (criza)
    for year in range(2008, 2011):
        current_price = current_price * (1 + np.random.uniform(CRISIS_MIN, CRISIS_MAX)) + np.random.normal(0, 5)
        add(year, current_price)

    # 2011 (stabilizare)
    current_price = current_price * (1 + np.random.uniform(0.02, 0.04)) + np.random.normal(0, 3)
    add(2011, current_price)

    df = pd.DataFrame(prices)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    #legam seria de preturile reale 2012-2025 (RON modern)
    real_csv = "../../2012_2025/craiova_apartment_prices_2012_2025.csv"

    df_real = pd.read_csv(real_csv, parse_dates=["date"])
    first_real_price = df_real.iloc[0]["price_per_sqm"]

    TARGET_2011 = first_real_price * TARGET_RATIO_2011

    df_hist = generate_historical_series(seed=SEED)
    price_2011_raw = df_hist.loc[df_hist["date"].dt.year == 2011, "price_per_sqm"].iloc[0]

    scale_factor = TARGET_2011 / price_2011_raw
    print("Pret 2011 brut:", price_2011_raw)
    print("Pret 2012 real:", first_real_price)
    print("Tintim 2011 la:", TARGET_2011)
    print("Factor scalare:", scale_factor)

    #Aici toata seria devine in RON “modern”
    df_hist["price_per_sqm"] *= scale_factor

    print("Pret 2011 dupa scalare:",
          df_hist.loc[df_hist["date"].dt.year == 2011, "price_per_sqm"].iloc[0])

    df_hist.to_csv("craiova_apartment_prices_1900_2011_gen7.csv", index=False)
    print("\nAm salvat:", "craiova_apartment_prices_1900_2011_gen7.csv")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    plt.plot(df_hist["date"], df_hist["price_per_sqm"], linewidth=2)
    plt.title("Evolutia pretului mediu pe metrul patrat (RON/sqm)\nCraiova, 1900–2011", fontsize=16, fontweight="bold")
    plt.xlabel("Anul", fontsize=13)
    plt.ylabel("Pret (RON/sqm)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()