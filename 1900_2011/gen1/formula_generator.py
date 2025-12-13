'''
Nu este bun pentru ca,  chiar daca datele sunt generate gradual si par a fi foarte aproape de realitatea economica a perioadei 1900-2011,
totusi exista o discrepanta prea mare dintre 2011(830 RON/sqm) si 2012(aprox 3260/m2) ceea ce este total gresit si inseamna ca datele nu comunica intre ele
Chit ca sunt perioade diferite, discrepanta este prea mare si ar fi ca si cum as vorbi despre doua lumi diferite...
si atunci, voi folosi un model statistic pentru a evita discrepantele date de aproximarea coeficientilor si a noise-ului.
'''

import numpy as np
import pandas as pd
from datetime import date

prices = []

#plecam de la aprox 250 lei/sqm in 1900, ceea ce e realist pentru salariile vremii
#un functionar avea in jur de 260 lei salariu pe luna
current_price = 250.0

def add(y, p):
    prices.append({
        "date": date(y, 12, 31),
        "price_per_sqm": float(p)     # convertim la float
    })


'''
  formula generala
  current_price = current_price * (1 + trend) + noise

  trend poate fi:
     -un procent fix (ex: +0.005, -0.02)
     -sau un procent random (ex: np.random.uniform(0.10, 0.15))

  zgomot_aleator este:
     -np.random.normal(media, deviația) - (ex: np.random.normal(0, 0.4))
'''

#crestere lenta
for year in range(1900, 1914):
    current_price = current_price * 1.005 + np.random.normal(0, 0.5)
    add(year, current_price)

#WW1
for year in range(1914, 1919):
    current_price = current_price * 0.99 + np.random.normal(0, 0.5)
    add(year, current_price)

#revenire lenta
for year in range(1919, 1939):
    current_price = current_price * 1.005 + np.random.normal(0, 0.5)
    add(year, current_price)

#WW2
for year in range(1939, 1946):
    current_price = current_price * 0.98 + np.random.normal(0, 1.0)
    add(year, current_price)

#comunism 1946–1989
#creștere stabila si predictibila
for year in range(1946, 1990):
    current_price = current_price * 1.015 + np.random.normal(0, 2.0)
    add(year, current_price)

#tranzitia, revolutia 1990–1999
for year in range(1990, 2000):
    current_price = current_price * (1 + np.random.uniform(0.00, 0.02)) + np.random.normal(0, 5)
    add(year, current_price)

#boom 2000–2007
for year in range(2000, 2008):
    current_price = current_price * (1 + np.random.uniform(0.10, 0.15)) + np.random.normal(0, 10)
    add(year, current_price)

#crisis 2008–2010
for year in range(2008, 2011):
    current_price = current_price * (1 + np.random.uniform(-0.15, -0.10)) + np.random.normal(0, 10)
    add(year, current_price)

#2011 - stabilizare)
current_price = current_price * (1 + np.random.uniform(0.02, 0.04)) + np.random.normal(0, 5)
add(2011, current_price)

df_hist = pd.DataFrame(prices)

df_hist["date"] = pd.to_datetime(df_hist["date"])

df_hist = df_hist.sort_values("date").reset_index(drop=True)

df_hist.to_csv("craiova_apartment_prices_1900_2011.csv", index=False)

print("\nSaved successfully!\n!")
print("craiova_apartment_prices_1900_2011.csv")

def generate_chart():
    import matplotlib.pyplot as plt

    df = pd.read_csv("craiova_apartment_prices_1900_2011.csv", parse_dates=["date"])

    plt.figure(figsize=(14, 6))

    plt.plot(df["date"], df["price_per_sqm"], color="steelblue", linewidth=2)

    plt.title(
        "Evolution of the average price per square meter (RON/sqm)\nCraiova, 1900–2011",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Year", fontsize=13)
    plt.ylabel("Price (RON/sqm)", fontsize=13)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig("craiova_prices_1900_2011_lei_gen1.png", dpi=200)
    plt.show()

    print("Chart saved: craiova_prices_1900_2011_gen1.png")

generate_chart()