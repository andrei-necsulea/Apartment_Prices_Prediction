import pandas as pd

df_hist = pd.read_csv("../1900_2011/gen6/craiova_apartment_prices_1900_2011_gen6.csv", parse_dates=["date"])
df_real = pd.read_csv("../2012_2025/craiova_apartment_prices_2012_2025.csv", parse_dates=["date"])

df_full = pd.concat([df_hist, df_real], ignore_index=True)
df_full = df_full.sort_values("date").reset_index(drop=True)

df_full["price_per_sqm"] = df_full["price_per_sqm"].round(2)

df_full.to_csv("craiova_apartment_prices_1900_2025.csv", index=False)

print("\nAm salvat fisierul cu succes!\n")
print("craiova_apartment_prices_1900_2025.csv")

def generate_chart():
    import matplotlib.pyplot as plt

    # Citim seria istorica deja generata
    df = pd.read_csv("craiova_apartment_prices_1900_2025.csv", parse_dates=["date"])

    plt.figure(figsize=(14, 6))

    plt.plot(df["date"], df["price_per_sqm"], color="steelblue", linewidth=2)

    plt.title(
        "Evoluția prețului mediu pe metrul pătrat (LEI/m²)\nCraiova, 1900–2025",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Anul", fontsize=13)
    plt.ylabel("Preț (LEI/m²)", fontsize=13)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig("craiova_prices_1900_2025.png", dpi=200)
    plt.show()

    print("Grafic salvat ca: craiova_prices_1900_2025.png")

generate_chart()