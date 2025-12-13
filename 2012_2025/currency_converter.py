# currency_converter.py

import pandas as pd

# Cursuri medii anuale EUR -> RON
EUR_RON_AVG = {
    2012: 4.4577,
    2013: 4.4189,
    2014: 4.4441,
    2015: 4.4457,
    2016: 4.4899,
    2017: 4.5697,
    2018: 4.6542,
    2019: 4.7464,
    2020: 4.8364,
    2021: 4.9212,
    2022: 4.9307,
    2023: 4.9482,
    2024: 4.9753,
    2025: 5.0382,
}

def convert_euro_to_lei(
    input_csv="craiova_apartment_prices_2012_2025_euro.csv",
    output_csv="craiova_apartment_prices_2012_2025_lei.csv",
):

    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Mapam anul la cursul mediu
    df["eur_ron_rate"] = df["year"].map(EUR_RON_AVG)

    # Verificam daca avem ani fara curs definit
    missing_years = df[df["eur_ron_rate"].isna()]["year"].unique()
    if len(missing_years) > 0:
        print("ATENTIE! Urmatorii ani nu au curs definit in dictionar:")
        print(missing_years)
        print("Completeaza dictionarul EUR_RON_AVG pentru acesti ani.")
        return

    # Calculam pretul pe m2 in lei
    df["price_per_sqm_lei"] = df["price_per_sqm"] * df["eur_ron_rate"]

    # Optional: rotunjim putin
    df["price_per_sqm_lei"] = df["price_per_sqm_lei"].round(2)

    # Salvam noul CSV (pastram si pretul in euro ca referinta)
    df.to_csv(output_csv, index=False)

    # Stergem coloana veche in EURO, daca exista
    if "price_per_sqm" in df.columns:
        df = df.drop(columns=["price_per_sqm"])

    #noul csv doar cu datetime si price_per_sqm
    df = df.rename(columns={'price_per_sqm_lei': 'price_per_sqm'})

    df.to_csv(
        'craiova_apartment_prices_2012_2025.csv',
        columns=['date', 'price_per_sqm'],
        index=False
    )

    print(f"Conversie finalizata cu succes!")
    print(f"Fisier de iesire: {output_csv}")


if __name__ == "__main__":
    convert_euro_to_lei()