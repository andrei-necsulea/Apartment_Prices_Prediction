import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

REAL_CSV = "../../2012_2025/craiova_apartment_prices_2012_2025.csv"
TARGET_START = "1900-01-01"


def load_real_series(path):
    df = pd.read_csv(path, parse_dates=["date"])

    #sortam datele, setam un index lunar, la inceput de luna(MS = Month Start).
    df = df.sort_values("date").set_index("date").asfreq("MS")

    '''
    Interpolarea se folosește pentru:
     - a completa lunile lipsa dupa ce seria este fortata intr-o frecventa lunara
     - pentru a regula seria de valori si pentru a o completa in caz de nan-uri
    Aceasta regularizare a datelor conduce la o mai buna antrenare a modelului
    '''
    y = df["price_per_sqm"].interpolate()

    return y


def main():
    y_real = load_real_series(REAL_CSV)

    #inversam valorile seriei si o reindexam numeric, practic de la 2025 la 2012
    y_inv = y_real.iloc[::-1].reset_index(drop=True)

    #antrenam modelul State Space pe seria inversata si reindexata numeric mai sus
    mod = UnobservedComponents(
        y_inv,
        level="local linear trend",
        seasonal=12
    )

    res = mod.fit(disp=False)

    '''
    Aici se face backcasting, 
    adica folosim un model pentru a genera valori in trecut, 
    inainte de inceputul seriei reale.

    Exemplu:
      Seria reala incepe în 2012-01, 
      dar noi vrem valori simulate din 2000–2011,
      trebuie calculat cati pasi trebuie sa parcurgem inapoi.
    '''
    target_start = pd.to_datetime(TARGET_START)

    '''
    calculam deci cate luni trebuie sa ne miscam inapoi in timp:
    in cazul nostru : (2012 - 1900) * 12 + (1-1) = 112 * 12 + 0 = 1344 luni
    '''
    n_backcast_months = (y_real.index.min().year - target_start.year) * 12 + \
                        (y_real.index.min().month - target_start.month)

    fc = res.get_forecast(steps=n_backcast_months)

    backcast_inv = fc.predicted_mean

    #lipim seria inversata + predicțiile inversate
    full_inv = pd.concat([y_inv, backcast_inv], ignore_index=True)

    #reinversam la ordinea reala in timp
    full = full_inv.iloc[::-1].reset_index(drop=True)

    #construim indexul calendaristic real
    full_index = pd.date_range(start=target_start, periods=len(full), freq="MS")
    full_series = pd.Series(full.values, index=full_index, name="price_per_sqm_smoothed")

    full_series.to_csv("backcast_correct_1900_2025.csv")
    print("Saved: backcast_correct_1900_2025.csv")

if __name__ == "__main__":
    main()