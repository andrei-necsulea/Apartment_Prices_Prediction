import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.statespace.structural import UnobservedComponents
import matplotlib.pyplot as plt

# === CONFIG ===
REAL_CSV = "../../2012_2025/craiova_apartment_prices_2012_2025.csv"
TARGET_START = "1900-01-01"

# plafon maxim (upper asymptote) al pietei imobiliare
# poti ajusta 2500 / 3000 / 3500 daca vrei alt nivel
K_FIXED = 3000.0


# ----------------- helper functions -----------------

def logistic_fixed_K(t, A, B):
    """
    Model logistic cu plafon K fixat:
        y(t) = K_FIXED / (1 + A * exp(-B * t))
    """
    return K_FIXED / (1.0 + A * np.exp(-B * t))


def load_real_series(path: str) -> pd.Series:
    """
    Incarca seria reala 2012-2025.
    CSV-ul trebuie sa aiba coloanele: date, price_per_sqm.
    Returneaza un Series cu index lunar (MS).
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date").asfreq("MS")
    y = df["price_per_sqm"]

    if y.isna().any():
        y = y.interpolate()

    return y


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Numarul de luni intre doua date (start inclus, end exclus)."""
    return (end.year - start.year) * 12 + (end.month - start.month)


# ----------------- main pipeline -----------------

def main():
    # 1) incarcam seria reala
    y_real = load_real_series(REAL_CSV)          # 2012..2025
    first_date = y_real.index.min()
    last_date = y_real.index.max()

    # t_real = 0..N-1 (numar luni de la inceputul seriei reale)
    t_real = np.arange(len(y_real))
    y_vals = y_real.values

    # 2) fit logistic CU K FIXAT pe 2012-2025
    # modelul logistic_fixed_K are DOAR 2 parametri: A si B
    p0 = (10.0, 0.01)  # valori initiale pentru A si B

    popt, _ = curve_fit(
        logistic_fixed_K,
        t_real,
        y_vals,
        p0=p0,
        maxfev=20000
    )

    A_hat, B_hat = popt
    print(f"Parametri logistic cu K fixat={K_FIXED}: A={A_hat:.4f}, B={B_hat:.6f}")

    # 3) extindem logisticul pana in 1900-01-01
    target_start = pd.to_datetime(TARGET_START)
    n_back = months_between(target_start, first_date)

    # t_full: valori negative pentru trecut, apoi 0..N-1 pentru intervalul observat
    t_full = np.arange(-n_back, len(y_real))
    y_log_full = logistic_fixed_K(t_full, A_hat, B_hat)

    # index calendaristic pentru 1900..ultimul punct real
    full_index = pd.date_range(start=target_start,
                               end=last_date,
                               freq="MS")
    if len(full_index) != len(t_full):
        raise RuntimeError(
            f"Index length {len(full_index)} != t_full length {len(t_full)}"
        )

    # 4) reziduuri pe perioada observata (2012–2025)
    y_log_real = logistic_fixed_K(t_real, A_hat, B_hat)
    resid = y_vals - y_log_real
    resid_series = pd.Series(resid, index=y_real.index)

    # 5) model State Space pe reziduu (nivel + sezonalitate)
    mod_resid = UnobservedComponents(
        resid_series,
        level="local level",
        seasonal=12
    )
    res_resid = mod_resid.fit(disp=False)
    print(res_resid.summary())

    # reziduuri netezite pe perioada observata
    try:
        level_smoothed = pd.Series(
            res_resid.level.smoothed, index=resid_series.index
        )
        seasonal_smoothed = pd.Series(
            res_resid.seasonal.smoothed, index=resid_series.index
        )
        resid_smoothed = level_smoothed + seasonal_smoothed
    except AttributeError:
        resid_smoothed = pd.Series(
            res_resid.level.smoothed, index=resid_series.index
        )

    # 6) construim seria finala 1900-2025:
    #    inainte de 2012 -> doar logistic (reziduu = 0)
    #    dupa 2012      -> logistic + reziduu smoothed
    df_full = pd.DataFrame({
        "date": full_index,
        "price_trend_logistic": y_log_full,
    }).set_index("date")

    df_full["residual_smoothed"] = 0.0
    df_full.loc[resid_smoothed.index, "residual_smoothed"] = resid_smoothed

    df_full["price_final"] = (
        df_full["price_trend_logistic"] + df_full["residual_smoothed"]
    )
    df_full["is_observed"] = df_full.index >= first_date

    # 7) salvam CSV
    df_full.to_csv("craiova_logistic_state_space_1900_2025.csv")
    print("Am salvat craiova_logistic_state_space_1900_2025.csv")

    # 8) plot pentru verificare
    plt.figure(figsize=(14, 5))
    df_full["price_trend_logistic"].plot(
        label="Trend logistic (K fixat) 1900–2025", linewidth=2, alpha=0.7
    )
    df_full["price_final"].plot(
        label="Trend + reziduu smoothed", linewidth=2
    )
    y_real.plot(
        style="o", markersize=3, label="Date reale 2012–2025", color="orange"
    )
    plt.title("Indice apartamente Craiova – Logistic(K fix) + State Space (1900–2025)")
    plt.ylabel("€/mp")
    plt.legend()
    plt.tight_layout()
    plt.savefig("craiova_logistic_state_space_1900_2025.png", dpi=150)
    plt.close()
    print("Am salvat graficul craiova_logistic_state_space_1900_2025.png")


if __name__ == "__main__":
    main()