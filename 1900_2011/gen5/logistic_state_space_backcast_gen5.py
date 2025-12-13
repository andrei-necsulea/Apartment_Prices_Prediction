import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.statespace.structural import UnobservedComponents
import matplotlib.pyplot as plt

REAL_CSV = "../../2012_2025/craiova_apartment_prices_2012_2025.csv"
TARGET_START = "1900-01-01"

Y_LOW = 50.0#pret asimptotic in jur de anul 1900
Y_HIGH = 2500.0#plafon maxim pe termen lung

def logistic_unit(t, A, B):
    """
    Logistic pe intervalul [0,1] (plafon K = 1):
        L(t) = 1 / (1 + A * exp(-B * t))
    """
    return 1.0 / (1.0 + A * np.exp(-B * t))

def load_real_series(path: str) -> pd.Series:
    #Returneaza un Series cu index lunar (MS).
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date").asfreq("MS")
    y = df["price_per_sqm"]

    if y.isna().any():
        y = y.interpolate()

    return y


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)

def main():
    y_real = load_real_series(REAL_CSV)          # 2012..2025
    first_date = y_real.index.min()
    last_date = y_real.index.max()

    t_real = np.arange(len(y_real))
    y_vals = y_real.values

    #normalizam seria reala in functie de Y_LOW si Y_HIGH
    #y_norm aproximativ [0, 1]
    scale = Y_HIGH - Y_LOW
    y_norm = (y_vals - Y_LOW) / scale

    #clamp in [0,1] (in caz ca avem ceva peste plafon)
    y_norm = np.clip(y_norm, 0.0, 1.0)

    #fit logistic_unit pe y_norm
    p0 = (10.0, 0.01)

    popt, _ = curve_fit(
        logistic_unit,
        t_real,
        y_norm,
        p0=p0,
        maxfev=20000
    )

    A_hat, B_hat = popt
    print(f"Parametri logistici (normalizati): A={A_hat:.4f}, B={B_hat:.6f}")

    #extindem logisticul pana in 1900
    target_start = pd.to_datetime(TARGET_START)
    n_back = months_between(target_start, first_date)

    #t_full: valori negative pentru trecut, apoi 0...N-1 pentru intervalul observat
    t_full = np.arange(-n_back, len(y_real))
    L_full = logistic_unit(t_full, A_hat, B_hat)
    y_log_full = L_full * scale + Y_LOW

    #index calendaristic pentru 1900, ultimul punct real
    full_index = pd.date_range(start=target_start,
                               end=last_date,
                               freq="MS")
    if len(full_index) != len(t_full):
        raise RuntimeError(
            f"Index length {len(full_index)} != t_full length {len(t_full)}"
        )

    #logistic doar pe intervalul observat
    L_real = logistic_unit(t_real, A_hat, B_hat)
    y_log_real = L_real * scale + Y_LOW

    #reziduuri pe perioada observata (2012–2025)
    resid = y_vals - y_log_real
    resid_series = pd.Series(resid, index=y_real.index)

    #model State Space pe reziduu (nivel + sezonalitate lunara)
    mod_resid = UnobservedComponents(
        resid_series,
        level="local level",
        seasonal=12
    )
    res_resid = mod_resid.fit(disp=False)
    print(res_resid.summary())

    #reziduuri netezite pe perioada observata
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

    #construim seria finala 1900-2025:
    #inainte de 2012 : doar logistic (reziduu = 0)
    #dupa 2012 : logistic + reziduu smoothed
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

    df_full.to_csv("craiova_logistic_norm_state_space_1900_2025_gen5.csv")
    print("Saved: craiova_logistic_norm_state_space_1900_2025.csv")

    plt.figure(figsize=(14, 5))
    df_full["price_trend_logistic"].plot(
        label="Trend logistic (normalizat) 1900–2025", linewidth=2, alpha=0.7
    )
    df_full["price_final"].plot(
        label="Trend + reziduu smoothed", linewidth=2
    )
    y_real.plot(
        style="o", markersize=3, label="Date reale 2012–2025", color="orange"
    )
    plt.title("Indice apartamente Craiova – Logistic normalizat + State Space (1900–2025)")
    plt.ylabel("RON/sqm")
    plt.legend()
    plt.tight_layout()
    plt.savefig("craiova_logistic_norm_state_space_1900_2025.png", dpi=150)
    plt.close()
    print("Saved: craiova_logistic_norm_state_space_1900_2025.png")

if __name__ == "__main__":
    main()