import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

REAL_CSV = "../../2012_2025/craiova_apartment_prices_2012_2025.csv"
TARGET_START = "1900-01-01"

def load_real(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date")
    df = df.set_index("date").asfreq("MS")
    return df["price_per_sqm"].interpolate()

y_real = load_real(REAL_CSV)

#construim axa de timp reala 2012–2025
x_real = np.arange(len(y_real))
y = y_real.values

#construim spline-ul
spline = make_interp_spline(x_real, y, k=3)

#calculam cate luni sunt pana in 1900
years = y_real.index.max().year - 1900
months = years * 12

#extindem axa in trecut
x_full = np.arange(-months, len(y_real))
y_spline_full = spline(x_full)

#construim indexul real
date_index = pd.date_range(start="1900-01-01", periods=len(x_full), freq="MS")
series = pd.Series(y_spline_full, index=date_index)

series.to_csv("spline_backcast_1900_2025.csv")
print("OK – spline_backcast_1900_2025.csv")

plt.figure(figsize=(14,5))
plt.plot(series, label="Spline Backcast")
plt.scatter(y_real.index, y_real.values, c="orange", s=10, label="Date reale")
plt.legend()
plt.show()