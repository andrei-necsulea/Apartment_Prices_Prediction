import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Path to the final merged dataset (historical + real data + forecasts)
DATA_PATH = "https://github.com/andrei-necsulea/Apartment_Prices_Prediction/blob/main/results/craiova_apartment_prices_1900_2035.csv"


@st.cache_data
def load_data(path=DATA_PATH):
    """
    Load the apartment prices dataset, sort it by date,
    and add helper columns for 'year' and 'decade'.

    The CSV is expected to have at least:
        - 'date' (parseable as datetime)
        - 'price_per_sqm' (float, price in LEI/m²)
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    #Extract year and decade for grouping/filters
    df["year"] = df["date"].dt.year
    df["decade"] = (df["year"] // 10) * 10
    return df


#Load the full dataset once, using streamlit cache
df = load_data()

#Compute global min/max years for the slider
min_year = int(df["year"].min())
max_year = int(df["year"].max())

# Sidebar: Controls / Filters
st.sidebar.title("Visual Settings")

# Mode selector:
#-Full period
#-Only the real-data window (2012–2025)
mode = st.sidebar.radio(
    "Analysis mode:",
    ["All period 1900–2025", "Only real period (2012–2025)"],
)

#Default lower bound of the slider depends on the chosen mode
if mode == "Only real period (2012–2025)":
    default_min = 2012
else:
    default_min = min_year

#Year range slider for filtering the dataset
year_range = st.sidebar.slider(
    "Year interval",
    min_value=min_year,
    max_value=max_year,
    value=(default_min, max_year),
    step=1,
)

#Optional visual layers: historical events + rolling mean
show_events = st.sidebar.checkbox("Highlights historical periods", value=True)
show_rolling = st.sidebar.checkbox("Display rolling mean (5 years)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Date:**")
st.sidebar.write(f"Min: {min_year}, Max: {max_year}")

#Data filtering based on mode + selected year range
df_filtered = df.copy()

#If user selects "real only", restrict to 2012+ first
if mode == "Only real period (2012–2025)":
    df_filtered = df_filtered[df_filtered["year"] >= 2012]

#Then apply the year range filter from the slider
df_filtered = df_filtered[
    (df_filtered["year"] >= year_range[0]) & (df_filtered["year"] <= year_range[1])
].copy()

# Main title + description
st.title("Craiova real estate market – 1900–2025 (RON/sqm)")

st.markdown(
    """
This interactive dashboard uses:
- **1900–2011**: synthetic series generated based on the economic context (wars, communism, transition, boom, crisis),
- **2012–2025**: real data, converted to **RON/sqm**.

You can filter the period on the left and analyze price evolution, historical shocks and long-term trends.
"""
)

# Top Metrics for filtered interval
if not df_filtered.empty:
    #First/last price in the filtered window
    first_price = df_filtered["price_per_sqm"].iloc[0]
    last_price = df_filtered["price_per_sqm"].iloc[-1]
    pct_change = (last_price - first_price) / first_price * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Minimum price", f"{df_filtered['price_per_sqm'].min():.2f} RON/sqm")
    with col2:
        st.metric("Average price", f"{df_filtered['price_per_sqm'].mean():.2f} RON/sqm")
    with col3:
        st.metric("Maximum price", f"{df_filtered['price_per_sqm'].max():.2f} RON/sqm")
    with col4:
        #Percentage change vs. first value in the selected period
        st.metric(
            "diff % from the beginning",
            f"{pct_change:.1f} %",
            delta=f"{pct_change:.1f} %",
        )
else:
    st.warning("There is no data in the selected range.")

#Main time-series chart
st.subheader("Price evolution per square meter in the selected range")

#Create a Matplotlib figure embedded in Streamlit
fig, ax = plt.subplots(figsize=(14, 5))

if not df_filtered.empty:
    #Plot the raw price series
    ax.plot(df_filtered["date"], df_filtered["price_per_sqm"], label="Price (RON/sqm)")

    #Optional: rolling mean (5-year window)
    if show_rolling:
        df_filtered["rolling_5"] = (
            df_filtered["price_per_sqm"].rolling(window=5, min_periods=1).mean()
        )
        ax.plot(
            df_filtered["date"],
            df_filtered["rolling_5"],
            color="red",
            linewidth=2,
            alpha=0.7,
            label="Rolling mean (5 years)",
        )

    #Optional: highlight major historical/economic periods
    if show_events:
        #WW1
        ax.axvspan(
            pd.Timestamp("1914-01-01"),
            pd.Timestamp("1918-12-31"),
            color="red",
            alpha=0.15,
            label="WW1",
        )
        #WW2
        ax.axvspan(
            pd.Timestamp("1939-01-01"),
            pd.Timestamp("1945-12-31"),
            color="orange",
            alpha=0.15,
            label="WW2",
        )
        #Communism
        ax.axvspan(
            pd.Timestamp("1948-01-01"),
            pd.Timestamp("1989-12-31"),
            color="green",
            alpha=0.05,
            label="Comunism",
        )
        #Real-estate boom
        ax.axvspan(
            pd.Timestamp("2000-01-01"),
            pd.Timestamp("2007-12-31"),
            color="blue",
            alpha=0.10,
            label="Boom of prices",
        )
        #2008–2010 financial crisis
        ax.axvspan(
            pd.Timestamp("2008-01-01"),
            pd.Timestamp("2010-12-31"),
            color="purple",
            alpha=0.12,
            label="Crisis 2008–2010",
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Price (RON/sqm)")
    ax.grid(True, linestyle="--", alpha=0.3)

    #Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left")

#Render the matplotlib figure in streamlit
st.pyplot(fig)

#Decade-level overview (bar chart)
st.subheader("Average price over decades (historical overview)")

df_dec = (
    df.groupby("decade")["price_per_sqm"]
    .mean()
    .reset_index()
    .sort_values("decade")
)

#Simple bar chart using streamlit built-in chart function
st.bar_chart(
    df_dec.set_index("decade")["price_per_sqm"],
    height=300,
)

#Price distribution (histogram-like bar chart)
st.subheader("Price distribution in the selected range")

if not df_filtered.empty:
    #Choose number of bins via a slider
    bins = st.slider("Number of bins for histogram", 10, 80, 30)

    #Compute histogram using NumPy
    hist_values, bin_edges = np.histogram(
        df_filtered["price_per_sqm"], bins=bins
    )

    #Build a small dataframe so we can use st.bar_chart
    hist_df = pd.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "count": hist_values,
        }
    )
    #Center of each bin (for the x-axis)
    hist_df["bin_center"] = (hist_df["bin_left"] + hist_df["bin_right"]) / 2

    st.bar_chart(
        hist_df.set_index("bin_center")["count"],
        height=300,
    )

st.subheader("Raw data in the selected range")

st.dataframe(
    df_filtered[["date", "price_per_sqm"]]
    .reset_index(drop=True)
)

#Extra sidebar options for model-based data
show_forecast = st.sidebar.checkbox("Show forecast 2026–2050", value=True)
show_gen8_debug = st.sidebar.checkbox("Show generator (gen8) debug view", value=False)

#Cached loaders for forecast and gen8 datasets
@st.cache_data
def load_forecast(path="https://github.com/andrei-necsulea/Apartment_Prices_Prediction/blob/main/results/forecast_2026_2050.csv"):
    df_fc = pd.read_csv(path)

    # Build a proper datetime column
    if "date" in df_fc.columns:
        df_fc["date"] = pd.to_datetime(df_fc["date"])
    elif "Date" in df_fc.columns:
        df_fc["date"] = pd.to_datetime(df_fc["Date"])
    elif "An" in df_fc.columns:
        df_fc["date"] = pd.to_datetime(df_fc["An"].astype(str) + "-01-01")
    else:
        raise ValueError("Forecast CSV must contain 'date', 'Date' or 'An'.")

    #Normalize price column name
    if "predicted_price_sqm" in df_fc.columns:
        df_fc = df_fc.rename(columns={"predicted_price_sqm": "price_per_sqm"})
    elif "price_per_sqm" not in df_fc.columns:
        raise ValueError("Forecast CSV must contain 'predicted_price_sqm' or 'price_per_sqm'.")

    df_fc = df_fc[["date", "price_per_sqm"]].sort_values("date").reset_index(drop=True)
    return df_fc


@st.cache_data
def load_gen8(path="https://github.com/andrei-necsulea/Apartment_Prices_Prediction/blob/main/merged_data/full_gen8.csv"):
    """
    Load the gen8 generator dataset (historical model features).
    Tries to infer the main price column and keep 'An' as x-axis.
    """
    df_g8 = pd.read_csv(path)

    #Try to infer which column holds the price
    candidate_price_cols = [
        "price_per_sqm_final",
        "price_per_sqm_initial",
        "price_per_sqm",
    ]
    price_col = None
    for c in candidate_price_cols:
        if c in df_g8.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError("Could not find a price column in full_gen8.csv.")

    df_g8 = df_g8[["An", price_col]].rename(columns={price_col: "price_per_sqm"})
    df_g8 = df_g8.sort_values("An").reset_index(drop=True)
    return df_g8

@st.cache_data
def load_gen8(path="https://github.com/andrei-necsulea/Apartment_Prices_Prediction/blob/main/merged_data/full_gen8.csv"):
    """
    Load the gen8 generator dataset (historical model features).
    Keeps all columns (pib, inflatie, index_cost_mat etc.) and
    normalizes doar coloana de preț la 'price_per_sqm'.
    """
    df_g8 = pd.read_csv(path)

    if "An" not in df_g8.columns:
        raise ValueError("full_gen8.csv must contain column 'An'.")

    #try to infer which column holds the price
    candidate_price_cols = [
        "price_per_sqm_final",
        "price_per_sqm_initial",
        "price_per_sqm",
    ]
    price_col = None
    for c in candidate_price_cols:
        if c in df_g8.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError("Could not find a price column in full_gen8.csv.")

    # rename doar coloana de preț, restul rămân la fel
    if price_col != "price_per_sqm":
        df_g8 = df_g8.rename(columns={price_col: "price_per_sqm"})

    df_g8 = df_g8.sort_values("An").reset_index(drop=True)
    return df_g8


#optional gen8 debug / insight section
if show_gen8_debug:
    st.subheader("Formula historical generator – debug view")

    df_gen8 = load_gen8()

    fig_g8, ax_g8 = plt.subplots(figsize=(14, 4))
    ax_g8.plot(df_gen8["An"], df_gen8["price_per_sqm"], linewidth=1.8)
    ax_g8.set_xlabel("Year")
    ax_g8.set_ylabel("Price (RON/sqm)")
    ax_g8.set_title("Generated synthetic historical series (model-level view)")
    ax_g8.grid(True, linestyle="--", alpha=0.3)

    st.pyplot(fig_g8)

    st.markdown("**Full data :**")
    #afisam toate coloanele (An, index_cost_mat, urban_incr_proc, rata_infl, pib_dolj_mld_ron, price_per_sqm etc.)
    st.dataframe(df_gen8.head(50))

@st.cache_data
def load_ensemble_forecast(
    path="https://github.com/andrei-necsulea/Apartment_Prices_Prediction/blob/main/results/ensemble_ss_catboost_forecast_2026_2050.csv"
):
    df_ens = pd.read_csv(path)

    # build datetime
    if "date" in df_ens.columns:
        df_ens["date"] = pd.to_datetime(df_ens["date"])
    elif "An" in df_ens.columns:
        df_ens["date"] = pd.to_datetime(df_ens["An"].astype(str) + "-01-01")
    else:
        raise ValueError("Ensemble CSV must contain 'An' or 'date'.")

    # normalize final prediction column
    if "price_pred" in df_ens.columns:
        df_ens = df_ens.rename(columns={"price_pred": "price_per_sqm"})
    elif "price_per_sqm" not in df_ens.columns:
        raise ValueError(
            "Ensemble CSV must contain 'price_pred' or 'price_per_sqm'."
        )

    df_ens = df_ens.sort_values("date").reset_index(drop=True)
    return df_ens

if show_forecast:
    st.subheader(
        "Ensemble forecast 2026–2050 "
        "(State Space + CatBoost pe reziduuri)"
    )

    df_ens = load_ensemble_forecast()

    # context istoric (ultimii ani reali)
    hist_tail = df[df["year"] >= 2000][["date", "price_per_sqm"]]
    hist_tail = hist_tail.assign(source="Historical / real")

    ens_plot = df_ens.assign(source="Ensemble forecast")

    df_plot = pd.concat([hist_tail, ens_plot], ignore_index=True)

    fig_ens, ax_ens = plt.subplots(figsize=(14, 5))

    # historical
    mask_hist = df_plot["source"] == "Historical / real"
    ax_ens.plot(
        df_plot.loc[mask_hist, "date"],
        df_plot.loc[mask_hist, "price_per_sqm"],
        linewidth=2,
        label="Historical / real (from 2000)",
    )

    # ensemble forecast
    mask_ens = df_plot["source"] == "Ensemble forecast"
    ax_ens.plot(
        df_plot.loc[mask_ens, "date"],
        df_plot.loc[mask_ens, "price_per_sqm"],
        linestyle="--",
        linewidth=2.5,
        color="tab:green",
        label="Ensemble forecast (SS + CatBoost)",
    )

    ax_ens.set_xlabel("Year")
    ax_ens.set_ylabel("Price (RON/sqm)")
    ax_ens.set_title(
        "Apartment price forecast – Ensemble model "
        "(State Space + CatBoost residuals)"
    )
    ax_ens.grid(True, linestyle="--", alpha=0.3)
    ax_ens.legend(loc="upper left")

    st.pyplot(fig_ens)

    st.markdown("**Ensemble forecast data (2026–2050):**")
    st.dataframe(df_ens.reset_index(drop=True))

st.info(
    "This forecast is generated using a hybrid ensemble model:\n"
    "- State Space model captures the long-term economic trend\n"
    "- CatBoost models the residuals (non-linear corrections)\n"
    "- Final prediction = trend + learned residual adjustment\n\n"
    "This approach preserves economic coherence while improving "
    "short- and medium-term dynamics."
)
