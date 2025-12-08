import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Path to the final merged dataset (historical + real data + forecasts)
DATA_PATH = "../results/craiova_apartment_prices_1900_2035.csv"


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
if mode == "All period 1900–2025":
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

# Raw data table for the filtered range
st.subheader("Raw data in the selected range")

st.dataframe(
    df_filtered[["date", "price_per_sqm"]]
    .reset_index(drop=True)
)