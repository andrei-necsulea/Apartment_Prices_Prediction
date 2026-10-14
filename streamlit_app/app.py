import os
import sys
import io
import time
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Asigurare ca root-ul proiectului este in sys.path pentru importul modulelor locale
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_utils import add_custom_features, validate_classification_input, validate_regression_input

# Asigurare compatibilitate deserializare unpickle pentru pipeline
import __main__
__main__.add_custom_features = add_custom_features

# Configurare pagina Streamlit
st.set_page_config(
    page_title="Machine Learning Platform - Real Estate & Mortgages",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Selector de limba in sidebar
lang = st.sidebar.selectbox(
    "🌐 Limba / Language:",
    ["Romana 🇷🇴", "English 🇬🇧"],
    index=0,
    key="app_language"
)
is_ro = "Romana" in lang

# Functie helper pentru traducere instanta
def t(ro_text, en_text):
    return ro_text if is_ro else en_text

# Functie utilitara pentru localizarea fisierelor locale
def resolve_path(rel_path):
    candidate = os.path.join(project_root, rel_path)
    if os.path.exists(candidate):
        return candidate
    return rel_path

# Cale date istorice Craiova
DATA_PATH = "https://raw.githubusercontent.com/andrei-necsulea/Apartment_Prices_Prediction/refs/heads/main/results/craiova_apartment_prices_1900_2035.csv"

@st.cache_data
def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception:
        local_csv = resolve_path("results/craiova_apartment_prices_1900_2035.csv")
        df = pd.read_csv(local_csv, parse_dates=["date"])
        
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["decade"] = (df["year"] // 10) * 10
    return df

@st.cache_resource
def load_classification_model():
    model_path = resolve_path("models/best_classification_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_regression_model():
    model_path = resolve_path("models/best_regression_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Incarcare dataset principal Craiova
df = load_data()
min_year = int(df["year"].min())
max_year = int(df["year"].max())

# Titlu Principal
st.title(t("🏡 Platforma de Machine Learning: Imobiliare si Credite", "🏡 Machine Learning Platform: Real Estate & Mortgages"))
st.markdown(t(
    "Proiect integrat pentru **Regresie** (Preturi apartamente Craiova) si **Clasificare** (Eligibilitate credit ipotecar).",
    "Integrated platform for **Regression** (Craiova apartment prices) and **Classification** (Mortgage loan eligibility)."
))

# Selectie Tab-uri principale
tab_dashboard, tab_reg, tab_clf, tab_models = st.tabs([
    t("📊 Analiza Istorica & Prognoza Craiova", "📊 Historical Analysis & Craiova Forecast"),
    t("📈 Predictie Regresie (Pret Apartamente)", "📈 Regression Prediction (Apartment Prices)"),
    t("🏦 Predictie Clasificare (Aprobare Credit)", "🏦 Classification Prediction (Loan Approval)"),
    t("📑 Modele ML & Explicabilitate (SHAP)", "📑 ML Models & Explainability (SHAP)")
])

# ==============================================================================
# TAB 1: ANALIZA ISTORICA SI PROGNOZA CRAIOVA
# ==============================================================================
with tab_dashboard:
    st.subheader(t("Piata Imobiliara Craiova – 1900–2025 (RON/mp)", "Craiova Real Estate Market – 1900–2025 (RON/sqm)"))
    
    st.markdown(t(
        """
        Acest dashboard interactiv utilizeaza:
        - **1900–2011**: serie sintetica generata pe baza contextului economic istoric (razboaie, comunism, tranzitie, boom, crize),
        - **2012–2025**: date reale preluate si convertite in **RON/mp**.
        - **2026–2050**: prognoza hibrida utilizand modele de tip State Space si CatBoost pe reziduuri.
        """,
        """
        This interactive dashboard uses:
        - **1900–2011**: synthetic historical series generated based on economic context (wars, communism, transition, boom, crisis),
        - **2012–2025**: real market data scraped and converted to **RON/sqm**.
        - **2026–2050**: hybrid ensemble forecast combining State Space long-term trends and CatBoost residuals.
        """
    ))
    
    # Controale laterale specifice Tab-ului 1
    st.sidebar.markdown("---")
    st.sidebar.title(t("Setari Vizuale Craiova", "Visual Settings (Craiova)"))
    
    mode_options = [
        t("Toata perioada 1900–2025", "All period 1900–2025"),
        t("Doar perioada cu date reale (2012–2025)", "Only real period (2012–2025)")
    ]
    mode = st.sidebar.radio(
        t("Mod de analiza istoric:", "Historical analysis mode:"),
        mode_options,
        key="mode_radio"
    )

    default_min = 2012 if mode == mode_options[1] else min_year
    year_range = st.sidebar.slider(
        t("Interval de ani:", "Year interval:"),
        min_value=min_year,
        max_value=max_year,
        value=(default_min, max_year),
        step=1,
        key="year_range_slider"
    )

    show_events = st.sidebar.checkbox(t("Evidentiaza perioadele istorice majore", "Highlight major historical periods"), value=True, key="chk_events")
    show_rolling = st.sidebar.checkbox(t("Afiseaza media mobila (5 ani)", "Display rolling mean (5 years)"), value=True, key="chk_rolling")
    show_forecast = st.sidebar.checkbox(t("Afiseaza prognoza hibrida 2026–2050", "Show hybrid forecast 2026–2050"), value=True, key="chk_fc")

    # Filtrare date pentru Tab 1
    df_filtered = df.copy()
    if mode == mode_options[1]:
        df_filtered = df_filtered[df_filtered["year"] >= 2012]

    df_filtered = df_filtered[
        (df_filtered["year"] >= year_range[0]) & (df_filtered["year"] <= year_range[1])
    ].copy()

    # Metrici de top pentru intervalul filtrat
    if not df_filtered.empty:
        first_price = df_filtered["price_per_sqm"].iloc[0]
        last_price = df_filtered["price_per_sqm"].iloc[-1]
        pct_change = (last_price - first_price) / first_price * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t("Pret Minim", "Minimum Price"), f"{df_filtered['price_per_sqm'].min():.2f} RON/sqm")
        with col2:
            st.metric(t("Pret Mediu", "Average Price"), f"{df_filtered['price_per_sqm'].mean():.2f} RON/sqm")
        with col3:
            st.metric(t("Pret Maxim", "Maximum Price"), f"{df_filtered['price_per_sqm'].max():.2f} RON/sqm")
        with col4:
            st.metric(t("Variatie in Perioada", "Period Change"), f"{pct_change:.1f} %", delta=f"{pct_change:.1f} %")
    else:
        st.warning(t("Nu exista date disponibile pentru intervalul selectat.", "No data available in the selected range."))

    # Grafic serie de timp principala
    st.markdown(t("#### Evolutia pretului pe metru patrat in intervalul selectat", "#### Price evolution per square meter in selected range"))
    fig, ax = plt.subplots(figsize=(14, 5))

    if not df_filtered.empty:
        ax.plot(df_filtered["date"], df_filtered["price_per_sqm"], label=t("Pret (RON/mp)", "Price (RON/sqm)"), color="#1f77b4", linewidth=2)

        if show_rolling:
            df_filtered["rolling_5"] = df_filtered["price_per_sqm"].rolling(window=5, min_periods=1).mean()
            ax.plot(df_filtered["date"], df_filtered["rolling_5"], color="red", linewidth=2, alpha=0.7, label=t("Medie mobila (5 ani)", "Rolling mean (5 years)"))

        if show_events:
            ax.axvspan(pd.Timestamp("1914-01-01"), pd.Timestamp("1918-12-31"), color="red", alpha=0.15, label=t("Primul Razboi Mondial", "WW1"))
            ax.axvspan(pd.Timestamp("1939-01-01"), pd.Timestamp("1945-12-31"), color="orange", alpha=0.15, label=t("Al Doilea Razboi Mondial", "WW2"))
            ax.axvspan(pd.Timestamp("1948-01-01"), pd.Timestamp("1989-12-31"), color="green", alpha=0.06, label=t("Perioada Comunista", "Communism"))
            ax.axvspan(pd.Timestamp("2000-01-01"), pd.Timestamp("2007-12-31"), color="blue", alpha=0.10, label=t("Boom Imobiliar", "Real Estate Boom"))
            ax.axvspan(pd.Timestamp("2008-01-01"), pd.Timestamp("2010-12-31"), color="purple", alpha=0.12, label=t("Criza Financiara 2008–2010", "Financial Crisis 2008–2010"))

        ax.set_xlabel(t("An", "Year"))
        ax.set_ylabel(t("Pret (RON/mp)", "Price (RON/sqm)"))
        ax.grid(True, linestyle="--", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="upper left")

    st.pyplot(fig)

    # Privire pe decenii si distributie
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(t("#### Pretul mediu pe decenii", "#### Average price over decades"))
        df_dec = df.groupby("decade")["price_per_sqm"].mean().reset_index().sort_values("decade")
        st.bar_chart(df_dec.set_index("decade")["price_per_sqm"], height=280)

    with col_d2:
        st.markdown(t("#### Distributia preturilor in interval", "#### Price distribution in selected range"))
        if not df_filtered.empty:
            hist_vals, bin_edges = np.histogram(df_filtered["price_per_sqm"], bins=25)
            hist_df = pd.DataFrame({"bin_center": (bin_edges[:-1] + bin_edges[1:]) / 2, "count": hist_vals})
            st.bar_chart(hist_df.set_index("bin_center")["count"], height=280)

    # Sectiunea Prognoza Hibrida 2026-2050
    if show_forecast:
        st.markdown("---")
        st.markdown(t("#### Prognoza de Pret 2026–2050 (State Space + CatBoost pe Reziduuri)", "#### Price Forecast 2026–2050 (State Space + CatBoost Residuals)"))
        
        ens_csv_path = resolve_path("results/ensemble_ss_catboost_forecast_2026_2050.csv")
        if os.path.exists(ens_csv_path):
            df_ens = pd.read_csv(ens_csv_path)
            if "An" in df_ens.columns and "date" not in df_ens.columns:
                df_ens["date"] = pd.to_datetime(df_ens["An"].astype(str) + "-01-01")
            if "price_pred" in df_ens.columns:
                df_ens = df_ens.rename(columns={"price_pred": "price_per_sqm"})
                
            hist_tail = df[df["year"] >= 2000][["date", "price_per_sqm"]].assign(Sursa=t("Date Istorice / Reale", "Historical / Real Data"))
            ens_plot = df_ens[["date", "price_per_sqm"]].assign(Sursa=t("Prognoza Hibrida Ensemble", "Hybrid Ensemble Forecast"))
            df_comb = pd.concat([hist_tail, ens_plot], ignore_index=True)

            fig_fc, ax_fc = plt.subplots(figsize=(14, 4.5))
            m_hist = df_comb["Sursa"] == t("Date Istorice / Reale", "Historical / Real Data")
            ax_fc.plot(df_comb.loc[m_hist, "date"], df_comb.loc[m_hist, "price_per_sqm"], linewidth=2, label=t("Date Reale (din 2000)", "Real Data (from 2000)"), color="#2c3e50")
            
            m_ens = df_comb["Sursa"] == t("Prognoza Hibrida Ensemble", "Hybrid Ensemble Forecast")
            ax_fc.plot(df_comb.loc[m_ens, "date"], df_comb.loc[m_ens, "price_per_sqm"], linestyle="--", linewidth=2.5, color="#27ae60", label=t("Prognoza Hibrida (State Space + CatBoost)", "Hybrid Forecast (State Space + CatBoost)"))
            
            ax_fc.set_xlabel(t("An", "Year"))
            ax_fc.set_ylabel(t("Pret (RON/mp)", "Price (RON/sqm)"))
            ax_fc.grid(True, linestyle="--", alpha=0.3)
            ax_fc.legend(loc="upper left")
            st.pyplot(fig_fc)

# ==============================================================================
# TAB 2: PREDICTIE REGRESIE (Manual Input + Batch CSV Prediction)
# ==============================================================================
with tab_reg:
    st.subheader(t("📈 Predictie Regresie: Estimator Pret Apartamente Craiova", "📈 Regression Prediction: Craiova Apartment Price Estimator"))
    st.markdown(t(
        "Acest modul permite estimarea pretului mediu pe metru patrat pe baza indicatorilor macroeconomici.",
        "This module estimates the average price per square meter based on macroeconomic indicators."
    ))

    reg_bundle = load_regression_model()
    if reg_bundle is None:
        st.error(t("Modelul de regresie nu a fost gasit in 'models/best_regression_model.joblib'.", "Regression model not found in 'models/best_regression_model.joblib'."))
    else:
        reg_model = reg_bundle["model"]
        last_lags = reg_bundle.get("last_known_lags", {"price_lag1": 10017.45, "price_lag2": 8344.82})

        reg_mode_options = [
            t("Manual (Un singur an)", "Manual (Single Year)"),
            t("Batch (Incarcare fisier CSV)", "Batch (Upload CSV file)")
        ]
        reg_mode = st.radio(t("Alege modul de predictie:", "Select prediction mode:"), reg_mode_options, horizontal=True)

        if reg_mode == reg_mode_options[0]:
            st.markdown(t("##### Introdu valorile indicatorilor macroeconomici:", "##### Input macroeconomic indicator values:"))
            with st.form("form_regresie"):
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    input_an = st.number_input(t("Anul prognozat", "Target Year"), min_value=2024, max_value=2050, value=2026, step=1)
                    input_cost_mat = st.number_input(t("Index cost materiale constructii", "Building Materials Cost Index"), min_value=10.0, max_value=300.0, value=115.4, step=0.5)
                with col_r2:
                    input_urban_incr = st.number_input(t("Rata crestere populatie urbana (ex: 0.005)", "Urban population growth rate (e.g. 0.005)"), min_value=-0.05, max_value=0.05, value=0.0052, step=0.0005, format="%.4f")
                    input_inflatie = st.number_input(t("Rata inflatiei anuale (%)", "Annual Inflation Rate (%)"), min_value=-5.0, max_value=50.0, value=4.5, step=0.1)
                with col_r3:
                    input_pib_dolj = st.number_input(t("PIB Judetul Dolj (miliarde RON)", "Dolj County GDP (billion RON)"), min_value=1.0, max_value=200.0, value=45.2, step=1.0)
                    input_lag1 = st.number_input(t("Pret An Anterior (price_lag1)", "Previous Year Price (price_lag1)"), min_value=100.0, max_value=50000.0, value=float(last_lags["price_lag1"]), step=50.0)

                submit_reg = st.form_submit_button(t("Estimeaza Pretul (RON/mp)", "Estimate Price (RON/sqm)"))

            if submit_reg:
                sample_df = pd.DataFrame([{
                    "An": int(input_an),
                    "index_cost_mat": float(input_cost_mat),
                    "urban_incr_proc": float(input_urban_incr),
                    "rata_infl": float(input_inflatie),
                    "pib_dolj_mld_ron": float(input_pib_dolj),
                    "price_lag1": float(input_lag1),
                    "price_lag2": float(last_lags["price_lag2"])
                }])

                # Validare input bilingva
                is_valid, errs, warns = validate_regression_input(sample_df, lang="ro" if is_ro else "en")
                for w in warns:
                    st.warning(w)

                if not is_valid:
                    for e in errs:
                        st.error(e)
                else:
                    pred_price = reg_model.predict(sample_df)[0]
                    pred_eur = pred_price / 5.0 # Conversie aproximativa 5 RON / EUR

                    st.success(t("✅ Predictie calculata cu succes!", "✅ Prediction calculated successfully!"))
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(t("Pret Estimat (RON/mp)", "Estimated Price (RON/sqm)"), f"{pred_price:.2f} RON/sqm")
                    with c2:
                        st.metric(t("Pret Estimat (EUR/mp)", "Estimated Price (EUR/sqm)"), f"{pred_eur:.2f} €/sqm")
                    with c3:
                        diff_vs_2025 = ((pred_price - last_lags["price_lag1"]) / last_lags["price_lag1"]) * 100
                        st.metric(t("Diferenta vs. 2025", "Change vs. 2025"), f"{diff_vs_2025:+.1f} %", delta=f"{diff_vs_2025:+.1f} %")

        else:
            st.markdown(t("##### Incarca un fisier CSV pentru calculul predictiilor batch:", "##### Upload a CSV file for batch price predictions:"))
            
            # Buton descarcare template CSV
            template_path = resolve_path("loan_data/batch_template_regression.csv")
            if os.path.exists(template_path):
                with open(template_path, "rb") as f_tpl:
                    st.download_button(
                        label=t("📥 Descarca Sablon CSV Exemplu (Regresie)", "📥 Download Sample CSV Template (Regression)"),
                        data=f_tpl,
                        file_name="template_regresie_craiova.csv",
                        mime="text/csv"
                    )

            uploaded_file = st.file_uploader(t("Incarca fisier CSV cu indicatori", "Upload CSV with indicators"), type=["csv"], key="upload_reg")
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write(t(f"Fisier incarcat: {batch_df.shape[0]} randuri.", f"Uploaded file: {batch_df.shape[0]} rows."))
                    st.dataframe(batch_df.head(5))

                    is_valid, errs, warns = validate_regression_input(batch_df, lang="ro" if is_ro else "en")
                    for w in warns:
                        st.warning(w)

                    if not is_valid:
                        for e in errs:
                            st.error(e)
                    else:
                        if st.button(t("Executa Predictia Batch (Regresie)", "Execute Batch Prediction (Regression)")):
                            # Asigurare coloane lag daca lipsesc
                            if "price_lag1" not in batch_df.columns:
                                batch_df["price_lag1"] = float(last_lags["price_lag1"])
                            if "price_lag2" not in batch_df.columns:
                                batch_df["price_lag2"] = float(last_lags["price_lag2"])

                            preds = reg_model.predict(batch_df)
                            results_df = batch_df.copy()
                            results_df["predicted_price_sqm"] = np.round(preds, 2)
                            results_df["predicted_price_eur"] = np.round(preds / 5.0, 2)

                            st.success(t("✅ Predictii batch generate!", "✅ Batch predictions generated successfully!"))
                            st.dataframe(results_df)

                            # Descarcare rezultate
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label=t("💾 Descarca Rezultatele Predicției (CSV)", "💾 Download Predictions (CSV)"),
                                data=csv_buffer.getvalue(),
                                file_name="rezultate_predictie_preturi_craiova.csv",
                                mime="text/csv"
                            )
                except Exception as ex:
                    st.error(f"{t('Eroare la procesarea fisierului CSV', 'Error processing CSV file')}: {ex}")

# ==============================================================================
# TAB 3: PREDICTIE CLASIFICARE (Loan Approval - Dataset 2)
# ==============================================================================
with tab_clf:
    st.subheader(t("🏦 Predictie Clasificare: Evaluare Eligibilitate Credit Ipotecar", "🏦 Classification Prediction: Mortgage Loan Eligibility Evaluator"))
    st.markdown(t(
        "Verifica probabilitatea ca o cerere de credit pentru locuinta sa fie **Aprobata (Status 1)** sau **Respinsa (Status 0)**.",
        "Assess whether a mortgage application will be **Approved (Status 1)** or **Rejected (Status 0)**."
    ))

    clf_model = load_classification_model()
    if clf_model is None:
        st.error(t("Modelul de clasificare nu a fost gasit in 'models/best_classification_model.joblib'.", "Classification model not found in 'models/best_classification_model.joblib'."))
    else:
        clf_mode_options = [
            t("Manual (Un singur solicitant)", "Manual (Single Applicant)"),
            t("Batch (Incarcare fisier CSV)", "Batch (Upload CSV file)")
        ]
        clf_mode = st.radio(t("Alege modul de clasificare:", "Select classification mode:"), clf_mode_options, horizontal=True, key="clf_mode_radio")

        if clf_mode == clf_mode_options[0]:
            st.markdown(t("##### Completeaza datele solicitantului de credit:", "##### Enter applicant financial and profile information:"))
            with st.form("form_clasificare"):
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    app_income = st.number_input(t("Venit Solicitant (RON/luna)", "Applicant Income (RON/month)"), min_value=0, max_value=200000, value=5000, step=100)
                    coapp_income = st.number_input(t("Venit Codebitor / Sot(ie) (RON/luna)", "Co-applicant Income (RON/month)"), min_value=0, max_value=200000, value=2500, step=100)
                    gender = st.selectbox(t("Gen", "Gender"), ["Male", "Female"])
                    married = st.selectbox(t("Stare Civila", "Marital Status"), ["Yes", "No"])
                with col_c2:
                    loan_amount = st.number_input(t("Suma Credit Solicitata (mii RON)", "Loan Amount Requested (thousand RON)"), min_value=5.0, max_value=3000.0, value=150.0, step=5.0)
                    loan_term = st.selectbox(t("Termen Credit (Luni)", "Loan Term (Months)"), [120.0, 180.0, 240.0, 300.0, 360.0, 480.0], index=4)
                    dependents = st.selectbox(t("Numar Persoane in Intretinere", "Number of Dependents"), ["0", "1", "2", "3+"])
                    education = st.selectbox(t("Nivel Educatie", "Education Level"), ["Graduate", "Not Graduate"])
                with col_c3:
                    credit_hist_labels = [
                        t("1.0 - Fara restante anterioare", "1.0 - Meets credit guidelines (Good)"),
                        t("0.0 - Cu restante / Incident de plata", "0.0 - Past defaults / Poor credit history")
                    ]
                    credit_hist = st.selectbox(t("Istoric de Creditare", "Credit History"), credit_hist_labels)
                    self_emp = st.selectbox(t("Persoana Fizica Autorizata / Liber Profesionist", "Self-Employed"), ["No", "Yes"])
                    prop_area_labels = [
                        "Semiurban",
                        "Urban",
                        "Rural",
                        t("Suburban (Categorie Nevazuta)", "Suburban (Unseen Category Demo)")
                    ]
                    prop_area = st.selectbox(t("Zona Imobilului Finantat", "Property Area"), prop_area_labels)

                submit_clf = st.form_submit_button(t("Evalueaza Cererea de Credit", "Evaluate Mortgage Application"))

            if submit_clf:
                hist_val = 1.0 if "1.0" in credit_hist else 0.0
                clean_prop_area = prop_area.split(" ")[0]

                single_applicant_df = pd.DataFrame([{
                    "ApplicantIncome": int(app_income),
                    "CoapplicantIncome": float(coapp_income),
                    "LoanAmount": float(loan_amount),
                    "Loan_Amount_Term": float(loan_term),
                    "Credit_History": float(hist_val),
                    "Gender": gender,
                    "Married": married,
                    "Dependents": dependents,
                    "Education": education,
                    "Self_Employed": self_emp,
                    "Property_Area": clean_prop_area
                }])

                # Validare input si notificare categorii nevazute
                is_valid, errs, warns = validate_classification_input(single_applicant_df, lang="ro" if is_ro else "en")
                for w in warns:
                    st.info(w)

                if not is_valid:
                    for e in errs:
                        st.error(e)
                else:
                    t_start = time.perf_counter()
                    pred_class = clf_model.predict(single_applicant_df)[0]
                    pred_prob = clf_model.predict_proba(single_applicant_df)[0]
                    duration_ms = (time.perf_counter() - t_start) * 1000.0

                    prob_approved = pred_prob[1]
                    prob_rejected = pred_prob[0]

                    if pred_class == 1:
                        st.success(t(
                            f"🎉 **DECIZIE: CREDIT APROBAT** (Probabilitate: {prob_approved * 100:.1f}%)",
                            f"🎉 **DECISION: LOAN APPROVED** (Confidence: {prob_approved * 100:.1f}%)"
                        ))
                    else:
                        st.error(t(
                            f"❌ **DECIZIE: CREDIT RESPINS** (Probabilitate Refuz: {prob_rejected * 100:.1f}%)",
                            f"❌ **DECISION: LOAN REJECTED** (Rejection probability: {prob_rejected * 100:.1f}%)"
                        ))

                    st.progress(float(prob_approved), text=f"{t('Scor de Aprobare', 'Approval Score')}: {prob_approved * 100:.1f}%")

                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    tot_inc = app_income + coapp_income
                    emi_est = (loan_amount * 1000) / loan_term
                    dti = (emi_est / (tot_inc + 1e-5)) * 100

                    with col_m1:
                        st.metric(t("Venit Total Gospodarie", "Total Household Income"), f"{tot_inc:,} RON")
                    with col_m2:
                        st.metric(t("Rata Lunara Estimata", "Estimated Monthly Payment"), f"{emi_est:.1f} RON")
                    with col_m3:
                        delta_msg = t("-Risc Redus", "-Low Risk") if dti < 40 else t("+Risc Ridicat", "+High Risk")
                        st.metric(t("Grad de Indatorare (DTI)", "Debt-to-Income (DTI)"), f"{dti:.1f} %", delta=delta_msg)
                    with col_m4:
                        st.metric(t("Timp Inferenta", "Inference Time"), f"{duration_ms:.2f} ms")

        else:
            st.markdown(t("##### Incarca un fisier CSV pentru predictii batch de creditare:", "##### Upload a CSV file for batch mortgage credit evaluations:"))

            template_path = resolve_path("loan_data/batch_template_classification.csv")
            if os.path.exists(template_path):
                with open(template_path, "rb") as f_tpl:
                    st.download_button(
                        label=t("📥 Descarca Sablon CSV Exemplu (Credite)", "📥 Download Sample CSV Template (Mortgages)"),
                        data=f_tpl,
                        file_name="template_credite_ipotecare.csv",
                        mime="text/csv"
                    )

            uploaded_file = st.file_uploader(t("Incarca fisier CSV cu aplicanti", "Upload CSV with loan applicants"), type=["csv"], key="upload_clf")
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write(t(f"Fisier incarcat: {batch_df.shape[0]} solicitanti.", f"Uploaded file: {batch_df.shape[0]} applicants."))
                    st.dataframe(batch_df.head(5))

                    is_valid, errs, warns = validate_classification_input(batch_df, lang="ro" if is_ro else "en")
                    for w in warns:
                        st.warning(w)

                    if not is_valid:
                        for e in errs:
                            st.error(e)
                    else:
                        if st.button(t("Executa Evaluarea Batch (Clasificare)", "Execute Batch Evaluation (Classification)")):
                            preds = clf_model.predict(batch_df)
                            probas = clf_model.predict_proba(batch_df)[:, 1]

                            results_df = batch_df.copy()
                            results_df["Decision"] = np.where(preds == 1, t("APROBAT", "APPROVED"), t("RESPINS", "REJECTED"))
                            results_df["Approval_Probability (%)"] = np.round(probas * 100, 2)

                            st.success(t(f"✅ Evaluare batch finalizata pentru {len(results_df)} solicitanti!", f"✅ Batch evaluation completed for {len(results_df)} applicants!"))
                            
                            c_ap = (preds == 1).sum()
                            c_rej = (preds == 0).sum()
                            st.write(f"**{t('Rezumat', 'Summary')}:** {c_ap} {t('Aprobate', 'Approved')} ({c_ap/len(preds)*100:.1f}%), {c_rej} {t('Respinse', 'Rejected')} ({c_rej/len(preds)*100:.1f}%)")

                            st.dataframe(results_df)

                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label=t("💾 Descarca Rezultatele Evaluarii (CSV)", "💾 Download Evaluation Results (CSV)"),
                                data=csv_buffer.getvalue(),
                                file_name="rezultate_aprobare_credite.csv",
                                mime="text/csv"
                            )
                except Exception as ex:
                    st.error(f"{t('Eroare la procesarea fisierului CSV', 'Error processing CSV file')}: {ex}")

# ==============================================================================
# TAB 4: MODELE & EXPLICABILITATE (SHAP, Comparatie, Barem)
# ==============================================================================
with tab_models:
    st.subheader(t("📑 Modele Machine Learning, Evaluare si Explicabilitate", "📑 Machine Learning Models, Evaluation & Explainability"))
    st.markdown(t("Sinteza rezultatelor obtinute conform cerintelor formale de examen.", "Summary of results matching the course exam requirements."))

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(t("#### 1. Dataset Regresie (Preturi Craiova)", "#### 1. Regression Dataset (Craiova Prices)"))
        st.markdown(t(
            """
            - **Model Final:** CatBoost Regressor (Trend macroeconomic + Lags)
            - **Evaluare TimeSeriesSplit (5 Folds):**
              - `MAE Mediu:` 501.40 ± 767.56 RON/mp
              - `RMSE Mediu:` 623.62 ± 957.75 RON/mp
            - **Evaluare Test Set Neatins (2023–2025):**
              - `MAE:` 1542.66 RON/mp
              - `RMSE:` 1893.96 RON/mp
              - `Timp antrenare:` 0.0439 s | `Timp inferenta:` 0.32 ms
            """,
            """
            - **Final Model:** CatBoost Regressor (Macroeconomic trend + Lags)
            - **TimeSeriesSplit Evaluation (5 Folds):**
              - `Mean MAE:` 501.40 ± 767.56 RON/sqm
              - `Mean RMSE:` 623.62 ± 957.75 RON/sqm
            - **Untouched Test Set Evaluation (2023–2025):**
              - `MAE:` 1542.66 RON/sqm
              - `RMSE:` 1893.96 RON/sqm
              - `Training time:` 0.0439 s | `Inference time:` 0.32 ms
            """
        ))

    with col_t2:
        st.markdown(t("#### 2. Dataset Clasificare (Loan Approval)", "#### 2. Classification Dataset (Loan Approval)"))
        st.markdown(t(
            """
            - **Model Final:** CatBoost Classifier cu Preprocesare RobustScaler + One-Hot
            - **Evaluare StratifiedKFold (5 Folds):**
              - `ROC-AUC Mediu:` 0.7784 ± 0.0608
              - `F1-Score Mediu:` 0.8621 ± 0.0245
              - `Acuratete Medie:` 78.60% ± 3.2%
            - **Evaluare Test Set Neatins (Holdout):**
              - `Acuratete:` 85.37% | `F1-Score:` 0.8966 | `ROC-AUC:` 0.8508
              - `Timp antrenare:` 0.229 s | `Timp inferenta:` 6.14 ms
            """,
            """
            - **Final Model:** CatBoost Classifier with RobustScaler + OneHot pipeline
            - **StratifiedKFold Evaluation (5 Folds):**
              - `Mean ROC-AUC:` 0.7784 ± 0.0608
              - `Mean F1-Score:` 0.8621 ± 0.0245
              - `Mean Accuracy:` 78.60% ± 3.2%
            - **Untouched Test Set Evaluation (Holdout):**
              - `Accuracy:` 85.37% | `F1-Score:` 0.8966 | `ROC-AUC:` 0.8508
              - `Training time:` 0.229 s | `Inference time:` 6.14 ms
            """
        ))

    st.markdown("---")
    st.markdown(t("#### Explicabilitate: Influenta Trasaturilor (SHAP Summary Plot)", "#### Explainability: Feature Influence (SHAP Summary Plot)"))
    shap_img_path = resolve_path("loan_data/plots/shap_summary.png")
    cm_img_path = resolve_path("loan_data/plots/confusion_matrix.png")
    roc_img_path = resolve_path("loan_data/plots/roc_pr_curves.png")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists(shap_img_path):
            st.image(shap_img_path, caption=t("SHAP Summary Plot (Clasificare Credite)", "SHAP Summary Plot (Credit Classification)"))
        if os.path.exists(cm_img_path):
            st.image(cm_img_path, caption=t("Matricea de Confuzie pe Setul de Test Neatins", "Confusion Matrix on Untouched Test Set"))

    with col_img2:
        if os.path.exists(roc_img_path):
            st.image(roc_img_path, caption=t("Curbele ROC si Precision-Recall", "ROC and Precision-Recall Curves"))
        st.info(t(
            """
            **Interpretare SHAP:**
            - **Credit_History** este de departe cea mai importanta trasatura. Un istoric curat creste masiv probabilitatea de aprobare.
            - **Total_Income** si **Income_to_Loan_Ratio** au un impact puternic pozitiv: solicitantii cu venit mare raportat la valoarea creditului primesc aprobare rapida.
            - **Property_Area_Semiurban** prezinta o corelatie pozitiva cu rata de acceptare bancara comparativ cu zona rurala.
            """,
            """
            **SHAP Interpretation:**
            - **Credit_History** is by far the most influential feature. Meeting guidelines dramatically increases the likelihood of approval.
            - **Total_Income** and **Income_to_Loan_Ratio** have strong positive impact: applicants with healthy financial capacity receive quick approval.
            - **Property_Area_Semiurban** shows positive correlation with bank acceptance rates compared to rural areas.
            """
        ))
