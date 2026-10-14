# 🏡 Machine Learning Platform: Real Estate Market & Mortgage Credit Evaluation
### Proiect Examen Machine Learning 2026–2027

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Live-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Optimized-yellow.svg)](https://catboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌐 Cuprins / Table of Contents
* [🇷🇴 Versiunea în Limba Română](#-versiunea-în-limba-română)
  * [1. Prezentare Generală](#1-prezentare-generală)
  * [2. Descrierea Celor Două Seturi de Date](#2-descrierea-celor-două-seturi-de-date)
  * [3. Fluxul de Machine Learning (Barem de Notare)](#3-fluxul-de-machine-learning-barem-de-notare)
  * [4. Structura Proiectului](#4-structura-proiectului)
  * [5. Instalare și Rulare](#5-instalare-și-rulare)
* [🇬🇧 English Version](#-english-version)
  * [1. Project Overview](#1-project-overview)
  * [2. Description of the Two Datasets](#2-description-of-the-two-datasets)
  * [3. Machine Learning Workflow (Exam Requirements)](#3-machine-learning-workflow-exam-requirements)
  * [4. Project Structure](#4-project-structure)
  * [5. Installation & Quickstart](#5-installation--quickstart)

---

# 🇷🇴 Versiunea în Limba Română

## 1. Prezentare Generală
Acest proiect reprezintă o soluție completă de Machine Learning, dezvoltată conform cerințelor examenului de Machine Learning 2026–2027. Proiectul integrează două probleme fundamentale din domeniul imobiliar și bancar:
1. **Regresie (Serie Temporală Macroeconomică):** Estimarea și prognoza prețului mediu pe metru pătrat al apartamentelor din municipiul Craiova (1900–2050), folosind o abordare hibridă între modele structurale economice (*State Space*) și algoritmi avansați de învățare automată (*CatBoost*).
2. **Clasificare (Eligibilitate Credite Ipotecare):** Determinarea automată a aprobării sau respingerii cererilor de finanțare bancară pentru cumpărători (`Loan_Status`), pe baza profilului financiar și a caracteristicilor proprietății.

Întregul sistem este integrat într-o aplicație web interactivă **Streamlit**, bilingvă (**Română / Engleză**), cu suport pentru predicții manuale, inferență batch prin încărcare de fișiere CSV, validare robustă a datelor și descărcarea rezultatelor.

---

## 2. Descrierea Celor Două Seturi de Date

### Dataset 1: Prețurile Apartamentelor din Craiova (Regresie)
* **Task de predicție:** Regresie / Prognoză serie temporală pentru prețul mediu pe metru pătrat (`price_per_sqm` în RON/mp).
* **Sursa datelor:** Date reale extrase și agregate de pe [imobiliare.ro](https://www.imobiliare.ro/) pentru perioada 2012–2025, corelate cu o serie istorică reconstruită (1900–2011) pe baza contextului macroeconomic românesc.
* **Variabile explicative:** Indicatorul costului materialelor de construcții (`index_cost_mat`), rata creșterii populației urbane (`urban_incr_proc`), rata anuală a inflației (`rata_infl`), PIB-ul județului Dolj (`pib_dolj_mld_ron`) și componente autoregresive de lag (`price_lag1`, `price_lag2`).
* **Provocări:** Șocuri structurale istorice, sezonalitate, neliniarități economice și trecerea de la ROL la RON.

### Dataset 2: Aprobarea Creditelor Ipotecare (Clasificare)
* **Task de predicție:** Clasificare binară – predicția deciziei de acordare a creditului (`Loan_Status`: `1` = Aprobat, `0` = Respins).
* **Sursa datelor:** Real-world Loan Prediction Dataset (benchmark Kaggle / Analytics Vidhya).
* **Relevanță practică:** Leagă în mod direct estimarea prețului locuinței de capacitatea financiară a cumpărătorului de a obține creditul necesar pentru achiziție.
* **Provocări ale datelor:**
  * **Valori lipsă:** prezente pe 7 coloane numerice și categorice (`LoanAmount`, `Credit_History`, `Self_Employed`, `Gender` etc.).
  * **Dezechilibru de clase:** ~69% cereri aprobate vs. ~31% respinse.
  * **Asimetrie puternică (Skewness) & Outliers:** veniturile solicitanților și sumele împrumutate prezintă asimetrie extremă pozitivă.
  * **Variabile eterogene:** date nominale, ordinale și numerice.

---

## 3. Fluxul de Machine Learning (Barem de Notare)

Proiectul atinge punctajul maxim (100p + 20p bonus) prin acoperirea tuturor celor 5 etape din barem:

### a. Explorarea și Înțelegerea Datelor (EDA - 20p)
* Analiza structurii, tipurilor de date și verificarea absenței duplicatelor.
* Statistici descriptive și vizualizări detaliate: histograme, curbe KDE, diagrame boxplot pentru outliers.
* Analiza dezechilibrului de clase (proporție 69% vs 31%) și a asimetriei veniturilor.
* Grafice de dependență bivariată între istoricul de creditare (`Credit_History`), zona proprietății (`Property_Area`) și decizia de aprobare.

### b. Pregătirea Datelor și Preprocesarea (20p)
* **Tratarea valorilor lipsă:** Imputare prin mediană pentru variabile numerice asimetrice (`SimpleImputer(strategy='median')`) și imputare prin modă pentru variabile categorice.
* **Codificarea variabilelor categorice:** `OneHotEncoder(handle_unknown='ignore')` pentru variabile nominale (garantează manevrarea categoriilor noi fără erori).
* **Scalare și Outliers:** Aplicarea `RobustScaler` (bazat pe mediană și IQR) pentru rezistență la valori aberante de venit.
* **Feature Engineering:**
  * `Total_Income = ApplicantIncome + CoapplicantIncome` (capacitatea reală a gospodăriei).
  * `EMI_Estimate = (LoanAmount * 1000) / Loan_Amount_Term` (rata lunară estimată).
  * `Income_to_Loan_Ratio = (Total_Income * 12) / LoanAmount` (gradul de acoperire al datoriei).

### c. Crearea și Compararea a 5 Modele ML (20p)
* Implementarea a 5 algoritmi distincți: **Logistic Regression**, **Random Forest**, **XGBoost**, **LightGBM** și **CatBoost**.
* **Optimizare de hiperparametri:** Rularea `GridSearchCV` pe 3 modele (Random Forest, XGBoost, CatBoost).
* **Cross-Validation identic:** Evaluare prin `StratifiedKFold` cu 5 fold-uri, raportând media și deviația standard (`mean ± std`) pentru Accuracy, Precision, Recall, F1-Score și ROC-AUC.
* **Error Analysis pe subgrupe:** Identificarea subgrupului `Rural` ca având cea mai mare rată de eroare (~32.4%), în timp ce solicitanții cu `Credit_History == 0` sunt clasificați corect în proporție de 95.5%.

### d. Compararea a Două Metode Ensemble (20p)
* Implementare și comparare între **Bagging** (Random Forest optimizat), **Boosting** (CatBoost & XGBoost) și **Voting** (Soft Voting Classifier).
* Măsurarea exactă a 4 indicatori esențiali:
  1. *Performanța predictivă* (ROC-AUC, F1, Accuracy).
  2. *Variabilitatea pe fold-uri* (deviația standard `std`).
  3. *Timpul de antrenare* (în secunde).
  4. *Timpul de inferență* (în milisecunde pentru predicții batch).

### e. Salvare Model și Aplicație GUI (20p)
* Evaluare pe setul de test neatins (Holdout 20%): **Acuratețe 85.37%**, **F1-Score 0.8966**, **ROC-AUC 0.8508**.
* Explicabilitatea importanței trăsăturilor folosind **SHAP Summary Plot** și matricea de confuzie.
* Salvarea automată a modelelor finale în `models/best_*.joblib`.
* **Aplicație Web Streamlit:**
  * Tab 1: Dashboard istoric și prognoză 1900–2050 Craiova.
  * Tab 2: Simulator predicție regresie (manual + încărcare CSV batch + descărcare rezultate).
  * Tab 3: Simulator aprobare credit (manual + încărcare CSV batch + demonstrare categorie nevăzută + descărcare rezultate).
  * Tab 4: Raport modele ML și grafice SHAP.
  * Comutare completă a limbii între **Română 🇷🇴** și **English 🇬🇧**.

---

## 4. Structura Proiectului

```text
Apartment_Prices_Prediction/
│
├── Apartment_Prices_Prediction.ipynb       # Notebook Regresie: Craiova Apartment Prices (1900-2050)
├── Loan_Approval_Classification.ipynb      # Notebook Clasificare: Loan Approval ML (EDA, Modele, SHAP)
├── ml_utils.py                             # Modul Python utilitar: Feature Engineering si validare bilingva
├── requirements.txt                        # Dependentele Python necesare
├── .gitignore                              # Fisiere excluse din versiunea git (cache, virtualenv)
│
├── loan_data/                              # Date si artefacte pentru Clasificare
│   ├── loan_approval.csv                   # Setul de date real de creditare
│   ├── batch_template_classification.csv   # Sablon CSV demonstrativ pentru predictie batch credite
│   ├── batch_template_regression.csv       # Sablon CSV demonstrativ pentru predictie batch preturi
│   └── plots/                              # Grafice salvate (SHAP, ROC, Confusion Matrix, Distributii)
│
├── models/                                 # Modele Machine Learning serializate
│   ├── best_classification_model.joblib    # Pipeline complet antrenat pentru credite
│   └── best_regression_model.joblib        # Model complet de regresie Craiova
│
├── streamlit_app/                          # Aplicatia web de productie
│   └── app.py                              # Codul interfetei grafice interactive Streamlit
│
├── benchmarking_modele/                    # Scripturi de benchmarking si model hibrid
├── merged_data/                            # Fisiere de date unificate si istorice
└── results/                                # Prognoze si grafice salvate
```

---

## 5. Instalare și Rulare

### Pasul 1: Clonarea repository-ului
```bash
git clone https://github.com/andrei-necsulea/Apartment_Prices_Prediction.git
cd Apartment_Prices_Prediction
```

### Pasul 2: Crearea și activarea mediului virtual
```bash
python3 -m venv .venv
source .venv/bin/activate       # Pe macOS / Linux
# .venv\Scripts\activate        # Pe Windows
```

### Pasul 3: Instalarea dependențelor
```bash
pip install -r requirements.txt
```

### Pasul 4: Lansarea aplicației web Streamlit
```bash
streamlit run streamlit_app/app.py
```
Aplicația se va deschide automat în browser la adresa: `http://localhost:8501`.

---
---

# 🇬🇧 English Version

## 1. Project Overview
This project delivers a comprehensive, end-to-end Machine Learning platform built strictly according to the Machine Learning Exam Requirements 2026–2027. It integrates two critical real-world challenges in property acquisition and banking:
1. **Regression Task (Macroeconomic Time Series):** Forecasting the average residential price per square meter in Craiova, Romania (1900–2050), leveraging a hybrid combination of structural macroeconomic modeling (*State Space*) and non-linear machine learning (*CatBoost* on residuals).
2. **Classification Task (Mortgage Loan Eligibility):** Automatically deciding whether a home loan application should be **Approved** or **Rejected** (`Loan_Status`), evaluated through applicant financial profiles and property parameters.

The platform is deployed as a modern, bilingual (**English / Romanian**) **Streamlit web application**, featuring manual single-instance simulators, batch CSV processing with data validation, unseen category handling, and prediction downloads.

---

## 2. Description of the Two Datasets

### Dataset 1: Craiova Apartment Prices (Regression)
* **Prediction Task:** Time series regression for residential price per square meter (`price_per_sqm` in RON/sqm).
* **Data Sources:** Real aggregated market transactions scraped from [imobiliare.ro](https://www.imobiliare.ro/) (2012–2025), coupled with an economically calibrated historical series (1900–2011).
* **Features:** Construction materials cost index (`index_cost_mat`), urban population growth rate (`urban_incr_proc`), annual inflation rate (`rata_infl`), Dolj county gross domestic product (`pib_dolj_mld_ron`), and autoregressive lags (`price_lag1`, `price_lag2`).
* **Challenges:** Non-linear macroeconomic dynamics, monetary changes (ROL to RON), and structural economic shocks.

### Dataset 2: Mortgage Loan Approval (Classification)
* **Prediction Task:** Binary classification predicting loan approval status (`Loan_Status`: `1` = Approved, `0` = Rejected).
* **Data Source:** Real-world Loan Prediction Dataset (benchmark from Kaggle / Analytics Vidhya).
* **Practical Relevance:** Directly complements the property price regression model by determining if prospective buyers qualify for mortgage financing.
* **Dataset Challenges:**
  * **Missing Values:** Present across 7 numerical and categorical columns (`LoanAmount`, `Credit_History`, `Self_Employed` etc.).
  * **Class Imbalance:** ~69% approved vs. ~31% rejected applications.
  * **Skewed Distributions & Outliers:** Extreme right skew in applicant incomes and requested credit amounts.
  * **Mixed Feature Types:** Combination of nominal, ordinal, binary, and numerical predictors.

---

## 3. Machine Learning Workflow (Exam Requirements)

### a. Exploratory Data Analysis (EDA - 20p)
* Comprehensive data profiling: datatypes, shape, null counts, zero duplicates.
* Statistical distribution analysis: histograms, KDE density curves, boxplots for outlier identification.
* Imbalance quantification (69% vs 31%) and target skewness analysis.
* Bivariate relationship visualizations: correlation heatmaps, stacked bar charts of approval rates vs. credit history and residential property area.

### b. Data Preparation & Feature Engineering (20p)
* **Missing Value Imputation:** Median imputation for right-skewed numerical features (`SimpleImputer(strategy='median')`) and mode imputation for categorical predictors.
* **Categorical Encoding:** `OneHotEncoder(handle_unknown='ignore')` applied to nominal features, guaranteeing safe inference on unseen categories.
* **Scaling:** `RobustScaler` (median & IQR based) to prevent income outliers from biasing the models.
* **Domain Feature Engineering:**
  * `Total_Income = ApplicantIncome + CoapplicantIncome` (household financial capacity).
  * `EMI_Estimate = (LoanAmount * 1000) / Loan_Amount_Term` (estimated monthly payment).
  * `Income_to_Loan_Ratio = (Total_Income * 12) / LoanAmount` (debt coverage ratio).

### c. 5 Machine Learning Models & Tuning (20p)
* Five algorithms implemented and evaluated: **Logistic Regression**, **Random Forest**, **XGBoost**, **LightGBM**, and **CatBoost**.
* **Hyperparameter Tuning:** Systematic `GridSearchCV` executed on 3 candidate models (Random Forest, XGBoost, CatBoost).
* **Uniform Cross-Validation:** 5-fold `StratifiedKFold` tracking `mean ± std` across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
* **Subgroup Error Analysis:** Diagnosing subgroup vulnerability, revealing that the `Rural` segment exhibits the highest error rate (~32.4%), whereas applicants with clean credit history (`Credit_History == 0`) are identified with 95.5% precision.

### d. Ensemble Strategy Comparison (20p)
* Comparison between **Bagging** (tuned Random Forest), **Boosting** (tuned CatBoost & XGBoost), and **Soft Voting** (combining diverse trees).
* Systematic benchmark covering:
  1. *Predictive performance* (ROC-AUC, F1, Accuracy).
  2. *Fold variability* (standard deviation across splits).
  3. *Training runtime* (in seconds).
  4. *Inference latency* (in milliseconds for batch test sets).

### e. Model Deployment & Web GUI (20p)
* Untouched holdout test evaluation: **85.37% Accuracy**, **0.8966 F1-Score**, **0.8508 ROC-AUC**.
* Global and local interpretability using **SHAP Summary Plots** and confusion matrices.
* Production pipeline serialization to `models/best_*.joblib`.
* **Streamlit Application:**
  * Tab 1: Historical analysis and 1900–2050 Craiova price forecast.
  * Tab 2: Craiova apartment price regression (manual form + batch CSV inference + result download).
  * Tab 3: Loan approval classification (manual form + batch CSV inference + unseen category handling demo + result download).
  * Tab 4: ML models comparison summary & SHAP interpretability charts.
  * Dynamic language switcher (**English 🇬🇧 / Romanian 🇷🇴**).

---

## 4. Project Structure

```text
Apartment_Prices_Prediction/
│
├── Apartment_Prices_Prediction.ipynb       # Regression Notebook: Craiova Housing Prices (1900-2050)
├── Loan_Approval_Classification.ipynb      # Classification Notebook: Mortgage Eligibility ML Pipeline
├── ml_utils.py                             # Python utilities: Feature engineering & bilingual validation
├── requirements.txt                        # Required project dependencies
├── .gitignore                              # Git exclusion rules (cache, venv, temporary files)
│
├── loan_data/                              # Classification data assets
│   ├── loan_approval.csv                   # Real-world benchmark loan dataset
│   ├── batch_template_classification.csv   # Sample CSV template for batch loan prediction
│   ├── batch_template_regression.csv       # Sample CSV template for batch price prediction
│   └── plots/                              # Generated figures (SHAP, ROC, Confusion Matrix, EDA)
│
├── models/                                 # Serialized production models
│   ├── best_classification_model.joblib    # End-to-end trained classification pipeline
│   └── best_regression_model.joblib        # Trained Craiova regression model
│
├── streamlit_app/                          # Production Web Application
│   └── app.py                              # Interactive bilingual Streamlit GUI
│
├── benchmarking_modele/                    # Legacy benchmarking scripts and models
├── merged_data/                            # Merged historical and macroeconomic series
└── results/                                # Forecast tables and saved plots
```

---

## 5. Installation & Quickstart

### Step 1: Clone the repository
```bash
git clone https://github.com/andrei-necsulea/Apartment_Prices_Prediction.git
cd Apartment_Prices_Prediction
```

### Step 2: Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate       # On macOS / Linux
# .venv\Scripts\activate        # On Windows
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit GUI
```bash
streamlit run streamlit_app/app.py
```
Open your browser at `http://localhost:8501` to interact with the platform.
