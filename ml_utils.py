"""
Modul de functii utilitare si transformari pentru proiectul de Machine Learning.
Contine functiile de Feature Engineering si validare bilingva (RO/EN) pentru regresie si clasificare.
"""

import numpy as np
import pandas as pd


def add_custom_features(df_input):
    """
    Functie de inginerie a trasaturilor pentru setul de date de clasificare credite.
    Calculeaza: Total_Income, EMI_Estimate, Income_to_Loan_Ratio.
    """
    df_out = df_input.copy()
    
    # 1. Venit total gospodarie (aplicant + co-aplicant)
    app_inc = pd.to_numeric(df_out["ApplicantIncome"], errors="coerce").fillna(0)
    coapp_inc = pd.to_numeric(df_out["CoapplicantIncome"], errors="coerce").fillna(0)
    df_out["Total_Income"] = app_inc + coapp_inc
    
    # 2. Estimare rata lunara (EMI Estimate)
    loan_term = pd.to_numeric(df_out["Loan_Amount_Term"], errors="coerce").replace(0, np.nan).fillna(360)
    loan_val = pd.to_numeric(df_out["LoanAmount"], errors="coerce").fillna(0) * 1000.0
    df_out["EMI_Estimate"] = loan_val / loan_term
    
    # 3. Raport venit anual raportat la valoarea creditului
    df_out["Income_to_Loan_Ratio"] = (df_out["Total_Income"] * 12.0) / (loan_val + 1.0)
    
    return df_out


def validate_classification_input(df_in, lang="ro"):
    """
    Valideaza intrarile pentru clasificarea creditelor (bilingv RO/EN).
    Verifica valori lipsa, tipuri de date, valori out-of-range si categorii nevazute.
    Returneaza (este_valid, lista_erori, lista_avertismente).
    """
    errors = []
    warnings = []
    is_ro = (lang == "ro")
    
    required_cols = [
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "Property_Area"
    ]
    
    # Verificare coloane lipsa
    missing_cols = [c for c in required_cols if c not in df_in.columns]
    if missing_cols:
        msg = f"Lipsesc urmatoarele coloane obligatorii: {', '.join(missing_cols)}" if is_ro else f"Missing required columns: {', '.join(missing_cols)}"
        errors.append(msg)
        return False, errors, warnings
        
    # Verificare intervale rezonabile (out-of-range)
    if (df_in["ApplicantIncome"] < 0).any():
        errors.append("Venitul aplicantului (ApplicantIncome) nu poate fi negativ." if is_ro else "ApplicantIncome cannot be negative.")
    if (df_in["CoapplicantIncome"] < 0).any():
        errors.append("Venitul codebitorului (CoapplicantIncome) nu poate fi negativ." if is_ro else "CoapplicantIncome cannot be negative.")
    if (df_in["LoanAmount"] <= 0).any():
        errors.append("Suma creditului (LoanAmount) trebuie sa fie strict pozitiva." if is_ro else "LoanAmount must be strictly positive.")
    if (df_in["Loan_Amount_Term"] <= 0).any():
        errors.append("Termenul creditului (Loan_Amount_Term) trebuie sa fie mai mare de 0." if is_ro else "Loan_Amount_Term must be greater than 0.")
        
    # Avertismente pentru valori neobisnuite dar valide
    if (df_in["ApplicantIncome"] > 100000).any():
        warnings.append("Atentie: Venit aplicant foarte mare (> 100.000)." if is_ro else "Warning: Very high applicant income (> 100,000).")
    if (df_in["LoanAmount"] > 1000).any():
        warnings.append("Atentie: Suma solicitata este foarte mare (> 1.000 mii RON)." if is_ro else "Warning: Very high requested loan amount (> 1,000k).")
        
    # Verificare categorii nevazute (demonstrare manevrare sigura)
    known_areas = {"Urban", "Semiurban", "Rural"}
    unseen_areas = set(df_in["Property_Area"].dropna().unique()) - known_areas
    if unseen_areas:
        msg = (f"S-au detectat categorii noi/nevazute pentru Property_Area: {unseen_areas}. "
               f"Acestea vor fi tratate automat cu valoarea 0 in One-Hot Encoding fara erori." if is_ro
               else f"Unseen categories detected for Property_Area: {unseen_areas}. "
                    f"These are automatically handled gracefully with all-zeros in One-Hot Encoding.")
        warnings.append(msg)
        
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def validate_regression_input(df_in, lang="ro"):
    """
    Valideaza intrarile pentru regresia preturilor Craiova (bilingv RO/EN).
    Verifica intervale rezonabile si prezenta coloanelor necesare.
    """
    errors = []
    warnings = []
    is_ro = (lang == "ro")
    
    required_cols = ["An", "index_cost_mat", "urban_incr_proc", "rata_infl", "pib_dolj_mld_ron"]
    missing_cols = [c for c in required_cols if c not in df_in.columns]
    if missing_cols:
        msg = f"Lipsesc coloanele obligatorii: {', '.join(missing_cols)}" if is_ro else f"Missing required columns: {', '.join(missing_cols)}"
        errors.append(msg)
        return False, errors, warnings
        
    if (df_in["An"] < 1900).any() or (df_in["An"] > 2100).any():
        errors.append("Anul trebuie sa fie cuprins intre 1900 si 2100." if is_ro else "Year must be between 1900 and 2100.")
    if (df_in["index_cost_mat"] <= 0).any():
        errors.append("Indexul costului materialelor trebuie sa fie strict pozitiv." if is_ro else "Building materials index must be strictly positive.")
    if (df_in["pib_dolj_mld_ron"] <= 0).any():
        errors.append("PIB-ul judetului Dolj trebuie sa fie pozitiv." if is_ro else "Dolj GDP must be strictly positive.")
        
    if (df_in["rata_infl"] > 50).any():
        warnings.append("Atentie: Rata inflatiei este foarte ridicata (> 50%)." if is_ro else "Warning: Inflation rate is very high (> 50%).")
        
    is_valid = len(errors) == 0
    return is_valid, errors, warnings
