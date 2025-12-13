import pandas as pd

df = pd.read_csv("ins_indici_cost_mat_construct.csv")

month_map = {
    "ianuarie": "January",
    "februarie": "February",
    "martie": "March",
    "aprilie": "April",
    "mai": "May",
    "iunie": "June",
    "iulie": "July",
    "august": "August",
    "septembrie": "September",
    "octombrie": "October",
    "noiembrie": "November",
    "decembrie": "December"
}

df["Timp"] = df["Timp"].str.replace("Luna ", "", regex=False)

for ro, en in month_map.items():
    df["Timp"] = df["Timp"].str.replace(ro, en, regex=False)

df["date"] = pd.to_datetime(df["Timp"], format="mixed")
df.drop(columns=["Timp"], inplace=True)

df.to_csv("ins_indici_cost_mat_construct_clean.csv", index=False)

print(df.head())