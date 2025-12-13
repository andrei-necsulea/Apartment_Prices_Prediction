import pandas as pd

df_infl = pd.read_csv("../../new_features/web_data/INFL/inflation_1900_2025.csv")
df_mat_construct = pd.read_csv("../../new_features/web_data/mat_construct_increase/index_cost_mat_1900_2025.csv")
df_PIB = pd.read_csv("../../new_features/web_data/PIB/pib_dolj_1900_2025.csv")
df_urban_pop_increase = pd.read_csv("../../new_features/web_data/URBAN_INCREASE/urban_population_increase_dolj_1900_2025.csv")

df_new_features = (
    df_mat_construct
    .merge(df_urban_pop_increase, on="An", how="left")
    .merge(df_infl, on="An", how="left")
    .merge(df_PIB, on="An", how="left")
)

df_new_features = df_new_features.rename(
    columns={'crestere_procente': 'urban_incr_proc',
             'pib_dolj_mld_lei' : 'pib_dolj_mld_ron'
    }
)

df_new_features.to_csv("new_features.csv", index=False)