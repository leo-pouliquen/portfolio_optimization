import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.covariance import LedoitWolf

#-------------------------------------------------------------------------------------------------------------------------------

def check_missing_values(df_close):

    # Calcul de la somme des valeurs manquantes
    missing_values_count = df_close.isnull().sum().sum()

    if missing_values_count > 0 : 
        print(f"\nAttention : il y a un total de {missing_values_count} valeurs manquantes dans les prix de clôtures")
    else:
        print("\nAucune valeur manquante détectée")

#-------------------------------------------------------------------------------------------------------------------------------

def calculate_returns(df_close):
    # Rendements logarithmiques quotidens
    daily_log_returns = np.log(df_close / df_close.shift(1)).dropna()

    # Rendements annuels moyens
    annual_log_returns = daily_log_returns.mean()*252

    return daily_log_returns, annual_log_returns

#-------------------------------------------------------------------------------------------------------------------------------

def calculate_volatility(daily_returns):
    # Volatilité quotidienne (Ecart-type)
    daily_volatility = daily_returns.std()

    return daily_volatility

def calculate_cov_matrix(daily_returns):
    # Matrice de covariance
    daily_cov_matrix = daily_returns.cov()
    annual_cov_matrix = daily_cov_matrix*252

    return daily_cov_matrix, annual_cov_matrix

#-------------------------------------------------------------------------------------------------------------------------------

def check_stationnarity(daily_returns):

    #Effectue le test de Dickey-Fuller Augmenté (ADF) sur chaque ticker pour vérifier la stationnarité des rendements
    print("\n--- Test de Stationnarité (ADF) ---")

    non_stationnary_found = False

    # Boucle sur chaque colonne (chaque ticker)
    for ticker in daily_returns.columns:
        adf_test = adfuller(daily_returns[ticker])
        p_value = adf_test[1]

        print(f"\n{ticker}: p-value : {adf_test[1]:.4f}")

        if p_value < 0.05:
            print(f"[OK] Série STATIONNAIRE (p-value < 0.05)")
        else:
            print(f"[NO] Série NON STATIONNAIRE (p-value ≥ 0.05)")
            non_stationnary_found = True

    if non_stationnary_found:
        print(f"\nAttention : Certaines séries ne sont pas stationnaires")
    else : 
        print(f"\n Toutes les séries sont stationnnaires")

#-------------------------------------------------------------------------------------------------------------------------------

# Calcul de la covariance Ledoit Wolf
def calculate_ledoitwolf_cov(daily_returns):
    lw = LedoitWolf()
    lw.fit(daily_returns)

    lw_daily_cov = lw.covariance_
    lw_annual_cov = lw_daily_cov * 252

    shrinkage_intensity  = lw.shrinkage_

    print(f"L'intensité de shrinkage est de : {shrinkage_intensity}")

    return lw_annual_cov

#-------------------------------------------------------------------------------------------------------------------------------