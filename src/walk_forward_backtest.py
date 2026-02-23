import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.covariance import LedoitWolf
import inspect


def walk_forward_analysis(df_close, model_function, train_years=2, test_months=3, risk_free_rate=0.02, transaction_cost=0.001, min_w=0.0, max_w=1.0):
    dates = df_close.index
    current_date = dates[0] + relativedelta(years=train_years)

    model_returns = []
    weights_history = {}
    previous_weights = np.zeros(df_close.shape[1])

    while current_date + relativedelta(months=test_months) <= dates[-1]:
        # --- 1. Préparation des données ---
        train_start = current_date - relativedelta(years=train_years)
        train_data = df_close.loc[train_start:current_date]
        
        test_end = current_date + relativedelta(months=test_months)
        # On inclut la dernière ligne du train pour avoir le premier rendement du test
        test_data = df_close.loc[current_date:test_end]

        # --- 2. Entraînement et Optimisation ---
        train_log_returns = np.log(train_data / train_data.shift(1)).dropna()
        annual_train_log_returns = train_log_returns.mean() * 252
        
        lw = LedoitWolf().fit(train_log_returns)
        annual_cov_matrix = lw.covariance_ * 252
        
        # On convertit en DataFrame pour être sûr de garder le nom des colonnes/actifs
        cov_df = pd.DataFrame(annual_cov_matrix, index=df_close.columns, columns=df_close.columns)

        # On met tous les noms de paramètres possibles ici
        available_params = {
            'annual_returns': annual_train_log_returns,
            'annual_log_returns': annual_train_log_returns, # Alias
            'cov_matrix': cov_df,
            'annual_cov_matrix': cov_df,                    # Alias
            'risk_free_rate': risk_free_rate,
            'min_w': min_w,
            'max_w': max_w,
            'daily_returns': train_log_returns              # Anticipation pour CVaR !
        }
        
        # Le module inspect regarde ce dont la fonction (model_function) a réellement besoin
        sig = inspect.signature(model_function)
        kwargs_to_pass = {k: v for k, v in available_params.items() if k in sig.parameters}
        
        try:
            # On exécute la fonction avec son "panier" d'arguments sur-mesure
            weights = model_function(**kwargs_to_pass)
        except Exception as e:
            print(f"⚠️ Échec de l'optimisation le {current_date.strftime('%Y-%m-%d')} : {e}")
            weights = np.array([1/len(df_close.columns)] * len(df_close.columns)) # Secours: Equal Weights
            
        weights_history[current_date.strftime('%Y-%m-%d')] = weights

        # --- 3. Calcul des rendements du Test ---
        test_log_returns = np.log(test_data / test_data.shift(1)).dropna()
        # Calcul du rendement brut du portefeuille (série temporelle)
        portfolio_returns = pd.Series(test_log_returns.dot(weights), index=test_log_returns.index)

        # --- 4. Application des coûts de transaction---
        # Le turnover est calculé une fois par rééquilibrage
        turnover = np.abs(weights - previous_weights).sum()
        total_cost = turnover * transaction_cost
        
        # On déduit les frais uniquement sur le premier jour de la période de test
        if not portfolio_returns.empty:
            portfolio_returns.iloc[0] -= total_cost

        model_returns.append(portfolio_returns)

        # Mise à jour pour la prochaine itération
        previous_weights = weights.copy()
        current_date += relativedelta(months=test_months)

    # Concaténation et suppression des doublons de dates dus au chevauchement
    final_returns = pd.concat(model_returns)
    final_returns = final_returns[~final_returns.index.duplicated(keep='first')]

    return final_returns, pd.DataFrame(weights_history, index=df_close.columns).T