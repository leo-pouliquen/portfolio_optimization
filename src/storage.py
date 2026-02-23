import pandas as pd
import numpy as np
from . import metrics

#-------------------------------------------------------------------------------------------------------------------------------

# Stockage des portefeuilles dans un dictionnaire
def store_portfolio_model(storage_dict, name, weights, annual_returns, cov_matrix, risk_free_rate):
    returns = metrics.portfolio_return(weights, annual_returns)
    volatility = metrics.portfolio_volatility(weights, cov_matrix)
    sharpe = metrics.portfolio_sharpe_ratio(weights, annual_returns, cov_matrix, risk_free_rate)

    storage_dict[name] = {
        'weights': weights,
        'returns': returns,
        'volatility':volatility,
        'sharpe': sharpe
    }

    print(f"\n[OK] Modèle '{name}' stocké avec succès")
    print(f"{len(storage_dict)} modèle(s) au total dans le dictionnaire")

#-------------------------------------------------------------------------------------------------------------------------------

# Stockage des portefeuilles avec les nouvelles métriques dans un dictionnaire
def store_portfolio_model_metrics(storage_dict, name, weights, annual_returns, daily_returns, cov_matrix, risk_free_rate):

    portfolio_metrics = metrics.portfolio_metrics(weights, daily_returns, risk_free_rate)

    storage_dict[name] = {
        'weights': weights.copy(),
        'returns': metrics.portfolio_return(weights, annual_returns),
        'volatility': metrics.portfolio_volatility(weights, cov_matrix),
        'sharpe': metrics.portfolio_sharpe_ratio(weights, annual_returns, cov_matrix, risk_free_rate),
        'sortino': portfolio_metrics.get('sortino'),
        'calmar': portfolio_metrics.get('calmar'),
        'max_drawdown': portfolio_metrics.get('max_drawdown'),
        'recovery_time_days': portfolio_metrics.get('recovery_time_days'),
        'var_95': portfolio_metrics.get('var_95'),
        'cvar_95': portfolio_metrics.get('cvar_95')
    }

    print(f"\n[OK] Modèle '{name}' stocké avec calcul des métriques")
    print(f"{len(storage_dict)} modèle(s) au total dans le dictionnaire")

#-------------------------------------------------------------------------------------------------------------------------------

# Stockage des portefeuilles walk forward 
def store_portfolio_model_walkforward(storage_dict, name, wf_returns, risk_free_rate):
    wf_metrics = metrics.wf_portfolio_metrics(wf_returns, risk_free_rate)

    annual_wf_returns = wf_returns.mean()* 252
    annual_wf_volatility = wf_returns.std() * np.sqrt(252)
    sharpe_wf = (annual_wf_returns - risk_free_rate) / annual_wf_volatility if annual_wf_volatility != 0 else np.nan

    storage_dict[name] = {
      'weights': 'Dynamic (WF)', 
        'returns': annual_wf_returns,
        'volatility': annual_wf_volatility,
        'sharpe': sharpe_wf,
        'sortino': wf_metrics.get('sortino'),
        'calmar': wf_metrics.get('calmar'),
        'max_drawdown': wf_metrics.get('max_drawdown'),
        'recovery_time_days': wf_metrics.get('recovery_time_days'),
        'var_95': wf_metrics.get('var_95'),
        'cvar_95': wf_metrics.get('cvar_95')
    }

    print(f"\n[OK] Stratégie Out Of Sample '{name}' stockée avec succès")  
    print(f"{len(storage_dict)} modèle(s) au total dans le dictionnaire")

#-------------------------------------------------------------------------------------------------------------------------------

# Stockage des portefeuille avec calcul des métriques relatives
def store_relative_metrics(storage_dict, name, ptf_daily_returns, benchmark_returns, risk_free_rate):
    relative_metrics = metrics.portfolio_relative_metrics(ptf_daily_returns, benchmark_returns, risk_free_rate)

    storage_dict[name] = {
        'alpha' : relative_metrics.get('alpha'),
        'beta' : relative_metrics.get('beta'),
        'tracking_error': relative_metrics.get('tracking_error'),
        'information_ratio' : relative_metrics.get('information_ratio'),
        'win_rate' : relative_metrics.get('win_rate'),
        'correlation' : relative_metrics.get('correlation')
    }

    print(f"\n[OK] Métriques relatives pour '{name}' stockées avec succès")
    print(f"{len(storage_dict)} modèle(s) au total dans le dictionnaire")
