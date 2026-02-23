import numpy as np
import pandas as pd
from tabulate import tabulate

#-------------------------------------------------------------------------------------------------------------------------------

# Calcul le rendement attendu du portefeuille
def portfolio_return(weights, annual_returns):
    return np.sum(weights * np.asarray(annual_returns))

#-------------------------------------------------------------------------------------------------------------------------------

# Calcul la volatilité attendue du portefeuille
def portfolio_volatility(weights, cov_matrix):
    cov_arr = np.asarray(cov_matrix)
    variance = weights.T @ cov_arr @ weights

    return np.sqrt(variance).item()

#-------------------------------------------------------------------------------------------------------------------------------

# Calcul du ratio de Sharpe du portefeuille
def portfolio_sharpe_ratio(weights, annual_returns, cov_matrix, risk_free_rate):
    p_return = portfolio_return(weights, annual_returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    
    if p_volatility == 0:
        return 0
        
    sharpe_ratio = (p_return - risk_free_rate) / p_volatility
    return float(sharpe_ratio)

#-------------------------------------------------------------------------------------------------------------------------------

# Calcul des métriques de performance des modèles
def portfolio_metrics(weights, daily_returns, risk_free_rate):
    portfolio_daily_returns = daily_returns.dot(weights) #On calcul les rendements quotidiens du portefeuille
    cumulates_returns = np.exp(portfolio_daily_returns.cumsum()) # Rendements cumulés

    # Max Drawdown
    running_max = cumulates_returns.cummax()
    drawdown = (cumulates_returns - running_max) / running_max
    max_drawdown = drawdown.min() # Le plus faible des drawdown = au maximum des drawdown

    # Recovery time
    is_in_drawdown = drawdown < 0 
    runs = is_in_drawdown.astype(int).groupby((~is_in_drawdown).cumsum()).cumsum()
    recovery_time = runs.max()

    # VaR et CVaR 1 jours à 95%
    var_95 = np.percentile(portfolio_daily_returns, 5)
    cvar_95 = portfolio_daily_returns[portfolio_daily_returns <= var_95].mean()

    #Sortino Ratio
    downside_returns = portfolio_daily_returns[portfolio_daily_returns < 0] 
    downside_deviation = downside_returns.std() * np.sqrt(252)
    expected_return = portfolio_daily_returns.mean() * 252
    sortino = (expected_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan

    #Calmar ratio
    calmar = expected_return / abs(max_drawdown) if max_drawdown != 0 else np.nan


    return {
        'max_drawdown': max_drawdown,
        'recovery_time_days': recovery_time,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'sortino': sortino,
        'calmar': calmar
    }

#-------------------------------------------------------------------------------------------------------------------------------

# Calcul des métriques de performance des modèles en analyse walk forward
# Il faut créer une nouvelle fonction comme les poids varient dans le temps
def wf_portfolio_metrics(wf_returns, risk_free_rate):
   
    # Max Drawdown
    cum_returns = np.exp(wf_returns.cumsum()) 
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
   
    #Recovery Time
    is_in_drawdown = drawdown < 0 
    runs = is_in_drawdown.astype(int).groupby((~is_in_drawdown).cumsum()).cumsum()
    recovery_time = runs.max()

    # VaR et CVaR
    var_95 = np.percentile(wf_returns, 5)
    cvar_95 = wf_returns[wf_returns <= var_95].mean()

    # Sortino ratio
    downside_returns = wf_returns[wf_returns < 0] 
    downside_deviation = downside_returns.std() * np.sqrt(252)
    expected_return = wf_returns.mean() * 252
    sortino = (expected_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan

    # Calmar ratio
    calmar = expected_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return {
        'max_drawdown': max_drawdown,
        'recovery_time_days': recovery_time,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'sortino': sortino,
        'calmar': calmar
    }

#-------------------------------------------------------------------------------------------------------------------------------

# Calcul des métriques relatives, par comparaison avec un benchmark
def portfolio_relative_metrics(ptf_daily_returns, benchmark_daily_returns, risk_free_rate):
    aligned_data = pd.DataFrame({
        'portfolio': ptf_daily_returns,
        'benchmark': benchmark_daily_returns }).dropna()
    
    ptf_returns = aligned_data['portfolio']
    benchmark_returns = aligned_data['benchmark']

    # Beta
    cov_matrix = np.cov(ptf_returns, benchmark_returns) 
    covariance = cov_matrix[0, 1]
    benchmark_variance = cov_matrix[1, 1]
    beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan

    # Alpha annualisé
    rf_daily = risk_free_rate / 252
    excess_port_ret = ptf_returns - rf_daily
    excess_bench_ret = benchmark_returns - rf_daily
    # On calcule l'alpha moyen journalier puis on l'annualise
    daily_alpha = excess_port_ret.mean() - beta * excess_bench_ret.mean()
    alpha = daily_alpha * 252

    # Tracking Error (annualisé)
    active_returns = ptf_returns - benchmark_returns
    tracking_error = active_returns.std(ddof=1) * np.sqrt(252)

    # Information Ratio
    mean_active_return = active_returns.mean() * 252
    information_ratio = mean_active_return / tracking_error if tracking_error != 0 else np.nan
    
    # Win Rate vs Benchmark (% de jours où on bat le benchmark)
    win_rate = (ptf_returns > benchmark_returns).mean() 
    
    # Correlation
    correlation = ptf_returns.corr(benchmark_returns)

    return {
        'alpha': alpha,
        'beta': beta,
        'information_ratio': information_ratio,
        'tracking_error': tracking_error,
        'win_rate': win_rate,
        'correlation': correlation
    }

#-------------------------------------------------------------------------------------------------------------------------------