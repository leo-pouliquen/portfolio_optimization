import numpy as np
from scipy.optimize import minimize
from . import metrics

#-------------------------------------------------------------------------------------------------------------------------------

# Modèle Monte Carlo
def monte_carlo(nb_simulation, annual_returns, cov_matrix, risk_free_rate, min_w, max_w):
    print(f"\nLancement de la simulation Monte Carlo...({nb_simulation} itérations)")

    np.random.seed(42)

    nb_tickers = len(annual_returns)

    #Liste de stockage des résultats
    mc_weights = []
    mc_returns = []
    mc_vols = []
    mc_sharpes = []

    for i in range(nb_simulation):
        #Génération des poids
        weights = np.random.dirichlet(np.ones(nb_tickers))

        #Vérification des contraintes
        while not (np.all(weights>= min_w) and np.all(weights <= max_w)):
            weights = np.random.dirichlet(np.ones(nb_tickers))

        # Calcul des métriques pour chaque portefeuille
        returns = metrics.portfolio_return(weights, annual_returns)
        volatility = metrics.portfolio_volatility(weights, cov_matrix)
        sharpe = metrics.portfolio_sharpe_ratio(weights, annual_returns, cov_matrix, risk_free_rate)

        # Stockage dans les liste

        mc_weights.append(weights)
        mc_returns.append(returns)
        mc_vols.append(volatility)
        mc_sharpes.append(sharpe)

    print(f"Simulation terminée : {nb_simulation} portefeuilles générés")

    return (np.array(mc_weights),
            np.array(mc_returns),
            np.array(mc_vols),
            np.array(mc_sharpes))

#-------------------------------------------------------------------------------------------------------------------------------

# Optimisation du ratio de sharpe

# Calcul du ratio sharpe négatif 
def negative_sharpe(weights, annual_returns, cov_matrix, risk_free_rate):
    # On inverse le sharpe ratio pour la minimisation
    return -metrics.portfolio_sharpe_ratio(weights, annual_returns, cov_matrix,risk_free_rate)

# Optimisation (Max Sharpe)
def optimize_max_sharpe(annual_returns, cov_matrix, risk_free_rate, min_w, max_w):
    print("\nLancement de l'optimisation Max Sharpe")
    nb_tickers = len(annual_returns)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((min_w, max_w) for _ in range (nb_tickers))
    init_weights = np.array([1. / nb_tickers] * nb_tickers)

    result = minimize(negative_sharpe, 
                      init_weights, 
                      args=(annual_returns, cov_matrix, risk_free_rate),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    
    if not result.success:
        print("L'optimisation n'a pas convergé")

    return result.x

#-------------------------------------------------------------------------------------------------------------------------------

# Optimisation de la volatilité minimale
# 
def optimize_min_vol(annual_returns, cov_matrix, min_w, max_w):
    print("\nLancement de l'optimisation Min Volatility")
    nb_tickers = len(annual_returns)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((min_w, max_w) for _ in range (nb_tickers))
    init_weights = np.array([1. / nb_tickers] * nb_tickers)

    result = minimize(metrics.portfolio_volatility, 
                      init_weights, 
                      args=(cov_matrix),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    
    if not result.success:
        print("L'optimisation n'a pas convergé")

    return result.x

#-------------------------------------------------------------------------------------------------------------------------------