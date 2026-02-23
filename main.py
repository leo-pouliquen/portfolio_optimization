import config
from src import data_loader, data_processing, visualization, metrics, models, storage, walk_forward_backtest
import pandas as pd
import numpy as np

def main():
    print("--- Démarrage du projet d'Optimisation de Portefeuille ---")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    #-----------------------------------------------------------------------------------------------------------------
    # ÉTAPE 1 : Chargement des données
    #-----------------------------------------------------------------------------------------------------------------
    print(f"Téléchargement des données pour : {config.tickers}")

    df_close = data_loader.download_portfolio_data(tickers=config.tickers,
                                                   start_date=config.start_date,
                                                   end_date=config.end_date)


    #-----------------------------------------------------------------------------------------------------------------
    # ÉTAPE 2 : Retraitement et analyse des données
    #-----------------------------------------------------------------------------------------------------------------

    #Valeurs manquantes : 
    missing_values = data_processing.check_missing_values(df_close)

    #Calcul des rendements, de la volatilité et de la covariance

    daily_log_returns, annual_log_returns = data_processing.calculate_returns(df_close)
    daily_volatility = data_processing.calculate_volatility(daily_log_returns)
    daily_cov_matrix, annual_cov_matrix = data_processing.calculate_cov_matrix(daily_log_returns)


    # Tests de stationnarité 
    data_processing.check_stationnarity(daily_log_returns)

    # Matrice de corrélation des rendements
    visualization.plot_correlation_matrix(daily_log_returns)


    #-----------------------------------------------------------------------------------------------------------------
    # ÉTAPE 3 : Préparation de la modélisation
    #-----------------------------------------------------------------------------------------------------------------

    # Initialisation du dictionnaire de stockage
    portfolio_results = {}

    # Stockage du portefeuille equal weights
    ptf_ew_weights = np.array([1/len(config.tickers)] * len(config.tickers))

    storage.store_portfolio_model(
        storage_dict=portfolio_results,
        name = "Equal Weights",
        weights=ptf_ew_weights,
        annual_returns=annual_log_returns,
        cov_matrix=annual_cov_matrix,
        risk_free_rate=config.risk_free_rate)


    #-----------------------------------------------------------------------------------------------------------------
    # ÉTAPE 4 : Monte Carlo
    #-----------------------------------------------------------------------------------------------------------------

    # Lancement de la simulation Monte Carlo
    mc_weights, mc_returns, mc_vols, mc_sharpes = models.monte_carlo(
        nb_simulation=15000,
        annual_returns=annual_log_returns,
        cov_matrix=annual_cov_matrix,
        risk_free_rate=config.risk_free_rate,
        min_w=config.Min_weights,
        max_w=config.Max_weights)
    
    # Récupération du portefeuille Max Sharpe
    mc_max_sharpe_idx = mc_sharpes.argmax()
    mc_max_sharpe_weights = mc_weights[mc_max_sharpe_idx]

    # Récupération du portefeuille Min Volatility
    mc_min_vol_idx = mc_vols.argmin()
    mc_min_vol_weights = mc_weights[mc_min_vol_idx]

    # Stockage dans le dictionnaire
    # Max Sharpe (Monte Carlo)
    storage.store_portfolio_model(
        storage_dict=portfolio_results,
        name = "Max Sharpe (Monte Carlo)",
        weights=mc_max_sharpe_weights,
        annual_returns=annual_log_returns,
        cov_matrix=annual_cov_matrix,
        risk_free_rate=config.risk_free_rate)
    
    # Min Volatility (Monte Carlo)
    storage.store_portfolio_model(
        storage_dict=portfolio_results,
        name = "Min Volatility (Monte Carlo)",
        weights=mc_min_vol_weights,
        annual_returns=annual_log_returns,
        cov_matrix=annual_cov_matrix,
        risk_free_rate=config.risk_free_rate)

    # Visualisation de la frontière efficiente
    visualization.plot_efficient_frontier(
        mc_returns=mc_returns,
        mc_vols=mc_vols,
        mc_sharpes=mc_sharpes,
        portfolio_results=portfolio_results,
        risk_free_rate=config.risk_free_rate,
        tickers=config.tickers)

    #-----------------------------------------------------------------------------------------------------------------
    # ÉTAPE 5 : Optimisation
    #-----------------------------------------------------------------------------------------------------------------

    # Optimisation Max Sharpe
    opti_max_sharpe_weights = models.optimize_max_sharpe(
        annual_returns=annual_log_returns,
        cov_matrix=annual_cov_matrix,
        risk_free_rate=config.risk_free_rate,
        min_w=config.Min_weights,
        max_w=config.Max_weights
    )
    

    # Optimisation Min Vol
    opti_min_vol_weights = models.optimize_min_vol(
        annual_returns=annual_log_returns,
        cov_matrix=annual_cov_matrix,
        min_w=config.Min_weights,
        max_w=config.Max_weights
    )

    # Stockage
    # Max Sharpe (Covariance Standard)
    storage.store_portfolio_model(
        storage_dict=portfolio_results,
        name= "Max Sharpe (Optimisation | Covariance Standard)",
        weights=opti_max_sharpe_weights,
        annual_returns=annual_log_returns,
        cov_matrix=annual_cov_matrix,
        risk_free_rate=config.risk_free_rate
    )

    # Min Vol (Covariance Standard)
    storage.store_portfolio_model(
        storage_dict=portfolio_results,
        name= "Min Volatility  (Optimisation | Covariance Standard)",
        weights=opti_min_vol_weights,
        annual_returns=annual_log_returns,
        cov_matrix=annual_cov_matrix,
        risk_free_rate=config.risk_free_rate
    )


    # Tableau comparatif des modèles
    visualization.display_model_comparison(portfolio_results)

    visualization.plot_model_allocations(portfolio_results, config.tickers)


    #-----------------------------------------------------------------------------------------------------------------
    # ÉTAPE 6 : Optimisation avec Covariance Ledoit Wolf
    #-----------------------------------------------------------------------------------------------------------------


    # Calcul de la covariance Ledoit Wolf
    lw_annual_cov_matrix = data_processing.calculate_ledoitwolf_cov(daily_log_returns)

    # Optimisation Max Sharpe | Covariance Ledoit Wolf
    lw_opti_max_sharpe_weights = models.optimize_max_sharpe(
        annual_returns=annual_log_returns,
        cov_matrix=lw_annual_cov_matrix,
        risk_free_rate=config.risk_free_rate,
        min_w=config.Min_weights,
        max_w=config.Max_weights
    )

    # Optimisation Min Vol | Covariance Ledoit Wolf
    lw_opti_min_vol_weights = models.optimize_min_vol(
        annual_returns=annual_log_returns,
        cov_matrix=lw_annual_cov_matrix,
        min_w=config.Min_weights,
        max_w=config.Max_weights
    )


    # Stockage
    # Max Sharpe (Covariance Ledoit Wolf)
    storage.store_portfolio_model(
        storage_dict=portfolio_results,
        name= "Max Sharpe (Optimisation | Covariance Ledoit Wolf)",
        weights=lw_opti_max_sharpe_weights,
        annual_returns=annual_log_returns,
        cov_matrix=annual_cov_matrix, # On utilise ici la covariance standard pour avoir une même base de comparaison entre les modèles
        risk_free_rate=config.risk_free_rate
    )

    # Min Vol (Covariance Ledoit Wolf)
    storage.store_portfolio_model(
        storage_dict=portfolio_results,
        name= "Min Volatility  (Optimisation | Covariance Ledoit Wolf)",
        weights=lw_opti_min_vol_weights,
        annual_returns=annual_log_returns,
        cov_matrix=annual_cov_matrix, # On utilise ici la covariance standard pour avoir une même base de comparaison entre les modèles
        risk_free_rate=config.risk_free_rate
    )


    # Tableau comparatif des modèles
    visualization.display_model_comparison(portfolio_results)

    visualization.plot_model_allocations(portfolio_results, config.tickers)


    #-----------------------------------------------------------------------------------------------------------------
    # ÉTAPE 7 : Calcul des métriques de performance et comparaison des modèles
    #-----------------------------------------------------------------------------------------------------------------

    # Nouveau dictionnaire de stockage des modèles avec calcul des métriques
    portfolio_results_metrics = {}

    # Equal Weights
    storage.store_portfolio_model_metrics(
        storage_dict=portfolio_results_metrics,
        name="Equal Weights",
        weights=ptf_ew_weights,
        annual_returns=annual_log_returns,
        daily_returns=daily_log_returns,
        cov_matrix=annual_cov_matrix,
        risk_free_rate=config.risk_free_rate
    )

    # Max Sharpe (Monte Carlo)
    storage.store_portfolio_model_metrics(
        storage_dict=portfolio_results_metrics,
        name="Max Sharpe (Monte Carlo)",
        weights=mc_max_sharpe_weights,
        annual_returns=annual_log_returns,
        daily_returns=daily_log_returns,
        cov_matrix=annual_cov_matrix,
        risk_free_rate=config.risk_free_rate
    )

    # Min Volatility (Monte Carlo)
    storage.store_portfolio_model_metrics(
        storage_dict=portfolio_results_metrics,
        name="Min Volatility (Monte Carlo)",
        weights=mc_min_vol_weights,
        annual_returns=annual_log_returns,
        daily_returns=daily_log_returns,
        cov_matrix=annual_cov_matrix,
        risk_free_rate=config.risk_free_rate
    )

   # Max Sharpe (Covariance Standard)
    storage.store_portfolio_model_metrics(
        storage_dict=portfolio_results_metrics,
        name="Max Sharpe (Optimisation | Covariance Standard)",
        weights=opti_max_sharpe_weights,
        annual_returns=annual_log_returns,
        daily_returns=daily_log_returns,
        cov_matrix=annual_cov_matrix,
        risk_free_rate=config.risk_free_rate
    )

    # Min Vol (Covariance Standard)
    storage.store_portfolio_model_metrics(
        storage_dict=portfolio_results_metrics,
        name="Min Volatility  (Optimisation | Covariance Standard)",
        weights=opti_min_vol_weights,
        annual_returns=annual_log_returns,
        daily_returns=daily_log_returns,
        cov_matrix=annual_cov_matrix,
        risk_free_rate=config.risk_free_rate
    )

    # Max Sharpe (Covariance Ledoit Wolf)
    storage.store_portfolio_model_metrics(
        storage_dict=portfolio_results_metrics,
        name="Max Sharpe (Optimisation | Covariance Ledoit Wolf)",
        weights=lw_opti_max_sharpe_weights,
        annual_returns=annual_log_returns,
        daily_returns=daily_log_returns,
        cov_matrix=annual_cov_matrix, # On utilise ici la covariance standard pour avoir une même base de comparaison entre les modèles
        risk_free_rate=config.risk_free_rate
    )

    # Min Vol (Covariance Ledoit Wolf)
    storage.store_portfolio_model_metrics(
        storage_dict=portfolio_results_metrics,
        name="Min Volatility  (Optimisation | Covariance Ledoit Wolf)",
        weights=lw_opti_min_vol_weights,
        annual_returns=annual_log_returns,
        daily_returns=daily_log_returns,
        cov_matrix=annual_cov_matrix, # On utilise ici la covariance standard pour avoir une même base de comparaison entre les modèles
        risk_free_rate=config.risk_free_rate
    )


    # Tableau comparatif des modèles
    visualization.display_model_comparison_metrics(portfolio_results_metrics)


    #-----------------------------------------------------------------------------------------------------------------
    # ÉTAPE 8 : Analyse Walk Foward
    #-----------------------------------------------------------------------------------------------------------------

    # Dictionnaire de stockage des résultats de l'analyse walk forward
    walk_forward_result = {}

    # WF Max Sharpe (Covariance Ledoit Wolf)
    wf_max_sharpe_returns, wf_max_sharpe_weights = walk_forward_backtest.walk_forward_analysis(
        df_close = df_close,
        model_function=models.optimize_max_sharpe,
        train_years=config.train_years,
        test_months=config.test_months,
        risk_free_rate=config.risk_free_rate,
        max_w=config.Max_weights,
        min_w=config.Min_weights,
        transaction_cost=config.transaction_cost
    )

    # WF Min Volatility (Covariance Ledoit Wolf)
    wf_min_vol_returns, wf_min_vol_weights = walk_forward_backtest.walk_forward_analysis(
        df_close = df_close,
        model_function=models.optimize_min_vol,
        train_years=config.train_years,
        test_months=config.test_months,
        risk_free_rate=config.risk_free_rate,
        max_w=config.Max_weights,
        min_w=config.Min_weights,
        transaction_cost=config.transaction_cost
    )

    # Stockage des résultats
    # WF Max Sharpe (Covariance Ledoit Wolf)
    storage.store_portfolio_model_walkforward(
        storage_dict=walk_forward_result,
        name="[WF] Max Sharpe",
        wf_returns=wf_max_sharpe_returns,
        risk_free_rate=config.risk_free_rate
    )


    # WF Min Volatility (Covariance Ledoit Wolf)
    storage.store_portfolio_model_walkforward(
        storage_dict=walk_forward_result,
        name="[WF] Min Volatility",
        wf_returns=wf_min_vol_returns,
        risk_free_rate=config.risk_free_rate
    )


    # Visualisation graphique 
    print("Génération des graphiques de backtest walk forward")

    visualization.plot_walk_foward_weights(
        wf_weights_max_sharpe=wf_max_sharpe_weights,
        wf_weights_min_vol=wf_min_vol_weights)
    

    visualization.plot_wf_performance(
        wf_returns_max_sharpe=wf_max_sharpe_returns,
        wf_returns_min_vol=wf_min_vol_returns)
    


    #-----------------------------------------------------------------------------------------------------------------
    # ÉTAPE 9 : Comparaison avec benchmark (S&P500 & NASQAD100)
    #-----------------------------------------------------------------------------------------------------------------

    # Téléchargement des données
    df_benchmark = data_loader.download_portfolio_data(tickers=config.benchmark_tickers,
                                                   start_date=config.start_date,
                                                   end_date=config.end_date)


    # Valeurs manquantes
    benchmark_missing_values = data_processing.check_missing_values(df_benchmark)
    print(benchmark_missing_values)


    # Calcul des rendements
    benchmark_daily_log_returns, _ = data_processing.calculate_returns(df_benchmark)


    # On récupère le portefeuille Equal weights
    ew_daily_returns_series = daily_log_returns.dot(ptf_ew_weights)

    # Alignement des dates avec celles de l'analyse walf forward
    aligned_benchmark_log_returns = benchmark_daily_log_returns.reindex(wf_max_sharpe_returns.index).fillna(0)
    aligned_ew_returns = ew_daily_returns_series.reindex(wf_max_sharpe_returns.index).fillna(0)

    # Calcul des métriques de performance
    # S&P500
    storage.store_portfolio_model_walkforward(
        storage_dict=walk_forward_result,
        name="[Benchmark] S&P 500",
        wf_returns=aligned_benchmark_log_returns['SPY'],
        risk_free_rate=config.risk_free_rate
    )

    # NASDAQ 100
    storage.store_portfolio_model_walkforward(
        storage_dict=walk_forward_result,
        name="[Benchmark] NASDAQ 100",
        wf_returns=aligned_benchmark_log_returns['QQQ'],
        risk_free_rate=config.risk_free_rate
    )

    # Equal Weights
    storage.store_portfolio_model_walkforward(
        storage_dict=walk_forward_result,
        name="[Benchmark] Equal Weights",
        wf_returns=aligned_ew_returns,
        risk_free_rate=config.risk_free_rate
    )

    # Comparaison des métriques de performances
    visualization.display_model_comparison_metrics(walk_forward_result)


    #Initialisation du dictionnaire pour calcul des métriques relatives
    wf_relative_metrics = {}


    # Calcul des métriques relatives

    # Max Sharpe vs S&P500
    storage.store_relative_metrics(
        storage_dict=wf_relative_metrics,
        name='[WF] Max Sharpe vs S&P500',
        ptf_daily_returns=wf_max_sharpe_returns,
        benchmark_returns=aligned_benchmark_log_returns['SPY'],
        risk_free_rate=config.risk_free_rate
    )

    # Max Sharpe vs NASDAQ
    storage.store_relative_metrics(
        storage_dict=wf_relative_metrics,
        name='[WF] Max Sharpe vs NASDAQ 100',
        ptf_daily_returns=wf_max_sharpe_returns,
        benchmark_returns=aligned_benchmark_log_returns['QQQ'],
        risk_free_rate=config.risk_free_rate
    )


    # Min Volatility vs S&P500
    storage.store_relative_metrics(
        storage_dict=wf_relative_metrics,
        name='[WF] Min Volatility vs S&P500',
        ptf_daily_returns=wf_min_vol_returns,
        benchmark_returns=aligned_benchmark_log_returns['SPY'],
        risk_free_rate=config.risk_free_rate
    )

    # Min Volatility vs NASDAQ 100
    storage.store_relative_metrics(
        storage_dict=wf_relative_metrics,
        name='[WF] Min Volatility vs NASDAQ 100',
        ptf_daily_returns=wf_min_vol_returns,
        benchmark_returns=aligned_benchmark_log_returns['QQQ'],
        risk_free_rate=config.risk_free_rate
    )

    # Comparaison des métriques de performances
    visualization.display_relative_metrics(wf_relative_metrics)


    # Représentation graphique des performances cumulées
    visualization.plot_comparative_performance(
        wf_max_sharpe_returns=wf_max_sharpe_returns,
        wf_min_vol_returns=wf_min_vol_returns,
        sp500_returns=benchmark_daily_log_returns['SPY'],
        nasdaq_returns=benchmark_daily_log_returns['QQQ'],
        ew_returns=aligned_ew_returns
    )


    # Représentation graphique des drawdown
    visualization.plot_comparative_drawdowns(
        wf_max_sharpe_returns=wf_max_sharpe_returns,
        wf_min_vol_returns=wf_min_vol_returns,
        sp500_returns=benchmark_daily_log_returns['SPY'],
        nasdaq_returns=benchmark_daily_log_returns['QQQ'],
        ew_returns=aligned_ew_returns
    )

    # Représentation graphique de la distribution des rendements
    visualization.plot_return_distributions(
        wf_max_sharpe_returns=wf_max_sharpe_returns,
        wf_min_vol_returns=wf_min_vol_returns,
        sp500_returns=benchmark_daily_log_returns['SPY'],
        nasdaq_returns=benchmark_daily_log_returns['QQQ'],
        ew_returns=aligned_ew_returns
    )

     
    print("\n--- Fin de l'exécution du projet ---")



if __name__ == "__main__":
    main()


