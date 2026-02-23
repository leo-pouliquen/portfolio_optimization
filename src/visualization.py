import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import skew, kurtosis

#-------------------------------------------------------------------------------------------------------------------------------

def plot_correlation_matrix(daily_returns):

    print("Génération du graphique de corrélation des rendements journaliers")
    
    # Calcul de la matrice
    correlation_matrix = daily_returns.corr()

    # Création de la figure
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matrice de Corrélation des Actifs du Portefeuille")
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------

def plot_efficient_frontier(mc_returns, mc_vols, mc_sharpes, portfolio_results, risk_free_rate, tickers):

    print("Génération du graphique de la frontière efficiente")

    fig, ax = plt.subplots(figsize=(14,8))
    
    # Nuage de points
    scatter = ax.scatter(mc_vols, mc_returns, 
                         c=mc_sharpes, cmap="plasma",
                         alpha=0.6, s=20, edgecolors="none")
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Ratio de Sharpe", fontsize=12)

    # Portefeuille Max Sharpe
    max_sharpe_data = portfolio_results.get("Max Sharpe (Monte Carlo)")

    if max_sharpe_data:
        max_sharpe_vol = max_sharpe_data["volatility"]
        max_sharpe_returns = max_sharpe_data["returns"]
        max_sharpe_sharpe = max_sharpe_data["sharpe"]

        # Point Max Sharpe
        ax.scatter(max_sharpe_vol, max_sharpe_returns,
                   marker=".", color="red", s=500,
                   label=f"Sharpe Max ({max_sharpe_sharpe:.2f})",
                   edgecolors="black", linewidths=2, zorder=5)
        
        
    # Portefeuille Min Volatility
    min_volatiliy_data = portfolio_results.get("Min Volatility (Monte Carlo)")

    if min_volatiliy_data:
        min_volatiliy_vol = min_volatiliy_data["volatility"]
        min_volatiliy_returns = min_volatiliy_data["returns"]
        min_volatiliy_sharpe = min_volatiliy_data["sharpe"]

        # Point Min Volatiliy
        ax.scatter(min_volatiliy_vol, min_volatiliy_returns,
                   marker=".", color="lime", s=500,
                   label=f"Volatility Min ({min_volatiliy_sharpe:.2f})",
                   edgecolors="black", linewidths=2, zorder=5)   

    # Taux sans risques  
    ax.scatter(0, risk_free_rate, marker='.', color='blue', s=300,
               label=f'Taux sans risque ({risk_free_rate:.2%})',
               edgecolors='black', linewidths=2, zorder=5)
    
    # Capital Market Line (CML)
    # Il s'agit de la tangente au portefeuille Max Sharpe et qui coupe l'axe des ordonnées par le taux sans risques
    max_vol_axis = mc_vols.max()
    cml_x = np.linspace(0, max_vol_axis * 1.1, 100)
    # Equation droite : y = Rf + Sharpe * x
    cml_y = risk_free_rate + max_sharpe_sharpe * cml_x

    ax.plot(cml_x, cml_y, 
        color='darkblue', linewidth=2.5, linestyle='-',
        label='Capital Market Line (CML)', zorder=4)
    
    # Mise en forme
    ax.set_xlabel('Volatilité Annuelle (Risque)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Rendement Annuel Attendu', fontsize=13, fontweight='bold')

    ticker_str = ", ".join(tickers) if len(tickers) < 10 else f"{len(tickers)} actifs"
    ax.set_title(f'Frontière Efficiente - Simulation Monte Carlo\nPortefeuille: {ticker_str}', 
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.01)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------

# Tableau de comparaison des modèles
def display_model_comparison(storage_dict):
    # Conversion du dictionnaire en dataframe
    df = pd.DataFrame(storage_dict).T
    df = df.sort_index()

    df = df[['returns', 'volatility', 'sharpe']]
    df.columns = ['Rendement', 'Volatilité', 'Sharpe']

    print('\n' + '='*80)
    print("TABLEAU COMPARATIF DES MODÈLES")
    print("="*80)

    # On applique un formatage aux données pour gérer l'alignement
    df_formatted = df.copy()
    df_formatted['Rendement'] = df['Rendement'].apply(lambda x: f"{x:.2%}")
    df_formatted['Volatilité'] = df['Volatilité'].apply(lambda x: f"{x:.2%}")
    df_formatted['Sharpe'] = df['Sharpe'].apply(lambda x: f"{x:.2f}")

    # Affichage avec le style 'psql' (style SQL avec des bordures)
    print(tabulate(df_formatted, headers='keys', tablefmt='psql', stralign="center"))
    
    print("="*80 + "\n")


#-------------------------------------------------------------------------------------------------------------------------------

# Graphique de répartition des poids par modèles
def plot_model_allocations(storage_dict, tickers):
    print("Génération du graphique de comparaison des allocations")

    model_names = list(storage_dict.keys())

    # Extraction des données
    weights_matrix = np.array([storage_dict[m]['weights'] for m in model_names])
    weights_df = pd.DataFrame(weights_matrix, columns=tickers, index=model_names)

    # Inverser l'ordre pour que le premier modèle soit en haut du graphique
    weights_df = weights_df.iloc[::-1]
    weights_df = weights_df.sort_index()

    sns.set_theme(style="whitegrid") 
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = sns.color_palette("husl", len(tickers))

    weights_df.plot(kind='barh', stacked=True, ax=ax, color=colors, 
                    edgecolor='white', linewidth=1.5, width=0.8)
    
    for container in ax.containers:
        labels = [f'{v:.1%}' if v > 0.04 else '' for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center', 
                     color='white', fontweight='bold', fontsize=9)
    
    sns.despine(left=True, bottom=True)

    ax.set_title('Répartition du Capital par Stratégie', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Allocation Totale (100%)', fontsize=12, labelpad=15)
    ax.set_ylabel('', fontsize=12)
    
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.tick_params(axis='y', labelsize=11)

    ax.legend(title='Actifs', title_fontsize='11', bbox_to_anchor=(1.02, 1), 
              loc='upper left', borderaxespad=0., frameon=False)
    
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------

# Tableau de comparaison des modèles avec métriques
def display_model_comparison_metrics(storage_dict):
    # Conversion du dictionnaire en dataframe
    df = pd.DataFrame(storage_dict).T
    df = df.sort_index()

    # Sélection des colonnes dans le dictionnaire de stockage
    cols = ['returns', 'volatility', 'sharpe', 'sortino', 'calmar', 'max_drawdown', 'recovery_time_days', 'var_95', 'cvar_95']
    df = df.reindex(columns=cols)

    # Renommage des colonnes
    df.columns = [
        'Rendement', 'Volatilité', 'Sharpe', 'Sortino', 'Calmar', 
        'Max Drawdown', 'Recovery (Jours)', 'VaR (95%)', 'CVaR (95%)'
    ]

    print('\n' + '='*120)
    print("TABLEAU COMPARATIF AVANCÉ DES MODÈLES")
    print("="*120)

    # Application d'un formatage groupé
    df_formatted = df.copy()
    
    # Formatage en pourcentages
    pct_cols = ['Rendement', 'Volatilité', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)']
    for col in pct_cols:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
            
    # Formatage des ratios (2 décimales)
    ratio_cols = ['Sharpe', 'Sortino', 'Calmar']
    for col in ratio_cols:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
    # Formatage des jours (Nombres entiers)
    df_formatted['Recovery (Jours)'] = df_formatted['Recovery (Jours)'].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "N/A")

    # Affichage avec le style 'psql'
    print(tabulate(df_formatted, headers='keys', tablefmt='psql', stralign="center"))
    
    print("="*120 + "\n")

#-------------------------------------------------------------------------------------------------------------------------------

# Graphique d'évolutions des poids dans le portefeuille
def plot_walk_foward_weights(wf_weights_max_sharpe, wf_weights_min_vol):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

    # Max Sharpe
    ax1.stackplot(wf_weights_max_sharpe.index, 
                  *[wf_weights_max_sharpe[col] for col in wf_weights_max_sharpe.columns],
                  labels=wf_weights_max_sharpe.columns,
                  alpha=0.8)
    
    ax1.set_ylabel('Poids du portefeuille', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date de rééquilibrage', fontsize=12, fontweight='bold')
    ax1.set_title('Portefeuille Max Sharpe Ratio', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1)

    # Min Volatility
    ax2.stackplot(wf_weights_min_vol.index, 
                  *[wf_weights_min_vol[col] for col in wf_weights_min_vol.columns],
                  labels=wf_weights_min_vol.columns,
                  alpha=0.8)

    ax2.set_xlabel('Date de rééquilibrage', fontsize=12, fontweight='bold')
    ax2.set_title('Portefeuille Min Volatilité', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 1)

    # Légende
    lines, labels = ax1.get_legend_handles_labels()
    ax2.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, shadow=True)

    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------

# Graphique de comparaison des performances cumulées
def plot_wf_performance(wf_returns_max_sharpe, wf_returns_min_vol):
    plt.figure(figsize=(12, 6))

    # Calcul des indices de richesse (Base 100)
    wealth_max_sharpe = 100 * np.exp(wf_returns_max_sharpe.cumsum())
    wealth_min_vol = 100 * np.exp(wf_returns_min_vol.cumsum())

    # Tracé
    plt.plot(wealth_max_sharpe.index, wealth_max_sharpe, label='WF Max Sharpe', linewidth=2, color='tab:blue')
    plt.plot(wealth_min_vol.index, wealth_min_vol, label='WF Min Volatility', linewidth=2, color='tab:green')

    plt.title('Comparaison de Performance Cumulée (Base 100)', fontsize=14, fontweight='bold')
    plt.ylabel('Valeur du Portefeuille', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout() # Ajouté pour éviter que les labels ne soient coupés
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------

# Tableau de comparaison des métriques relatives
def display_relative_metrics(storage_dict):
    # Conversion du dictionnaire en dataframe
    df = pd.DataFrame(storage_dict).T
    df = df.sort_index()

    # Sélection des colonnes dans le dictionnaire de stockage
    cols = ['alpha', 'beta', 'tracking_error', 'information_ratio', 'win_rate', 'correlation']
    df = df.reindex(columns=cols)

    # Renommage des colonnes pour un affichage propre
    df.columns = [
        'Alpha (Ann.)', 'Beta', 'Tracking Error', 
        'Info Ratio', 'Win Rate', 'Corrélation'
    ]

    print('\n' + '='*100)
    print("TABLEAU DES MÉTRIQUES RELATIVES (VS BENCHMARK)")
    print("="*100)

    # Application d'un formatage groupé
    df_formatted = df.copy()

    # Formatage en pourcentages
    pct_cols = ['Alpha (Ann.)', 'Tracking Error', 'Win Rate']
    for col in pct_cols:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
            
    # Formatage des ratios et multiplicateurs
    ratio_cols = ['Beta', 'Info Ratio', 'Corrélation']
    for col in ratio_cols:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

    # Affichage avec le style 'psql'
    print(tabulate(df_formatted, headers='keys', tablefmt='psql', stralign="center"))
    
    print("="*100 + "\n")

#-------------------------------------------------------------------------------------------------------------------------------

# Graphique de comparaison des rendements des stratégies
def plot_comparative_performance(wf_max_sharpe_returns, wf_min_vol_returns, sp500_returns, nasdaq_returns, ew_returns):
    print("\nGénération du graphique de performance cumulée comparée...")

    # Calcul de l'indice de richesse (Wealth Index) ---
    # On part de 100 et on applique les rendements cumulés
    wealth_max_sharpe = 100 * np.exp(wf_max_sharpe_returns.cumsum())
    wealth_min_vol = 100 * np.exp(wf_min_vol_returns.cumsum())
    wealth_sp500 = 100 * np.exp(sp500_returns.cumsum())
    wealth_nasdaq = 100 * np.exp(nasdaq_returns.cumsum())
    wealth_ew = 100 * np.exp(ew_returns.cumsum())

    # Création du graphique ---
    fig, ax = plt.subplots(figsize=(15, 8))

    # Stratégies 
    ax.plot(wealth_max_sharpe.index, wealth_max_sharpe, 
             label='Stratégie Max Sharpe (WF)', linewidth=2.5, color='#1f77b4') # Bleu
    ax.plot(wealth_min_vol.index, wealth_min_vol, 
             label='Stratégie Min Volatilité (WF)', linewidth=2.5, color='#2ca02c') # Vert

    # Benchmarks
    ax.plot(wealth_nasdaq.index, wealth_nasdaq, 
             label='Nasdaq-100 (QQQ)', linestyle='-', linewidth=1.5, color='#ff7f0e', alpha=0.8) # Orange
    ax.plot(wealth_sp500.index, wealth_sp500, 
             label='S&P 500 (SPY)', linestyle='-', linewidth=1.5, color='#d62728', alpha=0.8) # Rouge
    ax.plot(wealth_ew.index, wealth_ew, 
             label='Equal Weights', linestyle='-', linewidth=1.5, color='gray', alpha=0.7) # Gris
    
    # Mise en forme
    ax.set_title('Performance Cumulée : Stratégies vs Marché (Base 100)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valeur du Portefeuille (Base 100)', fontsize=12, fontweight='bold')
    
    # Ajout d'une zone de couleur pour mieux voir les périodes de baisse sous le capital initial
    ax.fill_between(wealth_max_sharpe.index, wealth_max_sharpe, 100, 
                    where=(wealth_max_sharpe < 100), 
                    color='red', alpha=0.1, label='Zone de perte (Max Sharpe < 100)')

    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, shadow=True)
    ax.grid(True, which='major', linestyle='-', alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    ax.minorticks_on()

    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------

# Graphique de comparaison des drawdown
def plot_comparative_drawdowns(wf_max_sharpe_returns, wf_min_vol_returns, sp500_returns, nasdaq_returns, ew_returns):
    print("\nGénération du graphique comparatif des Drawdowns...")

    data_dict = {
        'WF Max Sharpe': wf_max_sharpe_returns,
        'WF Min Volatility': wf_min_vol_returns,
        'S&P 500 (SPY)': sp500_returns,
        'Nasdaq-100 (QQQ)': nasdaq_returns,
        'Equal Weight': ew_returns
    }
    # .dropna() pour aligner  le début et la fin des séries
    df_returns_aligned = pd.DataFrame(data_dict).dropna()

    # Fonction Interne de Calcul du Drawdown
    def calculate_drawdown_series(returns_series):
        compounded_growth = np.exp(returns_series.cumsum())
        # Calcul du sommet historique à chaque instant
        running_max = compounded_growth.cummax()
        # Calcul du % de perte par rapport au sommet
        drawdown = (compounded_growth - running_max) / running_max
        return drawdown

    # Calcul des drawdowns pour toutes les colonnes
    df_drawdowns = df_returns_aligned.apply(calculate_drawdown_series)

    # Création de la Figure
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24, 5), sharey=True, sharex=True)

    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', 'purple']
    strategies = df_drawdowns.columns

    for i, ax in enumerate(axes):
        strat_name = strategies[i]
        series = df_drawdowns[strat_name]
        color = colors[i]
        
        # Tracé de la ligne
        ax.plot(series.index, series, color=color, linewidth=1.5)
        
        ax.fill_between(series.index, series, 0, color=color, alpha=0.3)
        
        # Ligne de surface (0%)
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        
        ax.set_title(strat_name, fontweight='bold', fontsize=11)
        
        # Grille légère
        ax.grid(True, alpha=0.3)
        
        # Rotation des dates en bas
        ax.tick_params(axis='x', rotation=45)

        # Annotation du Max Drawdown (MDD) sur le graphique
        mdd_value = series.min()
        mid_idx = len(series) // 2  # On place le texte au milieu de l'axe X
        ax.text(series.index[mid_idx], mdd_value - 0.02, 
                f'MDD: {mdd_value:.1%}', 
                ha='center', color='darkred', fontweight='bold')
        
      # Mise en forme globale 
    # Formatage de l'axe Y en pourcentage
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].set_ylabel('Perte depuis le sommet (%)', fontsize=12, fontweight='bold')

    fig.suptitle('Comparaison Visuelle des Drawdowns (Profondeur et Durée des pertes)', 
                 fontsize=16, fontweight='bold', y=1.05)

    plt.tight_layout()
    plt.show()  

#-------------------------------------------------------------------------------------------------------------------------------

def plot_return_distributions(wf_max_sharpe_returns, wf_min_vol_returns, sp500_returns, nasdaq_returns, ew_returns):
    print("\nGénération du graphique de distribution des rendements...")

    # Préparation et Alignement des Données
    data_dict = {
        'WF Max Sharpe': wf_max_sharpe_returns,
        'WF Min Volatility': wf_min_vol_returns,
        'S&P 500 (SPY)': sp500_returns,
        'Nasdaq-100 (QQQ)': nasdaq_returns,
        'Equal Weight': ew_returns
    }
    df_returns_aligned = pd.DataFrame(data_dict).dropna()

    # Création de la Figure
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24, 6), sharey=True, sharex=True)

    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', 'purple']
    strategies = df_returns_aligned.columns

    for i, ax in enumerate(axes):
        name = strategies[i]
        returns = df_returns_aligned[name]
        color = colors[i]
        
        # Histogramme et KDE (Densité)
        sns.histplot(returns, bins=50, kde=True, stat="density", 
                     color=color, alpha=0.4, edgecolor=None, ax=ax)
        
        # Ligne verticale à 0 (pour voir si la cloche est centrée à droite ou gauche)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        
        # Calcul des Statistiques 
        mu = returns.mean() * 252 # Rendement Annuel
        sigma = returns.std() * np.sqrt(252) # Volatilité Annuelle
        sk = skew(returns) # Asymétrie (Négatif = risque de gros crashs)
        ku = kurtosis(returns) # Aplatissement (Élevé = événements extrêmes fréquents "Fat Tails")
        
        # Affichage des Stats sur le Graphique
        stats_text = (
            f"Ann. Return: {mu:.1%}\n"
            f"Ann. Vol: {sigma:.1%}\n"
            f"Skewness: {sk:.2f}\n"
            f"Kurtosis: {ku:.2f}"
        )
        
        # Placement du texte en haut à gauche
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Mise en Forme
        ax.set_title(name, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Rendement Journalier', fontsize=10)
        
        if i == 0:
            ax.set_ylabel('Densité de Fréquence', fontsize=10)
        else:
            ax.set_ylabel('') # On cache le label Y pour les autres graphiques
        
        ax.grid(True, alpha=0.2)

    # Titre Global
    fig.suptitle('Analyse de la Distribution des Rendements (Risque de "Queues Épaisses")', 
                 fontsize=16, fontweight='bold', y=1.05)

    plt.tight_layout()
    plt.show()