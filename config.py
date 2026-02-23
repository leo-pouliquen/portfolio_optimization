from datetime import datetime
from dateutil.relativedelta import relativedelta

#Paramètres des actifs
tickers = ['MSFT', 'NVDA', 'GOOGL', 'TSM', 'AMZN', 'V', 'BKNG', 'JNJ', 'CAT', 'WMT']
benchmark_tickers = ['SPY', 'QQQ']

#Paramètres temporels
end_date = datetime.now()
start_date = end_date - relativedelta(years = 5)
start_date_wf = end_date - relativedelta(years = 7)

#Paramètre financiers
risk_free_rate = 0.0367
transaction_cost = 0.001 #coûts de transaction 0,01%

#Contraintes d'optimisation
Min_weights = 0.00
Max_weights = 0.40

#Paramètres analyse walk-forward
train_years=2
test_months=3