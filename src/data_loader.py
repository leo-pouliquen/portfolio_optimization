import yfinance as yf
import pandas as pd

def download_portfolio_data(tickers, start_date, end_date):

    print(f"n/ Téléchargements des données")
    df = yf.download(tickers, start= start_date, end= end_date, auto_adjust= True) #Auto adjust = True pour avoir les données ajustée (si il y a eu un split par exemple et les dividendes)
    df_close = df['Close'] #on ne garde que les données de clôture

    return df_close

#-------------------------------------------------------------------------------------------------------------------------------