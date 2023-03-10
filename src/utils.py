import pandas as pd
from pandas_datareader import data as pdr
from datetime import date as dt

import yfinance as yf
yf.pdr_override()

TICKERS = ['VWCE.MI','LCWD.MI','EUNA.F']

PARAMS = {}
PARAMS['EUNA.F']={'changepoint_prior_scale': 0.007, 'seasonality_prior_scale': 10.0, 'interval_width': 0.95}
PARAMS['LCWD.MI']={'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 0.01, 'interval_width': 0.95}
PARAMS['VWCE.MI']={'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 0.01, 'interval_width': 0.99}

def ExtractData(list_of_tickers):
    data = {}
    print('\nDownload the following tickers:')
    for ticker in list_of_tickers:
        print('\n'+ticker)
        data[ticker] = pdr.get_data_yahoo(ticker, start="2017-01-01", end=str(dt.today()))
        data[ticker]['ticker'] = ticker
    df = pd.concat(data[ticker] for ticker in list_of_tickers)
    return df

def LoadStoredData():
    
    dfs = pd.read_parquet('../data/tickers_source.parquet')
    dfs.drop(columns='Adj Close', inplace=True)
    dfs = dfs.groupby('ticker')
    return dfs