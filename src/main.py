from utils import *
import pandas as pd
import tqdm.notebook as tq
import warnings
warnings.filterwarnings("ignore")
import sys
import os

import logging

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

def main(periods=30*6, show = True, save_fig = False):
    data = ExtractData(TICKERS)
    os.makedirs('data',exist_ok=True)
    data.to_parquet('data/tickers_source.parquet')
    print('\nData stored!')
    
    dfs = data.drop(columns='Adj Close', inplace=False).copy()
    dfs = dfs.groupby('ticker')

    for ticker, df in tq.tqdm(dfs):

        print('\nWorking on '+ticker)

        data = TransformaData2Prophet(df)

        m, data_frc = TrainProphetModel(data, ticker=ticker, periods=periods)

        PlotResults(m, data_frc,ticker=ticker, report_folder='data/report', show = show, save_fig=save_fig)

        ModelPerformance(m).to_csv('data/performance/'+ticker+'_model.csv')
        print('\n'+ticker+': DONE!')




    

if __name__ == '__main__':
    main(show = True, save_fig=False)