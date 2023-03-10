from utils import *
import pandas as pd
import tqdm.notebook as tq
import warnings
warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

def main(periods=30*6):
    data = ExtractData(TICKERS)
    data.to_parquet('data/tickers_source.parquet')
    print('\nData stored!')
    
    dfs = data.drop(columns='Adj Close', inplace=False).copy()
    dfs = dfs.groupby('ticker')

    for ticker, df in tq.tqdm(dfs):

        print('\nWorking on '+ticker)

        data = TransformaData2Prophet(df)

        m, data_frc = TrainProphetModel(data, ticker=ticker, periods=periods)

        PlotResults(m, data_frc,ticker=ticker)

        ModelPerformance(m).to_csv('data/performance/'+ticker+'_model.csv')
        print('\n'+ticker+': DONE!')




    

if __name__ == '__main__':
    main()