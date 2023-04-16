import pandas as pd
from pandas_datareader import data as pdr
from datetime import date as dt
import warnings


import yfinance as yf
yf.pdr_override()

TICKERS = ['VWCE.AS','LCWD.PA','EUNA.F']

PARAMS = {}
PARAMS['EUNA.F']={'changepoint_prior_scale': 0.007, 'seasonality_prior_scale': 10.0, 'interval_width': 0.99}
PARAMS['LCWD.PA']={'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 0.01, 'interval_width': 0.99}
PARAMS['VWCE.AS']={'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 0.01, 'interval_width': 0.99}


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
    '''
    Load source file with ETF data and returns a groupedby DataFrame
    '''
    dfs = pd.read_parquet('data/tickers_source.parquet')
    dfs.drop(columns='Adj Close', inplace=True)
    dfs = dfs.groupby('ticker')
    return dfs

def TransformaData2Prophet(df: pd.DataFrame):
    '''Transform the input DataFrame and given in output a DataFrame which can be used for Prophet Forecasting.'''
    df['Span'] = df.Close - df.Open 
    df['Range'] = df.High - df.Low 
    df['Night Span'] = df.Open - df.Close.shift(1)

    df.dropna(inplace=True)
    df.drop(columns=['Open','High','Low','Volume'], inplace=True)

    data = df.copy()
    data['ds'] = data.index
    data['y'] = data['Close']
    data = data[['ds','y']]

    return data

def TrainProphetModel(data: pd.DataFrame, ticker: str, periods: 30*6):
    from prophet import Prophet

    m = Prophet(**PARAMS[ticker]) 

    m.add_country_holidays(country_name='US')
    m.add_country_holidays(country_name='IT')

    m.fit(data)

    future = m.make_future_dataframe(periods=periods)
    future = future[future.ds.dt.day_of_week<5]
    forecast = m.predict(future)

    return m, forecast

def PlotResults(m, forecast,ticker,report_folder, save_fig = False,show = False):
    from prophet import Prophet
    import matplotlib.pyplot as mp
    from prophet.plot import plot_plotly, plot_components_plotly, plot, plot_components
    from prophet.plot import add_changepoints_to_plot
    
    fig1 = plot_plotly(m,forecast, ylabel=ticker)
    # fig3 = m.plot(forecast, ylabel = ticker)
    if show:
        fig1.show()
    if save_fig:
        fig1.write_image(report_folder+'/'+ticker+'.png')
    # a = add_changepoints_to_plot(fig3.gca(), m, forecast)
    # fig2 = plot_components_plotly(m,forecast)
    # if show:
    #    fig2.show()

def ModelPerformance(m):

    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric

    data_cv = cross_validation(m, horizon = '90 days', parallel='processes')
    data_p = performance_metrics(data_cv)
    return data_p