# portfolio.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd
import yfinance as yf
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np


@dataclass
class ETFMetrics:
    """Stores calculated metrics for an ETF."""
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    var_95: float  # Value at Risk (95% confidence)
    current_price: float
    price_change_1d: float
    price_change_1m: float
    price_change_3m: float
    price_change_ytd: float

@dataclass
class ETFHolding:
    """Represents a single ETF holding in the portfolio."""
    ticker: str
    allocation: float
    last_updated: datetime = None
    historical_data: Optional[pd.DataFrame] = None
    metrics: Optional[ETFMetrics] = None

class Portfolio:
    def __init__(self, data_dir: Path = Path("data"), benchmark_ticker: str = "SPY"):
        self.data_dir = data_dir
        self.portfolio_file = data_dir / "portfolio.json"
        self.etf_data_dir = data_dir / "etf_data"
        self.benchmark_ticker = benchmark_ticker
        
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.etf_data_dir.mkdir(exist_ok=True)
        
        # Initialize holdings
        self.holdings: Dict[str, ETFHolding] = {}
        self.benchmark_data = None
        self.load_portfolio()
        self._fetch_benchmark_data()

    def _calculate_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns.min()

    def _calculate_etf_metrics(self, ticker: str, data: pd.DataFrame) -> ETFMetrics:
        """Calculate comprehensive metrics for a single ETF."""
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        benchmark_returns = self.benchmark_data['Close'].pct_change().dropna()
        
        # Align returns with benchmark
        returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
        
        # Calculate beta and alpha
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance
        
        expected_return = beta * benchmark_returns.mean()
        alpha = returns.mean() - expected_return
        
        # Calculate tracking error and information ratio
        tracking_diff = returns - benchmark_returns
        tracking_error = tracking_diff.std() * np.sqrt(252)
        information_ratio = (returns.mean() - benchmark_returns.mean()) * 252 / tracking_error
        
        # Calculate VaR
        var_95 = np.percentile(returns, 5)
        
        # Calculate price changes
        current_price = data['Close'].iloc[-1]
        price_change_1d = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) if len(data) > 1 else 0
        price_change_1m = (data['Close'].iloc[-1] / data['Close'].iloc[-22] - 1) if len(data) > 22 else 0
        price_change_3m = (data['Close'].iloc[-1] / data['Close'].iloc[-66] - 1) if len(data) > 66 else 0
        
        # YTD calculation
        current_year = datetime.now().year
        ytd_start = data[data.index.year == current_year].iloc[0]['Close']
        price_change_ytd = (current_price / ytd_start - 1)
        
        return ETFMetrics(
            annual_return=(returns + 1).prod() ** (252 / len(returns)) - 1,
            annual_volatility=returns.std() * np.sqrt(252),
            sharpe_ratio=(returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            max_drawdown=self._calculate_drawdown(returns),
            beta=beta,
            alpha=alpha * 252,  # Annualized
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            var_95=var_95,
            current_price=current_price,
            price_change_1d=price_change_1d,
            price_change_1m=price_change_1m,
            price_change_3m=price_change_3m,
            price_change_ytd=price_change_ytd
        )

    def update_data(self) -> None:
        """Update historical data and metrics for all holdings."""
        self._fetch_benchmark_data()
        for ticker in self.holdings:
            self.holdings[ticker].historical_data = self._fetch_etf_data(ticker)
            if self.holdings[ticker].historical_data is not None:
                self.holdings[ticker].metrics = self._calculate_etf_metrics(
                    ticker,
                    self.holdings[ticker].historical_data
                )
            self.holdings[ticker].last_updated = datetime.now()

    def get_etf_metrics_summary(self) -> pd.DataFrame:
        """Get a summary of metrics for all ETFs."""
        metrics_data = []
        for ticker, holding in self.holdings.items():
            if holding.metrics:
                metrics_data.append({
                    'ETF': ticker,
                    'Allocation (%)': holding.allocation,
                    'Current Price': holding.metrics.current_price,
                    '1D Change (%)': holding.metrics.price_change_1d * 100,
                    '1M Change (%)': holding.metrics.price_change_1m * 100,
                    '3M Change (%)': holding.metrics.price_change_3m * 100,
                    'YTD Change (%)': holding.metrics.price_change_ytd * 100,
                    'Annual Return (%)': holding.metrics.annual_return * 100,
                    'Annual Volatility (%)': holding.metrics.annual_volatility * 100,
                    'Sharpe Ratio': holding.metrics.sharpe_ratio,
                    'Max Drawdown (%)': holding.metrics.max_drawdown * 100,
                    'Beta': holding.metrics.beta,
                    'Alpha (%)': holding.metrics.alpha * 100,
                    'Tracking Error (%)': holding.metrics.tracking_error * 100,
                    'Information Ratio': holding.metrics.information_ratio,
                    'VaR 95% (%)': holding.metrics.var_95 * 100
                })
        return pd.DataFrame(metrics_data)

    def load_portfolio(self) -> None:
        """Load portfolio data from JSON file."""
        if self.portfolio_file.exists():
            with open(self.portfolio_file, 'r') as f:
                data = json.load(f)
                self.holdings = {
                    ticker: ETFHolding(
                        ticker=ticker,
                        allocation=float(allocation),
                        last_updated=datetime.now()
                    )
                    for ticker, allocation in data.items()
                }

    def save_portfolio(self) -> None:
        """Save portfolio data to JSON file."""
        data = {
            holding.ticker: holding.allocation
            for holding in self.holdings.values()
        }
        with open(self.portfolio_file, 'w') as f:
            json.dump(data, f)

    def add_holding(self, ticker: str, allocation: float) -> bool:
        """
        Add a new ETF holding to the portfolio.
        Returns True if successful, False otherwise.
        """
        # Verify ETF exists and data can be fetched
        if self._fetch_etf_data(ticker) is not None:
            self.holdings[ticker] = ETFHolding(
                ticker=ticker,
                allocation=allocation,
                last_updated=datetime.now()
            )
            self.save_portfolio()
            return True
        return False

    def remove_holding(self, ticker: str) -> None:
        """Remove an ETF holding from the portfolio."""
        if ticker in self.holdings:
            del self.holdings[ticker]
            self.save_portfolio()

    def _fetch_etf_data(self, ticker: str, start_date: datetime = None, end_date: datetime = None) -> Optional[pd.DataFrame]:
        """Fetch and cache ETF data."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        file_path = self.etf_data_dir / f"{ticker}.parquet"

        # Check cached data
        if file_path.exists():
            df = pd.read_parquet(file_path)
            if df.index[-1].date() >= end_date.date() - timedelta(days=1):
                return df

        try:
            etf = yf.Ticker(ticker)
            df = etf.history(start=start_date, end=end_date)
            
            # Clean data
            df = self._clean_data(df)
            
            # Cache data
            df.to_parquet(file_path)
            return df
            
        except Exception:
            return None

    def _fetch_benchmark_data(self) -> None:
        """Fetch benchmark (S&P 500) data."""
        self.benchmark_data = self._fetch_etf_data(self.benchmark_ticker)

    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize ETF data."""
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        return df

#    def update_data(self) -> None:
#        """Update historical data for all holdings."""
#        for ticker in self.holdings:
#            self.holdings[ticker].historical_data = self._fetch_etf_data(ticker)
#            self.holdings[ticker].last_updated = datetime.now()

    def calculate_metrics(self) -> Tuple[Optional[dict], Optional[pd.Series]]:
        """Calculate portfolio metrics and returns."""
        if not self.holdings:
            return None, None

        # Update all historical data
        self.update_data()

        # Calculate returns for each holding
        returns = {}
        for ticker, holding in self.holdings.items():
            if holding.historical_data is not None:
                returns[ticker] = holding.historical_data['Close'].pct_change()

        if not returns:
            return None, None

        # Combine returns into a DataFrame
        returns_df = pd.DataFrame(returns)

        # Calculate portfolio returns
        portfolio_returns = pd.Series(0, index=returns_df.index)
        for ticker, holding in self.holdings.items():
            if ticker in returns_df.columns:
                portfolio_returns += returns_df[ticker] * (holding.allocation / 100)

        # Calculate metrics
        metrics = {
            'Annual Return': (portfolio_returns + 1).prod() ** (252 / len(portfolio_returns)) - 1,
            'Annual Volatility': portfolio_returns.std() * (252 ** 0.5),
            'Sharpe Ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * (252 ** 0.5))
        }

        return metrics, portfolio_returns

    def get_holdings_summary(self) -> pd.DataFrame:
        """Get a summary of current holdings."""
        return pd.DataFrame([
            {
                'ETF': holding.ticker,
                'Allocation (%)': holding.allocation,
                'Last Updated': holding.last_updated.strftime('%Y-%m-%d %H:%M:%S')
            }
            for holding in self.holdings.values()
        ])

