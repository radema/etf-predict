# portfolio.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd
import yfinance as yf
import json
from datetime import datetime, timedelta
from pathlib import Path

@dataclass
class ETFHolding:
    """Represents a single ETF holding in the portfolio."""
    ticker: str
    allocation: float
    last_updated: datetime = None
    historical_data: Optional[pd.DataFrame] = None

class Portfolio:
    """Manages a portfolio of ETF holdings with associated data and calculations."""
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.portfolio_file = data_dir / "portfolio.json"
        self.etf_data_dir = data_dir / "etf_data"
        
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.etf_data_dir.mkdir(exist_ok=True)
        
        # Initialize holdings
        self.holdings: Dict[str, ETFHolding] = {}
        self.load_portfolio()

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

    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize ETF data."""
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        return df

    def update_data(self) -> None:
        """Update historical data for all holdings."""
        for ticker in self.holdings:
            self.holdings[ticker].historical_data = self._fetch_etf_data(ticker)
            self.holdings[ticker].last_updated = datetime.now()

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

