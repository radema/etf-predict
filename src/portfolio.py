import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd
import yfinance as yf
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import sys

# Configure logging
def setup_logging(log_file: str = 'portfolio.log'):
    """Set up logging configuration."""
    logger = logging.getLogger('PortfolioLogger')
    logger.setLevel(logging.DEBUG)
    
    return logger

# Create logger instance
logger = setup_logging()

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
    var_95: float
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
        
        logger.info(f"Initializing Portfolio with data directory: {data_dir}")
        
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.etf_data_dir.mkdir(exist_ok=True)
        
        # Initialize holdings
        self.holdings: Dict[str, ETFHolding] = {}
        self.benchmark_data = None
        self.load_portfolio()
        self._fetch_benchmark_data()

    def _calculate_etf_metrics(self, ticker: str, data: pd.DataFrame) -> Optional[ETFMetrics]:
        """Calculate comprehensive metrics for a single ETF."""
        logger.info(f"Calculating metrics for {ticker}")
        
        try:
            # Check if we have enough data
            if data is None or len(data) < 2:
                logger.warning(f"Insufficient data for {ticker}. Data length: {0 if data is None else len(data)}")
                return None
                
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            logger.debug(f"Calculated returns for {ticker}. Length: {len(returns)}")
            
            # Check if we have enough return data
            if len(returns) < 2:
                logger.warning(f"Insufficient return data for {ticker}")
                return None
                
            # Get benchmark data and calculate benchmark returns
            if self.benchmark_data is None or len(self.benchmark_data) < 2:
                logger.debug("Fetching benchmark data")
                self._fetch_benchmark_data()
                
            if self.benchmark_data is None:
                logger.warning("No benchmark data available")
                return None
                
            benchmark_returns = self.benchmark_data['Close'].pct_change().dropna()
            logging.info(f"Calculated benchmark returns. Length: {len(benchmark_returns)}")
            # Align returns with benchmark
            returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
            logger.debug(f"Aligned returns length: {len(returns)}")
            
            # Check if we have enough aligned data
            if len(returns) < 2:
                logger.warning(f"Insufficient aligned data for {ticker}")
                return None
                
            # Calculate metrics with detailed logging
            logger.debug(f"Calculating detailed metrics for {ticker}")
            
            # Beta and Alpha
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            logger.debug(f"Beta for {ticker}: {beta}")
            
            expected_return = beta * benchmark_returns.mean()
            alpha = returns.mean() - expected_return
            logger.debug(f"Alpha for {ticker}: {alpha}")
            
            # Tracking error and information ratio
            tracking_diff = returns - benchmark_returns
            tracking_error = tracking_diff.std() * np.sqrt(252) if len(tracking_diff) > 0 else 0
            information_ratio = (returns.mean() - benchmark_returns.mean()) * 252 / tracking_error if tracking_error != 0 else 0
            
            # Price changes
            current_price = data['Close'].iloc[-1]
            price_changes = {
                '1D': (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) if len(data) > 1 else 0,
                '1M': (data['Close'].iloc[-1] / data['Close'].iloc[-22] - 1) if len(data) > 22 else 0,
                '3M': (data['Close'].iloc[-1] / data['Close'].iloc[-66] - 1) if len(data) > 66 else 0
            }
            logger.debug(f"Price changes for {ticker}: {price_changes}")
            
            # YTD calculation
            current_year = datetime.now().year
            ytd_data = data[data.index.year == current_year]
            if len(ytd_data) > 0:
                ytd_start = ytd_data.iloc[0]['Close']
                price_change_ytd = (current_price / ytd_start - 1) if ytd_start != 0 else 0
            else:
                price_change_ytd = 0
                
            logger.debug(f"YTD change for {ticker}: {price_change_ytd}")
            
            # Calculate VaR
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
            
            metrics = ETFMetrics(
                annual_return=(returns + 1).prod() ** (252 / len(returns)) - 1 if len(returns) > 0 else 0,
                annual_volatility=returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
                sharpe_ratio=(returns.mean() * 252) / (returns.std() * np.sqrt(252)) if len(returns) > 0 and returns.std() != 0 else 0,
                max_drawdown=self._calculate_drawdown(returns) if len(returns) > 0 else 0,
                beta=beta,
                alpha=alpha * 252,  # Annualized
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                var_95=var_95,
                current_price=current_price,
                price_change_1d=price_changes['1D'],
                price_change_1m=price_changes['1M'],
                price_change_3m=price_changes['3M'],
                price_change_ytd=price_change_ytd
            )
            
            logger.info(f"Successfully calculated metrics for {ticker}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {ticker}: {str(e)}", exc_info=True)
            return None

    def update_data(self) -> None:
        """Update historical data and metrics for all holdings."""
        logger.info("Updating data for all holdings")
        self._fetch_benchmark_data()
        for ticker in self.holdings:
            logger.debug(f"Updating data for {ticker}")
            try:
                self.holdings[ticker].historical_data = self._fetch_etf_data(ticker)
                if self.holdings[ticker].historical_data is not None:
                    self.holdings[ticker].metrics = self._calculate_etf_metrics(
                        ticker,
                        self.holdings[ticker].historical_data
                    )
                self.holdings[ticker].last_updated = datetime.now()
                logger.debug(f"Successfully updated data for {ticker}")
            except Exception as e:
                logger.error(f"Error updating data for {ticker}: {str(e)}", exc_info=True)

    def _fetch_etf_data(self, ticker: str, start_date: datetime = None, end_date: datetime = None) -> Optional[pd.DataFrame]:
        """Fetch and cache ETF data."""
        logger.info(f"Fetching data for {ticker}")
        
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.utcnow()

        file_path = self.etf_data_dir / f"{ticker}.parquet"

        # Check cached data
        if file_path.exists():
            logger.debug(f"Found cached data for {ticker}")
            df = pd.read_parquet(file_path)
            if df.index[-1].date() >= end_date.date() - timedelta(days=1):
                logger.debug(f"Using cached data for {ticker}")
                return df

        try:
            logger.debug(f"Fetching new data for {ticker}")
            etf = yf.Ticker(ticker)
            df = etf.history(start=start_date, end=end_date)
            
            df.index = df.index.tz_convert("UTC").normalize()
            if df.empty:
                logger.warning(f"No data received for {ticker}")
                return None
                
            # Clean data
            df = self._clean_data(df)
            
            # Cache data
            df.to_parquet(file_path)
            logger.info(f"Successfully fetched and cached data for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}", exc_info=True)
            return None

    def calculate_metrics(self) -> Tuple[Optional[dict], Optional[pd.Series]]:
        """Calculate portfolio metrics and returns."""
        logger.info("Calculating portfolio metrics")
        
        if not self.holdings:
            logger.warning("No holdings in portfolio")
            return None, None

        try:
            # Update all historical data
            self.update_data()

            # Calculate returns for each holding
            returns = {}
            for ticker, holding in self.holdings.items():
                if holding.historical_data is not None and len(holding.historical_data) > 1:
                    returns[ticker] = holding.historical_data['Close'].pct_change()
                    logger.debug(f"Calculated returns for {ticker}")

            if not returns:
                logger.warning("No return data available")
                return None, None

            # Combine returns into a DataFrame
            returns_df = pd.DataFrame(returns)
            logger.debug(f"Combined returns shape: {returns_df.shape}")
            
            if returns_df.empty:
                logger.warning("Empty returns DataFrame")
                return None, None

            # Calculate portfolio returns
            portfolio_returns = pd.Series(0, index=returns_df.index)
            for ticker, holding in self.holdings.items():
                if ticker in returns_df.columns:
                    portfolio_returns += returns_df[ticker] * (holding.allocation / 100)
            
            logger.debug("Calculated portfolio returns")

            # Calculate metrics
            metrics = {
                'Annual Return': (portfolio_returns + 1).prod() ** (252 / len(portfolio_returns)) - 1,
                'Annual Volatility': portfolio_returns.std() * (252 ** 0.5),
                'Sharpe Ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * (252 ** 0.5)) 
                    if portfolio_returns.std() != 0 else 0
            }
            
            logger.info("Successfully calculated portfolio metrics")
            return metrics, portfolio_returns
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}", exc_info=True)
            return None, None

    def add_holding(self, ticker: str, allocation: float) -> bool:
        """Add a new ETF holding to the portfolio."""
        logger.info(f"Adding new holding: {ticker} with allocation {allocation}%")
        
        try:
            # Verify ETF exists and data can be fetched
            if self._fetch_etf_data(ticker) is not None:
                self.holdings[ticker] = ETFHolding(
                    ticker=ticker,
                    allocation=allocation,
                    last_updated=datetime.now()
                )
                self.save_portfolio()
                logger.info(f"Successfully added {ticker} to portfolio")
                return True
            else:
                logger.warning(f"Failed to add {ticker} - unable to fetch data")
                return False
        except Exception as e:
            logger.error(f"Error adding holding {ticker}: {str(e)}", exc_info=True)
            return False

    def remove_holding(self, ticker: str) -> None:
        """Remove an ETF holding from the portfolio."""
        logger.info(f"Removing holding: {ticker}")
        try:
            if ticker in self.holdings:
                del self.holdings[ticker]
                self.save_portfolio()
                logger.info(f"Successfully removed {ticker} from portfolio")
            else:
                logger.warning(f"Attempted to remove non-existent holding: {ticker}")
        except Exception as e:
            logger.error(f"Error removing holding {ticker}: {str(e)}", exc_info=True)



    def _fetch_benchmark_data(self) -> None:
        """Fetch benchmark (S&P 500) data."""
        self.benchmark_data = self._fetch_etf_data(self.benchmark_ticker)

    def _calculate_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown with safety checks."""
        try:
            if len(returns) < 2:
                return 0.0
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            return drawdowns.min()
        except Exception:
            return 0.0

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

    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize ETF data."""
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        return df


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

