# Import necessary libraries and modules
from lumibot.brokers import Alpaca  # For connecting to Alpaca trading API
from lumibot.backtesting import YahooDataBacktesting  # For backtesting strategies
from lumibot.strategies.strategy import Strategy  # Base class for trading strategies
from lumibot.traders import Trader  # For executing trades
from datetime import datetime  # For handling dates and times
from alpaca_trade_api import REST  # Alpaca's REST API client
from timedelta import Timedelta  # For time calculations
from finbert_utils import estimate_sentiment  # For sentiment analysis
import os  # Import os to load environment variables

# Load credentials from .env file
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = os.getenv('ALPACA_BASE_URL')

# Store API credentials in a dictionary
ALPACA_CREDS = {
    "API_KEY": API_KEY,  # Alpaca API key
    "API_SECRET": API_SECRET,  # Alpaca API secret
    "PAPER": True  # Use paper trading mode (simulated trading)
}

class MLTrader(Strategy):
    """
    Main trading strategy class that uses sentiment analysis to make trading decisions
    
    Parameters:
    - symbol: str - The stock symbol to trade (default: SPY - S&P 500 ETF)
    - cash_at_risk: float - Percentage of cash to risk per trade (default: 0.5 = 50%)
    """
    def initialize(self, symbol:str="SPY", cash_at_risk:float=.5):
        """
        Initialize the trading strategy
        
        Sets up initial parameters and connects to Alpaca API
        """
        self.symbol = symbol  # Stock symbol to trade
        self.sleeptime = "24H"  # Time between trading iterations
        self.last_trade = None  # Track the last trade type
        self.cash_at_risk = cash_at_risk  # Percentage of cash to risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)  # Initialize Alpaca API client

    def position_sizing(self):
        """
        Calculate the position size based on available cash and current price
        
        Returns:
        - cash: float - Available cash
        - last_price: float - Current price of the stock
        - quantity: int - Number of shares to trade
        """
        cash = self.get_cash()  # Get available cash
        last_price = self.get_last_price(self.symbol)  # Get current stock price
        # Calculate number of shares to trade based on risk percentage
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        """
        Get current date and date 3 days prior
        
        Used for fetching recent news articles
        """
        today = self.get_datetime()  # Get current datetime
        three_days_prior = today - Timedelta(days=3)  # Get date 3 days ago
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        """
        Analyze market sentiment based on recent news
        
        Returns:
        - probability: float - Confidence score of the sentiment analysis
        - sentiment: str - Either "positive", "negative", or "neutral"
        """
        today, three_days_prior = self.get_dates()  # Get date range
        # Fetch news articles for the stock
        news = self.api.get_news(symbol=self.symbol, 
                                start=three_days_prior, 
                                end=today)
        # Extract headlines from news articles
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        # Analyze sentiment of the news articles
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def on_trading_iteration(self):
        """
        Main trading logic executed on each iteration
        """
        cash, last_price, quantity = self.position_sizing()  # Calculate position size
        probability, sentiment = self.get_sentiment()  # Analyze market sentiment

        # Only trade if we have enough cash
        if cash > last_price:
            # If sentiment is strongly positive (confidence > 99.9%)
            if sentiment == "positive" and probability > .999:
                # Close any existing sell positions
                if self.last_trade == "sell":
                    self.sell_all()
                # Create a buy order with take-profit and stop-loss
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price*1.20,  # Take profit at 20% gain
                    stop_loss_price=last_price*.95  # Stop loss at 5% loss
                )
                self.submit_order(order)  # Execute the order
                self.last_trade = "buy"  # Update last trade type

            # If sentiment is strongly negative (confidence > 99.9%)
            elif sentiment == "negative" and probability > .999:
                # Close any existing buy positions
                if self.last_trade == "buy":
                    self.sell_all()
                # Create a sell order with take-profit and stop-loss
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price*.8,  # Take profit at 20% loss
                    stop_loss_price=last_price*1.05  # Stop loss at 5% gain
                )
                self.submit_order(order)  # Execute the order
                self.last_trade = "sell"  # Update last trade type

# Set up backtesting parameters
start_date = datetime(2020,1,1)  # Start date for backtesting
end_date = datetime(2023,12,31)  # End date for backtesting
broker = Alpaca(ALPACA_CREDS)  # Initialize Alpaca broker

# Create and configure the trading strategy
strategy = MLTrader(name='mlstrat', broker=broker,
                    parameters={"symbol":"SPY",  # Trade SPY ETF
                                "cash_at_risk":.5})  # Risk 50% of cash per trade

# Run backtest using Yahoo historical data
strategy.backtest(
    YahooDataBacktesting,  # Use Yahoo data for backtesting
    start_date,  # Start date
    end_date,  # End date
    parameters={"symbol":"SPY", "cash_at_risk":.5}  # Strategy parameters
)

# Uncomment to run live trading
# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
