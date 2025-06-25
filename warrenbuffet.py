from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.entities import Asset, TradingFee, Order
from lumibot.backtesting import YahooDataBacktesting
from lumibot.credentials import IS_BACKTESTING
from lumibot.components.perplexity_helper import PerplexityHelper
from lumibot.components.drift_rebalancer_logic import DriftRebalancerLogic, DriftType

from datetime import timedelta
from decimal import Decimal
import os

"""
WarrenBuffettStyleStrategy
-------------------------
A LumiBot strategy that once a week (Monday morning, before markets open) asks
Perplexity AI for stocks that fit a classic "Warren Buffett" value-investing
profile.  It then adjusts the portfolio so that **only** the three most
attractive symbols returned are held, equally weighted, selling anything else.

This code was generated based on the user prompt: 'Ask Perplexity every Monday morning before trading starts for stocks matching Warren Buffett's style. Around 6:35 AM PST on Mondays, adjust the portfolio to hold only the top 3 stocks found (equal weight), selling any others.'

Important notes
---------------
* PerplexityHelper is used to perform the query.  Make sure you have set the
  PERPLEXITY_API_KEY environment variable, otherwise the strategy will warn and
  skip trading for that week.
* DriftRebalancerLogic is leveraged to do the heavy lifting of selling the
  unwanted holdings and buying the new ones to reach the desired equal weights.
* All dates/times are handled with self.get_datetime() which is automatically
  timezone-aware inside LumiBot.
* The strategy only runs a rebalance once per Monday.  The date of the last
  refresh is stored in self.vars.last_refresh_date so that additional calls on
  the same day are ignored.
* The strategy back-tests on daily Yahoo! Finance candles – adequate because we
  only trade once a week and all assets are stocks (no options data required).
"""


class WarrenBuffettStyleStrategy(Strategy):

    def initialize(self):
        """Runs once when the bot starts."""
        # The bot only needs to wake up a few times per day; once an hour is fine
        self.sleeptime = "1H"  # 1-hour iterations during the trading day

        # Instantiate helpers – they make the rest of the code much simpler
        self.perplexity_helper = PerplexityHelper(api_key=os.getenv("PERPLEXITY_API_KEY"))
        self.drift_rebalancer = DriftRebalancerLogic(
            strategy=self,
            drift_type=DriftType.ABSOLUTE,
            drift_threshold=Decimal("0.02"),  # Rebalance if >2 % off target
            order_type=Order.OrderType.MARKET,  # Simpler fills for weekly trades
            acceptable_slippage=Decimal("0.002"),
            fractional_shares=True,
        )

        # Persistent variables (survive restarts)
        if not hasattr(self.vars, "last_refresh_date"):
            self.vars.last_refresh_date = None  # Date of the last Monday query
        if not hasattr(self.vars, "target_weights"):
            self.vars.target_weights = []  # List[dict] for the rebalancer
        if not hasattr(self.vars, "needs_rebalance"):
            self.vars.needs_rebalance = False  # Flag set by Monday query

    # ----------------------------------------------------------------------
    # Lifecycle method: runs shortly before the market session opens.
    # ----------------------------------------------------------------------
    def before_market_opens(self):
        dt = self.get_datetime()  # Safer than datetime.now()

        # We only care about Monday (weekday()==0).  If today is not Monday,
        # simply return.  This event runs every morning, so this is cheap.
        if dt.weekday() != 0:
            return

        # Ensure we haven't already refreshed today.
        if self.vars.last_refresh_date == dt.date():
            return

        # If the API key isn't set, log a warning and skip.
        if self.perplexity_helper.api_key is None:
            self.log_message(
                "PERPLEXITY_API_KEY not found. Skipping weekly stock screen.",
                color="red",
            )
            return

        # ------------------------------------------------------------------
        # 1) Ask Perplexity for Buffett-style stocks.
        # ------------------------------------------------------------------
        query = (
            "Provide a ranked list (tickers only) of U.S. stocks that today best "
            "match Warren Buffett's value-investing principles: durable moat, "
            "consistent earnings, strong ROE, manageable debt, and trading at "
            "a reasonable valuation. Return up to 10 symbols ranked from best "
            "to worst."
        )
        self.log_message("Querying Perplexity for Buffett-style stocks...", color="blue")

        try:
            response = self.perplexity_helper.execute_general_query(query)
            symbols = response.get("symbols", [])
        except Exception as e:
            self.log_message(f"Perplexity query failed: {e}", color="red")
            return

        if not symbols:
            self.log_message("Perplexity returned no symbols. Skipping.", color="red")
            return

        # Take the top 3 symbols (less if fewer available)
        top_symbols = symbols[:3]
        self.log_message(f"Top symbols: {top_symbols}", color="green")

        # ------------------------------------------------------------------
        # 2) Build equal-weight target list for the rebalancer.
        # ------------------------------------------------------------------
        weight = Decimal("1") / Decimal(len(top_symbols))
        target_weights = [
            {"base_asset": Asset(sym), "weight": weight} for sym in top_symbols
        ]

        # Store for use in on_trading_iteration
        self.vars.target_weights = target_weights
        self.vars.needs_rebalance = True
        self.vars.last_refresh_date = dt.date()

    # ----------------------------------------------------------------------
    # Main loop – runs according to self.sleeptime.
    # ----------------------------------------------------------------------
    def on_trading_iteration(self):
        # If we don't need to rebalance, exit early.
        if not self.vars.needs_rebalance:
            return

        # Defensive check: do we have targets?
        if not self.vars.target_weights:
            self.log_message("No target weights available – skipping rebalance.", color="yellow")
            self.vars.needs_rebalance = False
            return

        # ------------------------------------------------------------------
        # 1) Calculate drift vs. desired equal weights.
        # ------------------------------------------------------------------
        drift_df = self.drift_rebalancer.calculate(self.vars.target_weights)
        self.log_message("Calculated portfolio drift:")
        self.log_message(str(drift_df))

        # ------------------------------------------------------------------
        # 2) Rebalance if necessary.
        # ------------------------------------------------------------------
        if self.drift_rebalancer.rebalance(drift_df):
            self.log_message("Portfolio rebalanced to new Buffett basket.", color="green")
        else:
            self.log_message("No trades needed – portfolio already aligned.", color="white")

        # Rebalance completed (or determined unnecessary).  Clear the flag.
        self.vars.needs_rebalance = False


# --------------------------------------------------------------------------
#                Main-block – allows the same file to run BOTH
#                     in backtest mode and in live trading.
# --------------------------------------------------------------------------
if __name__ == "__main__":

    if IS_BACKTESTING:
        # -----------------------
        # Backtesting path
        # -----------------------
        trading_fee = TradingFee(percent_fee=0.001)

        result = WarrenBuffettStyleStrategy.backtest(
            datasource_class=YahooDataBacktesting,
            benchmark_asset=Asset("SPY", Asset.AssetType.STOCK),
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            quote_asset=Asset("USD", Asset.AssetType.FOREX),
            budget=100000,  # Default budget
        )

    else:
        # -----------------------
        # Live trading path
        # -----------------------
        trader = Trader()
        strategy = WarrenBuffettStyleStrategy(
            quote_asset=Asset("USD", Asset.AssetType.FOREX)
        )
        trader.add_strategy(strategy)
        strategies = trader.run_all()