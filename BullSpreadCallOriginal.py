from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.entities import Asset, TradingFee, Order
from lumibot.credentials import IS_BACKTESTING
from lumibot.backtesting import PolygonDataBacktesting
from datetime import datetime, timedelta

"""
Strategy Description
--------------------
Monthly Bull-Call Spread on PLTR.

How it works (high-level):
1.  Once per *calendar* month the bot looks for the nearest option expiration
    that is **at least 30 days away**.
2.  It BUYS a call ~10 % above the current stock price and SELLS a call ~20 %
    above the price (classic debit-spread / bull-call spread).
3.  It trades **20 contracts per leg** (i.e. 20 complete spreads) in one shot.
4.  Because option chains often have missing “open” prices early in the day (and
    that causes the “`<=` not supported between float and NoneType” error inside
    the back-tester) the bot **waits until after 12 PM New-York time** before it
    will place any orders.  By midday most contracts have traded at least once
    so the required OHLC data is available.
5.  Orders are sent as *market* orders — this eliminates the broker’s internal
    limit-price comparisons that triggered the original error.

This code was refined based on the user prompt: "Polygon rate-limit pauses then
still crashes with '<=' not supported between float and NoneType'.  Please fix
so the back-test can complete without errors."
"""

class BullCallSpreadPLTR(Strategy):

    def initialize(self):
        # We only need to wake up a few times per day.  30-minute cadence keeps
        # things responsive but avoids hammering Polygon unnecessarily.
        self.sleeptime = "12H"

        # Persistent variable – remembers the month of the last trade so we do
        # not open more than one spread per calendar month.
        if not hasattr(self.vars, "last_trade_month"):
            self.vars.last_trade_month = None

    # ---------------------------------------------------------------------
    # Main trading loop – executes every `self.sleeptime` (30 min here).
    # ---------------------------------------------------------------------
    def on_trading_iteration(self):
        dt = self.get_datetime()                        # Exchange-aware timestamp
        today = dt.date()
        current_month = today.month

        # 0) Trade ONLY between 12:00 and 15:30 ET so that most option series
        #    have produced an "open" price in Polygon’s data.
        if dt.hour < 12 or dt.hour >= 15:               # Simple time window
            self.log_message("Waiting until after noon ET to check trades…", color="blue")
            return

        # 1) Skip if we already traded this month --------------------------------
        if self.vars.last_trade_month == current_month:
            self.log_message(f"Spread already placed for month #{current_month} – skipping.", color="blue")
            return

        # 2) Get the underlying stock price --------------------------------------
        underlying = Asset("PLTR", asset_type=Asset.AssetType.STOCK)
        last_price = self.get_last_price(underlying)
        if last_price is None:
            self.log_message("PLTR price unavailable – will try later.", color="red")
            return

        # Draw / update a price line on the chart so you can visually follow PLTR
        self.add_line("PLTR", last_price, color="black", width=2, detail_text="PLTR Price")

        # 3) Download CALL chains -------------------------------------------------
        chains_raw = self.get_chains(underlying)
        if not chains_raw:
            self.log_message("Unable to download option chains for PLTR.", color="red")
            return

        call_chains = chains_raw.get("Chains", {}).get("CALL", {})
        if not call_chains:
            self.log_message("No CALL option chains found for PLTR.", color="red")
            return

        # 4) Find the soonest expiration ≥ 30 days away --------------------------
        min_expiry_date = today + timedelta(days=30)
        valid_expiries = [
            datetime.strptime(e, "%Y-%m-%d").date()
            for e in call_chains.keys()
            if datetime.strptime(e, "%Y-%m-%d").date() >= min_expiry_date
        ]
        if not valid_expiries:
            self.log_message("No PLTR expirations 30+ days out – skipping.", color="red")
            return

        target_expiry = min(valid_expiries)                # soonest eligible expiry
        expiry_str = target_expiry.strftime("%Y-%m-%d")
        strikes = sorted(call_chains.get(expiry_str, []))  # strikes for that expiry
        if not strikes:
            self.log_message(f"No strikes listed for expiry {expiry_str}", color="red")
            return

        # 5) Choose strikes ≈10 % & ≈20 % OTM ------------------------------------
        buy_target  = last_price * 1.10
        sell_target = last_price * 1.20

        buy_strike  = next((s for s in strikes if s >= buy_target), strikes[-1])
        sell_strike = next((s for s in strikes if s >= sell_target), strikes[-1])

        if buy_strike >= sell_strike:
            self.log_message("Calculated buy-strike ≥ sell-strike – cannot form spread.", color="red")
            return

        # 6) Build the two option Asset objects ----------------------------------
        buy_option_asset = Asset(
            "PLTR",
            asset_type=Asset.AssetType.OPTION,
            expiration=target_expiry,
            strike=buy_strike,
            right=Asset.OptionRight.CALL,
            multiplier=100,
        )
        sell_option_asset = Asset(
            "PLTR",
            asset_type=Asset.AssetType.OPTION,
            expiration=target_expiry,
            strike=sell_strike,
            right=Asset.OptionRight.CALL,
            multiplier=100,
        )

        # 7) Safety check – ensure BOTH legs have a *recent* trade so back-tester
        #    has valid OHLC data (get_last_price() returns None if no bar yet).
        if (self.get_last_price(buy_option_asset) is None or
                self.get_last_price(sell_option_asset) is None):
            self.log_message("One or both option legs have no recent trades – skipping for now.", color="yellow")
            return

        # 8) Create MARKET orders for both legs (20 contracts each) ---------------
        qty = 20  # contracts per leg

        buy_order = self.create_order(
            buy_option_asset,
            qty,
            Order.OrderSide.BUY,
            type=Order.OrderType.MARKET,  # market order – avoids limit/open comparisons
        )
        sell_order = self.create_order(
            sell_option_asset,
            qty,
            Order.OrderSide.SELL,
            type=Order.OrderType.MARKET,
        )

        # Submit both legs together so they land in the same timestamp
        submitted_orders = self.submit_orders([buy_order, sell_order])

        if submitted_orders:
            # Mark the trade on the chart so it’s easy to see when the spread was opened
            self.add_marker(
                name="Bull-Call Spread Opened",
                value=last_price,
                color="green",
                symbol="star",
                size=12,
                detail_text=f"Buy {buy_strike} / Sell {sell_strike} exp {expiry_str} (20×)"
            )
            self.vars.last_trade_month = current_month
            self.log_message(
                f"Executed 20× bull-call spread | Buy {buy_strike} / Sell {sell_strike} exp {expiry_str}",
                color="green",
            )
        else:
            self.log_message("Order submission failed – will retry later.", color="red")


# -----------------------------------------------------------------------------
# Back-testing vs. live trading entry-point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if IS_BACKTESTING:
        trading_fee = TradingFee(percent_fee=0.001)  # 0.1 % per side

        BullCallSpreadPLTR.backtest(
            datasource_class=PolygonDataBacktesting,
            benchmark_asset=Asset("SPY", Asset.AssetType.STOCK),
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
            quote_asset=Asset("USD", Asset.AssetType.FOREX),
            budget=1000,  # as requested
        )
    else:
        trader = Trader()
        strategy = BullCallSpreadPLTR(
            quote_asset=Asset("USD", Asset.AssetType.FOREX)
        )
        trader.add_strategy(strategy)
        trader.run_all()