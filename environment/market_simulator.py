
"""
QuantGuard-MAS
Indian Equity Market Simulator

Purpose:
- Simulate Indian equity prices in INR.
- Generate reproducible market movements.
- Inject artificial market shocks.
- Maintain historical price data.
- Track normal market movements separately from artificial shocks.
- Provide structured market snapshots for multi-agent decision-making.

NOTE:
This is a research simulator, not a live trading system.
"""

import random
import math
from datetime import datetime, timezone


class MarketSimulator:

    DEFAULT_ASSETS = {
        "RELIANCE.NS": {
            "name": "Reliance Industries",
            "price": 1450.0,
            "daily_drift": 0.0004,
            "daily_volatility": 0.018
        },
        "TCS.NS": {
            "name": "Tata Consultancy Services",
            "price": 3200.0,
            "daily_drift": 0.0003,
            "daily_volatility": 0.014
        },
        "HDFCBANK.NS": {
            "name": "HDFC Bank",
            "price": 1700.0,
            "daily_drift": 0.00035,
            "daily_volatility": 0.016
        },
        "INFY.NS": {
            "name": "Infosys",
            "price": 1500.0,
            "daily_drift": 0.0003,
            "daily_volatility": 0.019
        },
        "ICICIBANK.NS": {
            "name": "ICICI Bank",
            "price": 1100.0,
            "daily_drift": 0.0004,
            "daily_volatility": 0.017
        },
        "ITC.NS": {
            "name": "ITC",
            "price": 430.0,
            "daily_drift": 0.0002,
            "daily_volatility": 0.012
        },
        "SBIN.NS": {
            "name": "State Bank of India",
            "price": 750.0,
            "daily_drift": 0.00035,
            "daily_volatility": 0.021
        },
        "LT.NS": {
            "name": "Larsen & Toubro",
            "price": 3400.0,
            "daily_drift": 0.0004,
            "daily_volatility": 0.018
        }
    }

    MIN_PRICE = 0.01
    MIN_SHOCK = -0.99
    MAX_SHOCK = 5.0

    def __init__(self, seed=42):

        self.seed = seed
        self.rng = random.Random(seed)

        # Initialize asset information.
        self.assets = {
            symbol: details.copy()
            for symbol, details in self.DEFAULT_ASSETS.items()
        }

        # Initialize current prices.
        self.prices = {
            symbol: details["price"]
            for symbol, details in self.assets.items()
        }

        # Historical price tracking.
        self.history = {
            symbol: [price]
            for symbol, price in self.prices.items()
        }

        # Normal market returns.
        self.last_normal_returns = {
            symbol: 0.0
            for symbol in self.assets
        }

        # Artificial shock returns.
        self.last_shock_returns = {
            symbol: 0.0
            for symbol in self.assets
        }

        # Combined daily returns.
        self.last_returns = {
            symbol: 0.0
            for symbol in self.assets
        }

        # Historical return tracking.
        self.return_history = {
            symbol: []
            for symbol in self.assets
        }

        # Historical shock tracking.
        self.shock_history = {
            symbol: []
            for symbol in self.assets
        }

        self.step_count = 0
        self.last_update = None

    # --------------------------------------------------
    # ASSET INFORMATION
    # --------------------------------------------------

    def get_available_assets(self):

        return {
            symbol: {
                "name": details["name"],
                "price": round(self.prices[symbol], 2)
            }
            for symbol, details in self.assets.items()
        }

    # --------------------------------------------------
    # CURRENT MARKET PRICES
    # --------------------------------------------------

    def get_prices(self):

        return {
            symbol: round(price, 2)
            for symbol, price in self.prices.items()
        }

    # --------------------------------------------------
    # HISTORICAL PRICE DATA
    # --------------------------------------------------

    def get_history(self, symbol):

        if symbol not in self.history:
            raise ValueError(f"Unknown asset: {symbol}")

        return self.history[symbol].copy()

    # --------------------------------------------------
    # HISTORICAL RETURNS
    # --------------------------------------------------

    def get_return_history(self, symbol):

        if symbol not in self.return_history:
            raise ValueError(f"Unknown asset: {symbol}")

        return self.return_history[symbol].copy()

    # --------------------------------------------------
    # HISTORICAL SHOCK DATA
    # --------------------------------------------------

    def get_shock_history(self, symbol):

        if symbol not in self.shock_history:
            raise ValueError(f"Unknown asset: {symbol}")

        return self.shock_history[symbol].copy()

    # --------------------------------------------------
    # LAST MARKET RETURNS
    # --------------------------------------------------

    def get_last_returns(self):

        return self.last_returns.copy()

    def get_last_normal_returns(self):

        return self.last_normal_returns.copy()

    def get_last_shock_returns(self):

        return self.last_shock_returns.copy()

    # --------------------------------------------------
    # SHOCK VALIDATION
    # --------------------------------------------------

    def validate_shocks(self, shocks):

        if shocks is None:
            return {}

        if not isinstance(shocks, dict):
            raise ValueError("Shocks must be provided as a dictionary")

        unknown_symbols = set(shocks) - set(self.assets)

        if unknown_symbols:
            raise ValueError(
                f"Unknown shock symbols: {unknown_symbols}"
            )

        validated_shocks = {}

        for symbol, shock in shocks.items():

            if isinstance(shock, bool) or not isinstance(
                shock, (int, float)
            ):
                raise ValueError(
                    f"Shock for {symbol} must be numeric"
                )

            if not math.isfinite(shock):
                raise ValueError(
                    f"Shock for {symbol} must be finite"
                )

            if not self.MIN_SHOCK <= shock <= self.MAX_SHOCK:
                raise ValueError(
                    f"Shock for {symbol} must be between "
                    "-99% and +500%"
                )

            validated_shocks[symbol] = float(shock)

        return validated_shocks

    # --------------------------------------------------
    # MARKET SIMULATION
    # --------------------------------------------------

    def simulate_step(self, shocks=None):

        """
        Advance the market by one simulated trading day.

        Parameters:
            shocks:
                Optional dictionary containing artificial
                percentage changes.

        Example:
            {
                "RELIANCE.NS": -0.12
            }

        Represents a 12% artificial price decline.

        Returns:
            Complete market snapshot.
        """

        shocks = self.validate_shocks(shocks)

        daily_returns = {}
        normal_returns = {}
        shock_returns = {}

        for symbol, details in self.assets.items():

            drift = details["daily_drift"]
            volatility = details["daily_volatility"]

            # Generate normal market movement.
            random_return = self.rng.gauss(
                drift,
                volatility
            )

            # Retrieve artificial shock.
            shock_return = shocks.get(symbol, 0.0)

            # Combine normal market movement and shock.
            combined_return = (
                (1 + random_return) *
                (1 + shock_return)
            ) - 1

            # Calculate new asset price.
            new_price = (
                self.prices[symbol] *
                (1 + combined_return)
            )

            # FIX: Apply minimum price consistently.
            new_price = max(
                self.MIN_PRICE,
                new_price
            )

            # Update current market price.
            self.prices[symbol] = new_price

            # Store the exact same value in price history.
            self.history[symbol].append(new_price)

            # Store individual return components.
            normal_returns[symbol] = random_return
            shock_returns[symbol] = shock_return
            daily_returns[symbol] = combined_return

            # Maintain historical return data.
            self.return_history[symbol].append(
                combined_return
            )

            # Maintain historical shock data.
            self.shock_history[symbol].append(
                shock_return
            )

        # Update latest market information.
        self.last_normal_returns = normal_returns
        self.last_shock_returns = shock_returns
        self.last_returns = daily_returns

        self.step_count += 1

        self.last_update = datetime.now(
            timezone.utc
        ).isoformat()

        return self.get_snapshot()

    # --------------------------------------------------
    # ARTIFICIAL SHOCK INJECTION
    # --------------------------------------------------

    def inject_shock(self, symbol, percentage_change):

        """
        Apply an artificial shock to one asset.

        Example:
            inject_shock("RELIANCE.NS", -0.12)

        The entire market advances by one simulated day.
        Other assets experience normal market movement.
        """

        if symbol not in self.assets:
            raise ValueError(f"Unknown asset: {symbol}")

        return self.simulate_step({
            symbol: percentage_change
        })

    # --------------------------------------------------
    # MARKET SNAPSHOT
    # --------------------------------------------------

    def get_snapshot(self):

        """
        Return complete market information.

        Includes:
        - Current prices
        - Normal market returns
        - Artificial shock returns
        - Combined daily returns
        - Simulation step
        - Timestamp
        """

        return {
            "step": self.step_count,

            "timestamp": self.last_update,

            "currency": "INR",

            "prices": self.get_prices(),

            "normal_returns": {
                symbol: round(value, 6)
                for symbol, value
                in self.last_normal_returns.items()
            },

            "shock_returns": {
                symbol: round(value, 6)
                for symbol, value
                in self.last_shock_returns.items()
            },

            "daily_returns": {
                symbol: round(value, 6)
                for symbol, value
                in self.last_returns.items()
            }
        }


# ======================================================
# MAIN EXECUTION
# ======================================================

if __name__ == "__main__":

    market = MarketSimulator(seed=42)

    print("=" * 55)
    print("       QUANTGUARD-MAS MARKET SIMULATOR")
    print("       Indian Equity Market | INR")
    print("=" * 55)

    # Display initial market.
    print("\nINITIAL MARKET PRICES")
    print("-" * 55)

    print(market.get_prices())

    # Simulate normal market movement.
    print("\nSIMULATING NORMAL MARKET MOVEMENT")
    print("-" * 55)

    normal_snapshot = market.simulate_step()

    print(normal_snapshot)

    # Inject artificial market shock.
    print("\nINJECTING 12% RELIANCE MARKET SHOCK")
    print("-" * 55)

    shock_snapshot = market.inject_shock(
        "RELIANCE.NS",
        -0.12
    )

    print(shock_snapshot)

    # Display normal versus artificial returns.
    print("\nSHOCK ANALYSIS")
    print("-" * 55)

    print(
        "Normal Returns:",
        market.get_last_normal_returns()
    )

    print(
        "Artificial Shock Returns:",
        market.get_last_shock_returns()
    )

    print(
        "Combined Returns:",
        market.get_last_returns()
    )

    # Display historical prices.
    print("\nRELIANCE HISTORICAL PRICES")
    print("-" * 55)

    print(
        market.get_history("RELIANCE.NS")
    )

    print("\nSIMULATION COMPLETED")
    print("=" * 55)
