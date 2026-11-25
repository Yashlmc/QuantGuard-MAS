"""
QuantGuard-MAS
Portfolio State Manager

All values are denominated in INR.
This module simulates portfolio operations only.
"""

import math


def _is_finite_number(value):
    """Return True if value is a real finite int or float (excluding bool)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


class Portfolio:

    DEFAULT_WEIGHT_TOLERANCE = 1e-3  # 0.1% weight tolerance for completion check

    def __init__(
        self,
        initial_capital=1_000_000,
        transaction_cost_rate=0.001,
        weight_tolerance=DEFAULT_WEIGHT_TOLERANCE,
    ):
        if not _is_finite_number(initial_capital) or initial_capital <= 0:
            raise ValueError("Initial capital must be a positive finite number")

        if (
            not _is_finite_number(transaction_cost_rate)
            or not (0.0 <= transaction_cost_rate < 1.0)
        ):
            raise ValueError("Invalid transaction cost rate")

        if not _is_finite_number(weight_tolerance) or weight_tolerance < 0.0:
            raise ValueError("Weight tolerance must be a non-negative finite number")

        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.weight_tolerance = float(weight_tolerance)

        self.holdings = {}
        self.total_transaction_costs = 0.0
        self.total_realized_pnl = 0.0
        self.trade_history = []

    def _validate_price_map(self, prices, symbols=None):
        """Ensure required symbols have finite positive prices."""
        if not isinstance(prices, dict):
            raise ValueError("Prices must be provided as a dictionary")

        target_symbols = symbols if symbols is not None else self.holdings.keys()
        for symbol in target_symbols:
            if symbol not in prices:
                raise ValueError(f"Missing market price for {symbol}")
            price = prices[symbol]
            if not _is_finite_number(price) or price <= 0.0:
                raise ValueError(f"Invalid market price for {symbol}")

    def get_quantity(self, symbol):
        return self.holdings.get(
            symbol,
            {"quantity": 0.0, "average_cost": 0.0},
        )["quantity"]

    def get_market_value(self, prices):
        self._validate_price_map(prices, self.holdings.keys())

        value = 0.0
        for symbol, holding in self.holdings.items():
            value += holding["quantity"] * float(prices[symbol])

        return value

    def get_total_value(self, prices):
        return self.cash + self.get_market_value(prices)

    def get_weights(self, prices):
        total_value = self.get_total_value(prices)

        if total_value <= 0.0 or not math.isfinite(total_value):
            return {}

        weights = {}
        for symbol, holding in self.holdings.items():
            market_value = holding["quantity"] * float(prices[symbol])
            weights[symbol] = market_value / total_value

        weights["CASH"] = self.cash / total_value
        return weights

    def get_unrealized_pnl(self, prices):
        self._validate_price_map(prices, self.holdings.keys())

        pnl = {}
        for symbol, holding in self.holdings.items():
            current_price = float(prices[symbol])
            pnl[symbol] = (
                current_price - holding["average_cost"]
            ) * holding["quantity"]

        return pnl

    def buy(self, symbol, quantity, price):
        if not _is_finite_number(quantity) or quantity <= 0.0:
            raise ValueError("Quantity must be a positive finite number")

        if not _is_finite_number(price) or price <= 0.0:
            raise ValueError("Price must be a positive finite number")

        quantity = float(quantity)
        price = float(price)

        gross_value = quantity * price
        transaction_cost = gross_value * self.transaction_cost_rate
        total_cost = gross_value + transaction_cost

        if not math.isfinite(total_cost):
            raise ValueError("Computed trade cost overflowed")

        if total_cost > self.cash + 1e-12:
            raise ValueError("Insufficient cash for purchase")

        # Clamp tiny floating-point rounding overshoot when buying full available cash
        if total_cost > self.cash:
            total_cost = self.cash

        existing = self.holdings.get(
            symbol,
            {"quantity": 0.0, "average_cost": 0.0},
        )

        old_quantity = existing["quantity"]
        new_quantity = old_quantity + quantity
        new_average_cost = (
            (old_quantity * existing["average_cost"])
            + gross_value
            + transaction_cost
        ) / new_quantity

        self.holdings[symbol] = {
            "quantity": new_quantity,
            "average_cost": new_average_cost,
        }

        self.cash -= total_cost
        if abs(self.cash) < 1e-12:
            self.cash = 0.0

        self._record_trade(
            action="BUY",
            symbol=symbol,
            quantity=quantity,
            price=price,
            transaction_cost=transaction_cost,
            realized_pnl=None,
        )

    def sell(self, symbol, quantity, price):
        if not _is_finite_number(quantity) or quantity <= 0.0:
            raise ValueError("Quantity must be a positive finite number")

        if not _is_finite_number(price) or price <= 0.0:
            raise ValueError("Price must be a positive finite number")

        if symbol not in self.holdings:
            raise ValueError(f"No holdings available for {symbol}")

        quantity = float(quantity)
        price = float(price)
        holding = self.holdings[symbol]

        if quantity > holding["quantity"] + 1e-12:
            raise ValueError("Cannot sell more than available quantity")

        # Clamp tiny floating-point rounding overshoot when selling entire position
        sell_qty = min(quantity, holding["quantity"])
        avg_cost = holding["average_cost"]

        gross_value = sell_qty * price
        transaction_cost = gross_value * self.transaction_cost_rate
        proceeds = gross_value - transaction_cost

        # Net realized P&L: net sale proceeds minus cost basis of shares sold
        realized_pnl = proceeds - (sell_qty * avg_cost)

        holding["quantity"] -= sell_qty
        self.cash += proceeds
        self.total_realized_pnl += realized_pnl

        if holding["quantity"] <= 1e-10:
            del self.holdings[symbol]

        self._record_trade(
            action="SELL",
            symbol=symbol,
            quantity=sell_qty,
            price=price,
            transaction_cost=transaction_cost,
            realized_pnl=realized_pnl,
        )

    def _record_trade(
        self,
        action,
        symbol,
        quantity,
        price,
        transaction_cost,
        realized_pnl=None,
    ):
        self.total_transaction_costs += transaction_cost
        self.trade_history.append({
            "action": action,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "transaction_cost": transaction_cost,
            "realized_pnl": realized_pnl,
        })

    def rebalance(self, target_weights, prices):
        """
        Rebalance the portfolio to target weights.

        Example:
        {
            "RELIANCE.NS": 0.30,
            "TCS.NS": 0.25,
            "HDFCBANK.NS": 0.25,
            "CASH": 0.20
        }

        Weights are fractions, not percentages.
        """
        if not isinstance(target_weights, dict) or not target_weights:
            raise ValueError("Target weights cannot be empty")

        if not isinstance(prices, dict):
            raise ValueError("Prices must be provided as a dictionary")

        cash_weight = float(target_weights.get("CASH", 0.0))
        if not _is_finite_number(cash_weight) or not (0.0 <= cash_weight <= 1.0):
            raise ValueError("Invalid cash weight")

        stock_weights = {}
        for symbol, weight in target_weights.items():
            if symbol == "CASH":
                continue

            if not _is_finite_number(weight) or not (0.0 <= weight <= 1.0):
                raise ValueError(f"Invalid weight for {symbol}")

            if symbol not in prices:
                raise ValueError(f"Missing price for {symbol}")

            if not _is_finite_number(prices[symbol]) or prices[symbol] <= 0.0:
                raise ValueError(f"Invalid price for {symbol}")

            stock_weights[symbol] = float(weight)

        # Also validate prices for any current holdings being evaluated/sold
        self._validate_price_map(prices, self.holdings.keys())

        total_weight = sum(stock_weights.values()) + cash_weight
        if not math.isfinite(total_weight) or abs(total_weight - 1.0) > 1e-6:
            raise ValueError("Target weights must sum to 1.0")

        requested_allocation = dict(stock_weights)
        requested_allocation["CASH"] = cash_weight

        cost_before = self.total_transaction_costs
        initial_value = self.get_total_value(prices)

        # 1. Sell excess holdings first to generate cash
        for symbol in list(self.holdings.keys()):
            current_quantity = self.holdings[symbol]["quantity"]
            price = float(prices[symbol])
            current_value = current_quantity * price
            target_value = initial_value * stock_weights.get(symbol, 0.0)
            excess_value = current_value - target_value

            if excess_value > 1e-9:
                quantity_to_sell = min(current_quantity, excess_value / price)
                if quantity_to_sell > 1e-12:
                    self.sell(symbol, quantity_to_sell, price)

        # 2. Buy assets below target weight within remaining cash & fee budget
        for symbol, weight in stock_weights.items():
            price = float(prices[symbol])
            target_value = initial_value * weight
            current_quantity = self.get_quantity(symbol)
            current_value = current_quantity * price
            deficit = target_value - current_value

            if deficit <= 1e-9:
                continue

            affordable_value = self.cash / (1.0 + self.transaction_cost_rate)
            purchase_value = min(deficit, affordable_value)
            quantity_to_buy = purchase_value / price

            if quantity_to_buy > 1e-12:
                self.buy(symbol, quantity_to_buy, price)

        actual_weights_raw = self.get_weights(prices)
        tracked_symbols = list(
            dict.fromkeys(
                list(requested_allocation.keys()) + list(actual_weights_raw.keys())
            )
        )

        actual_allocation = {
            sym: round(actual_weights_raw.get(sym, 0.0), 6)
            for sym in tracked_symbols
        }
        allocation_deviation = {
            sym: round(
                actual_weights_raw.get(sym, 0.0) - requested_allocation.get(sym, 0.0),
                6,
            )
            for sym in tracked_symbols
        }

        max_deviation = max(
            (abs(val) for val in allocation_deviation.values()),
            default=0.0,
        )
        rebalance_completed = max_deviation <= self.weight_tolerance

        target_cash_amount = initial_value * cash_weight
        uninvested_cash = max(0.0, self.cash - target_cash_amount)
        rebalance_costs = self.total_transaction_costs - cost_before

        return {
            "rebalance_completed": rebalance_completed,
            "requested_allocation": {
                sym: round(w, 6) for sym, w in requested_allocation.items()
            },
            "actual_allocation": actual_allocation,
            "allocation_deviation": allocation_deviation,
            "max_allocation_deviation": round(max_deviation, 6),
            "uninvested_cash": round(uninvested_cash, 2),
            "rebalance_transaction_costs": round(rebalance_costs, 2),
            "total_transaction_costs": round(self.total_transaction_costs, 2),
            "summary": self.get_summary(prices),
        }

    def get_summary(self, prices):
        self._validate_price_map(prices, self.holdings.keys())

        return {
            "initial_capital": round(self.initial_capital, 2),
            "cash": round(self.cash, 2),
            "holdings": {
                symbol: {
                    "quantity": round(holding["quantity"], 6),
                    "average_cost": round(holding["average_cost"], 2),
                    "market_price": round(float(prices[symbol]), 2),
                    "market_value": round(
                        holding["quantity"] * float(prices[symbol]), 2
                    ),
                }
                for symbol, holding in self.holdings.items()
            },
            "market_value": round(self.get_market_value(prices), 2),
            "total_value": round(self.get_total_value(prices), 2),
            "weights": {
                symbol: round(weight * 100, 2)
                for symbol, weight in self.get_weights(prices).items()
            },
            "unrealized_pnl": {
                symbol: round(value, 2)
                for symbol, value in self.get_unrealized_pnl(prices).items()
            },
            "realized_pnl": round(self.total_realized_pnl, 2),
            "transaction_costs": round(self.total_transaction_costs, 2),
        }


if __name__ == "__main__":
    from environment.market_simulator import MarketSimulator

    market = MarketSimulator()
    portfolio = Portfolio(initial_capital=1_000_000)

    print("INITIAL PORTFOLIO")
    print(portfolio.get_summary(market.get_prices()))

    print("\nAPPLYING TEST ALLOCATION")
    allocation = {
        "RELIANCE.NS": 0.30,
        "TCS.NS": 0.25,
        "HDFCBANK.NS": 0.25,
        "CASH": 0.20,
    }

    result = portfolio.rebalance(allocation, market.get_prices())
    print(result)