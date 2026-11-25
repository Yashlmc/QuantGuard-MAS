"""
QuantGuard-MAS
PEAS Environment

P - Performance Measures
E - Environment
A - Actuators
S - Sensors

Connects the market simulator and portfolio manager.
"""

import math
import statistics

from environment.market_simulator import MarketSimulator
from environment.portfolio import Portfolio


def _compute_max_drawdown(values):
    """Compute peak-to-trough maximum drawdown from a sequence of equity values."""
    if not values:
        return 0.0

    peak = values[0]
    maximum_drawdown = 0.0

    for value in values:
        peak = max(peak, value)
        if peak > 0:
            drawdown = (value / peak) - 1.0
            maximum_drawdown = min(maximum_drawdown, drawdown)

    return abs(maximum_drawdown)


class QuantGuardEnvironment:

    # Minimum number of return observations required
    # before calculating meaningful risk metrics.
    MIN_RETURN_OBSERVATIONS = 30

    def __init__(
        self,
        initial_capital=1_000_000,
        risk_free_rate=0.06,
        seed=42,
    ):
        if not isinstance(initial_capital, (int, float)) or isinstance(initial_capital, bool) or not math.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("Capital must be a positive finite number")

        if not isinstance(risk_free_rate, (int, float)) or isinstance(risk_free_rate, bool) or not math.isfinite(risk_free_rate) or not (0.0 <= risk_free_rate < 1.0):
            raise ValueError("Invalid risk-free rate")

        self.market = MarketSimulator(seed=seed)

        self.portfolio = Portfolio(
            initial_capital=initial_capital
        )

        self.initial_capital = float(initial_capital)
        self.risk_free_rate = float(risk_free_rate)

        self.equity_curve = [float(initial_capital)]
        self.episode_equity_curve = [float(initial_capital)]

        self.shock_events = []
        self.post_shock_equity_curve = None

        self.daily_portfolio_returns = []
        self.performance_history = []

    # =================================================
    # SENSORS
    # =================================================

    def observe(self):
        prices = self.market.get_prices()
        portfolio_summary = self.portfolio.get_summary(prices)
        market_returns = self.market.get_last_returns()

        absolute_returns = [
            abs(value)
            for value in market_returns.values()
        ]

        average_market_movement = (
            statistics.mean(absolute_returns)
            if absolute_returns else 0.0
        )

        if average_market_movement >= 0.025:
            volatility_regime = "HIGH"
        elif average_market_movement >= 0.012:
            volatility_regime = "MODERATE"
        else:
            volatility_regime = "LOW"

        return {
            "environment": {
                "name": "QuantGuard Indian Equity Simulator",
                "currency": "INR",
                "market": "INDIA",
                "data_source": "synthetic",
                "market_status": "simulated",
                "simulation_day": self.market.step_count,
                "active_shock_count": len(self.shock_events),
            },
            "market": {
                "prices": prices,
                "daily_returns": market_returns,
                "volatility_regime": volatility_regime,
            },
            "portfolio": portfolio_summary,
            "performance": self.get_performance(),
        }

    # =================================================
    # ACTUATORS
    # =================================================

    def advance_market(self, shocks=None):
        previous_value = self.portfolio.get_total_value(
            self.market.get_prices()
        )

        snapshot = self.market.simulate_step(shocks=shocks)

        current_value = self.portfolio.get_total_value(
            self.market.get_prices()
        )

        if previous_value > 0:
            portfolio_return = (current_value / previous_value) - 1.0
            self.daily_portfolio_returns.append(portfolio_return)

        shock_occurred = shocks is not None and len(shocks) > 0
        if shock_occurred:
            self.shock_events.append({
                "simulation_day": self.market.step_count,
                "shocks": dict(shocks),
                "post_shock_value": current_value,
            })
            # Start a fresh post-shock trajectory anchored at the shocked value
            self.post_shock_equity_curve = [current_value]

        self._record_equity(record_history=True)

        return {
            "market": snapshot,
            "shock_injected": shock_occurred,
            "observation": self.observe(),
        }

    def apply_allocation(self, target_weights):
        """
        Apply a simulated portfolio allocation.

        This is a simulation actuator, not a live trade.
        """
        result = self.portfolio.rebalance(
            target_weights,
            self.market.get_prices()
        )

        # Record full-episode equity without adding a duplicate history step on rebalance
        self._record_equity(record_history=False)

        rebalance_completed = result.get("rebalance_completed", True)

        return {
            "action": "REBALANCE",
            "status": "SUCCESS" if rebalance_completed else "PARTIAL",
            "portfolio": result,
            "performance": self.get_performance(),
        }

    # =================================================
    # PERFORMANCE MEASURES
    # =================================================

    def _record_equity(self, record_history=True):
        value = float(
            self.portfolio.get_total_value(
                self.market.get_prices()
            )
        )

        self.episode_equity_curve.append(value)

        if record_history:
            self.equity_curve.append(value)

        if self.post_shock_equity_curve is not None:
            # Avoid duplicating the initial anchor point when shock just recorded it
            if not (record_history and self.shock_events and self.shock_events[-1]["simulation_day"] == self.market.step_count and len(self.post_shock_equity_curve) == 1):
                self.post_shock_equity_curve.append(value)

    def get_history_drawdown(self):
        """Drawdown measured from recorded portfolio equity history."""
        return _compute_max_drawdown(self.equity_curve)

    def get_post_shock_drawdown(self):
        """Drawdown measured since the most recent market shock, or None if no shock occurred."""
        if self.post_shock_equity_curve is None:
            return None
        return _compute_max_drawdown(self.post_shock_equity_curve)

    def get_episode_max_drawdown(self):
        """Maximum drawdown across the entire evaluation episode."""
        return _compute_max_drawdown(self.episode_equity_curve)

    def get_max_drawdown(self):
        """Backward-compatible alias for portfolio history maximum drawdown."""
        return self.get_history_drawdown()

    def get_volatility(self):
        if len(self.daily_portfolio_returns) < self.MIN_RETURN_OBSERVATIONS:
            return None

        daily_volatility = statistics.stdev(self.daily_portfolio_returns)
        return daily_volatility * math.sqrt(252)

    def get_sharpe_ratio(self):
        if len(self.daily_portfolio_returns) < self.MIN_RETURN_OBSERVATIONS:
            return None

        daily_risk_free_rate = ((1.0 + self.risk_free_rate) ** (1.0 / 252.0)) - 1.0

        excess_returns = [
            value - daily_risk_free_rate
            for value in self.daily_portfolio_returns
        ]

        daily_volatility = statistics.stdev(self.daily_portfolio_returns)
        if daily_volatility == 0:
            return None

        return (
            statistics.mean(excess_returns) / daily_volatility
        ) * math.sqrt(252)

    def get_total_return(self):
        current_value = self.portfolio.get_total_value(
            self.market.get_prices()
        )
        return (current_value / self.initial_capital) - 1.0

    def get_performance(self):
        total_return = self.get_total_return()
        history_drawdown = self.get_history_drawdown()
        post_shock_drawdown = self.get_post_shock_drawdown()
        episode_max_drawdown = self.get_episode_max_drawdown()

        volatility = self.get_volatility()
        sharpe = self.get_sharpe_ratio()

        observations = len(self.daily_portfolio_returns)
        sufficient_history = observations >= self.MIN_RETURN_OBSERVATIONS

        perf = {
            "initial_capital": round(self.initial_capital, 2),
            "current_portfolio_value": round(
                self.portfolio.get_total_value(self.market.get_prices()), 2
            ),
            "total_return_pct": round(total_return * 100, 4),
            "max_drawdown_pct": round(history_drawdown * 100, 4),
            "history_drawdown_pct": round(history_drawdown * 100, 4),
            "post_shock_drawdown_pct": (
                round(post_shock_drawdown * 100, 4)
                if post_shock_drawdown is not None else None
            ),
            "episode_max_drawdown_pct": round(episode_max_drawdown * 100, 4),
            "shock_count": len(self.shock_events),
            "annualized_volatility_pct": (
                round(volatility * 100, 4)
                if volatility is not None else None
            ),
            "sharpe_ratio": (
                round(sharpe, 4)
                if sharpe is not None else None
            ),
            "risk_metrics_status": (
                "sufficient_history"
                if sufficient_history
                else "insufficient_history"
            ),
            "return_observations": observations,
            "required_observations": self.MIN_RETURN_OBSERVATIONS,
            "simulation_days": self.market.step_count,
            "realized_pnl": round(self.portfolio.total_realized_pnl, 2),
            "transaction_costs": round(self.portfolio.total_transaction_costs, 2),
        }

        self.performance_history.append(perf)
        return perf

    # =================================================
    # PEAS DESCRIPTION
    # =================================================

    @staticmethod
    def get_peas_definition():
        return {
            "performance_measures": [
                "Total portfolio return",
                "History drawdown",
                "Post-shock drawdown",
                "Episode maximum drawdown",
                "Annualized volatility",
                "Sharpe ratio",
                "Realized and unrealized P&L",
                "Transaction costs",
                "Constraint compliance",
            ],
            "environment": [
                "Indian equity market",
                "INR-denominated portfolio",
                "Simulated market prices",
                "Market volatility regimes",
                "Artificial market shocks",
                "Synthetic market data",
            ],
            "actuators": [
                "Advance market simulation",
                "Inject market shock",
                "Apply portfolio allocation",
            ],
            "sensors": [
                "Stock prices",
                "Daily stock returns",
                "Portfolio holdings",
                "Portfolio weights",
                "Unrealized and realized profit/loss",
                "Volatility regime",
            ],
        }


if __name__ == "__main__":
    env = QuantGuardEnvironment(
        initial_capital=1_000_000,
        seed=42
    )

    print("=" * 60)
    print("QUANTGUARD PEAS ENVIRONMENT")
    print("=" * 60)

    print("\nPEAS DEFINITION:")
    print(env.get_peas_definition())

    print("\nINITIAL OBSERVATION:")
    print(env.observe())

    print("\nAPPLYING INITIAL ALLOCATION:")
    allocation = {
        "RELIANCE.NS": 0.30,
        "TCS.NS": 0.25,
        "HDFCBANK.NS": 0.25,
        "CASH": 0.20
    }
    print(env.apply_allocation(allocation))

    print("\nSIMULATING FIVE MARKET DAYS:")
    for day in range(5):
        result = env.advance_market()
        print(
            f"Day {day + 1}:",
            result["observation"]["performance"]
        )

    print("\nINJECTING MARKET SHOCK:")
    shock_result = env.advance_market(
        shocks={
            "RELIANCE.NS": -0.12
        }
    )
    print(shock_result["observation"])

    print("\nFINAL PERFORMANCE:")
    print(env.get_performance())