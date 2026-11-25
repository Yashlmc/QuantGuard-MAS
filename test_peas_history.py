
from environment.peas import QuantGuardEnvironment

env = QuantGuardEnvironment(
    initial_capital=1_000_000,
    seed=42
)

allocation = {
    "RELIANCE.NS": 0.30,
    "TCS.NS": 0.25,
    "HDFCBANK.NS": 0.25,
    "CASH": 0.20
}

env.apply_allocation(allocation)

for day in range(35):
    env.advance_market()

print("30+ DAY PERFORMANCE TEST")
print("=" * 45)

print(env.get_performance())
