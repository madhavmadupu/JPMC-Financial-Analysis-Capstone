import pandas as pd
import numpy as np
from datetime import date, timedelta
import math

# ----------------------------
# TASK 1: PRICE ESTIMATION MODEL
# ----------------------------

# Load and prepare data
df = pd.read_csv('Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df.sort_values('Dates', inplace=True)
df.reset_index(drop=True, inplace=True)

prices = df['Prices'].values
start_date = date(2020, 10, 31)
end_date = date(2024, 9, 30)

# Generate month-end dates matching the data
months = []
year = start_date.year
month = start_date.month + 1
while True:
    current = date(year, month, 1) - timedelta(days=1)
    months.append(current)
    if current.month == end_date.month and current.year == end_date.year:
        break
    month = ((month + 1) % 12) or 12
    if month == 1:
        year += 1

days_from_start = [(d - start_date).days for d in months]
time = np.array(days_from_start)

# Linear trend
def simple_regression(x, y):
    xbar, ybar = np.mean(x), np.mean(y)
    slope = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar) ** 2)
    intercept = ybar - slope * xbar
    return slope, intercept

slope, intercept = simple_regression(time, prices)

# Seasonal component
sin_prices = prices - (time * slope + intercept)
sin_time = np.sin(time * 2 * np.pi / 365)
cos_time = np.cos(time * 2 * np.pi / 365)

def bilinear_regression(y, x1, x2):
    slope1 = np.sum(y * x1) / np.sum(x1 ** 2)
    slope2 = np.sum(y * x2) / np.sum(x2 ** 2)
    return slope1, slope2

slope1, slope2 = bilinear_regression(sin_prices, sin_time, cos_time)
amplitude = np.sqrt(slope1**2 + slope2**2)
shift = np.arctan2(slope2, slope1)

def estimate_price(date_str):
    """Estimate natural gas price for any date (YYYY-MM-DD)."""
    target = pd.to_datetime(date_str)
    days = (target - pd.Timestamp(start_date)).days
    return amplitude * np.sin(days * 2 * np.pi / 365 + shift) + days * slope + intercept


# ----------------------------
# TASK 2: STORAGE CONTRACT PRICING
# ----------------------------

def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    volume_mmbtu,
    max_storage_capacity_mmbtu,
    injection_withdrawal_cost_per_mmbtu=0.10,
    transport_cost_per_mmbtu=0.50,
    storage_cost_per_month=10000.0
):
    """
    Prices a natural gas storage contract using dynamic inventory tracking.
    
    Args:
        injection_dates (list of str): Dates in 'YYYY-MM-DD' format to inject gas.
        withdrawal_dates (list of str): Dates in 'YYYY-MM-DD' format to withdraw gas.
        volume_mmbtu (float): Volume injected/withdrawn per transaction (assumed equal for simplicity).
        max_storage_capacity_mmbtu (float): Max storage capacity.
        injection_withdrawal_cost_per_mmbtu (float): Cost per MMBtu for inject/withdraw.
        transport_cost_per_mmbtu (float): One-way transport cost per MMBtu.
        storage_cost_per_month (float): Fixed monthly storage fee.

    Returns:
        dict: Contract value and cost breakdown.
    """
    # Convert string dates to datetime
    inj_dates = [pd.to_datetime(d) for d in injection_dates]
    wth_dates = [pd.to_datetime(d) for d in withdrawal_dates]
    all_events = [(d, 'inject') for d in inj_dates] + [(d, 'withdraw') for d in wth_dates]
    all_events.sort(key=lambda x: x[0])  # Sort chronologically

    inventory = 0.0
    total_buy_cost = 0.0
    total_sell_revenue = 0.0
    total_inj_wth_cost = 0.0
    total_transport_cost = 0.0

    for event_date, action in all_events:
        date_str = event_date.strftime('%Y-%m-%d')
        price = estimate_price(date_str)

        if action == 'inject':
            if inventory + volume_mmbtu > max_storage_capacity_mmbtu:
                raise ValueError(f"Cannot inject on {date_str}: exceeds storage capacity.")
            inventory += volume_mmbtu
            total_buy_cost += volume_mmbtu * price
            total_inj_wth_cost += volume_mmbtu * injection_withdrawal_cost_per_mmbtu
            total_transport_cost += volume_mmbtu * transport_cost_per_mmbtu  # one-way to site

        elif action == 'withdraw':
            if inventory < volume_mmbtu:
                raise ValueError(f"Cannot withdraw on {date_str}: insufficient inventory.")
            inventory -= volume_mmbtu
            total_sell_revenue += volume_mmbtu * price
            total_inj_wth_cost += volume_mmbtu * injection_withdrawal_cost_per_mmbtu
            total_transport_cost += volume_mmbtu * transport_cost_per_mmbtu  # one-way from site

    # Compute total storage duration (from first injection to last withdrawal)
    first_inj = min(inj_dates)
    last_wth = max(wth_dates)
    months_held = (last_wth.year - first_inj.year) * 12 + (last_wth.month - first_inj.month)
    if last_wth.day < first_inj.day:
        months_held -= 1
    months_held = max(0, months_held)
    total_storage_cost = months_held * storage_cost_per_month

    gross_profit = total_sell_revenue - total_buy_cost
    total_costs = total_inj_wth_cost + total_transport_cost + total_storage_cost
    net_value = gross_profit - total_costs

    return {
        "net_contract_value_usd": round(net_value, 2),
        "gross_profit_usd": round(gross_profit, 2),
        "total_injection_withdrawal_cost_usd": round(total_inj_wth_cost, 2),
        "total_transport_cost_usd": round(total_transport_cost, 2),
        "total_storage_cost_usd": round(total_storage_cost, 2),
        "inventory_remaining_mmbtu": round(inventory, 2),
        "storage_duration_months": months_held
    }

# ----------------------------
# EXAMPLE USAGE
# ----------------------------

if __name__ == "__main__":
    result = price_storage_contract(
        injection_dates=["2025-04-15", "2025-05-15"],
        withdrawal_dates=["2025-12-15", "2026-01-15"],
        volume_mmbtu=500000,            # 500,000 MMBtu per transaction
        max_storage_capacity_mmbtu=1e6, # 1 million MMBtu max capacity
        injection_withdrawal_cost_per_mmbtu=0.10,
        transport_cost_per_mmbtu=0.50,
        storage_cost_per_month=10000.0
    )
    print("Storage Contract Valuation:\n")
    for k, v in result.items():
        print(f"{k.replace('_', ' ').title()}: {v}")