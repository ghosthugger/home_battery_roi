#!/usr/bin/env python3
"""
Basic test script to verify calculation logic without external dependencies.
"""

import sqlite3
from datetime import datetime, timedelta
import json

def test_api_parsing():
    """Test parsing of mgrey.se API response format."""
    # Sample API response for SE3 on 2024-01-01
    sample_response = {
        "date": "2024-01-01",
        "SE3": [
            {"hour": 0, "price_eur": 2.96, "price_sek": 32.8, "kmeans": 0},
            {"hour": 1, "price_eur": 2.85, "price_sek": 31.58, "kmeans": 0},
            {"hour": 2, "price_eur": 2.67, "price_sek": 29.58, "kmeans": 0},
            {"hour": 3, "price_eur": 2.45, "price_sek": 27.16, "kmeans": 0},
            {"hour": 4, "price_eur": 2.4, "price_sek": 26.64, "kmeans": 0},
            {"hour": 5, "price_eur": 2.12, "price_sek": 23.56, "kmeans": 0},
            {"hour": 6, "price_eur": 2.26, "price_sek": 25.1, "kmeans": 0},
            {"hour": 7, "price_eur": 2.5, "price_sek": 27.79, "kmeans": 0},
            {"hour": 8, "price_eur": 2.62, "price_sek": 29.12, "kmeans": 0},
            {"hour": 9, "price_eur": 3.22, "price_sek": 35.74, "kmeans": 0},
            {"hour": 10, "price_eur": 4.13, "price_sek": 45.87, "kmeans": 1},
            {"hour": 11, "price_eur": 4.35, "price_sek": 48.28, "kmeans": 1},
            {"hour": 12, "price_eur": 4.3, "price_sek": 47.74, "kmeans": 1},
            {"hour": 13, "price_eur": 4.43, "price_sek": 49.15, "kmeans": 1},
            {"hour": 14, "price_eur": 4.62, "price_sek": 51.31, "kmeans": 1},
            {"hour": 15, "price_eur": 5.06, "price_sek": 56.16, "kmeans": 2},
            {"hour": 16, "price_eur": 5.95, "price_sek": 65.99, "kmeans": 3},
            {"hour": 17, "price_eur": 6.5, "price_sek": 72.11, "kmeans": 3},
            {"hour": 18, "price_eur": 6.17, "price_sek": 68.51, "kmeans": 3},
            {"hour": 19, "price_eur": 5.51, "price_sek": 61.11, "kmeans": 2},
            {"hour": 20, "price_eur": 4.8, "price_sek": 53.27, "kmeans": 2},
            {"hour": 21, "price_eur": 4.4, "price_sek": 48.83, "kmeans": 1},
            {"hour": 22, "price_eur": 4.52, "price_sek": 50.16, "kmeans": 1},
            {"hour": 23, "price_eur": 3.8, "price_sek": 42.17, "kmeans": 1}
        ]
    }

    # Test data parsing
    region = "SE3"
    date_str = "2024-01-01"
    all_data = []

    for hour_data in sample_response[region]:
        timestamp = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(hours=hour_data['hour'])
        price_sek_per_kwh = hour_data['price_sek'] / 100.0

        all_data.append({
            'timestamp': timestamp,
            'price_sek_per_kwh': price_sek_per_kwh
        })

    print(f"Parsed {len(all_data)} hours of data")
    print("First few hours:")
    for i in range(5):
        print(f"  {all_data[i]['timestamp']}: {all_data[i]['price_sek_per_kwh']:.2f} SEK/kWh")

    # Test sorting by price
    sorted_by_price = sorted(all_data, key=lambda x: x['price_sek_per_kwh'])
    print("\nCheapest hours:")
    for i in range(5):
        print(f"  {sorted_by_price[i]['timestamp']}: {sorted_by_price[i]['price_sek_per_kwh']:.2f} SEK/kWh")

    print("\nMost expensive hours:")
    for i in range(1, 6):
        print(f"  {sorted_by_price[-i]['timestamp']}: {sorted_by_price[-i]['price_sek_per_kwh']:.2f} SEK/kWh")

    return all_data

def test_optimization_logic():
    """Test the battery optimization logic with sample data."""
    # Get sample data
    prices_data = test_api_parsing()

    # Simple optimization: charge during 5 cheapest hours, discharge during 4 most expensive hours
    # (accounting for 80% efficiency)
    capacity_kwh = 10.0
    efficiency = 0.8

    # Sort by price
    sorted_prices = sorted(prices_data, key=lambda x: x['price_sek_per_kwh'])

    # Take cheapest hours for charging
    charge_hours = sorted_prices[:int(capacity_kwh)]
    avg_charge_price = sum(h['price_sek_per_kwh'] for h in charge_hours) / len(charge_hours)

    # Take most expensive hours for discharging
    discharge_hours = sorted_prices[-int(capacity_kwh * efficiency):]
    avg_discharge_price = sum(h['price_sek_per_kwh'] for h in discharge_hours) / len(discharge_hours)

    # Calculate profit
    energy_stored = len(charge_hours)
    energy_retrieved = len(discharge_hours)

    charge_cost = energy_stored * avg_charge_price
    discharge_revenue = energy_retrieved * avg_discharge_price

    daily_profit = discharge_revenue - charge_cost

    print("\nOptimization Results:")
    print(f"Charge hours: {len(charge_hours)}")
    print(f"Discharge hours: {len(discharge_hours)}")
    print(f"Avg charge price: {avg_charge_price:.2f} SEK/kWh")
    print(f"Avg discharge price: {avg_discharge_price:.2f} SEK/kWh")
    print(f"Energy stored: {energy_stored} kWh")
    print(f"Energy retrieved: {energy_retrieved} kWh")
    print(f"Charge cost: {charge_cost:.2f} SEK")
    print(f"Discharge revenue: {discharge_revenue:.2f} SEK")
    print(f"Daily profit: {daily_profit:.2f} SEK")

    return daily_profit

if __name__ == "__main__":
    print("Testing basic functionality...")
    test_api_parsing()
    test_optimization_logic()
    print("\nBasic tests completed successfully!")
