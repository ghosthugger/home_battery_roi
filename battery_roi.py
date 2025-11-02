#!/usr/bin/env python3
"""
Battery ROI Calculator for Swedish Home Battery Systems

This tool calculates the return on investment (ROI) for home battery systems in Sweden
using hourly electricity prices from the mgrey.se API. It assumes perfect price
prediction and optimizes charge/discharge schedules for maximum profit.

Assumptions and limitations:
- Perfect price prediction (PERFECT_PREDICT mode only)
- No taxes, network tariffs, or power-peak tariffs
- No battery degradation or charge-rate limits
- Hourly prices only (no 15-minute data)
- No solar panels (battery-only arbitrage)
"""

import argparse
import sqlite3
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import json


def fetch_single_day(region: str, date_str: str) -> pd.DataFrame:
    """
    Fetch hourly electricity prices for a single day from mgrey.se API.

    Args:
        region: Bidding zone (SE1, SE2, SE3, SE4)
        date_str: Date in YYYY-MM-DD format

    Returns:
        DataFrame with columns: timestamp, price_sek_per_kwh
    """
    all_data = []
    url = f"https://mgrey.se/espot?format=json&date={date_str}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Parse the response for the specific region
        if region in data:
            for hour_data in data[region]:
                # Create timestamp from date and hour
                timestamp = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(hours=hour_data['hour'])
                # Convert from öre/kWh to SEK/kWh
                price_sek_per_kwh = hour_data['price_sek'] / 100.0

                all_data.append({
                    'timestamp': timestamp,
                    'price_sek_per_kwh': price_sek_per_kwh
                })

    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Data parsing failed: {e}")

    df = pd.DataFrame(all_data)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


def fetch_prices(region: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly electricity prices from mgrey.se API.

    Args:
        region: Bidding zone (SE1, SE2, SE3, SE4)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with columns: timestamp, price_sek_per_kwh
    """
    all_data = []

    # Convert dates to datetime objects
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Fetch data day by day (API returns one day at a time)
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        url = f"https://mgrey.se/espot?format=json&date={date_str}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Parse the response for the specific region
            if region in data:
                for hour_data in data[region]:
                    # Create timestamp from date and hour
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(hours=hour_data['hour'])
                    # Convert from öre/kWh to SEK/kWh
                    price_sek_per_kwh = hour_data['price_sek'] / 100.0

                    all_data.append({
                        'timestamp': timestamp,
                        'price_sek_per_kwh': price_sek_per_kwh
                    })

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {date_str}: {e}")
            continue
        except (KeyError, ValueError) as e:
            print(f"Error parsing data for {date_str}: {e}")
            continue

        current += timedelta(days=1)

    df = pd.DataFrame(all_data)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def cache_prices_to_sqlite(df: pd.DataFrame, region: str):
    """
    Cache price data to SQLite database.

    Args:
        df: DataFrame with price data
        region: Region identifier for table naming
    """
    if df.empty:
        return

    conn = sqlite3.connect('battery_roi_cache.db')

    # Create table if it doesn't exist, append new data
    table_name = f"prices_{region.lower()}"
    df.to_sql(table_name, conn, if_exists='append', index=False)

    # Create index on timestamp for faster queries
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name}(timestamp)")

    conn.close()


def load_prices_from_cache(region: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load cached price data from SQLite database.

    Args:
        region: Bidding zone
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with cached price data, or empty DataFrame if no cache
    """
    conn = sqlite3.connect('battery_roi_cache.db')
    table_name = f"prices_{region.lower()}"

    try:
        query = f"""
        SELECT timestamp, price_sek_per_kwh
        FROM {table_name}
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn, params=[start_date + ' 00:00:00', end_date + ' 23:59:59'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def optimize_battery_schedule(prices_df: pd.DataFrame, capacity_kwh: float, efficiency: float) -> pd.DataFrame:
    """
    Optimize battery charge/discharge schedule using perfect price prediction.

    For each day, charge during the cheapest hours and discharge during the most
    expensive hours, respecting battery capacity and round-trip efficiency.

    Args:
        prices_df: DataFrame with hourly prices
        capacity_kwh: Battery capacity in kWh
        efficiency: Round-trip efficiency (0-1)

    Returns:
        DataFrame with daily optimization results
    """
    if prices_df.empty:
        return pd.DataFrame()

    results = []

    # Group by date for daily optimization
    prices_df = prices_df.copy()
    prices_df['date'] = prices_df['timestamp'].dt.date
    grouped = prices_df.groupby('date')

    for date, day_data in grouped:
        if len(day_data) != 24:
            # Skip incomplete days
            continue

        # Sort hours by price for this day
        sorted_hours = day_data.sort_values('price_sek_per_kwh').copy()

        # For perfect prediction, we want to:
        # 1. Charge during the lowest price hours
        # 2. Discharge during the highest price hours
        # 3. Assume we can fully cycle the battery daily

        # Take the cheapest hours for charging (up to capacity)
        charge_hours = sorted_hours.head(min(int(capacity_kwh), len(sorted_hours)))

        # Take the most expensive hours for discharging (accounting for efficiency)
        discharge_hours = sorted_hours.tail(min(int(capacity_kwh * efficiency), len(sorted_hours))).sort_values('price_sek_per_kwh', ascending=False)

        # Calculate metrics
        energy_charged = len(charge_hours)  # 1 kWh per hour
        energy_discharged = len(discharge_hours)  # 1 kWh per hour

        avg_charge_price = charge_hours['price_sek_per_kwh'].mean() if not charge_hours.empty else 0
        avg_discharge_price = discharge_hours['price_sek_per_kwh'].mean() if not discharge_hours.empty else 0

        charge_cost = energy_charged * avg_charge_price
        discharge_revenue = energy_discharged * avg_discharge_price

        # Apply efficiency loss to the revenue calculation
        effective_revenue = energy_discharged * avg_discharge_price
        effective_cost = energy_charged * avg_charge_price

        daily_profit = effective_revenue - effective_cost

        results.append({
            'date': date,
            'charge_hours': len(charge_hours),
            'discharge_hours': len(discharge_hours),
            'avg_charge_price': avg_charge_price,
            'avg_discharge_price': avg_discharge_price,
            'energy_stored_kwh': energy_charged,
            'energy_retrieved_kwh': energy_discharged,
            'charge_cost_sek': charge_cost,
            'discharge_revenue_sek': discharge_revenue,
            'daily_profit_sek': daily_profit
        })

    return pd.DataFrame(results)


def calculate_roi(schedule_df: pd.DataFrame, prices_df: pd.DataFrame, battery_cost_sek: float, capacity_kwh: float, efficiency: float) -> dict:
    """
    Calculate ROI metrics from the optimization schedule.

    Args:
        schedule_df: DataFrame from optimize_battery_schedule
        prices_df: Original price data for sensitivity analysis
        battery_cost_sek: Total battery system cost
        capacity_kwh: Battery capacity
        efficiency: Round-trip efficiency

    Returns:
        Dictionary with ROI calculations
    """
    if schedule_df.empty:
        return {
            'total_profit_sek': 0,
            'annual_profit_sek': 0,
            'payback_years': float('inf'),
            'sensitivity': {}
        }

    total_profit = schedule_df['daily_profit_sek'].sum()
    days = len(schedule_df)
    annual_profit = total_profit * (365 / days) if days > 0 else 0
    payback_years = battery_cost_sek / annual_profit if annual_profit > 0 else float('inf')

    # Sensitivity analysis
    sensitivity = {}

    # Efficiency variation ±10%
    for eff_change in [-0.1, 0.1]:
        new_eff = max(0.1, min(1.0, efficiency + eff_change))
        # Recalculate with new efficiency - scale discharge energy
        schedule_df_sens = schedule_df.copy()
        schedule_df_sens['energy_retrieved_kwh'] = schedule_df_sens['energy_stored_kwh'] * (new_eff / efficiency)
        schedule_df_sens['discharge_revenue_sek'] = schedule_df_sens['energy_retrieved_kwh'] * schedule_df_sens['avg_discharge_price']
        schedule_df_sens['daily_profit_sek'] = schedule_df_sens['discharge_revenue_sek'] - schedule_df_sens['charge_cost_sek']

        profit_sens = schedule_df_sens['daily_profit_sek'].sum()
        sensitivity[f'efficiency_{new_eff:.2f}'] = profit_sens

    # Price spread variation ±10% (multiply all prices by factor)
    for price_change in [-0.1, 0.1]:
        price_factor = 1.0 + price_change
        # Create modified price data
        prices_sens = prices_df.copy()
        prices_sens['price_sek_per_kwh'] = prices_sens['price_sek_per_kwh'] * price_factor

        # Re-run optimization with modified prices
        schedule_sens = optimize_battery_schedule(prices_sens, capacity_kwh, efficiency)
        profit_sens = schedule_sens['daily_profit_sek'].sum() if not schedule_sens.empty else 0
        sensitivity[f'price_spread_{price_factor:.2f}'] = profit_sens

    return {
        'total_profit_sek': total_profit,
        'annual_profit_sek': annual_profit,
        'payback_years': payback_years,
        'sensitivity': sensitivity
    }


def plot_results(prices_df: pd.DataFrame, schedule_df: pd.DataFrame, region: str, efficiency: float):
    """
    Generate plots showing battery optimization results.

    Args:
        prices_df: Original price data
        schedule_df: Optimization schedule
        region: Region for plot titles
    """
    if prices_df.empty or schedule_df.empty:
        return

    # Create plots directory
    plots_dir = Path('./plots')
    plots_dir.mkdir(exist_ok=True)

    # 1. Daily charge/discharge schedule for a representative day
    # Find a day with typical activity
    active_days = schedule_df[schedule_df['daily_profit_sek'] > 0]
    if not active_days.empty:
        representative_date = active_days.iloc[len(active_days)//2]['date']

        # Get hourly prices for that day
        day_prices = prices_df[prices_df['timestamp'].dt.date == representative_date].copy()
        day_prices = day_prices.sort_values('timestamp')

        plt.figure(figsize=(12, 6))
        plt.plot(day_prices['timestamp'], day_prices['price_sek_per_kwh'], 'b-', linewidth=2)
        plt.title(f'Hourly Electricity Prices - {representative_date} ({region})')
        plt.xlabel('Time')
        plt.ylabel('Price (SEK/kWh)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / f'price_profile_{representative_date}_{region}.png', dpi=150)
        plt.close()

    # 2. Cumulative revenue vs cost over time
    schedule_df = schedule_df.sort_values('date')
    schedule_df['cumulative_profit'] = schedule_df['daily_profit_sek'].cumsum()
    schedule_df['cumulative_cost'] = schedule_df['charge_cost_sek'].cumsum()
    schedule_df['cumulative_revenue'] = schedule_df['discharge_revenue_sek'].cumsum()

    plt.figure(figsize=(14, 8))
    plt.plot(schedule_df['date'], schedule_df['cumulative_revenue'], 'g-', label='Cumulative Revenue', linewidth=2)
    plt.plot(schedule_df['date'], schedule_df['cumulative_cost'], 'r-', label='Cumulative Cost', linewidth=2)
    plt.plot(schedule_df['date'], schedule_df['cumulative_profit'], 'b-', label='Cumulative Profit', linewidth=2)
    plt.title(f'Cumulative Revenue vs Cost ({region})', fontsize=14, pad=20)
    plt.xlabel('Date')
    plt.ylabel('SEK')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Add explanatory text box
    explanation_text = f"""Home battery Economics:

Revenue (Green): Savings from using stored electricity during high-price hours
• Battery discharges when spot prices are highest
• Accounts for round-trip efficiency losses ({efficiency:.1%})

Cost (Red): Expense of buying electricity during low-price hours
• Battery charges when spot prices are lowest
• Raw charging costs before efficiency losses

Profit (Blue): Net earnings from home battery system
• Revenue - Cost = Daily profit
• Cumulative profit shows total savings over time

Strategy: Buy low, use at high using time-shifted consumption"""

    plt.text(0.02, 0.98, explanation_text,
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment='top',
             horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
             family='monospace')

    plt.tight_layout()
    plt.savefig(plots_dir / f'cumulative_roi_{region}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Battery utilization histogram
    plt.figure(figsize=(10, 6))
    plt.hist(schedule_df['energy_stored_kwh'], bins=20, alpha=0.7, edgecolor='black')
    plt.title(f'Daily Battery Utilization Histogram ({region})')
    plt.xlabel('Energy Stored (kWh)')
    plt.ylabel('Number of Days')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f'utilization_histogram_{region}.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Calculate ROI for Swedish home battery systems')
    parser.add_argument('--region', required=True, choices=['SE1', 'SE2', 'SE3', 'SE4'],
                       help='Electricity bidding zone')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capacity', type=float, required=True, help='Battery capacity (kWh)')
    parser.add_argument('--efficiency', type=float, default=0.9, help='Round-trip efficiency (default: 0.9)')
    parser.add_argument('--cost', type=float, required=True, help='Battery system cost (SEK)')

    args = parser.parse_args()

    # Validate inputs
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        print("Error: Invalid date format. Use YYYY-MM-DD")
        return

    if not (0 < args.efficiency <= 1):
        print("Error: Efficiency must be between 0 and 1")
        return

    if args.capacity <= 0:
        print("Error: Capacity must be positive")
        return

    if args.cost <= 0:
        print("Error: Cost must be positive")
        return

    print(f"Fetching/caching price data for {args.region} from {args.start} to {args.end}...")

    # Smart caching: Load existing data and fetch only missing dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")

    # Load any existing cached data
    prices_df = load_prices_from_cache(args.region, args.start, args.end)

    # Generate list of all expected dates
    expected_dates = []
    current_date = start_date
    while current_date <= end_date:
        expected_dates.append(current_date.date())
        current_date += timedelta(days=1)

    # Check which dates we have data for
    if not prices_df.empty:
        cached_dates = set(prices_df['timestamp'].dt.date.unique())
        missing_dates = [d for d in expected_dates if d not in cached_dates]
    else:
        missing_dates = expected_dates

    # Fetch missing data
    if missing_dates:
        if not prices_df.empty:
            print(f"Found {len(cached_dates)} days in cache. Fetching {len(missing_dates)} missing days...")
        else:
            print(f"No cached data found. Fetching {len(missing_dates)} days from API...")

        # Fetch missing dates in batches or individually
        new_data_frames = []
        for missing_date in missing_dates:
            date_str = missing_date.strftime("%Y-%m-%d")
            try:
                day_df = fetch_single_day(args.region, date_str)
                if not day_df.empty:
                    new_data_frames.append(day_df)
            except Exception as e:
                print(f"Failed to fetch data for {date_str}: {e}")
                continue

        # Combine and cache new data
        if new_data_frames:
            new_prices_df = pd.concat(new_data_frames, ignore_index=True)
            cache_prices_to_sqlite(new_prices_df, args.region)
            print(f"Cached {len(new_prices_df)} additional price records")

            # Reload complete dataset
            prices_df = load_prices_from_cache(args.region, args.start, args.end)

    if prices_df.empty:
        print("Failed to load or fetch any price data")
        return

    print(f"Using {len(prices_df)} price records from cache")

    # Calculate total days in range and days with data
    total_days_in_range = len(expected_dates)

    # Count days with data
    if not prices_df.empty:
        days_with_data = len(prices_df['timestamp'].dt.date.unique())
        print(f"Date range: {total_days_in_range} days total")
        print(f"Days with price data: {days_with_data}")
    else:
        days_with_data = 0
        print(f"Date range: {total_days_in_range} days total")
        print("Days with price data: 0")

    print("Optimizing battery schedule...")
    schedule_df = optimize_battery_schedule(prices_df, args.capacity, args.efficiency)

    if not schedule_df.empty:
        days_processed = len(schedule_df)
        print(f"Days processed (complete 24h data): {days_processed}")
        if days_with_data > 0:
            skip_percentage = ((days_with_data - days_processed) / days_with_data) * 100
            print(f"Days skipped due to incomplete data: {skip_percentage:.1f}%")
    else:
        print("Days processed (complete 24h data): 0")

    print("Calculating ROI...")
    roi_results = calculate_roi(schedule_df, prices_df, args.cost, args.capacity, args.efficiency)

    print("Generating plots...")
    plot_results(prices_df, schedule_df, args.region, args.efficiency)

    # Print results
    print("\n--- Battery ROI Summary ---")
    print(f"Region: {args.region}")
    print(f"Period: {args.start} → {args.end}")
    print(f"Battery Capacity: {args.capacity} kWh")
    print(f"Efficiency: {args.efficiency:.2f}")
    print(f"Battery Cost: {args.cost:,.0f} SEK")
    print()
    print(f"Total Profit: {roi_results['total_profit_sek']:,.0f} SEK")
    print(f"Annual Profit: {roi_results['annual_profit_sek']:,.0f} SEK/year")
    print(f"Pay-back Time: {roi_results['payback_years']:.1f} years")
    print()
    print("Sensitivity Analysis:")
    for key, profit in roi_results['sensitivity'].items():
        if 'efficiency' in key:
            eff_val = float(key.split('_')[1])
            print(f"  Efficiency {eff_val:.2f} → Profit {profit:,.0f} SEK")
        elif 'price_spread' in key:
            spread_val = float(key.split('_')[2])
            price_pct = (spread_val - 1.0) * 100
            print(f"  Price spread {price_pct:+.0f}% → Profit {profit:,.0f} SEK")
    print()
    print("Charts saved to ./plots/")


if __name__ == "__main__":
    main()
