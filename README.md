# Battery ROI Calculator for Swedish Home Battery Systems

This tool calculates the return on investment (ROI) for home battery systems in Sweden using hourly electricity prices from the mgrey.se API.

## Features

- Fetches hourly electricity prices from the mgrey.se API for Swedish bidding zones (SE1-SE4)
- Caches data locally in SQLite database for faster subsequent runs
- Perfect prediction optimization for charge/discharge scheduling
- ROI calculations with sensitivity analysis
- Visual plots showing price profiles, cumulative profits, and battery utilization

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python battery_roi.py \
    --region SE3 \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --capacity 10 \
    --efficiency 0.9 \
    --cost 80000
```

### Command Line Arguments

- `--region`: Electricity bidding zone (SE1, SE2, SE3, SE4)
- `--start`: Start date in YYYY-MM-DD format
- `--end`: End date in YYYY-MM-DD format
- `--capacity`: Battery capacity in kWh
- `--efficiency`: Round-trip efficiency (0.0-1.0, default: 0.9)
- `--cost`: Battery system cost in SEK

## Assumptions and Limitations

- **PERFECT_PREDICT mode only**: Algorithm knows all future prices
- No taxes, network tariffs, or power-peak tariffs
- No battery degradation or charge-rate limits
- Hourly prices only (no 15-minute data)
- Daily cycling assumption (battery fully cycles each day)

## Output

The tool generates:
- Text summary with total profit, annual profit, and payback time
- Sensitivity analysis for efficiency (±10%) and price spreads (±10%)
- Three plots saved to `./plots/` directory:
  - Daily price profile for a representative day
  - Cumulative revenue vs cost over time
  - Battery utilization histogram

## Example Output

```
--- Battery ROI Summary ---
Region: SE3
Period: 2024-01-01 → 2024-12-31
Battery Capacity: 10 kWh
Efficiency: 0.90
Battery Cost: 80,000 SEK

Total Profit: 6,850 SEK
Annual Profit: 6,850 SEK/year
Pay-back Time: 11.7 years

Sensitivity Analysis:
  Efficiency 0.81 → Profit 5,950 SEK
  Efficiency 0.99 → Profit 7,520 SEK
  Price spread -10% → Profit 6,165 SEK
  Price spread +10% → Profit 7,535 SEK

Charts saved to ./plots/
```

## Data Source

Electricity prices are fetched from [mgrey.se](https://mgrey.se/espot/api), which provides day-ahead prices from ENTSO-E. Historical data is available from 2022-09-01.
