# Option Pricing (Black–Scholes) + IV & Greeks

Single-file Python project that prices European calls/puts with dividend yield `q`, computes Greeks, solves **implied volatility** via Brent’s method, and scans a simple option chain to surface buyer/seller candidates. Outputs a CSV snapshot for quick review.

---

## What’s inside
- **Pricing:** `call_price`, `put_price` (with continuous dividend yield `q`)
- **Greeks:** `delta`, `gamma`, `vega`, `theta`, `rho`
- **IV solver:** `implied_volatility` (root-finds σ from a market price)
- **Chain scan:** reads `options.csv` (or builds a synthetic chain) → computes IV & Δ → prints best picks and saves `iv_scan_snapshot.csv`
- **Main guard:** can be imported without auto-running

> **Units in the printed commentary**
> - `theta` is shown **per day** (yearly theta ÷ 365).  
> - `vega` and `rho` are shown **per +1% change** (yearly vega/rho × 0.01).

---

## Quick start
```bash
# Python 3.10+ recommended
pip install -r requirements.txt
python option_pricer.py

