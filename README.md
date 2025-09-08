# Option Pricing (Black–Scholes) + IV & Greeks

Single-file Python project that prices European calls/puts with a continuous dividend yield `q`, computes Greeks, solves implied volatility via Brent’s method, and scans a simple option chain to surface buyer/seller candidates. Outputs a CSV snapshot for quick review.

## What’s inside
- **Pricing:** `call_price`, `put_price` (with dividend yield `q`). :contentReference[oaicite:0]{index=0}
- **Greeks:** `delta`, `gamma`, `vega`, `theta`, `rho`. :contentReference[oaicite:1]{index=1}
- **IV solver:** `implied_volatility` (root-finds σ from a market price with Brent’s method). :contentReference[oaicite:2]{index=2}
- **Chain scan:** reads `options.csv` (or builds a realistic synthetic chain) → computes IV & Δ → picks “preferred” buyer/seller contracts and saves `iv_scan_snapshot.csv`. :contentReference[oaicite:3]{index=3}
- **Main guard:** can be imported without auto-running (`if __name__ == "__main__":`). :contentReference[oaicite:4]{index=4}

## Units in printed output
- **Theta** shown **per day** (yearly theta ÷ 365 at print). :contentReference[oaicite:5]{index=5}
- **Vega** shown **per +1% vol** (yearly vega × 0.01 at print). :contentReference[oaicite:6]{index=6}
- **Rho** shown **per +1% rate** (yearly rho × 0.01 at print). :contentReference[oaicite:7]{index=7}

## CSV format for `options.csv`
Minimum columns:
- `option_type` = `call_price` or `put_price` *(or)* `type` = `call`/`put`
- `strike`
- `lastPrice` *(or)* both `bid` and `ask` (mid is computed)
Optional:
- `expiration` in `YYYY-MM-DD` (else falls back to global `T`)

> If `options.csv` is missing or unusable, the script creates a synthetic chain around `S` so the scan still runs. :contentReference[oaicite:8]{index=8}

## Quick start
```bash
# Python 3.10+ recommended
pip install -r requirements.txt
python app.py

