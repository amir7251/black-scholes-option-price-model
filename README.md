# Option Pricing (Black–Scholes) + Implied Volatility & Greeks

A single-file Python project that:

- Prices European calls/puts under Black–Scholes–Merton with continuous dividend yield `q`.
- Computes closed-form **Greeks**.
- Solves **implied volatility (IV)** from market prices (Brent’s method).
- Scans an option chain to surface **preferred (rule-based) contracts for buyers and sellers** using IV.
- Exports a **CSV snapshot** of all evaluated contracts.

> If `options.csv` isn’t present, the script builds a **realistic synthetic chain** around your spot so the demo always runs.

---

## Why this project

Option quotes embed the market’s expectations via **implied volatility**. This tool inverts for IV per contract and then **systematically chooses**:

- **Buyer picks:** lowest IV **near the money** (pay less implied variance where gamma matters most).
- **Seller picks:** **OTM** strikes with **high IV** and **small |Δ|** (collect richer premium with less directional exposure).

Selections are printed as **clear one-liners** and saved to `iv_scan_snapshot.csv` for auditability.

---

## Features

- **Pricing:** `call_price`, `put_price` (supports dividend yield `q`).
- **Greeks:** `delta`, `gamma`, `vega`, `theta`, `rho` (closed-form).
- **IV solver:** `implied_volatility` (Brent root-finder; robust bracketing).
- **Chain scan:** reads `options.csv` (or builds synthetic) → computes IV & Δ → picks buyer/seller candidates → prints concise results → saves `iv_scan_snapshot.csv`.

**Units in printed output**
- **Theta:** per **day** (annual θ ÷ 365).
- **Vega & Rho:** per **+1%** move (annual vega/rho × 0.01).

**Model assumptions**
- European exercise, constant `r`, `q`, and σ; continuous compounding.

---

## Quick start

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
python app.py

