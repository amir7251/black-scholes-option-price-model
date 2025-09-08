import numpy as np
from numpy import log, exp, sqrt
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
from datetime import datetime

# Functions for call and put prices


def call_price(S, K, T, r, sigma, q):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma, q):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)

# Greeks calculations


def delta(S, K, T, r, sigma, q, option_type='call_price'):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if option_type == 'call_price':
        return exp(-q * T) * norm.cdf(d1)
    elif option_type == 'put_price':
        return exp(-q * T) * (norm.cdf(d1) - 1)


def gamma(S, K, T, r, sigma, q):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt(T))


def vega(S, K, T, r, sigma, q):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return S * exp(-q * T) * norm.pdf(d1) * sqrt(T)  # per +1.00 vol


def theta(S, K, T, r, sigma, q, option_type='call_price'):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == 'call_price':
        return (-S * sigma * exp(-q * T) * norm.pdf(d1) / (2 * sqrt(T))
                + q * S * exp(-q * T) * norm.cdf(d1)
                - r * K * exp(-r * T) * norm.cdf(d2))
    else:
        return (-S * sigma * exp(-q * T) * norm.pdf(d1) / (2 * sqrt(T))
                - q * S * exp(-q * T) * norm.cdf(-d1)
                + r * K * exp(-r * T) * norm.cdf(-d2))


def rho(S, K, T, r, sigma, q, option_type='call_price'):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == 'call_price':
        return K * T * exp(-r * T) * norm.cdf(d2)   # per +1.00 in rate
    else:
        return -K * T * exp(-r * T) * norm.cdf(-d2)  # per +1.00 in rate

# implied volatility


def implied_volatility(option_market_price, S, K, T, r, q, option_type='call_price'):
    price_function = call_price if option_type == 'call_price' else put_price

    def objective_function(sigma):
        return price_function(S, K, T, r, sigma, q) - option_market_price
    try:
        return brentq(objective_function, 1e-6, 5.0)
    except ValueError:
        return None

# helpers for chain scan


def _ensure_chain_dataframe():
    try:
        df = pd.read_csv("options.csv")
        df.columns = [c.strip() for c in df.columns]
        if "option_type" not in df.columns:
            if "type" in df.columns:
                df["option_type"] = df["type"].str.lower().str.strip()
            else:
                raise ValueError("CSV must include 'option_type' or 'type'.")
        df["option_type"] = df["option_type"].str.lower().str.strip()
        lower = {c.lower(): c for c in df.columns}
        if "lastprice" in lower and "lastPrice" not in df.columns:
            df.rename(columns={lower["lastprice"]: "lastPrice"}, inplace=True)
        if "lastPrice" not in df.columns:
            if "bid" in lower and "ask" in lower:
                bid_col = lower["bid"]
                ask_col = lower["ask"]
                df["lastPrice"] = (df[bid_col].astype(
                    float) + df[ask_col].astype(float)) / 2.0
            else:
                raise ValueError(
                    "CSV needs 'lastPrice' OR both 'bid' and 'ask'.")
        return df
    except Exception:
        expiry = (pd.Timestamp.now().date() +
                  pd.Timedelta(days=45)).isoformat()
        T_synth = 45 / 365.0
        sigma_synth = 0.27
        strikes = [210, 220, 230, 235, 240, 245, 250, 260]
        rows = []
        for K_ in strikes:
            rows.append({"option_type": "call_price", "strike": K_,
                         "lastPrice": float(call_price(239, K_, T_synth, 0.048, sigma_synth, 0.0045)),
                         "expiration": expiry})
            rows.append({"option_type": "put_price", "strike": K_,
                         "lastPrice": float(put_price(239, K_, T_synth, 0.048, sigma_synth, 0.0045)),
                         "expiration": expiry})
        print("options.csv not found or unusable — using a realistic synthetic chain.")
        return pd.DataFrame(rows)


def _row_T(row, fallback_T):
    exp_str = row.get("expiration", None) if isinstance(
        row, dict) else row["expiration"] if "expiration" in row else None
    if pd.notna(exp_str):
        try:
            ex = pd.to_datetime(exp_str).date()
            days = max((ex - datetime.now().date()).days, 1)
            return days / 365.0
        except Exception:
            pass
    return float(fallback_T)


def _breakeven(row):
    return row["strike"] + row["lastPrice"] if row["option_type"] == "call_price" else row["strike"] - row["lastPrice"]


def _fmt_pick(title, row):
    days = int(round(row["T_row"] * 365))
    otype = "Call" if row["option_type"] == "call_price" else "Put"
    return (title + ": " + otype + " K=" + str(int(round(row["strike"]))) +
            ", Px=" + f"{row['lastPrice']:.2f}" +
            ", IV=" + f"{row['IV']*100:.1f}%" +
            ", Δ=" + f"{row['Delta']:+.3f}" +
            ", T≈" + str(days) + "d" +
            ", breakeven≈" + f"{_breakeven(row):.2f}")


if __name__ == "__main__":

    # variables defined specific to Apple stock ($)
    S = 239  # Example: live price
    K = 190  # strike price
    T = 0.5  # time to maturity in years
    r = 0.048  # risk-free interest rate
    sigma = 0.25  # assumed volatility
    q = 0.0045  # Dividend yield for Apple

    # Pricing
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call_px = S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    put_px = K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)
    print("d1:", d1)
    print("d2:", d2)
    print("Call Price: $", call_px)
    print("Put Price: $", put_px)

    # parity check
    parity_check = call_price(S, K, T, r, sigma, q) - put_price(S,
                                                                K, T, r, sigma, q) - S * exp(-q * T) + K * exp(-r * T)
    print("Put-Call Parity Check (should be close to 0):", parity_check)

    # Greeks
    print("Call Delta:", delta(S, K, T, r, sigma, q, 'call_price'),
          "Apples share price rises $1, the call option price rises by ~$0.93.")
    print("Put Delta:", delta(S, K, T, r, sigma, q, 'put_price'),
          "Apples share price rises $1, the put option price falls by ~$0.07.")
    print("Gamma:", gamma(S, K, T, r, sigma, q),
          "Apples share price changes by $1, the delta of the option changes by ~0.003.")
    print("Vega:", 0.01 * vega(S, K, T, r, sigma, q),  # per +1% vol
          "Apples share price volatility changes by 1%, the option price changes by ~$0.215.")
    print("Call Theta:", theta(S, K, T, r, sigma, q, 'call_price') / 365.0,  # per day
          "Each day that passes, the call option price decreases by ~$0.09.")
    print("Put Theta:", theta(S, K, T, r, sigma, q, 'put_price') / 365.0,    # per day
          "Each day that passes, the put option price decreases by ~$0.034.")
    print("Call Rho:", 0.01 * rho(S, K, T, r, sigma, q, 'call_price'),       # per +1% rate
          "The risk-free interest rate rises by 1%, the call option price rises by ~$0.843.")
    print("Put Rho:", 0.01 * rho(S, K, T, r, sigma, q, 'put_price'),         # per +1% rate
          "The risk-free interest rate rises by 1%, the put option price falls by ~$0.085.")

    # implied volatility examples
    market_call_price = 100
    implied_vol_call = implied_volatility(
        market_call_price, S, K, T, r, q, 'call_price')
    print("Implied Volatility from market call price of $100.0:", implied_vol_call)
    market_put_price = 10.0
    implied_vol_put = implied_volatility(
        market_put_price, S, K, T, r, q, 'put_price')
    print("Implied Volatility from market put price of $10.0:", implied_vol_put)

    # Calculating preferred options based on implied volatility
    options_chain = _ensure_chain_dataframe()
    options_chain = options_chain.dropna(subset=["strike", "lastPrice"]).copy()
    options_chain["strike"] = options_chain["strike"].astype(float)
    options_chain["lastPrice"] = options_chain["lastPrice"].astype(float)
    options_chain = options_chain[(options_chain["strike"] > 0) & (
        options_chain["lastPrice"] > 0)]
    options_chain["T_row"] = options_chain.apply(
        lambda r_: _row_T(r_, T), axis=1)
    options_chain["IV"] = options_chain.apply(
        lambda r_: implied_volatility(r_["lastPrice"], S, r_["strike"], r_[
                                      "T_row"], r, q, r_["option_type"]),
        axis=1
    )
    options_chain = options_chain.dropna(subset=["IV"]).copy()
    options_chain["Delta"] = options_chain.apply(
        lambda r_: delta(S, r_["strike"], r_["T_row"], r,
                         r_["IV"], q, r_["option_type"]),
        axis=1
    )
    options_chain["absDelta"] = options_chain["Delta"].abs()
    options_chain["moneyness"] = options_chain["strike"] / float(S)

    BUYER_MONEYNESS_MIN = 0.95
    BUYER_MONEYNESS_MAX = 1.05
    SELLER_MAX_ABS_DELTA = 0.25

    buyers_calls = options_chain[
        (options_chain["option_type"] == "call_price") &
        (options_chain["moneyness"].between(
            BUYER_MONEYNESS_MIN, BUYER_MONEYNESS_MAX))
    ].sort_values("IV")
    buyers_puts = options_chain[
        (options_chain["option_type"] == "put_price") &
        (options_chain["moneyness"].between(
            BUYER_MONEYNESS_MIN, BUYER_MONEYNESS_MAX))
    ].sort_values("IV")
    sellers_calls = options_chain[
        (options_chain["option_type"] == "call_price") &
        (options_chain["strike"] > S) &
        (options_chain["absDelta"] <= SELLER_MAX_ABS_DELTA)
    ].sort_values("IV", ascending=False)
    sellers_puts = options_chain[
        (options_chain["option_type"] == "put_price") &
        (options_chain["strike"] < S) &
        (options_chain["absDelta"] <= SELLER_MAX_ABS_DELTA)
    ].sort_values("IV", ascending=False)

    best_call_for_buyer = buyers_calls.head(1)
    best_put_for_buyer = buyers_puts.head(1)
    best_call_for_seller = sellers_calls.head(1)
    best_put_for_seller = sellers_puts.head(1)

    print("\nPreferred Options by Implied Volatility")
    print("Spot S:", S, "| r:", r, "| q:", q)
    if best_call_for_buyer.empty:
        print("Buyer (Call, low IV near ATM): no contract met the filter")
    else:
        print(_fmt_pick("Buyer (Call, low IV near ATM)",
              best_call_for_buyer.iloc[0]))
    if best_put_for_buyer.empty:
        print("Buyer (Put, low IV near ATM): no contract met the filter")
    else:
        print(_fmt_pick("Buyer (Put, low IV near ATM)",
              best_put_for_buyer.iloc[0]))
    if best_call_for_seller.empty:
        print("Seller (Call, OTM, high IV, |Δ|≤0.25): no contract met the filter")
    else:
        print(_fmt_pick("Seller (Call, OTM, high IV, |Δ|≤0.25)",
              best_call_for_seller.iloc[0]))
    if best_put_for_seller.empty:
        print("Seller (Put, OTM, high IV, |Δ|≤0.25): no contract met the filter")
    else:
        print(_fmt_pick("Seller (Put, OTM, high IV, |Δ|≤0.25)",
              best_put_for_seller.iloc[0]))

    cols = ["option_type", "expiration", "strike", "lastPrice",
            "T_row", "IV", "Delta", "absDelta", "moneyness"]
    for c in cols:
        if c not in options_chain.columns:
            options_chain[c] = np.nan
    options_chain[cols].to_csv("iv_scan_snapshot.csv", index=False)
