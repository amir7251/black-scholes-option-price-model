import numpy as np
from numpy import log, exp, sqrt
from scipy.stats import norm
from scipy.optimize import brentq

# variables defined specific to tesco stock (£)
S = 425.0  # current stock price at 03/09/2025
K = 430.0  # strike price
T = 0.5    # time to maturity in years
r = 0.048   # risk-free interest rate at 03/09/2025
sigma = 0.25  # volatility of the stock
q = 0.032  # dividend yield

# Direct calculation
d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
call_price = S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
put_price = K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)
print("d1:", d1)
print("d2:", d2)
print("Call Price: £", call_price)
print("Put Price: £", put_price)

# Functions for call and put prices


def call_price(S, K, T, r, sigma, q):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call_price = S * exp(-q * T) * norm.cdf(d1) - K * \
        exp(-r * T) * norm.cdf(d2)
    return call_price


def put_price(S, K, T, r, sigma, q):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    put_price = K * exp(-r * T) * norm.cdf(-d2) - S * \
        exp(-q * T) * norm.cdf(-d1)
    return put_price


parity_check = call_price(S, K, T, r, sigma, q) - put_price(S,
                                                            K, T, r, sigma, q) - S * exp(-q * T) + K * exp(-r * T)
# as expected, very close to 0 making the formulas correct
print("Put-Call Parity Check (should be close to 0):", parity_check)

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
    return S * exp(-q * T) * norm.pdf(d1) * sqrt(T)


def theta(S, K, T, r, sigma, q, option_type='call_price'):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == 'call_price':
        theta = (-S * sigma * exp(-q * T) * norm.pdf(d1) / (2 * sqrt(T))
                 + q * S * exp(-q * T) * norm.cdf(d1)
                 - r * K * exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put_price':
        theta = (-S * sigma * exp(-q * T) * norm.pdf(d1) / (2 * sqrt(T))
                 - q * S * exp(-q * T) * norm.cdf(-d1)
                 + r * K * exp(-r * T) * norm.cdf(-d2))
    return theta


def rho(S, K, T, r, sigma, q, option_type='call_price'):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == 'call_price':
        return K * T * exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put_price':
        return -K * T * exp(-r * T) * norm.cdf(-d2)


print("Call Delta:", delta(S, K, T, r, sigma, q, 'call_price'),
      "Tescos share price rises £1, the call option price rises by ~£0.52.")
print("Put Delta:", delta(S, K, T, r, sigma, q, 'put_price'),
      "Tescos share price rises £1, the put option price falls by ~£0.47.")
print("Gamma:", gamma(S, K, T, r, sigma, q),
      "Tescos share price changes by £1, the delta of the option changes by ~0.0052.")
print("Vega:", vega(S, K, T, r, sigma, q),
      "Tescos share price volatility changes by 1%, the option price changes by ~£1.18.")
print("Call Theta:", theta(S, K, T, r, sigma, q, 'call_price'),
      "Each day that passes, the call option price decreases by ~£0.09.")
print("Put Theta:", theta(S, K, T, r, sigma, q, 'put_price'),
      "Each day that passes, the put option price decreases by ~£0.068.")
print("Call Rho:", rho(S, K, T, r, sigma, q, 'call_price'),
      "The risk-free interest rate rises by 1%, the call option price rises by ~£0.96.")
print("Put Rho:", rho(S, K, T, r, sigma, q, 'put_price'),
      "The risk-free interest rate rises by 1%, the put option price falls by ~£1.14.")
