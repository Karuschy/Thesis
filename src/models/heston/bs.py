"""
Black-Scholes pricing and implied volatility inversion.

Provides:
- bs_call / bs_put: closed-form European option prices
- bs_iv: implied volatility via Brent root-finding
"""

from __future__ import annotations

import math
from typing import Literal

from scipy.optimize import brentq
from scipy.stats import norm


def bs_d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Calculate d1 in Black-Scholes formula."""
    return (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def bs_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Calculate d2 in Black-Scholes formula."""
    return bs_d1(S, K, T, r, q, sigma) - sigma * math.sqrt(T)


def bs_call(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Black-Scholes European call price.

    Parameters
    ----------
    S : float
        Spot price of underlying.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free rate (continuously compounded).
    q : float
        Dividend yield (continuously compounded).
    sigma : float
        Volatility (annualized).

    Returns
    -------
    float
        Call option price.
    """
    if T <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    
    d1 = bs_d1(S, K, T, r, q, sigma)
    d2 = bs_d2(S, K, T, r, q, sigma)
    
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Black-Scholes European put price.

    Parameters
    ----------
    S : float
        Spot price of underlying.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free rate (continuously compounded).
    q : float
        Dividend yield (continuously compounded).
    sigma : float
        Volatility (annualized).

    Returns
    -------
    float
        Put option price.
    """
    if T <= 0:
        return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    
    d1 = bs_d1(S, K, T, r, q, sigma)
    d2 = bs_d2(S, K, T, r, q, sigma)
    
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    cp_flag: Literal["C", "P"],
) -> float:
    """
    Black-Scholes European option price.

    Parameters
    ----------
    S, K, T, r, q, sigma : float
        See bs_call / bs_put.
    cp_flag : {"C", "P", "call", "put"}
        "C" or "call" for call, "P" or "put" for put.

    Returns
    -------
    float
        Option price.
    """
    is_call = cp_flag in ("C", "c", "call", "Call", "CALL")
    if is_call:
        return bs_call(S, K, T, r, q, sigma)
    else:
        return bs_put(S, K, T, r, q, sigma)


def bs_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Black-Scholes vega (sensitivity to volatility).

    Returns
    -------
    float
        Vega (dPrice/dSigma).
    """
    if T <= 0:
        return 0.0
    
    d1 = bs_d1(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)


def bs_iv(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    cp_flag: Literal["C", "P"],
    sigma_low: float = 1e-6,
    sigma_high: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float | None:
    """
    Invert Black-Scholes to find implied volatility via Brent's method.

    Parameters
    ----------
    price : float
        Market price of the option.
    S, K, T, r, q : float
        Underlying, strike, time, rate, dividend yield.
    cp_flag : {"C", "P"}
        Call or put.
    sigma_low : float
        Lower bound for volatility search.
    sigma_high : float
        Upper bound for volatility search.
    tol : float
        Tolerance for root finding.
    max_iter : int
        Maximum iterations for Brent.

    Returns
    -------
    float or None
        Implied volatility, or None if root not found.
    """
    # Normalize cp_flag
    is_call = cp_flag in ("C", "c", "call", "Call", "CALL")
    normalized_flag = "C" if is_call else "P"
    
    # Intrinsic value bounds
    forward = S * math.exp((r - q) * T)
    df = math.exp(-r * T)
    
    if is_call:
        intrinsic = max(forward - K, 0.0) * df
        upper_bound = S * math.exp(-q * T)  # Max call value
    else:
        intrinsic = max(K - forward, 0.0) * df
        upper_bound = K * df  # Max put value
    
    # Price must be between intrinsic and upper bound
    if price < intrinsic - tol or price > upper_bound + tol:
        return None
    
    # Edge case: price equals intrinsic (sigma -> 0)
    if price <= intrinsic + tol:
        return sigma_low
    
    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, q, sigma, normalized_flag) - price
    
    try:
        # Check bounds have opposite signs
        f_low = objective(sigma_low)
        f_high = objective(sigma_high)
        
        if f_low * f_high > 0:
            # Try expanding upper bound
            for mult in [10.0, 20.0, 50.0]:
                sigma_high_exp = sigma_high * mult
                f_high = objective(sigma_high_exp)
                if f_low * f_high <= 0:
                    sigma_high = sigma_high_exp
                    break
            else:
                return None
        
        iv = brentq(objective, sigma_low, sigma_high, xtol=tol, maxiter=max_iter)
        return iv
    
    except (ValueError, RuntimeError):
        return None


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    cp_flag: Literal["C", "P"],
) -> float:
    """
    Black-Scholes delta.

    Parameters
    ----------
    S, K, T, r, q, sigma : float
        Standard BS parameters.
    cp_flag : {"C", "P", "call", "put"}
        Call or put.

    Returns
    -------
    float
        Delta (dPrice/dS).
    """
    is_call = cp_flag in ("C", "c", "call", "Call", "CALL")
    
    if T <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    d1 = bs_d1(S, K, T, r, q, sigma)
    
    if is_call:
        return math.exp(-q * T) * norm.cdf(d1)
    else:
        return -math.exp(-q * T) * norm.cdf(-d1)


def strike_from_delta(
    S: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    delta: float,
    cp_flag: Literal["C", "P"],
) -> float:
    """
    Convert delta to strike using Black-Scholes.

    This uses the "spot delta" convention common in equity options.

    Parameters
    ----------
    S, T, r, q, sigma : float
        Standard BS parameters.
    delta : float
        Target delta (positive for calls, negative for puts,
        or absolute value if cp_flag specified).
    cp_flag : {"C", "P", "call", "put"}
        Call or put.

    Returns
    -------
    float
        Strike corresponding to the given delta.
    """
    is_call = cp_flag in ("C", "c", "call", "Call", "CALL")
    
    # Ensure delta is in proper range
    if is_call:
        # Call delta in (0, 1)
        delta = abs(delta)
        d1 = norm.ppf(delta * math.exp(q * T))
    else:
        # Put delta in (-1, 0), we use absolute value
        delta = -abs(delta)
        d1 = norm.ppf((delta + 1) * math.exp(q * T))
    
    # Invert d1 formula to get K
    # d1 = [ln(S/K) + (r - q + 0.5*sigma^2)*T] / (sigma*sqrt(T))
    # K = S * exp(-d1 * sigma * sqrt(T) + (r - q + 0.5*sigma^2)*T)
    
    K = S * math.exp(-d1 * sigma * math.sqrt(T) + (r - q + 0.5 * sigma**2) * T)
    return K
