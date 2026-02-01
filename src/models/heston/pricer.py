"""
Heston model pricing via QuantLib.

Provides:
- heston_price: European call/put under Heston dynamics
- heston_iv: implied volatility from Heston price
- HestonParams: dataclass for Heston parameters
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import QuantLib as ql

from .bs import bs_iv


@dataclass
class HestonParams:
    """
    Heston model parameters.

    Attributes
    ----------
    v0 : float
        Initial variance (σ² at t=0).
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run variance (mean reversion level).
    sigma : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between spot and variance processes.
    """
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def __post_init__(self):
        """Validate parameters."""
        if self.v0 < 0:
            raise ValueError(f"v0 must be non-negative, got {self.v0}")
        if self.kappa < 0:
            raise ValueError(f"kappa must be non-negative, got {self.kappa}")
        if self.theta < 0:
            raise ValueError(f"theta must be non-negative, got {self.theta}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")

    @property
    def feller_condition(self) -> bool:
        """Check if Feller condition is satisfied: 2*kappa*theta > sigma^2."""
        return 2 * self.kappa * self.theta > self.sigma**2

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        """Return parameters as tuple (v0, kappa, theta, sigma, rho)."""
        return (self.v0, self.kappa, self.theta, self.sigma, self.rho)


def _build_heston_process(
    S0: float,
    r: float,
    q: float,
    params: HestonParams,
    eval_date: ql.Date | None = None,
) -> tuple[ql.HestonProcess, ql.Settings]:
    """
    Build a QuantLib HestonProcess.

    Parameters
    ----------
    S0 : float
        Spot price.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    params : HestonParams
        Heston model parameters.
    eval_date : ql.Date, optional
        Evaluation date. Defaults to today.

    Returns
    -------
    tuple[ql.HestonProcess, ql.Settings]
        The Heston process and QuantLib settings.
    """
    settings = ql.Settings.instance()
    
    if eval_date is None:
        eval_date = ql.Date.todaysDate()
    
    settings.evaluationDate = eval_date
    
    # Flat term structures
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    
    risk_free_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(eval_date, r, day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(eval_date, q, day_count)
    )
    
    heston_process = ql.HestonProcess(
        risk_free_ts,
        dividend_ts,
        spot_handle,
        params.v0,
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
    )
    
    return heston_process, settings


def heston_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    cp_flag: Literal["C", "P"] = "C",
) -> float:
    """
    Price a European option under the Heston model.

    Uses QuantLib's AnalyticHestonEngine with default settings.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    params : HestonParams
        Heston model parameters.
    cp_flag : {"C", "P"}
        Call or put.

    Returns
    -------
    float
        Option price.
    """
    if T <= 0:
        # Handle expired options
        if cp_flag == "C":
            return max(S0 - K, 0.0)
        else:
            return max(K - S0, 0.0)
    
    # Build process
    eval_date = ql.Date.todaysDate()
    heston_process, _ = _build_heston_process(S0, r, q, params, eval_date)
    
    # Build option
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    
    # Calculate maturity date
    maturity_date = eval_date + ql.Period(int(T * 365), ql.Days)
    
    option_type = ql.Option.Call if cp_flag == "C" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(option_type, K)
    exercise = ql.EuropeanExercise(maturity_date)
    
    option = ql.VanillaOption(payoff, exercise)
    
    # Build model and engine
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)
    
    option.setPricingEngine(engine)
    
    return option.NPV()


def heston_iv(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    cp_flag: Literal["C", "P"] = "C",
) -> float | None:
    """
    Compute Black-Scholes implied volatility from Heston price.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    params : HestonParams
        Heston model parameters.
    cp_flag : {"C", "P"}
        Call or put.

    Returns
    -------
    float or None
        Implied volatility, or None if inversion fails.
    """
    price = heston_price(S0, K, T, r, q, params, cp_flag)
    return bs_iv(price, S0, K, T, r, q, cp_flag)


def heston_price_surface(
    S0: float,
    strikes: list[float],
    maturities: list[float],
    r: float,
    q: float,
    params: HestonParams,
    cp_flag: Literal["C", "P"] = "C",
) -> list[list[float]]:
    """
    Price a grid of European options under Heston.

    Parameters
    ----------
    S0 : float
        Spot price.
    strikes : list[float]
        List of strike prices.
    maturities : list[float]
        List of times to maturity in years.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    params : HestonParams
        Heston model parameters.
    cp_flag : {"C", "P"}
        Call or put.

    Returns
    -------
    list[list[float]]
        Price grid of shape [len(maturities), len(strikes)].
    """
    prices = []
    for T in maturities:
        row = [heston_price(S0, K, T, r, q, params, cp_flag) for K in strikes]
        prices.append(row)
    return prices


def heston_iv_surface(
    S0: float,
    strikes: list[float],
    maturities: list[float],
    r: float,
    q: float,
    params: HestonParams,
    cp_flag: Literal["C", "P"] = "C",
) -> list[list[float | None]]:
    """
    Compute BS implied volatility surface under Heston.

    Parameters
    ----------
    S0, strikes, maturities, r, q, params, cp_flag
        See heston_price_surface.

    Returns
    -------
    list[list[float | None]]
        IV grid of shape [len(maturities), len(strikes)].
        None for points where IV inversion fails.
    """
    ivs = []
    for T in maturities:
        row = [heston_iv(S0, K, T, r, q, params, cp_flag) for K in strikes]
        ivs.append(row)
    return ivs
