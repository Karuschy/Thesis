"""
Heston model calibration using QuantLib's built-in engine.

Uses HestonModelHelper with Levenberg-Marquardt optimizer
to calibrate (v0, kappa, theta, sigma, rho) to market IVs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import QuantLib as ql

from .pricer import HestonParams


@dataclass
class CalibrationResult:
    """Result of Heston calibration."""
    params: HestonParams
    error: float  # RMSE of model vs market IVs
    n_iterations: int
    success: bool
    message: str
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"CalibrationResult({status}, error={self.error:.6f})\n"
            f"  v0={self.params.v0:.6f}, kappa={self.params.kappa:.4f}, "
            f"theta={self.params.theta:.6f}, sigma={self.params.sigma:.4f}, rho={self.params.rho:.4f}"
        )


def calibrate_heston(
    S0: float,
    r: float,
    q: float,
    maturities: np.ndarray,  # in years
    strikes: np.ndarray,
    market_ivs: np.ndarray,
    cp_flags: np.ndarray,  # 'C' or 'P' for each option
    *,
    initial_params: HestonParams | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
    use_market_iv_for_v0: bool = True,
) -> CalibrationResult:
    """
    Calibrate Heston model to market implied volatilities.
    
    Uses QuantLib's HestonModelHelper and Levenberg-Marquardt optimizer.
    
    Parameters
    ----------
    S0 : float
        Spot price
    r : float
        Risk-free rate (annualized, continuous)
    q : float
        Dividend yield (annualized, continuous)
    maturities : np.ndarray
        Time to maturity in years for each option
    strikes : np.ndarray
        Strike prices
    market_ivs : np.ndarray
        Market implied volatilities (as decimals, e.g., 0.20 for 20%)
    cp_flags : np.ndarray
        'C' for call, 'P' for put
    initial_params : HestonParams, optional
        Starting point for optimization. If None, uses reasonable defaults.
    max_iterations : int
        Maximum optimization iterations
    tolerance : float
        Convergence tolerance
    use_market_iv_for_v0 : bool
        If True and initial_params is None, use ATM IV^2 as initial v0
        
    Returns
    -------
    CalibrationResult
        Contains calibrated params, error, and convergence info
    """
    assert len(maturities) == len(strikes) == len(market_ivs) == len(cp_flags)
    
    # Ensure Python floats (QuantLib doesn't accept numpy types)
    S0 = float(S0)
    r = float(r)
    q = float(q)
    
    # Setup QuantLib environment
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    
    # Market data handles
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed())
    )
    div_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, q, ql.Actual365Fixed())
    )
    
    # Initial parameters
    if initial_params is None:
        # Smart defaults based on market
        if use_market_iv_for_v0:
            # Use ATM IV^2 as initial variance
            atm_mask = np.abs(strikes / S0 - 1.0) < 0.1
            if atm_mask.any():
                v0_init = float(np.mean(market_ivs[atm_mask]) ** 2)
            else:
                v0_init = float(np.mean(market_ivs) ** 2)
        else:
            v0_init = 0.04  # 20% vol
        
        initial_params = HestonParams(
            v0=v0_init,
            kappa=1.0,
            theta=v0_init,
            sigma=0.3,
            rho=-0.5,
        )
    
    # Build Heston model and process
    heston_process = ql.HestonProcess(
        rate_handle,
        div_handle,
        spot_handle,
        initial_params.v0,
        initial_params.kappa,
        initial_params.theta,
        initial_params.sigma,
        initial_params.rho,
    )
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)
    
    # Build calibration helpers
    helpers = []
    for T, K, iv, cp in zip(maturities, strikes, market_ivs, cp_flags):
        # Convert to Python floats (QuantLib doesn't accept numpy types)
        T = float(T)
        K = float(K)
        iv = float(iv)
        
        if np.isnan(iv) or iv <= 0 or np.isnan(K) or K <= 0:
            continue
            
        # QuantLib Period from years
        days = int(round(T * 365))
        if days <= 0:
            continue
        period = ql.Period(days, ql.Days)
        
        # Option type
        option_type = ql.Option.Call if cp == "C" else ql.Option.Put
        
        # HestonModelHelper takes IV as volatility quote
        vol_quote = ql.QuoteHandle(ql.SimpleQuote(iv))
        
        helper = ql.HestonModelHelper(
            period,
            ql.TARGET(),  # Calendar
            float(S0),  # Ensure float
            K,
            vol_quote,
            rate_handle,
            div_handle,
            ql.BlackCalibrationHelper.ImpliedVolError,  # Calibrate to IV
        )
        helper.setPricingEngine(engine)
        helpers.append(helper)
    
    if len(helpers) == 0:
        return CalibrationResult(
            params=initial_params,
            error=np.inf,
            n_iterations=0,
            success=False,
            message="No valid calibration helpers could be built",
        )
    
    # Levenberg-Marquardt optimizer
    lm = ql.LevenbergMarquardt()
    
    # Calibrate
    end_criteria = ql.EndCriteria(max_iterations, 100, tolerance, tolerance, tolerance)
    
    try:
        heston_model.calibrate(helpers, lm, end_criteria)
        success = True
        message = "Calibration converged"
    except Exception as e:
        success = False
        message = str(e)
    
    # Extract calibrated parameters
    v0 = heston_model.v0()
    kappa = heston_model.kappa()
    theta = heston_model.theta()
    sigma = heston_model.sigma()
    rho = heston_model.rho()
    
    calibrated_params = HestonParams(
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
    )
    
    # Compute calibration error
    errors = []
    for helper in helpers:
        try:
            err = helper.calibrationError()
            errors.append(err ** 2)
        except:
            pass
    
    rmse = np.sqrt(np.mean(errors)) if errors else np.inf
    
    return CalibrationResult(
        params=calibrated_params,
        error=rmse,
        n_iterations=max_iterations,  # QuantLib doesn't expose iteration count
        success=success,
        message=message,
    )


def calibrate_heston_by_date(
    date_data: dict,
    *,
    initial_params: HestonParams | None = None,
    max_iterations: int = 500,
) -> CalibrationResult:
    """
    Convenience wrapper for calibrating one date from a DataFrame row dict.
    
    Parameters
    ----------
    date_data : dict
        Contains: S0, r, q, and arrays T, K, iv_market, cp_flag
        
    Returns
    -------
    CalibrationResult
    """
    return calibrate_heston(
        S0=date_data["S0"],
        r=date_data["r"],
        q=date_data["q"],
        maturities=np.array(date_data["T"]),
        strikes=np.array(date_data["K"]),
        market_ivs=np.array(date_data["iv_market"]),
        cp_flags=np.array(date_data["cp_flag"]),
        initial_params=initial_params,
        max_iterations=max_iterations,
    )


if __name__ == "__main__":
    # Quick test
    import numpy as np
    
    # Synthetic test data
    S0 = 100.0
    r = 0.05
    q = 0.02
    
    # True params
    true_params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)
    
    # Generate synthetic market IVs using Heston model
    from .pricer import heston_iv
    
    maturities = np.array([0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    strikes = np.array([90, 100, 110, 90, 100, 110, 90, 100, 110])
    cp_flags = np.array(["C"] * 9)
    
    market_ivs = np.array([
        heston_iv(S0, K, T, r, q, true_params, "call")
        for T, K in zip(maturities, strikes)
    ])
    
    print("Synthetic market IVs:", market_ivs)
    
    # Calibrate
    result = calibrate_heston(
        S0, r, q, maturities, strikes, market_ivs, cp_flags
    )
    
    print("\nCalibration result:")
    print(result)
    print(f"\nTrue params: v0={true_params.v0}, kappa={true_params.kappa}, "
          f"theta={true_params.theta}, sigma={true_params.sigma}, rho={true_params.rho}")
