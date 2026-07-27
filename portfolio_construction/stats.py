"""Shared statistical helpers used by both portfolio_analysis and portfolio_optimization.

These used to be duplicated (with the exact same implementation) in both modules;
they now live here as the single source of truth.
"""

import numpy as np
import scipy.stats as sc


def time_series_frequence_inference(dates):
    """Find the most frequent Timedelta days"""
    delta_days = dates.to_series().diff().value_counts().idxmax().days

    if delta_days > 20:
        return "Monthly"
    elif delta_days > 4:
        return "Weekly"
    elif delta_days < 4:
        return "Daily"
    else:
        return "Annualy"


def annualization_factor(freq):
    """Define the annualization factor according to time series frequence"""
    if freq == "Monthly":
        return 12
    elif freq == "Weekly":
        return 52
    elif freq == "Daily":
        return 252
    else:
        return 1


def covariance_parametric_var(
    w: list, S: np.ndarray, alpha: float = 0.05, distrib: str = "normal"
) -> float:
    """Function to estimate the value-at-risk using only the covariance matrix and
    the weights of the underlyings

    Args:
        w (list): portfolio weights
        S (np.ndarray): covariance matrix
        alpha (float): the percentile of the distribution
        distrib (str, optional): hypothetical underlying distribution. Defaults to "normal".

    Returns:
        float: VaR for one day
    """
    portfolioSigma = np.sqrt(np.dot(np.dot(w, S), w))

    if distrib == "student":
        VaR = sc.t.ppf(q=alpha, loc=0, scale=1, df=4) * portfolioSigma
    else:
        VaR = sc.norm.ppf(q=alpha) * portfolioSigma

    return VaR
