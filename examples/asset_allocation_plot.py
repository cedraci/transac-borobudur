"""Sweep an efficient frontier over a range of target returns and plot how the
optimal asset allocation (weights) shifts as the target return increases.

Run from the project root with:
    python examples/asset_allocation_plot.py

Reads `Asset Class Historical Data.xlsx` (sheet "rebased") from this same
`examples/` folder - swap DATA_FILE below to point at your own workbook.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import portfolio_construction.portfolio_analysis as pa
import portfolio_construction.portfolio_optimization as po

EXAMPLES_DIR = Path(__file__).parent
DATA_FILE = EXAMPLES_DIR / "Asset Class Historical Data.xlsx"

# Min/max weight allowed for each asset class, in the same order as the
# columns of the "rebased" sheet.
BOUNDARIES = [
    (0.0, 0.4),  # Actions EU
    (0.0, 0.4),  # Actions US
    (0.0, 0.10),  # Actions Japon
    (0.0, 0.05),  # Actions EM
    (0.0, 0.6),  # Souverain US
    (0.0, 0.6),  # Souverain EU
    (0.0, 0.3),  # High Yield US
    (0.0, 0.3),  # High Yield EU
    (0.0, 0.4),  # Credit IG US
    (0.0, 0.4),  # Credit IG EU
    (0.0, 0.4),  # Credit IG EM
    (0.0, 0.2),  # Private Equity US
    (0.0, 0.1),  # Gold
    (0.0, 0.6),  # Diversification Monde
]

TARGET_RETURNS = np.arange(0.04, 0.085, 0.001)


def mean_variance_optimization(expected_return, covariance_matrix, target_return, bounds=None):
    """Find the minimum-variance weights that hit `target_return` exactly."""
    TOLERANCE = 1e-30

    w = np.ones(covariance_matrix.shape[0]) / covariance_matrix.shape[0]
    e = np.ones(covariance_matrix.shape[0])
    bds = bounds if bounds is not None else [(0.0, 1.0) for _ in range(len(e))]

    const = (
        {
            "type": "eq",
            "fun": lambda w: np.dot(w, expected_return) - float(target_return),
        },  # hit the target return exactly
        {
            "type": "eq",
            "fun": lambda w: np.dot(w, e) - 1.0,
        },  # weights sum to 1
    )

    solution = minimize(
        fun=po.portfolio_variance,
        x0=w,
        args=(covariance_matrix),
        bounds=bds,
        method="SLSQP",
        constraints=const,
        tol=TOLERANCE,
    )

    return solution.x.round(6)


def main():
    historical_data = pd.read_excel(DATA_FILE, sheet_name="rebased", header=0, index_col=0)
    exp_returns = historical_data.apply(lambda x: pa.annualized_return(x))
    covariance_matrix = historical_data.pct_change().dropna().cov()

    optim_results = {}
    for i, current_target in enumerate(TARGET_RETURNS):
        w_star = mean_variance_optimization(
            exp_returns.values,
            covariance_matrix,
            current_target,
            BOUNDARIES,
        )
        optim_results[i] = {
            "perf": po.portfolio_return(w_star, exp_returns.values),
            "risk": np.sqrt(po.portfolio_variance(w_star, covariance_matrix)) * np.sqrt(252),
            "weights": w_star,
        }

    df_optim = pd.DataFrame.from_dict(optim_results, orient="index")
    df_weights = pd.DataFrame([w for w in df_optim["weights"]])
    df_weights.columns = exp_returns.index

    df_weights.plot(kind="area", stacked=True)
    plt.show()

    return df_weights, df_optim


if __name__ == "__main__":
    main()
