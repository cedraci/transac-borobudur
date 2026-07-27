import numpy as np
import pandas as pd

from portfolio_construction import stats


def test_time_series_frequence_inference_daily():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    assert stats.time_series_frequence_inference(dates) == "Daily"


def test_time_series_frequence_inference_monthly():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    assert stats.time_series_frequence_inference(dates) == "Monthly"


def test_time_series_frequence_inference_weekly():
    dates = pd.date_range("2020-01-01", periods=20, freq="W")
    assert stats.time_series_frequence_inference(dates) == "Weekly"


def test_annualization_factor():
    assert stats.annualization_factor("Daily") == 252
    assert stats.annualization_factor("Weekly") == 52
    assert stats.annualization_factor("Monthly") == 12
    assert stats.annualization_factor("Annualy") == 1


def test_covariance_parametric_var_normal():
    w = np.array([0.5, 0.5])
    S = np.array([[0.04, 0.0], [0.0, 0.09]])
    var = stats.covariance_parametric_var(w, S, alpha=0.05, distrib="normal")
    # VaR should be negative (a loss) at a 5% left-tail
    assert var < 0


def test_covariance_parametric_var_student_more_extreme_than_normal():
    w = np.array([0.5, 0.5])
    S = np.array([[0.04, 0.0], [0.0, 0.09]])
    var_normal = stats.covariance_parametric_var(w, S, alpha=0.05, distrib="normal")
    var_student = stats.covariance_parametric_var(w, S, alpha=0.05, distrib="student")
    # fatter-tailed Student distribution should show a larger (more negative) loss
    assert var_student < var_normal
