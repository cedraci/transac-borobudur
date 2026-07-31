import pandas as pd
import numpy as np
import scipy.stats as sc
from typing import List
from datetime import date
from decimal import Decimal
from dataclasses import dataclass

from portfolio_construction.stats import (
    time_series_frequence_inference,
    annualization_factor,
    covariance_parametric_var,
)


def annualized_return(bkt, trace=False):
    """Annualized return from pandas DF or Series (index = dates)"""
    freq = time_series_frequence_inference(bkt.index)

    if trace == True:
        print("Time Series Frequence : %s" % (freq))

    nb_days = (bkt.index[-1] - bkt.index[0]).days
    result = (bkt.tail(1).values / bkt.head(1).values) ** (1 / (nb_days / 365)) - 1.0
    return float(result.item())


def annualized_volatility(bkt, trace=False):
    """Annualized volatility from pandas DF or Series (index=dates)"""
    freq = time_series_frequence_inference(bkt.index)
    ANN_FACT = annualization_factor(freq)

    if trace == True:
        print("Time Series Frequence : %s" % (freq))
        print("ANN_FACT : %s" % (ANN_FACT))

    returns = bkt.pct_change().dropna()
    return float(returns.std() * np.sqrt(ANN_FACT))


def annualized_sharpe_ratio(bkt, rf=0.0):
    """Annualized sharpe ratio from pandas DF or Series (index= dates)"""
    ann_ret = annualized_return(bkt)
    ann_vol = annualized_volatility(bkt)
    return float((ann_ret - rf) / ann_vol)


def historical_drawdown(bkt):
    """Return historical drawdown of backtest"""
    return bkt / bkt.cummax() - 1


def maximum_drawdown(bkt):
    """Return the maximum drawdown of backtest"""
    return historical_drawdown(bkt).min()


def monthly_returns(bkt):
    """Return monthly returns of a time series"""
    if isinstance(bkt, pd.DataFrame):
        new = bkt.copy()
    else:
        new = bkt.copy().to_frame()  # convert to DataFrame if Series.

    eom = new.resample("ME").last()
    return eom.pct_change().dropna()


def calendar_performances(strat):
    """Generate calendar table (dataframe) of monthly and yearly returns"""
    bkt_series = strat.copy()
    returns = monthly_returns(
        bkt_series
    )  # extract monthly returns from dataframe indexed by dates

    if returns.empty:
        # Fewer than two month-ends in the input, so there is no complete month
        # to report. Fail with something a caller can act on rather than an
        # IndexError from returns.index[0] below.
        raise ValueError(
            "calendar_performances needs at least one complete month of data"
        )

    # reformat dates in index to ensure that it is the last day of the month
    returns.index = returns.index.map(lambda x: x + pd.tseries.offsets.MonthEnd(0))

    eoy_date = returns.index[0] + pd.tseries.offsets.YearEnd(
        -1
    )  # extract last day of previous year
    first_date = eoy_date + pd.tseries.offsets.MonthEnd(
        1
    )  # last day of the first month of the year
    last_date = returns.index[-1] + pd.tseries.offsets.YearEnd(
        0
    )  # extract the last day of the year

    # create monthly dates series between first and last dates
    dates_rng = pd.date_range(start=first_date, end=last_date, freq="ME")

    # df_dates is created two ensure that dates series always begin by january
    # and end by december. That way, it is easy to next reshape the data by 12 columns
    df_dates = pd.DataFrame(np.zeros(len(dates_rng)), index=dates_rng)

    # concat allow to merge df without join logic which allow that months with no returns
    # are filled with NaN and stay in the dataframe (which is necessary)
    merged = pd.concat([df_dates, returns], axis=1)

    months_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    years_names = np.unique(returns.index.year)  # extract years in array

    # the merged dataframe is reshaped with 12 columns
    cal_table = pd.DataFrame(
        np.asarray(merged[bkt_series.name].values).reshape(len(years_names), 12),
        index=years_names,
        columns=months_names,
    )

    # yearly performances calculation as the cumulative product of monthly returns
    yearly_perf = cal_table.fillna(0).apply(lambda x: np.cumprod(1 + x), axis=1) - 1
    # fill NaN by zero is necessary for calculating yearly_perf because it's cumprod of
    # monthly returns. If monthly return equal NaN it's not possible.
    yearly_perf = yearly_perf.December
    yearly_perf.name = "Year"

    # final concatenation of monthly and yearly performances
    final_table = pd.concat([cal_table, yearly_perf], axis=1)
    return final_table


def historical_var(prices, alpha, duration):
    """estimate historical VaR given confidence and time horizon"""
    duration_returns = prices.pct_change(duration).dropna()
    return duration_returns.quantile(alpha)


def gbm_multiple_path(
    num: int,
    startprice: float,
    mu: float,
    sigma: float,
    days: int,
    distrib: str = "normal",
):
    """
    Génère plusieurs trajectoires de GBM (Monte Carlo)

    Simule un Mouvement Brownien Géométrique (GBM)

    Parameters:
    -----------
    num : int
        Nombre de simulations à effectuer
    startprice : float
        Prix initial
    mu : float
        Rendement annuel attendu (drift)
    sigma : float
        Volatilité annuelle
    days : int
        Nombre de jours de trading
    distrib : str
        "normal" ou "student"

    Returns:
    --------
    price : np.array
        Trajectoire des prix
    """

    days = int(days)
    dt = 1.0 / 252.0

    # Génération vectorisée de TOUTES les trajectoires en une fois
    if distrib == "normal":
        Z = np.random.normal(0, 1, (days, num))
    else:
        df = 5.0
        Z = sc.t.rvs(df, loc=0, scale=1, size=(days, num))
        # Z = Z / np.sqrt(df / (df - 2)) if df > 2 else Z

    # Calcul vectorisé
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = np.exp(drift + diffusion)

    # Trajectoires des prix
    price_paths = np.zeros((days + 1, num))
    price_paths[0] = startprice

    price_paths[1:] = startprice * np.cumprod(log_returns, axis=0)

    return price_paths


def monteCarlo_var(num, startprice, mu, sigma, alpha, duration, distrib):
    """VaR estimation based on Monte Carlo simulations"""
    multi_path = gbm_multiple_path(num, startprice, mu, sigma, duration, distrib)
    # Quantile over the terminal (final-day) distribution only - pooling in the
    # intermediate time steps (row 0 is always startprice, i.e. a return of 0)
    # biases the quantile toward zero and silently measures the wrong alpha.
    return np.quantile(multi_path[-1, :], alpha) / startprice - 1


def parametric_var(prices, vec_w, alpha, duration, distrib):
    """estimate parametric VaR given a distribution law, confidence and time horizon"""

    # rendements nécessaires au calcul de la matrice de variance-covariance
    duration_returns = prices.pct_change(duration).dropna()

    if vec_w == None:
        mean_return = duration_returns.mean()
        sigma = duration_returns.std()
    else:
        # rendement du portefeuille, pondéré des poids de chaque action
        mean_return = np.dot(vec_w, duration_returns.mean())
        varcov_mat = duration_returns.cov()  # matrice de variance-covariance
        sigma = np.sqrt(
            np.dot(np.dot(vec_w, varcov_mat), vec_w)
        )  # Volatilité du portefeuille

    if distrib == "normal":
        var = sc.norm.ppf(q=alpha, loc=mean_return, scale=sigma)
    else:
        if vec_w != None:
            params = sc.t.fit(np.dot(duration_returns, vec_w))
            var = sc.t.ppf(q=alpha, df=params[0], loc=mean_return, scale=sigma)
        else:
            params = sc.t.fit(duration_returns)
            var = sc.t.ppf(q=alpha, df=params[0], loc=mean_return, scale=sigma)
    return var


def historical_es(prices, alpha, duration):
    """estimate expected shortfall based on historical data, given parameters"""
    duration_returns = prices.pct_change(duration).dropna()
    var = duration_returns.quantile(alpha)
    return duration_returns[duration_returns <= var].mean()


def parametric_es(prices, vec_w, alpha, duration):
    """estimate parametric expected shortfall based on normal distribution"""
    duration_returns = prices.pct_change(duration).dropna()
    if vec_w == None:
        mean_return = duration_returns.mean()
        sigma = duration_returns.std()
    else:
        mean_return = np.dot(vec_w, duration_returns.mean())
        varcov_mat = duration_returns.cov()
        sigma = np.sqrt(np.dot(np.dot(vec_w, varcov_mat), vec_w))

    es = -sigma * sc.norm.pdf(sc.norm.ppf(alpha)) / (alpha) + mean_return
    return es


def monteCarlo_es(num, startprice, mean, sigma, alpha, duration, distrib):
    """Expected Shortfall estimation based on Monte Carlo simulations"""
    multi_path = np.asarray(
        gbm_multiple_path(num, startprice, mean, sigma, duration, distrib)
    )  # simulation des MBG

    # création d'une matrice aux mêmes dimensions que notre matrice de simulations
    div_matrix = np.full(multi_path.shape, startprice)
    returns = (
        multi_path / div_matrix - 1.0
    )  # calcul des rendements cumulés sur la période
    # rendements cumulés au jour final, pour chaque trajectoire simulée
    final_returns = returns[-1, :]
    var_threeshold = np.quantile(final_returns, alpha)  # extraction du seuil de la VaR

    # extraction des indices où le rendement cumulé final est en dessous du seuil
    below_var = np.where(final_returns <= var_threeshold)[0]
    return np.mean(final_returns[below_var])


def stats_report(df, rf=0.0, trace=False, rebased_plot=False):
    """function that compile statistics for dataframe of strategies"""
    global_results = []

    for fd in df.columns:
        cal_perf = calendar_performances(df[fd])
        # Build a fresh dict per column - reusing one dict across iterations
        # (the previous implementation) let trailing-year keys from an
        # earlier column with a longer/later history leak into a later
        # column whose own history doesn't cover those years.
        fd_results = {
            "Name": fd,
            "Ann. Return": "{:.2%}".format(annualized_return(df[fd])),
            "Ann. Volatility": "{:.2%}".format(annualized_volatility(df[fd])),
            "Ann. Sharpe": "{:.2%}".format(annualized_sharpe_ratio(df[fd], rf=rf)),
            "Value-at-Risk": "{:.2%}".format(historical_var(df[fd], 0.05, 1)),
            "Expected Shortfall (histo)": "{:.2%}".format(
                historical_es(df[fd], 0.05, 1)
            ),
            "Max Drawdown": "{:.2%}".format(maximum_drawdown(df[fd])),
        }
        # Report up to the three most recent calendar periods, fewer when
        # the history is shorter than that instead of raising IndexError.
        n_periods = min(3, len(cal_perf))
        for i in range(1, n_periods + 1):
            fd_results[cal_perf.index[-i]] = "{:.2%}".format(cal_perf.iloc[-i, -1])
        global_results.append(fd_results)
    df_results = pd.DataFrame(global_results)

    if trace == True:
        print(df_results)

    return df_results


def rebased_from_date(prices, rebased_dt):
    if isinstance(prices, pd.DataFrame):
        if rebased_dt in prices.index:
            return (
                prices[prices.index >= rebased_dt].div(
                    prices[prices.index == rebased_dt].values
                )
                * 100.0
            )
        else:
            print("Given date doesn't exists in the index")
            return None


def cagr_rolled(
    yearly_returns: pd.DataFrame, max_holding_period: int = 20
) -> pd.DataFrame:
    """
    function that construct a matrix of cagr for different starting point
    and for different holding periods. That allow to measure the robustness
    of a strategy.

    Args:
        yearly_returns (pd.DataFrame): yearly returns from strategy
        max_holding_period (int, optional): maximum of holding period to test. Defaults to 20.

    Returns:
        pd.DataFrame: cagr table
    """
    # get the total period
    periods = yearly_returns.shape[0]
    current_max_holding_period = np.min([max_holding_period, periods])

    # instanciate an empty matrix
    result_matrix = np.empty((periods, current_max_holding_period))
    result_matrix[:] = np.nan

    for year in range(0, periods):
        # set the first value, the yearly return
        result_matrix[year][0] = float(yearly_returns.iloc[year].iloc[0])

        # loop to compile cagr through all holding periods
        for holding_period in range(2, current_max_holding_period + 1):

            # ensure that the remaining period is coherent with holding period
            if holding_period > periods - year:
                continue
            else:
                sub_period = yearly_returns.iloc[year : year + holding_period]

                # cagr calculation
                result_matrix[year, holding_period - 1] = float(
                    (sub_period.apply(lambda x: np.cumprod(1 + x)).tail(1).values.item())
                    ** (1 / holding_period)
                    - 1.0
                )

    # Built unconditionally: a dataset confined to a single calendar year gives
    # zero yearly returns, the loop above never runs, and this used to fall
    # through to an unbound result_df.
    result_df = pd.DataFrame(
        result_matrix,
        columns=range(1, current_max_holding_period + 1),
        index=yearly_returns.index,
    )
    return result_df


def portfolio_path_cholesky(
    n_sims: int,
    start_price: float,
    weights: List[float],
    mu: List[float] | None,
    cov_matrix: pd.DataFrame,
    days: int,
    distrib: str = "normal",
):
    """
    Simule la valeur globale d'un portefeuille composé d'actifs corrélés.

    Arguments:
    n_sims      -- Nombre de simulations
    start_price -- Valeur totale initiale du portefeuille (ex: 100.0)
    weights     -- VECTEUR des poids initiaux (ex: [0.6, 0.4])
    mu          -- VECTEUR des drifts annuels
    cov_matrix  -- MATRICE de Covariance

    Retourne:
    portfolio_paths -- Matrice (Jours, Simulations) -> Valeur du PF global
    """

    # CONVERSION CRITIQUE : On transforme tout en tableau NumPy pur
    # Cela évite définitivement le KeyError: (2, 2)
    if hasattr(cov_matrix, "values"):
        cov_matrix = cov_matrix.values
    else:
        cov_matrix = np.array(cov_matrix)

    days = int(days)
    dt = 1.0 / 252.0

    # Sécurités sur les inputs
    mu = np.array(mu)
    weights = np.array(weights)
    n_assets = cov_matrix.shape[0]

    # Calcul des Montants Initiaux par Ligne (Buy & Hold)
    # Si start_price=100 et weights=[0.6, 0.4], on commence avec [60, 40]
    initial_positions = start_price * weights

    # Cholesky
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        print(
            "Erreur: La matrice n'est pas définie positive. Vérifiez vos données (doublons, NaN)."
        )
        return None

    # Génération des bruits
    # Shape : (Jours, Sims, Actifs)
    if distrib == "normal":
        Z_uncorrelated = np.random.normal(0, 1, (days, n_sims, n_assets))
    else:
        # Loi de Student (Fat Tails)
        # 5 est le standard pour les actions (rendements quotidiens)
        df = 5.0  # Paramètre pour la loi de Student

        # Génération brute (Variance > 1)
        t_random = sc.t.rvs(df, loc=0, scale=1, size=(days, n_sims, n_assets))

        # --- AJOUT CRUCIAL : LE CLIPPING ---
        # On interdit au générateur de sortir des valeurs > 10 ou < -10
        # Cela empêche les valeurs aberrantes (comme le "70" que vous avez eu)
        t_random = np.clip(t_random, -15, 15)

        # Correction de Variance (CRUCIAL)
        # La variance d'une Student(df) est df / (df-2)
        # Pour df=5, la variance est 5/3 = 1.666...
        # L'écart-type est sqrt(1.666...) ~= 1.29
        std_correction = np.sqrt(df / (df - 2))

        # Normalisation
        # On divise pour que le bruit ait une variance de 1, comme la loi Normale
        Z_uncorrelated = t_random / std_correction

    # Corrélation des aléas via Cholesky
    # On utilise einsum pour multiplier efficacement (Jours, Sims, Actifs) par (Actifs, Actifs)
    Z_correlated_annual = np.einsum("ijk,lk->ijl", Z_uncorrelated, L)
    diffusion_term_daily = Z_correlated_annual * np.sqrt(dt)

    if mu.ndim == 0:
        mu = np.full(n_assets, mu)

    # Calcul des Rendements (Log returns) pour chaque actif
    variances = np.diag(cov_matrix)
    # diffusion_term = Z_correlated * np.sqrt(dt)

    max_move = np.max(np.abs(diffusion_term_daily))

    if max_move > 0.40:
        print(f"--- INFO VOLATILITÉ ---")
        print(f"Mouvement max détecté : {max_move*100:.2f}%")

        # Vérifions si c'est cohérent avec la volatilité de l'actif
        # Récupérons l'index de l'actif qui a fait ce saut
        indices = np.where(np.abs(diffusion_term_daily) == max_move)
        asset_idx = indices[2][0]
        vol_annuelle = np.sqrt(cov_matrix[asset_idx, asset_idx])
        print(f"Actif concerné (index) : {asset_idx}")
        print(f"Volatilité annuelle de cet actif : {vol_annuelle*100:.2f}%")

        # Règle de décision suggérée
        if max_move > 0.50:  # Si > 50% en un jour
            print(
                "CRITIQUE : Mouvement irréaliste. Il y a un problème dans cov_matrix."
            )
        else:
            print("ACCEPTABLE : C'est un scénario de Krach (Black Swan). On garde.")

    drift = (mu - 0.5 * variances) * dt
    drift_term = drift[np.newaxis, np.newaxis, :]

    cum_log_returns = np.cumsum(drift_term + diffusion_term_daily, axis=0)

    # Construction de la valeur de chaque ligne (Asset Value)
    # On crée un tableau pour les valeurs des actifs individuels
    # Shape : (Jours+1, Sims, Actifs)
    asset_values = np.zeros((days + 1, n_sims, n_assets))

    # Initialisation : [60, 40] pour toutes les simus
    asset_values[0] = initial_positions

    # Evolution : Montant * Cumul des rendements
    # Broadcasting : (n_assets) se multiplie bien sur la dernière dimension
    asset_values[1:] = initial_positions * np.exp(cum_log_returns)

    # Agrégation (Somme des lignes pour avoir le Portefeuille)
    # On écrase la dimension "Actifs" (axis=2) en faisant la somme
    portfolio_paths = np.sum(asset_values, axis=2)

    return portfolio_paths


@dataclass
class DrawdownPeriod:
    """Période de drawdown identifiée.

    Attributes:
        rank: Rang du drawdown (1 = pire).
        start_date: Date du pic précédant la baisse.
        trough_date: Date du point bas.
        recovery_date: Date de retour au niveau du pic (None si non récupéré).
        drawdown: Drawdown maximum de la période (valeur négative, ex: -0.15 pour -15%).
        days_to_trough: Nombre de jours calendaires du pic au creux.
        days_to_recovery: Nombre de jours calendaires du creux au recovery (None si non récupéré).
    """

    rank: int
    start_date: date
    trough_date: date
    recovery_date: date | None
    drawdown: float
    days_to_trough: int
    days_to_recovery: int | None


def compute_drawdown_periods(
    navs: list[tuple[date, Decimal]],
    top_n: int = 10,
    min_drawdown: float = 0.01,
) -> list[DrawdownPeriod]:
    """Identifie et classe les principales périodes de drawdown.

    Algorithme :
    1. Calcule le high-water mark (pic glissant) et la série de drawdown.
    2. Identifie chaque période contiguë de drawdown (entre deux retours au pic).
    3. Pour chaque période, extrait le creux, le drawdown max, et la date de recovery.
    4. Trie par sévérité (drawdown le plus profond en premier).

    Args:
        navs: Liste de tuples (date, nav_value), triée par date croissante.
        top_n: Nombre maximum de drawdowns à retourner.
        min_drawdown: Seuil minimum (en valeur absolue) pour retenir un drawdown.
            Ex: 0.01 = ignorer les drawdowns < 1%.

    Returns:
        Liste de DrawdownPeriod triée par sévérité (pire en premier).
    """
    if len(navs) < 3:
        return []

    # Construire un DataFrame à partir des NAV
    df = pd.DataFrame(navs, columns=["date", "nav"])
    df["nav"] = df["nav"].astype(float)
    df = df.set_index("date").sort_index()

    # High-water mark et série de drawdown
    df["hwm"] = df["nav"].cummax()
    df["dd"] = df["nav"] / df["hwm"] - 1.0

    # Identifier les périodes de drawdown
    # Un drawdown commence quand dd < 0 et se termine quand dd revient à 0
    periods: list[DrawdownPeriod] = []
    in_drawdown = False
    start_dt: date | None = None
    trough_dt: date | None = None
    trough_val = 0.0

    for dt, row in df.iterrows():
        dd_val = row["dd"]

        if not in_drawdown and dd_val < 0:
            # Début d'un drawdown : le pic est la date précédente (ou la première date)
            # On cherche la dernière date où dd == 0
            mask = df.loc[:dt, "dd"] == 0.0  # type: ignore[index]
            if mask.any():
                start_dt = mask[mask].index[-1]
            else:
                start_dt = df.index[0]
            in_drawdown = True
            trough_dt = dt
            trough_val = dd_val

        elif in_drawdown:
            if dd_val < trough_val:
                trough_dt = dt
                trough_val = dd_val

            if dd_val >= 0:
                # Recovery atteint
                if (
                    abs(trough_val) >= min_drawdown
                    and start_dt is not None
                    and trough_dt is not None
                ):
                    periods.append(
                        DrawdownPeriod(
                            rank=0,
                            start_date=start_dt,
                            trough_date=trough_dt,
                            recovery_date=dt,
                            drawdown=round(trough_val, 6),
                            days_to_trough=(trough_dt - start_dt).days,
                            days_to_recovery=(dt - trough_dt).days,
                        )
                    )
                in_drawdown = False

    # Drawdown en cours (pas de recovery)
    if (
        in_drawdown
        and abs(trough_val) >= min_drawdown
        and start_dt is not None
        and trough_dt is not None
    ):
        periods.append(
            DrawdownPeriod(
                rank=0,
                start_date=start_dt,
                trough_date=trough_dt,
                recovery_date=None,
                drawdown=round(trough_val, 6),
                days_to_trough=(trough_dt - start_dt).days,
                days_to_recovery=None,
            )
        )

    # Trier par sévérité (plus négatif = pire) et assigner les rangs
    periods.sort(key=lambda p: p.drawdown)
    for i, period in enumerate(periods[:top_n]):
        period.rank = i + 1

    return periods[:top_n]


def drawdowns_table(
    nav: pd.Series, top_n: int = 10, min_drawdown: float = 0.01
) -> pd.DataFrame:
    """Rank the worst drawdown episodes.

    Takes a Series: `nav.items()` must yield (timestamp, price) pairs, which a
    DataFrame does not do.
    """
    convert_ts = [(date.date(), Decimal(str(price))) for date, price in nav.items()]

    strat_dds = compute_drawdown_periods(
        convert_ts, top_n=top_n, min_drawdown=min_drawdown
    )

    return pd.DataFrame(strat_dds)
