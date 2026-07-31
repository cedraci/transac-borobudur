from sklearn.covariance import ShrunkCovariance, LedoitWolf, OAS
from scipy.optimize import minimize
import scipy.stats as sc
import numpy as np

from portfolio_construction.stats import (
    time_series_frequence_inference,
    annualization_factor,
    covariance_parametric_var,
)

def portfolio_return(w, mu):
    """estimate portfolio return with given weights and expected returns"""
    return np.dot(w, mu)

def MAD(returns):
    return np.mean(np.absolute(returns - np.mean(returns,axis=0)), axis=0).item()

def portfolio_variance(w, S):
    """ Objective function for minimum variance optimization problem """
    return np.dot(np.dot(w, S), w)

def minimum_VaR(w, mean_ret, S):
    """ Objective function for minimum Value-at-Risk optimization problem """
    ALPHA = 0.05
    return sc.norm.ppf(ALPHA, np.dot(mean_ret, w), np.dot(np.dot(w, S), w)) * -1.

def maximun_return(w, mean_ret):
    """ Objective function for the Max return portfolio """
    return -np.dot(mean_ret, w)

def maximum_sharpe(w, mean_ret, S):
    """ Objective function for maximum sharpe ratio optimization problem """
    return np.dot(mean_ret, w) / np.sqrt(np.dot(w.T, np.dot(S, w))) * -1

def risk_contribution(w, S):
    """ function that calculates asset risk contribution to total risk

    Returns one value per asset; they sum to the portfolio volatility.

    `S * w.T` is a matrix product only when S is an np.matrix. Every caller
    passes a plain ndarray (np.cov returns one), where `*` broadcasts
    elementwise and produced an n-by-n array instead. That corrupted ERC's
    error term, so "equal risk contribution" did not equalise risk.
    """
    w = np.asarray(w).ravel()
    S = np.asarray(S)
    sigma_p = np.sqrt(np.dot(np.dot(w, S), w))

    # Marginal Risk Contribution
    RC = np.multiply(np.dot(S, w), w)/sigma_p
    return RC

def ERC(w, S):
    """ Objective function to perform Equalization of Risk Contribution """
    risk_budget = np.asmatrix(np.ones(len(w))/len(w))
    RC = risk_contribution(w, S)
    sigma_p = np.sqrt(np.dot(np.dot(w, S), w))
    risk_target = np.asmatrix(np.multiply(sigma_p, risk_budget))
    error = np.sum(np.square(RC - risk_target)) * 1000
    return error

def MDP(w, S):
    """ Objective function for the Most Diversify Portfolio optimization problem

    The diversification ratio is (w.sigma)/sigma_p, so the numerator needs the
    assets' standard deviations. np.diag(S) on a 2-D covariance gives
    VARIANCES, which made the objective favour the noisiest asset.
    """
    volatilities = np.sqrt(np.diag(S))
    return -np.transpose(np.dot(w, volatilities))/np.sqrt(np.dot(np.dot(w, S), w))

def InverseVol(S):
    inverse_vol = 1 / np.sqrt(np.diagonal(S))
    return inverse_vol / np.sum(inverse_vol)

def minimum_var_objective(w, S, alpha=0.05, distrib="normal"):
    """Objective function for the `minimum_VaR` optimizer: the negative of
    `covariance_parametric_var`, so that minimizing it maximizes VaR
    (i.e. minimizes the magnitude of the downside)."""
    return -covariance_parametric_var(w, S, alpha, distrib)

"""
optim = arg max S.T * W
sous contrainte :
    sigma(W) <= sigma_star

Pour la stratégie total return :
    sigma_star = 6%
    0 =< W <= 1
    1.T * W = 1

Pour la stratégie absolute return :
    sigma_star = 4%
    -0.5% * 1 <= W <= 0.5 * 1
    1.T * W = 0
"""
def max_momentum(w, mom):
    return -np.dot(mom, w)

def max_momentum_optimization(ret, scores):
    TOLERANCE = 1e-15

    # ShrunkCovariance must see the RETURNS: handed an (n,n) covariance it
    # treats it as n observations of n features and returns a near-zero matrix.
    cov = ShrunkCovariance().fit(np.asarray(ret)).covariance_

    w = np.ones(cov.shape[0])/cov.shape[0]
    e = np.ones(cov.shape[0])

    # w'Sw is a DAILY VARIANCE; annualising it takes *252 and a square root to
    # become a volatility. The docstring specifies sigma(W) <= sigma_star, so
    # this is an inequality (scipy 'ineq' requires fun(w) >= 0).
    const = [{'type' : 'eq' , 'fun' : lambda w: np.dot(w, e) - 1.},
            {'type' : 'ineq' , 'fun' : lambda w: 0.06 - np.sqrt(np.dot(np.dot(w, cov), w) * 252)}]

    bds = [(0.0, 1.) for i in range(0, len(e))]

    solution = minimize(fun=max_momentum, x0=w, args=(scores), bounds=bds, method='SLSQP', constraints=const, tol=TOLERANCE)
    return np.round(solution.x, 5)

def longshort_momentum_optimization(ret, scores):

    """
    -0.5% * 1 <= W <= 0.5 * 1
    1.T * W = 0
    """
    TOLERANCE = 1e-12

    cov = ShrunkCovariance().fit(np.asarray(ret)).covariance_

    w = np.ones(cov.shape[0])/cov.shape[0]
    e = np.ones(cov.shape[0])

    const = [{'type' : 'eq' , 'fun' : lambda w: np.dot(w, e) - 0.},
            {'type' : 'ineq' , 'fun' : lambda w: 0.04 - np.sqrt(np.dot(np.dot(w, cov), w) * 252)}]

    bds = [(-0.5, 0.50) for i in range(0, len(e))]

    solution = minimize(fun=max_momentum, x0=w, args=(scores), bounds=bds, method='SLSQP', constraints=const, tol=TOLERANCE)
    return np.round(solution.x, 5)

def portfolio_optimization(ret, typeOpt, bounds=None, cov_mat="sample", target_return=0.):
    TOLERANCE = 1e-9
    
    data_frequence = time_series_frequence_inference(ret.index)
    annual_factor = annualization_factor(data_frequence)

    if cov_mat == "shrunked":
        # fitted on the returns, not on their covariance - see MDP note above
        cov = ShrunkCovariance().fit(np.asarray(ret)).covariance_
    elif cov_mat == "gerber":
        corr = gerber_correlation_matrix(ret, 0.5, True) # on calcul la matrice de correlation
        cov = corr_to_cov(corr, ret.std()) # de laquelle on dérive la matrice de covariance
    else:
        cov = np.cov(ret, rowvar=False)

    mean_ret = ret.mean()

    w = np.ones(cov.shape[0])/cov.shape[0]
    e = np.ones(cov.shape[0])
    
    const = ({'type' : 'eq' , 'fun' : lambda w: np.dot(w, e) - 1.})

    if bounds == None:
        bds = [(0.0, 1.0) for i in range(0, len(e))]
    else:
        bds = bounds
        
    if typeOpt == "minimum_variance":
        solution = minimize(fun=portfolio_variance, x0=w, args=(cov), bounds=bds, method='SLSQP', constraints=const, tol=TOLERANCE)
        w_star = solution.x.round(6)
    elif typeOpt == "most_diversified":
        solution = minimize(fun=MDP, x0=w, args=(cov), bounds=bds, method='SLSQP', constraints=const, tol=TOLERANCE)
        w_star = solution.x.round(6)
    elif typeOpt == "maximum_sharpe":
        solution = minimize(fun=maximum_sharpe, x0=w, args=(mean_ret, cov), bounds=bds, method='SLSQP', constraints=const, tol=TOLERANCE)
        w_star = solution.x.round(6)
    elif typeOpt == "minimum_VaR":
        solution = minimize(fun=minimum_var_objective, x0=w, args=(cov), bounds=bds, method='SLSQP', constraints=const, tol=TOLERANCE)
        w_star = solution.x.round(6)
    elif typeOpt == "equal_risk_contribution":
        solution = minimize(fun=ERC, x0=w, args=(cov), bounds=bds, method='SLSQP', constraints=const, tol=TOLERANCE)
        w_star = solution.x.round(6)
    elif typeOpt == "maximum_return":
        solution = minimize(fun=maximun_return, x0=w, args=(mean_ret), bounds=bds, method='SLSQP', constraints=const, tol=TOLERANCE)
        w_star = solution.x.round(6)
    elif typeOpt == "inverse_volatility":
        w_star = InverseVol(cov)
    elif typeOpt == "mean_variance":
        const = (
            {'type' : 'eq' , 'fun' : lambda w: np.dot(w, e) - 1.}, # contrainte de la somme des poids égale à 1
            {'type' : 'eq' , 'fun' : lambda w: np.dot(w, mean_ret) - float(target_return)/annual_factor}, # contrainte sur le rendement
        )
        solution = minimize(fun=portfolio_variance, x0=w, args=(cov), bounds=bds, method='SLSQP', constraints=const, tol=TOLERANCE)

        if np.sum(solution.x.round(6)) > np.dot(w, e):
            w_star = solution.x.round(6) / np.sum(solution.x.round(6))
        else:
            w_star = solution.x.round(6)

    else:
        w_star = w
        
    return dict(zip(ret.columns, w_star))

def estimate_bayes_stein(matrix):
    t, n = matrix.shape
    sample_means = matrix.mean(axis=0)
    ones = np.ones_like(sample_means)
    E = np.cov(matrix, rowvar=False)
    adj_Z_C = (float(t) - 1) / (t - n - 2)
    E *= round(adj_Z_C, 4)

    I = np.linalg.inv(E)
    grand_mean = sample_means.dot(I).dot(ones.T) / ones.dot(I).dot(ones.T)

    diff = sample_means - grand_mean
    denominator = n + 2 + diff.dot(t*I).dot(diff.T)
    w = round((n + 2) / denominator, 4)
    adjusted_means = (1 - w)*sample_means + w*grand_mean
    return adjusted_means


def gerber_statistic(df1, df2, threshold, returns=False):
    """Function for estimating the Gerber Statistic of co-movement between two time series.
    source : The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3880054

    Args:
        df1 (pd.DataFrame): time series of prices or returns.
        df2 (pd.DataFrame): time series of prices or returns.
        threshold (float): threshold parameter for seperating scatter regions of co-movements.
        returns (bool, defaults to False): if True, time series are prices returns. Defaults to False (prices).

    Returns:
        float: Gerber Statistic as robust co-movement measure for covariance matrix estimation.
    """

    new_df1 = df1.copy()
    new_df2 = df2.copy()

    # False means that we work with prices
    if returns == False:  # So we need to calculte returns
        new_df1 = new_df1.pct_change().fillna(0)
        new_df2 = new_df2.pct_change().fillna(0)

    # Returns are centered to be compared to threshold
    centered1 = new_df1 / new_df1.std()
    centered2 = new_df2 / new_df2.std()

    uu = dd = ud = du = nn = 0.0

    for i in range(centered1.shape[0]):
        if (float(centered1.iloc[i]) >= threshold) & (float(centered2.iloc[i]) >= threshold):
            uu += 1
        elif (-threshold <= float(centered1.iloc[i]) <= threshold) & (
            -threshold <= float(centered2.iloc[i]) <= threshold
        ):
            nn += 1
        elif (float(centered1.iloc[i]) <= -threshold) & (
            float(centered2.iloc[i]) <= -threshold
        ):
            dd += 1
        elif (float(centered1.iloc[i]) >= threshold) & (
            float(centered2.iloc[i]) <= -threshold
        ):
            ud += 1
        elif (float(centered1.iloc[i]) <= -threshold) & (
            float(centered2.iloc[i]) >= threshold
        ):
            du += 1
        else:
            next

    numerator = uu + dd - ud - du

    denominator = centered1.shape[0] - nn

    return numerator / denominator

def gerber_correlation_matrix(df, threshold, returns=False):
    """Function to estimate the covariance matrix of multiple time series, through prices or returns,
    based on the gerber statistic, which is a robust co-movement statistic.
    source : https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3880054

    Args:
        df (pd.DataFrame): DataFrame of time series (prices or returns).
        threshold (float): threshold parameter for seperating scatter regions of co-movements.
        returns (bool, optional): if True, time series are prices returns. Defaults to False (prices).

    Returns:
        np.ndarray: correlation matrix
    """
    new_df = df.copy()

    # False means that we work with prices
    if returns == False:  # So we need to calculte returns
        new_df = new_df.pct_change().fillna(0)

    flatten_matrix = []
    for i in range(new_df.shape[1]):
        for j in range(new_df.shape[1]):
            flatten_matrix.append(
                gerber_statistic(
                    new_df.iloc[0:, i], new_df.iloc[0:, j], threshold, True
                )
            )

    reshaped_matrix = np.asarray(flatten_matrix).reshape(
        new_df.shape[1], new_df.shape[1]
    )
    return reshaped_matrix

def corr_to_cov(corr_mat, sigma):
    """Function to convert correlation matrix to covariance matrix"""
    return np.dot(np.dot(np.diag(sigma), corr_mat), np.diag(sigma))


def optimize_capital_protection(mu, cov_mat, duration, confidence=0.95):
    """
    Trouve l'allocation optimale maximisant le rendement, sous la contrainte
    que le rendement cumulé au bout de 'duration' années soit >= 0
    avec un niveau de confiance donné.

    Paramètres:
    -----------
    mu : list ou np.array
        Vecteur des rendements espérés annualisés (ex: [0.05, 0.08...]).
    cov_mat : np.array
        Matrice de covariance annualisée.
    duration : float
        Horizon d'investissement en années (ex: 5.0).
    confidence : float, optionnel (défaut=0.95)
        Niveau de confiance pour la protection (0.95 = 95% de chance de ne pas perdre).

    Retourne:
    ---------
    dict : Dictionnaire contenant :
        - 'weights': L'allocation optimale trouvée.
        - 'expected_return': Le rendement espéré du portefeuille (annualisé).
        - 'volatility': La volatilité du portefeuille (annualisée).
        - 'worst_case_total_return': Le rendement cumulé dans le pire scénario (doit être >= 0).
        - 'success': Booléen indiquant si l'optimiseur a trouvé une solution.
        - 'message': Message de l'optimiseur.
    """

    # Conversion en numpy arrays pour la sécurité
    mu = np.array(mu)
    cov_mat = np.array(cov_mat)
    n_assets = len(mu)

    # 1. Calcul du Z-score pour le seuil de risque
    # Si confiance = 0.95, alpha = 0.05. norm.ppf(0.05) ~ -1.645
    alpha = 1.0 - confidence
    z_score = sc.norm.ppf(alpha)

    # 2. Fonction Objectif : Maximiser le rendement (donc Minimiser -Mu)
    def objective(w):
        return -np.dot(w, mu)

    # 3. La Contrainte de Sécurité (Capital Protection)
    def constraint_safety(w):
        # Stats du portefeuille
        port_mu = np.dot(w, mu)
        port_var = np.dot(w.T, np.dot(cov_mat, w))
        port_std = np.sqrt(port_var)

        # Modèle Brownien Géométrique (Log-Normal)
        # Drift géométrique = mu - 0.5 * sigma^2
        geo_drift = port_mu - 0.5 * (port_std**2)

        # Rendement cumulé au seuil de probabilité alpha
        # Formule : (Drift * T) + (Z * Sigma * sqrt(T))
        worst_case_log_return = (geo_drift * duration) + (
            z_score * port_std * np.sqrt(duration)
        )

        # On veut que ce log-return soit >= 0 (soit un prix final >= prix initial)
        return worst_case_log_return

    # 4. Configuration de l'optimiseur
    # Bornes : Poids entre 0% et 100% (pas de vente à découvert)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))

    # Contraintes :
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Somme des poids = 100%
        {"type": "ineq", "fun": constraint_safety},  # Contrainte de sécurité >= 0
    )

    # Point de départ (Equipondéré)
    init_guess = np.full(n_assets, 1.0 / n_assets)

    # 5. Exécution
    result = minimize(
        objective, init_guess, method="SLSQP", bounds=bounds, constraints=cons
    )

    # 6. Préparation des résultats
    opt_weights = result.x
    final_mu = np.dot(opt_weights, mu)
    final_std = np.sqrt(np.dot(opt_weights.T, np.dot(cov_mat, opt_weights)))
    safety_margin = constraint_safety(
        opt_weights
    )  # La valeur atteinte par la contrainte

    return {
        "weights": np.round(opt_weights, 4),
        "expected_return_annual": np.round(final_mu, 4),
        "volatility_annual": np.round(final_std, 4),
        "worst_case_total_return": np.round(safety_margin, 4),
        "success": result.success,
        "message": result.message,
    }

def get_stressed_covariance(cov_matrix, stress_factor=0.5):
    """
    Durcit la matrice de covariance en augmentant artificiellement les corrélations.

    Paramètres:
    -----------
    cov_matrix : np.array
        Matrice de covariance historique (annualisée).
    stress_factor : float (0 à 1)
        Force du stress.
        0.0 = Aucune modification (Matrice historique pure).
        1.0 = Panique totale (Corrélation de 1 entre tous les actifs).
        0.3 à 0.5 = Valeurs recommandées pour une allocation prudente.

    Retourne:
    ---------
    np.array : La nouvelle matrice de covariance "stressée".
    """
    cov_matrix = np.array(cov_matrix)

    # 1. Extraire les Volatilités (Écart-types)
    # La diagonale de la covariance contient les variances (sigma^2)
    variances = np.diag(cov_matrix)
    stds = np.sqrt(variances) * 1.1  # j'augmente la vol historique de 10%

    # 2. Calculer la Matrice de Corrélation Historique
    # Corr(i,j) = Cov(i,j) / (Std(i) * Std(j))
    # On utilise l'algèbre linéaire pour le faire proprement : D^-1 * S * D^-1
    D_inv = np.diag(1 / stds)
    corr_matrix_hist = D_inv @ cov_matrix @ D_inv

    # 3. Créer la Matrice de "Panique" (Corrélation = 1 partout)
    # Dans un krach absolu, tout bouge ensemble.
    corr_matrix_panic = np.ones_like(corr_matrix_hist)

    # 4. Mélanger les deux (Shrinkage)
    # Formule : (1 - alpha) * Hist + alpha * Panique
    corr_matrix_stressed = (
        1 - stress_factor
    ) * corr_matrix_hist + stress_factor * corr_matrix_panic

    # (Optionnel mais recommandé) : On remet des 1 parfaits sur la diagonale pour la propreté mathématique
    np.fill_diagonal(corr_matrix_stressed, 1.0)

    # 5. Reconstruire la Matrice de Covariance
    # Cov_stress = D * Corr_stress * D
    D = np.diag(stds)
    cov_matrix_stressed = D @ corr_matrix_stressed @ D

    return cov_matrix_stressed