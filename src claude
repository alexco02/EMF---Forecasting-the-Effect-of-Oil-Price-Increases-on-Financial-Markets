"""
EMIF Group Project — Helper Functions
=====================================
All analysis functions used by the main notebook.
Existing functions are kept. New additions marked with # [NEW].
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist


# ============================================================
# DATA LOADING
# ============================================================

DEFAULT_DATA_PATH = Path("data/raw/data_hec_projet_1.xlsx")
if not DEFAULT_DATA_PATH.exists():
    DEFAULT_DATA_PATH = Path("../data/raw/data_hec_projet_1.xlsx")

DAILY_COLUMNS = [
    "date", "wti", "brent", "bcom_energy", "tft", "natgas",
    "sp500", "msci_world", "msci_em", "russell2000",
    "us10y", "us2y", "hy_ytw", "gold",
]

MONTHLY_COLUMNS = [
    "date", "ip", "cfnai", "ism_mfg", "ism_prices",
    "ism_services", "ism_services_prices", "retail_sales", "richmond_fed",
]


def load_data_sheet(excel_path, sheet_name, columns):
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    data = raw.iloc[5:].copy()
    data.columns = columns

    first_value = str(data.iloc[0, 0]).strip().lower()
    if first_value == "dates":
        data = data.iloc[1:].copy()

    data["date"] = pd.to_datetime(data["date"])
    for column in columns[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.sort_values("date").reset_index(drop=True)
    return data


def aggregate_daily_to_monthly(daily_df):
    monthly_data = daily_df.resample("M", on="date").last().reset_index()

    sp500_daily_returns = daily_df[["date", "sp500"]].copy()
    sp500_daily_returns["sp500_daily_log_return"] = np.log(sp500_daily_returns["sp500"]).diff()

    monthly_volatility = (
        sp500_daily_returns.resample("M", on="date")["sp500_daily_log_return"]
        .std()
        .rename("sp500_realized_vol")
        .reset_index()
    )

    monthly_data = monthly_data.merge(monthly_volatility, on="date", how="left")
    return monthly_data


def prepare_monthly_dataset(excel_path=DEFAULT_DATA_PATH):
    daily = load_data_sheet(excel_path, "Daily", DAILY_COLUMNS)
    monthly_macro = load_data_sheet(excel_path, "Monthly", MONTHLY_COLUMNS)

    print("Daily raw shape:", daily.shape)
    print("Monthly macro raw shape:", monthly_macro.shape)

    monthly_market_data = aggregate_daily_to_monthly(daily)
    print("Aggregated daily-to-monthly shape:", monthly_market_data.shape)

    merged = monthly_market_data.merge(monthly_macro, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    print("Merged monthly dataset shape:", merged.shape)
    return daily, monthly_macro, merged


# ============================================================
# VARIABLE CONSTRUCTION
# ============================================================

def add_log_return(df, price_column, new_column):
    df[new_column] = np.log(df[price_column] / df[price_column].shift(1))


def add_growth_rate(df, level_column, new_column):
    df[new_column] = np.log(df[level_column] / df[level_column].shift(1))


def build_project_variables(monthly_df):
    df = monthly_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    add_log_return(df, "wti", "wti_return")
    add_log_return(df, "brent", "brent_return")
    add_log_return(df, "sp500", "sp500_return")
    add_log_return(df, "msci_em", "msci_em_return")
    add_log_return(df, "msci_world", "msci_world_return")
    add_log_return(df, "russell2000", "russell2000_return")
    add_log_return(df, "gold", "gold_return")

    df["term_spread"] = df["us10y"] - df["us2y"]
    df["hy_change"] = df["hy_ytw"].diff()
    df["us10y_change"] = df["us10y"].diff()

    add_growth_rate(df, "ip", "ip_growth")
    add_growth_rate(df, "retail_sales", "retail_sales_growth")

    print("\nCheck of expected columns after variable construction:")
    expected_columns = [
        "wti_return", "brent_return", "sp500_return", "msci_em_return",
        "gold_return", "term_spread", "hy_change", "sp500_realized_vol",
        "cfnai", "ism_mfg",
    ]
    for column in expected_columns:
        print(f"  {column}: {'OK' if column in df.columns else 'MISSING'}")

    return df


# ============================================================
# STATIONARITY TESTS (ADF — kept from original)
# ============================================================

def run_adf_table(df, columns):
    rows = []
    for column in columns:
        series = df[column].dropna()
        if len(series) < 20:
            rows.append({
                "variable": column, "n_obs": len(series),
                "adf_stat": np.nan, "p_value": np.nan, "stationary_5pct": np.nan,
            })
            continue

        result = adfuller(series, autolag="AIC")
        rows.append({
            "variable": column, "n_obs": len(series),
            "adf_stat": round(result[0], 4),
            "p_value": round(result[1], 4),
            "stationary_5pct": result[1] < 0.05,
        })

    return pd.DataFrame(rows)


# ============================================================
# [NEW] LJUNG-BOX + JARQUE-BERA DIAGNOSTICS
# ============================================================

def run_ljungbox_jb_table(df, columns, lags=(6, 12)):
    """Ljung-Box autocorrelation test + Jarque-Bera normality test for each series."""
    rows = []
    for col in columns:
        s = df[col].dropna()
        if len(s) < 20:
            continue

        lb = acorr_ljungbox(s, lags=list(lags), return_df=True)
        jb_stat, jb_p = stats.jarque_bera(s.values)

        row = {"variable": col}
        for lag in lags:
            row[f"LB({lag})_stat"] = round(float(lb.loc[lag, "lb_stat"]), 2)
            row[f"LB({lag})_pval"] = round(float(lb.loc[lag, "lb_pvalue"]), 4)
        row["JB_stat"] = round(float(jb_stat), 2)
        row["JB_pval"] = float(jb_p)
        row["normal_5pct"] = jb_p > 0.05
        rows.append(row)

    return pd.DataFrame(rows)


def run_diagnostic_residuals(resid, lags=(6, 12)):
    """ACF + Ljung-Box on residuals — used after every model."""
    resid = np.asarray(resid).flatten()
    resid = resid[np.isfinite(resid)]

    lb = acorr_ljungbox(resid, lags=list(lags), return_df=True)
    acf_values = acf(resid, nlags=max(lags), fft=True)

    rows = []
    for lag in lags:
        rows.append({
            "lag": lag,
            "ACF": round(float(acf_values[lag]), 4),
            "LB_stat": round(float(lb.loc[lag, "lb_stat"]), 2),
            "LB_pval": round(float(lb.loc[lag, "lb_pvalue"]), 4),
            "autocorrelated_5pct": lb.loc[lag, "lb_pvalue"] < 0.05,
        })

    return pd.DataFrame(rows)


# ============================================================
# [NEW] OLS FROM SCRATCH (Class 3 methodology)
# ============================================================

def ols_from_scratch(y, X, x_names=None):
    """
    OLS with intercept, implemented by hand: beta = (X'X)^-1 X'y.
    Returns dict with beta, se, t_stat, p_value, resid, fitted, R2.
    """
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[mask], X[mask]

    T = len(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    k = X.shape[1]

    Xc = np.column_stack([np.ones(T), X])
    XtX_inv = np.linalg.inv(Xc.T @ Xc)
    beta = XtX_inv @ (Xc.T @ y)

    y_hat = Xc @ beta
    resid = y - y_hat

    SSR = resid @ resid
    SST = (y - y.mean()) @ (y - y.mean())
    R2 = 1.0 - SSR / SST

    sigma2 = SSR / (T - k - 1)
    var_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(var_beta))
    t_stat = beta / se
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=T - k - 1))

    names = ["const"] + (x_names if x_names else [f"x{i}" for i in range(k)])

    return {
        "beta": beta, "se": se, "t_stat": t_stat, "p_value": p_value,
        "resid": resid, "fitted": y_hat, "R2": R2, "sigma2": sigma2,
        "T": T, "k": k + 1, "names": names, "y": y, "X": Xc,
    }


def ols_summary_df(res):
    """Format OLS from scratch results as a clean DataFrame."""
    return pd.DataFrame({
        "coef": res["beta"],
        "std_err": res["se"],
        "t_stat": res["t_stat"],
        "p_value": res["p_value"],
    }, index=res["names"]).round(4)


# ============================================================
# [NEW] MA(1) MLE WITH RECURSIVE INNOVATIONS (Class 3-4)
# ============================================================

def ma_innovations(u, theta):
    """Recursive MA(1) inversion: eps_t = u_t - theta * eps_{t-1}."""
    T = len(u)
    eps = np.zeros(T)
    eps[0] = u[0]
    for t in range(1, T):
        eps[t] = u[t] - theta * eps[t - 1]
    return eps


def _loglik_contributions(params, y, X):
    """Individual log-likelihood contributions for OPG."""
    k = X.shape[1]
    beta = params[:k]
    theta = params[k]

    u = y - X @ beta
    eps = ma_innovations(u, theta)
    sigma2 = np.mean(eps ** 2)

    if sigma2 <= 0:
        return np.full(len(y), -1e10)

    return -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * eps ** 2 / sigma2


def _negloglik_concentrated(params, y, X):
    """Concentrated negative log-likelihood for regression + MA(1)."""
    ll = _loglik_contributions(params, y, X)
    return -np.sum(ll)


def opg_standard_errors(params, y, X):
    """OPG standard errors from numerical score derivatives (Class 4)."""
    T = len(y)
    n_params = len(params)
    scores = np.zeros((T, n_params))
    h = 1e-5

    for j in range(n_params):
        p_up = params.copy(); p_up[j] += h
        p_dn = params.copy(); p_dn[j] -= h
        scores[:, j] = (_loglik_contributions(p_up, y, X) - _loglik_contributions(p_dn, y, X)) / (2 * h)

    OPG = scores.T @ scores
    try:
        se = np.sqrt(np.diag(np.linalg.inv(OPG)))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)

    return se


def estimate_ma1(y, X, x_names=None):
    """
    Estimate regression with MA(1) errors by MLE.
    Returns dict with beta, theta, se (OPG), innovations, sigma.
    """
    ols_res = ols_from_scratch(y, X, x_names)
    y_clean, X_clean = ols_res["y"], ols_res["X"]

    x0 = np.concatenate([ols_res["beta"], [0.0]])

    result = minimize(
        _negloglik_concentrated, x0, args=(y_clean, X_clean),
        method="Nelder-Mead", options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-10},
    )

    params = result.x
    se = opg_standard_errors(params, y_clean, X_clean)

    k = X_clean.shape[1]
    beta = params[:k]
    theta = params[k]

    u = y_clean - X_clean @ beta
    innovations = ma_innovations(u, theta)
    sigma = np.sqrt(np.mean(innovations ** 2))

    names = ols_res["names"] + ["theta_MA1"]

    return {
        "beta": beta, "theta": theta, "se": se, "sigma": sigma,
        "innovations": innovations, "residuals_ols": ols_res["resid"],
        "params": params, "names": names,
        "ols_beta": ols_res["beta"], "ols_se": ols_res["se"],
        "T": ols_res["T"], "R2_ols": ols_res["R2"],
    }


def ma1_comparison_table(ma1_res):
    """Side-by-side OLS vs MLE table."""
    names = ma1_res["names"]
    n_beta = len(ma1_res["ols_beta"])

    rows = []
    for i in range(n_beta):
        rows.append({
            "parameter": names[i],
            "OLS_coef": round(float(ma1_res["ols_beta"][i]), 4),
            "OLS_se": round(float(ma1_res["ols_se"][i]), 4),
            "MLE_coef": round(float(ma1_res["beta"][i]), 4),
            "MLE_se": round(float(ma1_res["se"][i]), 4),
        })
    rows.append({
        "parameter": "theta_MA1",
        "OLS_coef": np.nan, "OLS_se": np.nan,
        "MLE_coef": round(float(ma1_res["theta"]), 4),
        "MLE_se": round(float(ma1_res["se"][-1]), 4),
    })

    return pd.DataFrame(rows)


# ============================================================
# OIL SHOCK DECOMPOSITION
# ============================================================

def decompose_oil_returns(df, oil_return_col="wti_return", activity_col="cfnai", prefix="baseline"):
    sample = df[[oil_return_col, activity_col]].dropna().copy()
    X = sm.add_constant(sample[activity_col])
    model = sm.OLS(sample[oil_return_col], X).fit()

    fitted = pd.Series(index=sample.index, data=model.fittedvalues, name=f"{prefix}_oil_demand_component")
    residual = pd.Series(index=sample.index, data=model.resid, name=f"{prefix}_oil_supply_component")

    out = df.copy()
    out[fitted.name] = fitted
    out[residual.name] = residual
    return out, model


def decompose_oil_returns_scratch(df, oil_return_col="wti_return", activity_col="cfnai", prefix="baseline"):
    """[NEW] Same decomposition but using OLS from scratch."""
    sample = df[[oil_return_col, activity_col]].dropna().copy()

    y = sample[oil_return_col].values
    X = sample[activity_col].values.reshape(-1, 1)

    res = ols_from_scratch(y, X, [activity_col])

    out = df.copy()
    out[f"{prefix}_oil_demand_component"] = np.nan
    out[f"{prefix}_oil_supply_component"] = np.nan
    out.loc[sample.index, f"{prefix}_oil_demand_component"] = res["fitted"]
    out.loc[sample.index, f"{prefix}_oil_supply_component"] = res["resid"]

    return out, res


def add_regime_variables(df, oil_col="wti_return", ism_col="ism_mfg"):
    out = df.copy()
    out["expansion"] = np.where(out[ism_col] > 50, 1, 0)
    out["contraction"] = np.where(out[ism_col] <= 50, 1, 0)
    out["oil_expansion"] = out[oil_col] * out["expansion"]
    out["oil_contraction"] = out[oil_col] * out["contraction"]
    return out


# ============================================================
# PREDICTIVE REGRESSIONS
# ============================================================

def fit_predictive_regression(df, dependent_col, predictor_cols, horizon=1, cov_type="HC1"):
    work = df.copy()
    work["y_next"] = work[dependent_col].shift(-horizon)
    if dependent_col in predictor_cols:
        work = work.rename(columns={dependent_col: f"{dependent_col}_current"})
        predictor_cols = [
            f"{dependent_col}_current" if col == dependent_col else col for col in predictor_cols
        ]

    regression_columns = ["y_next"] + predictor_cols
    sample = work[regression_columns].dropna().copy()

    y = sample["y_next"]
    X = sm.add_constant(sample[predictor_cols])
    model = sm.OLS(y, X).fit(cov_type=cov_type)
    return model, sample


def regression_results_table(model):
    table = pd.DataFrame({
        "coef": model.params, "std_err": model.bse,
        "t_stat": model.tvalues, "p_value": model.pvalues,
    })
    return table.round(4)


def interpret_two_component_model(model, demand_name, supply_name):
    lines = []
    for variable, label in [(demand_name, "Demand-related oil component"), (supply_name, "Supply-related oil component")]:
        if variable not in model.params.index:
            continue
        coef = model.params[variable]
        p_value = model.pvalues[variable]
        sign = "positive" if coef > 0 else "negative"
        strength = "statistically significant" if p_value < 0.05 else "not statistically significant"
        lines.append(f"{label}: coefficient is {sign} ({coef:.4f}) and {strength} at the 5% level (p-value = {p_value:.4f}).")
    return lines


# ============================================================
# [NEW] ROLLING PREDICTIVE COEFFICIENTS (Class 3)
# ============================================================

def rolling_predictive_coefficients(df, dependent_col, predictor_cols, window=60, horizon=1):
    """Estimate predictive regression on a rolling window. Returns rolling betas + SE."""
    work = df.copy()
    work["y_next"] = work[dependent_col].shift(-horizon)

    cols_needed = ["date", "y_next"] + predictor_cols
    work = work[cols_needed].dropna().reset_index(drop=True)

    results = []
    for i in range(window, len(work)):
        train = work.iloc[i - window:i]
        y = train["y_next"].values
        X = sm.add_constant(train[predictor_cols].values)

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ (X.T @ y)
            resid = y - X @ beta
            sigma2 = (resid @ resid) / (len(y) - X.shape[1])
            se = np.sqrt(np.diag(sigma2 * XtX_inv))
        except np.linalg.LinAlgError:
            continue

        row = {"date": work.iloc[i]["date"]}
        all_names = ["const"] + predictor_cols
        for j, name in enumerate(all_names):
            row[f"beta_{name}"] = beta[j]
            row[f"se_{name}"] = se[j]
        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# GRANGER CAUSALITY (kept from original)
# ============================================================

def granger_pvalue_table(df, cause_col, effect_col, max_lag=3):
    sample = df[[effect_col, cause_col]].dropna().copy()
    results = grangercausalitytests(sample[[effect_col, cause_col]], maxlag=max_lag, verbose=False)

    rows = []
    for lag in range(1, max_lag + 1):
        p_value = results[lag][0]["ssr_ftest"][1]
        rows.append({"lag": lag, "p_value": round(p_value, 4)})

    out = pd.DataFrame(rows)
    out["cause"] = cause_col
    out["effect"] = effect_col
    return out[["cause", "effect", "lag", "p_value"]]


# ============================================================
# [NEW] GEWEKE CAUSALITY MEASURES (Class 6)
# ============================================================

def geweke_causality(df, var1, var2, lag=2):
    """
    Compute the four Geweke (1982) causality measures between two variables.
    Returns DataFrame with measure, statistic, p-value.
    """
    sample = df[[var1, var2]].dropna()

    model = VAR(sample)
    full_results = model.fit(lag)
    residuals = full_results.resid.values
    T = residuals.shape[0]
    Sigma = np.cov(residuals.T)
    Sigma11 = Sigma[0, 0]
    Sigma22 = Sigma[1, 1]
    det_Sigma = np.linalg.det(Sigma)

    # Auxiliary model 1: var1 on its own lags only
    Y1 = sample[var1].values
    X1_list = [pd.Series(Y1).shift(i).values for i in range(1, lag + 1)]
    X1 = np.column_stack(X1_list)[lag:]
    Y1_dep = Y1[lag:]
    mask1 = np.all(np.isfinite(X1), axis=1) & np.isfinite(Y1_dep)
    X1, Y1_dep = X1[mask1], Y1_dep[mask1]
    beta1 = np.linalg.inv(X1.T @ X1) @ (X1.T @ Y1_dep)
    Sigma1 = np.var(Y1_dep - X1 @ beta1)

    # Auxiliary model 2: var2 on its own lags only
    Y2 = sample[var2].values
    X2_list = [pd.Series(Y2).shift(i).values for i in range(1, lag + 1)]
    X2 = np.column_stack(X2_list)[lag:]
    Y2_dep = Y2[lag:]
    mask2 = np.all(np.isfinite(X2), axis=1) & np.isfinite(Y2_dep)
    X2, Y2_dep = X2[mask2], Y2_dep[mask2]
    beta2 = np.linalg.inv(X2.T @ X2) @ (X2.T @ Y2_dep)
    Sigma2 = np.var(Y2_dep - X2 @ beta2)

    # Geweke measures
    C_21 = np.log(Sigma1 / Sigma11)
    C_12 = np.log(Sigma2 / Sigma22)
    C_inst = np.log((Sigma11 * Sigma22) / det_Sigma)
    C_total = C_21 + C_12 + C_inst

    df_gc = lag
    results_list = [
        {"measure": f"{var2} -> {var1}", "C": C_21,
         "statistic": T * C_21, "df": df_gc,
         "p_value": 1.0 - chi2_dist.cdf(max(T * C_21, 0), df_gc)},
        {"measure": f"{var1} -> {var2}", "C": C_12,
         "statistic": T * C_12, "df": df_gc,
         "p_value": 1.0 - chi2_dist.cdf(max(T * C_12, 0), df_gc)},
        {"measure": "Instantaneous", "C": C_inst,
         "statistic": T * C_inst, "df": 1,
         "p_value": 1.0 - chi2_dist.cdf(max(T * C_inst, 0), 1)},
        {"measure": "Total", "C": C_total,
         "statistic": T * C_total, "df": 2 * lag + 1,
         "p_value": 1.0 - chi2_dist.cdf(max(T * C_total, 0), 2 * lag + 1)},
    ]

    out = pd.DataFrame(results_list)
    for col in ["C", "statistic", "p_value"]:
        out[col] = out[col].round(4)
    return out


# ============================================================
# VAR ANALYSIS
# ============================================================

def choose_var_lag(var_df, maxlags=6):
    model = VAR(var_df)
    lag_selection = model.select_order(maxlags=maxlags)
    lag_table = pd.DataFrame([lag_selection.selected_orders]).rename(index={0: "selected_lag"})
    return lag_selection, lag_table


def fit_var_model(var_df, chosen_lag):
    model = VAR(var_df)
    results = model.fit(chosen_lag)
    return results


# ============================================================
# FORECASTING
# ============================================================

def historical_mean_forecast(train_series):
    return train_series.mean()


def rolling_forecast_comparison(df, target_col, raw_oil_col, demand_col, supply_col, controls, start_share=0.6):
    work = df.copy()
    work["target_next"] = work[target_col].shift(-1)
    required_cols = ["date", "target_next", target_col, raw_oil_col, demand_col, supply_col] + controls
    work = work[required_cols].dropna().copy()

    start_index = max(int(len(work) * start_share), 36)
    forecasts = []

    for i in range(start_index, len(work)):
        train = work.iloc[:i].copy()
        test = work.iloc[i:i + 1].copy()
        if len(test) == 0:
            continue

        benchmark_forecast = historical_mean_forecast(train["target_next"])

        raw_predictors = [target_col, raw_oil_col] + controls
        X_raw = sm.add_constant(train[raw_predictors])
        raw_model = sm.OLS(train["target_next"], X_raw).fit()
        raw_forecast = raw_model.predict(sm.add_constant(test[raw_predictors], has_constant="add")).iloc[0]

        decomposition_predictors = [target_col, demand_col, supply_col] + controls
        X_decomp = sm.add_constant(train[decomposition_predictors])
        decomp_model = sm.OLS(train["target_next"], X_decomp).fit()
        decomp_forecast = decomp_model.predict(
            sm.add_constant(test[decomposition_predictors], has_constant="add")
        ).iloc[0]

        forecasts.append({
            "date": test["date"].iloc[0],
            "actual": test["target_next"].iloc[0],
            "benchmark": benchmark_forecast,
            "raw_oil_model": raw_forecast,
            "decomposition_model": decomp_forecast,
        })

    forecast_df = pd.DataFrame(forecasts)
    metrics = []
    for model_name in ["benchmark", "raw_oil_model", "decomposition_model"]:
        errors = forecast_df[model_name] - forecast_df["actual"]
        metrics.append({
            "model": model_name,
            "rmse": np.sqrt(np.mean(errors ** 2)),
            "mae": np.mean(np.abs(errors)),
        })

    metrics_df = pd.DataFrame(metrics).round(4)
    return forecast_df, metrics_df


def split_sample(df, split_date):
    early = df[df["date"] < split_date].copy()
    late = df[df["date"] >= split_date].copy()
    return early, late
