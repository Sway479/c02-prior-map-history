#!/usr/bin/env python3
"""Numerically stable patient-clustered logistic regression for C02.

The manuscript-facing association models are unpenalized logistic regressions
with patient-clustered sandwich standard errors.  This helper standardizes the
design only for optimization, transforms coefficients and covariance back to
the requested clinical units, applies the same HC1-style finite-sample
correction in every analysis, and fails closed on non-convergence or non-finite
results.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


def fit_clustered_logit(
    frame: pd.DataFrame,
    outcome: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    terms: list[str],
    *,
    maxiter: int = 5000,
) -> pd.DataFrame:
    """Fit an unpenalized logit and return aggregate robust inference.

    Missing predictors are median-imputed before fitting, matching the paper's
    interpretable association models.  Standardization is an internal numeric
    device only; returned coefficients use the original columns and units.
    """

    if not terms or len(set(terms)) != len(terms):
        raise ValueError("terms must be a non-empty unique list")
    missing_columns = [term for term in terms if term not in frame.columns]
    if missing_columns:
        raise KeyError(f"missing model terms: {missing_columns}")

    x = frame[terms].apply(pd.to_numeric, errors="coerce").copy()
    missing_counts = x.isna().sum().astype(int)
    for column in terms:
        median = float(x[column].median())
        if not math.isfinite(median):
            raise RuntimeError(f"all-missing predictor: {column}")
        x[column] = x[column].fillna(median)

    matrix_raw = x.to_numpy(float)
    y = np.asarray(outcome, dtype=float)
    cluster = np.asarray(groups)
    if len(matrix_raw) != len(y) or len(y) != len(cluster):
        raise ValueError("predictor, outcome and cluster lengths differ")
    if not np.isfinite(matrix_raw).all():
        raise RuntimeError("non-finite predictor after median imputation")
    if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise RuntimeError("outcome must be finite and binary")
    if pd.isna(cluster).any():
        raise RuntimeError("missing patient cluster")
    if np.unique(y).size != 2:
        raise RuntimeError("both outcome classes are required")

    means = matrix_raw.mean(axis=0)
    scales = matrix_raw.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (matrix_raw - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        # NumPy 2.0 linked to macOS Accelerate can emit spurious floating-point
        # matmul warnings even for finite operands.  Suppress those warnings,
        # then enforce explicit finite checks on every returned quantity.
        with np.errstate(all="ignore"):
            eta = design @ beta
            probability = expit(eta)
            loss = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
            gradient = design.T @ (probability - y)
        if not math.isfinite(loss) or not np.isfinite(gradient).all():
            return float("inf"), np.full_like(beta, 1e100)
        return loss, gradient

    initial = np.zeros(design.shape[1], dtype=float)
    initial[0] = math.log(float(y.mean()) / float(1.0 - y.mean()))
    fitted = minimize(
        lambda beta: objective(beta)[0],
        initial,
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    gradient_max = float(np.max(np.abs(objective(fitted.x)[1])))
    if (
        not fitted.success
        or not np.isfinite(fitted.x).all()
        or gradient_max > 1e-3
    ):
        raise RuntimeError(
            "clustered logit failed to converge: "
            f"success={fitted.success}, gradient={gradient_max:.3g}, "
            f"message={fitted.message}"
        )

    with np.errstate(all="ignore"):
        eta = design @ fitted.x
        probability = expit(eta)
        weight = probability * (1.0 - probability)
        bread = design.T @ (design * weight[:, None])
    if not np.isfinite(eta).all() or not np.isfinite(bread).all():
        raise RuntimeError("non-finite fitted design quantities")
    if np.any(probability <= 0) or np.any(probability >= 1):
        raise RuntimeError("degenerate fitted probability")
    # macOS Accelerate can emit floating-point RuntimeWarnings from pinv even
    # when both its input and output are finite. Suppress only that operation;
    # the explicit finite checks on both sides remain fail-closed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            bread_inverse = np.linalg.pinv(bread, rcond=1e-12)
    if not np.isfinite(bread_inverse).all():
        raise RuntimeError("non-finite inverse information matrix")

    scores = design * (y - probability)[:, None]
    score_frame = pd.DataFrame(scores)
    score_frame.insert(0, "cluster", cluster)
    cluster_scores = score_frame.groupby("cluster", sort=False).sum().to_numpy()
    cluster_count = len(cluster_scores)
    n, k = design.shape
    if cluster_count <= 1 or n <= k:
        raise RuntimeError("insufficient observations or clusters")
    correction = (cluster_count / (cluster_count - 1)) * ((n - 1) / (n - k))
    with np.errstate(all="ignore"):
        covariance_standardized = (
            correction
            * bread_inverse
            @ (cluster_scores.T @ cluster_scores)
            @ bread_inverse
        )
    if not np.isfinite(covariance_standardized).all():
        raise RuntimeError("non-finite clustered covariance")

    # Transform [intercept, standardized coefficients] to coefficients for the
    # original model matrix [intercept, raw clinical-unit predictors].
    transform = np.zeros((k, k), dtype=float)
    transform[0, 0] = 1.0
    transform[0, 1:] = -means / scales
    transform[1:, 1:] = np.diag(1.0 / scales)
    with np.errstate(all="ignore"):
        beta = transform @ fitted.x
        covariance = transform @ covariance_standardized @ transform.T
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    names = ["intercept", *terms]

    rows: list[dict[str, object]] = []
    for position, name in enumerate(names):
        estimate = float(beta[position])
        se = float(standard_error[position])
        if not math.isfinite(estimate) or not math.isfinite(se) or se <= 0:
            raise RuntimeError(f"invalid coefficient inference for {name}")
        rows.append(
            {
                "term": name,
                "log_odds": estimate,
                "cluster_se": se,
                "odds_ratio": math.exp(estimate),
                "ci_low": math.exp(estimate - 1.959963984540054 * se),
                "ci_high": math.exp(estimate + 1.959963984540054 * se),
                "p_value": float(2 * norm.sf(abs(estimate / se))),
                "n": int(n),
                "events": int(y.sum()),
                "clusters": int(cluster_count),
                "missing_before_median_imputation": (
                    0 if name == "intercept" else int(missing_counts[name])
                ),
                "fit_converged": bool(fitted.success),
                "fit_iterations": int(fitted.nit),
                "gradient_max_abs": gradient_max,
            }
        )
    return pd.DataFrame(rows)
