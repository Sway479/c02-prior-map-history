#!/usr/bin/env python3
"""Test whether prior MAP response marks fixed-opportunity hypotension burden.

MOVER operations are anchored to actual administered IV hypnotic time.  The
next 30 minutes are represented by six non-overlapping five-minute bins.  No
interpolation or carry-forward is used.  Aggregate analysis outputs stay in the
paper workspace; operation-level metrics remain in the restricted directory.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from mover_c02_procedure_family import procedure_family


from c02_runtime import private_workspace_root, protect_file, secure_directory


ROOT = private_workspace_root()
MAP_PATH = ROOT / "data/restricted/mover/extracted/mover_cleaned_full_intraop_map.csv.gz"
ANCHOR_PAIR = ROOT / "data/restricted/mover/extracted/mover_c02_hypnotic_anchored_pair.csv.gz"
RELATIVE_PAIR = ROOT / "data/restricted/mover/extracted/mover_c02_relative_hypnotic_pair.csv.gz"
RESTRICTED_METRICS = ROOT / "data/restricted/mover/extracted/mover_c02_posthypnotic_burden.csv.gz"
RESTRICTED_RELATIVE_METRICS = ROOT / (
    "data/restricted/mover/extracted/mover_c02_posthypnotic_relative_burden.csv.gz"
)
OUT = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild/"
    "post_hypnotic_burden"
)
SEED = 20260817


def wilson(events: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = events / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def load_pair_with_anchor() -> tuple[pd.DataFrame, dict]:
    columns = [
        "LOG_ID", "prior_LOG_ID", "patient_id", "age_years", "bmi_kg_m2",
        "asa_numeric", "sex_common", "patient_class_common", "procedure_common",
        "interval_days", "weight_kg", "current_baseline_MAP", "prior_relative_drop",
        "prior_relative_drop_20", "current_anchor_agent", "current_anchor_propofol_mg",
        "current_anchor_etomidate_mg", "current_anchor_ketamine_mg",
    ]
    pair = pd.read_csv(RELATIVE_PAIR, usecols=columns, dtype={"LOG_ID": str, "prior_LOG_ID": str})
    anchors = pd.read_csv(
        ANCHOR_PAIR,
        usecols=["LOG_ID", "current_anchor_rel"],
        dtype={"LOG_ID": str},
    )
    anchor_conflicts = int(anchors.groupby("LOG_ID").current_anchor_rel.nunique().gt(1).sum())
    if anchor_conflicts:
        raise RuntimeError("conflicting hypnotic anchors")
    anchors = anchors.drop_duplicates("LOG_ID")
    pair = pair.merge(anchors, on="LOG_ID", how="left", validate="one_to_one")
    missing_anchor = int(pair.current_anchor_rel.isna().sum())
    if missing_anchor:
        raise RuntimeError("relative pair is missing a current hypnotic anchor")
    audit = {
        "relative_pairs": int(len(pair)),
        "patients": int(pair.patient_id.nunique()),
        "anchor_conflicts": anchor_conflicts,
        "missing_current_anchor": missing_anchor,
    }
    return pair, audit


def clean_post_anchor_rows(pair: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ids = set(pair.LOG_ID.astype(str))
    maps = pd.read_csv(
        MAP_PATH,
        usecols=["LOG_ID", "RECORDED_TIME", "relative_min", "value", "modality_hint"],
        dtype={"LOG_ID": str},
        low_memory=False,
    )
    raw_rows = len(maps)
    maps = maps.loc[maps.LOG_ID.isin(ids)].copy()
    selected_operation_rows = len(maps)
    anchor = pair[["LOG_ID", "current_anchor_rel"]]
    maps = maps.merge(anchor, on="LOG_ID", how="inner", validate="many_to_one")
    maps["relative_min"] = pd.to_numeric(maps.relative_min, errors="coerce")
    maps["value"] = pd.to_numeric(maps.value, errors="coerce")
    maps["time"] = pd.to_datetime(maps.RECORDED_TIME, errors="coerce")
    maps["after_anchor_min"] = maps.relative_min - maps.current_anchor_rel
    maps = maps.loc[
        maps.after_anchor_min.ge(0) & maps.after_anchor_min.lt(30)
        & maps.value.between(20, 200, inclusive="both")
        & maps.time.notna()
        & maps.modality_hint.isin(["ART", "NIBP"])
    ].copy()
    in_window_rows = len(maps)
    keys = (
        maps.groupby(["LOG_ID", "modality_hint", "time"], as_index=False, observed=True)
        .agg(
            after_anchor_min=("after_anchor_min", "min"),
            value=("value", "first"),
            distinct_values=("value", "nunique"),
            duplicate_rows=("value", "size"),
        )
    )
    conflict_keys = int(keys.distinct_values.gt(1).sum())
    exact_duplicate_rows = int((keys.duplicate_rows - 1).clip(lower=0).sum())
    keys = keys.loc[keys.distinct_values.eq(1)].copy()
    keys["bin"] = np.floor(keys.after_anchor_min / 5.0).astype(int)
    audit = {
        "full_map_rows": int(raw_rows),
        "rows_for_current_pair_operations": int(selected_operation_rows),
        "valid_rows_in_0_30_after_anchor": int(in_window_rows),
        "same_time_modality_conflict_keys_excluded": conflict_keys,
        "exact_duplicate_rows_collapsed": exact_duplicate_rows,
        "operations_with_any_valid_row": int(keys.LOG_ID.nunique()),
    }
    return keys, audit


def operation_burden(keys: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_modality = (
        keys.groupby(["LOG_ID", "bin", "modality_hint"], as_index=False, observed=True)
        .agg(map_value=("value", "median"), records=("value", "size"))
    )
    primary_rows = []
    for (log_id, bin_index), frame in per_modality.groupby(["LOG_ID", "bin"], sort=False):
        art = frame.loc[frame.modality_hint.eq("ART")]
        chosen = art.iloc[0] if len(art) else frame.loc[frame.modality_hint.eq("NIBP")].iloc[0]
        primary_rows.append(
            {
                "LOG_ID": log_id,
                "bin": int(bin_index),
                "map_value": float(chosen.map_value),
                "modality_used": str(chosen.modality_hint),
                "records": int(chosen.records),
            }
        )
    primary = pd.DataFrame(primary_rows)

    metric_rows = []
    for rule in ["ART_priority", "ART_only", "NIBP_only"]:
        if rule == "ART_priority":
            source = primary.copy()
        else:
            modality = rule.replace("_only", "")
            source = per_modality.loc[per_modality.modality_hint.eq(modality)].rename(
                columns={"modality_hint": "modality_used"}
            )
        for log_id, frame in source.groupby("LOG_ID", sort=False, observed=True):
            frame = frame.loc[frame.bin.between(0, 5)].sort_values("bin")
            complete = len(frame) == 6 and frame.bin.nunique() == 6
            values = frame.map_value.to_numpy(float)
            low_bins = int((values < 65).sum())
            metric_rows.append(
                {
                    "LOG_ID": str(log_id),
                    "rule": rule,
                    "observed_bins": int(frame.bin.nunique()),
                    "complete_six_bins": bool(complete),
                    "low_bins": low_bins,
                    "low_minutes": float(5 * low_bins),
                    "any_low": int(low_bins >= 1),
                    "low_at_least_10min": int(low_bins >= 2),
                    "deficit_burden_mmhg_min": float(5 * np.maximum(65 - values, 0).sum()),
                    "nadir_map": float(np.min(values)) if len(values) else math.nan,
                    "mean_map": float(np.mean(values)) if len(values) else math.nan,
                    "art_bins": int(frame.modality_used.astype(str).eq("ART").sum()),
                    "nibp_bins": int(frame.modality_used.astype(str).eq("NIBP").sum()),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    return metrics, primary


def standardized_mean_difference(included: pd.Series, excluded: pd.Series) -> float:
    a = pd.to_numeric(included, errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(excluded, errors="coerce").dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return math.nan
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def selection_profile(pair: pd.DataFrame, primary_metrics: pd.DataFrame) -> pd.DataFrame:
    complete_ids = set(
        primary_metrics.loc[primary_metrics.complete_six_bins, "LOG_ID"].astype(str)
    )
    d = pair.copy()
    d["included"] = d.LOG_ID.astype(str).isin(complete_ids)
    d["female"] = d.sex_common.astype(str).eq("F").astype(int)
    d["inpatient"] = d.patient_class_common.astype(str).eq("Inpatient").astype(int)
    d["nonpropofol"] = ~d.current_anchor_agent.astype(str).eq("propofol")
    variables = {
        "age_years": "Age, years",
        "bmi_kg_m2": "BMI, kg/m2",
        "asa_numeric": "ASA class",
        "current_baseline_MAP": "Current pre-hypnotic MAP",
        "prior_relative_drop": "Prior relative MAP decline",
        "interval_days": "Prior-to-current interval, days",
        "female": "Female sex",
        "inpatient": "Inpatient",
        "nonpropofol": "Current non-propofol hypnotic",
    }
    rows = []
    for column, label in variables.items():
        included = d.loc[d.included, column]
        excluded = d.loc[~d.included, column]
        rows.append(
            {
                "variable": label,
                "included_n_nonmissing": int(pd.to_numeric(included, errors="coerce").notna().sum()),
                "excluded_n_nonmissing": int(pd.to_numeric(excluded, errors="coerce").notna().sum()),
                "included_mean": float(pd.to_numeric(included, errors="coerce").mean()),
                "excluded_mean": float(pd.to_numeric(excluded, errors="coerce").mean()),
                "standardized_mean_difference": standardized_mean_difference(included, excluded),
            }
        )
    return pd.DataFrame(rows)


def build_design(d: pd.DataFrame) -> tuple[np.ndarray, list[str], dict]:
    x = pd.DataFrame(index=d.index)
    x["intercept"] = 1.0
    x["prior_drop_per_10pp"] = pd.to_numeric(d.prior_relative_drop, errors="coerce") / 0.10
    x["current_baseline_per_10"] = pd.to_numeric(d.current_baseline_MAP, errors="coerce") / 10.0
    x["age_per_10"] = pd.to_numeric(d.age_years, errors="coerce") / 10.0
    x["bmi_per_5"] = pd.to_numeric(d.bmi_kg_m2, errors="coerce") / 5.0
    x["asa"] = pd.to_numeric(d.asa_numeric, errors="coerce")
    x["log_interval"] = np.log1p(pd.to_numeric(d.interval_days, errors="coerce").clip(lower=0))
    x["female"] = d.sex_common.astype(str).eq("F").astype(float)
    x["inpatient"] = d.patient_class_common.astype(str).eq("Inpatient").astype(float)
    x["propofol_per_100mg"] = pd.to_numeric(d.current_anchor_propofol_mg, errors="coerce").fillna(0) / 100
    x["etomidate_any"] = pd.to_numeric(d.current_anchor_etomidate_mg, errors="coerce").fillna(0).gt(0).astype(float)
    x["ketamine_any"] = pd.to_numeric(d.current_anchor_ketamine_mg, errors="coerce").fillna(0).gt(0).astype(float)
    family = d.procedure_common.map(procedure_family).fillna("missing")
    family = family.where(family.map(family.value_counts()).ge(30), "Other")
    x = pd.concat([x, pd.get_dummies(family, prefix="family", drop_first=True, dtype=float)], axis=1)
    imputation = {}
    for column in x.columns:
        value = pd.to_numeric(x[column], errors="coerce")
        if value.isna().any():
            median = float(value.median())
            imputation[column] = median
            x[column] = value.fillna(median)
    constant = [column for column in x.columns if x[column].nunique(dropna=False) <= 1 and column != "intercept"]
    x = x.drop(columns=constant)
    return x.to_numpy(float), list(x.columns), {"median_imputation": imputation, "dropped_constants": constant}


def fit_cluster_logistic(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.einsum("ij,j->i", x, beta)
        probability = expit(eta)
        loss = float(np.logaddexp(0, eta).sum() - (y * eta).sum())
        gradient = np.einsum("ij,i->j", x, probability - y)
        return loss, gradient

    fit = minimize(lambda beta: objective(beta), np.zeros(x.shape[1]), jac=True,
                   method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-8})
    loss, gradient = objective(fit.x)
    if not fit.success and np.max(np.abs(gradient)) > 1e-5:
        raise RuntimeError(f"logistic fit failed: {fit.message}")
    probability = expit(np.einsum("ij,j->i", x, fit.x))
    weight = probability * (1 - probability)
    information = np.einsum("ij,i,ik->jk", x, weight, x)
    if not np.isfinite(information).all():
        raise RuntimeError("non-finite logistic information matrix")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            bread = np.linalg.pinv(information)
    if not np.isfinite(bread).all():
        raise RuntimeError("non-finite logistic inverse information matrix")
    scores = x * (y - probability)[:, None]
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in pd.unique(groups):
        score = scores[groups == group].sum(axis=0)
        meat += np.outer(score, score)
    with np.errstate(all="ignore"):
        covariance = bread @ meat @ bread
    if not np.isfinite(covariance).all():
        raise RuntimeError("non-finite logistic clustered covariance")
    return fit.x, covariance, {
        "converged": bool(fit.success or np.max(np.abs(gradient)) <= 1e-5),
        "max_abs_gradient": float(np.max(np.abs(gradient))),
        "log_likelihood": float(-loss),
    }


def fit_cluster_linear(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
    if not np.isfinite(beta).all():
        raise RuntimeError("non-finite linear coefficient")
    residual = y - np.einsum("ij,j->i", x, beta)
    information = x.T @ x
    if not np.isfinite(information).all():
        raise RuntimeError("non-finite linear information matrix")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            bread = np.linalg.pinv(information)
    if not np.isfinite(bread).all():
        raise RuntimeError("non-finite linear inverse information matrix")
    meat = np.zeros((x.shape[1], x.shape[1]))
    scores = x * residual[:, None]
    for group in pd.unique(groups):
        score = scores[groups == group].sum(axis=0)
        meat += np.outer(score, score)
    with np.errstate(all="ignore"):
        covariance = bread @ meat @ bread
    if not np.isfinite(covariance).all():
        raise RuntimeError("non-finite linear clustered covariance")
    return beta, covariance, {"residual_sd": float(np.std(residual, ddof=x.shape[1]))}


def association_analysis(
    d: pd.DataFrame,
    binary_column: str = "low_at_least_10min",
    burden_column: str = "deficit_burden_mmhg_min",
    binary_label: str = "At least 10 fixed-bin minutes with MAP<65",
    burden_label: str = "log(1 + MAP deficit burden, mmHg*min)",
    burden_scale: str = "ratio of geometric mean (1 + burden)",
) -> pd.DataFrame:
    x, names, design_audit = build_design(d)
    exposure = names.index("prior_drop_per_10pp")
    groups = d.patient_id.astype(str).to_numpy()
    rows = []
    y_binary = d[binary_column].to_numpy(int)
    beta, covariance, audit = fit_cluster_logistic(x, y_binary, groups)
    se = math.sqrt(max(covariance[exposure, exposure], 0))
    b = beta[exposure]
    rows.append(
        {
            "outcome": binary_label,
            "model": "cluster-robust logistic",
            "effect_unit": "per 10-percentage-point larger prior relative MAP decline",
            "estimate": float(math.exp(b)),
            "ci_low": float(math.exp(b - 1.96 * se)),
            "ci_high": float(math.exp(b + 1.96 * se)),
            "p_value": float(2 * norm.sf(abs(b / se))),
            "scale": "odds ratio",
            "n": int(len(d)),
            "events": int(y_binary.sum()),
        }
    )
    x_shifted = x.copy()
    x_shifted[:, exposure] += 1.0
    p0 = expit(np.einsum("ij,j->i", x, beta))
    p1 = expit(np.einsum("ij,j->i", x_shifted, beta))
    risk_difference = float(np.mean(p1 - p0))
    gradient = np.mean(
        x_shifted * (p1 * (1 - p1))[:, None]
        - x * (p0 * (1 - p0))[:, None],
        axis=0,
    )
    rd_se = float(np.sqrt(max(gradient @ covariance @ gradient, 0.0)))
    rows.append(
        {
            "outcome": binary_label,
            "model": "marginal standardization from cluster-robust logistic",
            "effect_unit": "per 10-percentage-point larger prior relative MAP decline",
            "estimate": float(100 * risk_difference),
            "ci_low": float(100 * (risk_difference - 1.96 * rd_se)),
            "ci_high": float(100 * (risk_difference + 1.96 * rd_se)),
            "p_value": float(2 * norm.sf(abs(risk_difference / rd_se))),
            "scale": "risk difference, percentage points",
            "n": int(len(d)),
            "events": int(y_binary.sum()),
        }
    )
    y_burden = np.log1p(d[burden_column].to_numpy(float))
    beta_l, covariance_l, audit_l = fit_cluster_linear(x, y_burden, groups)
    se_l = math.sqrt(max(covariance_l[exposure, exposure], 0))
    b_l = beta_l[exposure]
    rows.append(
        {
            "outcome": burden_label,
            "model": "cluster-robust linear",
            "effect_unit": "per 10-percentage-point larger prior relative MAP decline",
            "estimate": float(math.exp(b_l)),
            "ci_low": float(math.exp(b_l - 1.96 * se_l)),
            "ci_high": float(math.exp(b_l + 1.96 * se_l)),
            "p_value": float(2 * norm.sf(abs(b_l / se_l))),
            "scale": burden_scale,
            "n": int(len(d)),
            "events": int((d[burden_column] > 0).sum()),
        }
    )
    return pd.DataFrame(rows)


def build_relative_nibp_sustained(pair: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    nibp = keys.loc[keys.modality_hint.eq("NIBP")].copy()
    bins = (
        nibp.groupby(["LOG_ID", "bin"], as_index=False, observed=True)
        .agg(map_value=("value", "median"))
    )
    bins = bins.merge(
        pair[["LOG_ID", "current_baseline_MAP"]], on="LOG_ID", how="inner", validate="many_to_one"
    )
    bins["baseline"] = pd.to_numeric(bins.current_baseline_MAP, errors="coerce")
    bins = bins.loc[bins.baseline.gt(0)].copy()
    bins["relative_ratio"] = bins.map_value / bins.baseline
    bins["relative_low"] = bins.relative_ratio.lt(0.80)
    bins["relative_deficit"] = np.maximum(0.80 - bins.relative_ratio, 0)
    rows = []
    for log_id, frame in bins.groupby("LOG_ID", sort=False, observed=True):
        frame = frame.loc[frame.bin.between(0, 5)]
        if len(frame) != 6 or frame.bin.nunique() != 6:
            continue
        low_bins = int(frame.relative_low.sum())
        rows.append(
            {
                "LOG_ID": str(log_id),
                "relative_low_bins": low_bins,
                "relative_low_at_least_10min": int(low_bins >= 2),
                "relative_any_low": int(low_bins >= 1),
                "relative_deficit_burden_pct_min": float(
                    100 * 5 * frame.relative_deficit.sum()
                ),
            }
        )
    metric = pd.DataFrame(rows)
    return pair.merge(metric, on="LOG_ID", how="inner", validate="one_to_one")


def prior_band_table(d: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 0.10, 0.20, 0.30, np.inf]
    labels = ["<10%", "10–19%", "20–29%", "≥30%"]
    d = d.copy()
    d["prior_drop_band"] = pd.cut(d.prior_relative_drop, bins=bins, labels=labels, right=False)
    rows = []
    for band, frame in d.groupby("prior_drop_band", observed=False):
        n = len(frame)
        events = int(frame.low_at_least_10min.sum())
        lo, hi = wilson(events, n)
        rows.append(
            {
                "prior_drop_band": str(band),
                "n": int(n),
                "any_low_n": int(frame.any_low.sum()),
                "any_low_rate": float(frame.any_low.mean()),
                "low_at_least_10min_n": events,
                "low_at_least_10min_rate": float(events / n),
                "low_at_least_10min_ci_low": lo,
                "low_at_least_10min_ci_high": hi,
                "mean_low_minutes": float(frame.low_minutes.mean()),
                "median_low_minutes": float(frame.low_minutes.median()),
                "mean_deficit_burden": float(frame.deficit_burden_mmhg_min.mean()),
                "median_deficit_burden": float(frame.deficit_burden_mmhg_min.median()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_band_burden(d: pd.DataFrame, reps: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    work = d[["patient_id", "prior_relative_drop", "deficit_burden_mmhg_min"]].copy()
    bins = [-np.inf, 0.10, 0.20, 0.30, np.inf]
    labels = ["<10%", "10–19%", "20–29%", "≥30%"]
    work["prior_drop_band"] = pd.cut(
        work.prior_relative_drop, bins=bins, labels=labels, right=False
    )
    patients = pd.Index(pd.unique(work.patient_id.astype(str)))
    grouped = (
        work.groupby([work.patient_id.astype(str), "prior_drop_band"], observed=False)
        .deficit_burden_mmhg_min.agg(["sum", "count"])
        .reset_index()
    )
    sum_matrix = (
        grouped.pivot(index="patient_id", columns="prior_drop_band", values="sum")
        .reindex(index=patients, columns=labels, fill_value=0).fillna(0).to_numpy(float)
    )
    count_matrix = (
        grouped.pivot(index="patient_id", columns="prior_drop_band", values="count")
        .reindex(index=patients, columns=labels, fill_value=0).fillna(0).to_numpy(float)
    )
    rows = []
    for rep in range(reps):
        sampled = rng.integers(0, len(patients), size=len(patients))
        sums = sum_matrix[sampled].sum(axis=0)
        counts = count_matrix[sampled].sum(axis=0)
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        for label, mean in zip(labels, means):
            rows.append(
                {"rep": rep, "prior_drop_band": label, "mean_deficit_burden": float(mean)}
            )
    return pd.DataFrame(rows)


def make_figure(
    bands: pd.DataFrame,
    bootstrap: pd.DataFrame | None,
    n: int,
    events: int,
    burden_ci_summary: pd.DataFrame | None = None,
) -> None:
    order = ["<10%", "10–19%", "20–29%", "≥30%"]
    bands = bands.set_index("prior_drop_band").loc[order].reset_index()
    if burden_ci_summary is None:
        if bootstrap is None:
            raise ValueError("bootstrap or burden_ci_summary is required")
        burden_ci = (
            bootstrap.groupby("prior_drop_band", observed=True).mean_deficit_burden
            .quantile([0.025, 0.975]).unstack(level=1)
        )
    else:
        burden_ci = (
            burden_ci_summary.set_index("prior_drop_band")
            .rename(
                columns={
                    "mean_burden_ci_low": 0.025,
                    "mean_burden_ci_high": 0.975,
                }
            )[[0.025, 0.975]]
        )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    x = np.arange(len(order))
    rate = bands.low_at_least_10min_rate.to_numpy(float)
    axes[0].errorbar(
        x, rate,
        yerr=[rate - bands.low_at_least_10min_ci_low, bands.low_at_least_10min_ci_high - rate],
        fmt="s", color="#D79B26", ecolor="#596273", capsize=3.5, markersize=7,
    )
    axes[0].set_xticks(x, order)
    axes[0].set_ylim(bottom=0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[0].set_ylabel("Current operations with ≥10 low minutes")
    axes[0].set_title("Sustained low-MAP proxy")
    for xi, value, count in zip(x, rate, bands.n):
        axes[0].text(xi, value + 0.012, f"{100*value:.1f}%\nn={int(count):,}", ha="center", fontsize=8.5)

    mean = bands.mean_deficit_burden.to_numpy(float)
    lo = np.array([burden_ci.loc[label, 0.025] for label in order], dtype=float)
    hi = np.array([burden_ci.loc[label, 0.975] for label in order], dtype=float)
    axes[1].errorbar(
        x, mean, yerr=[mean - lo, hi - mean], fmt="o", color="#2E5EAA",
        ecolor="#596273", capsize=3.5, markersize=7,
    )
    axes[1].set_xticks(x, order)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("Mean MAP deficit burden (mmHg·min)")
    axes[1].set_title("Thirty-minute deficit burden")
    for xi, value in zip(x, mean):
        axes[1].text(xi, value + max(mean) * 0.04, f"{value:.1f}", ha="center", fontsize=8.5)
    for ax in axes:
        ax.set_xlabel("Prior relative MAP decline")
        ax.grid(axis="y", color="#E1E5EA", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Post-hypnotic hypotension burden by prior response band")
    fig.text(
        0.5, 0.01,
        f"MOVER fixed six-bin cohort: {n:,} pairs; {events:,} with ≥10 low minutes; ART-priority without interpolation",
        ha="center", fontsize=9.2, color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUT / "fig_c02_post_hypnotic_burden.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "fig_c02_post_hypnotic_burden.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    secure_directory(OUT)
    secure_directory(RESTRICTED_METRICS.parent)
    pair, pair_audit = load_pair_with_anchor()
    keys, map_audit = clean_post_anchor_rows(pair)
    metrics, primary_bins = operation_burden(keys)
    art_observed_ids = set(keys.loc[keys.modality_hint.eq("ART"), "LOG_ID"].astype(str))
    nibp_no_art = metrics.loc[
        metrics.rule.eq("NIBP_only") & ~metrics.LOG_ID.astype(str).isin(art_observed_ids)
    ].copy()
    nibp_no_art["rule"] = "NIBP_no_ART"
    metrics = pd.concat([metrics, nibp_no_art], ignore_index=True)
    metrics.to_csv(RESTRICTED_METRICS, index=False, compression="gzip")
    protect_file(RESTRICTED_METRICS)

    coverage_rows = []
    for rule, frame in metrics.groupby("rule", observed=True):
        complete = frame.loc[frame.complete_six_bins]
        coverage_rows.append(
            {
                "rule": rule,
                "operations_with_any_bins": int(frame.LOG_ID.nunique()),
                "operations_complete_six_bins": int(len(complete)),
                "coverage_of_relative_pair": float(len(complete) / len(pair)),
                "any_low_events": int(complete.any_low.sum()),
                "at_least_10_low_minutes_events": int(complete.low_at_least_10min.sum()),
                "median_low_minutes": float(complete.low_minutes.median()) if len(complete) else math.nan,
                "mean_deficit_burden": float(complete.deficit_burden_mmhg_min.mean()) if len(complete) else math.nan,
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT / "coverage_by_modality_rule.csv", index=False)
    primary = metrics.loc[metrics.rule.eq("ART_priority") & metrics.complete_six_bins].copy()
    selection = selection_profile(pair, metrics.loc[metrics.rule.eq("ART_priority")])
    selection.to_csv(OUT / "included_vs_excluded_smd.csv", index=False)

    gate_n = len(primary) >= 1500
    gate_events = int(primary.any_low.sum()) >= 200
    if not (gate_n and gate_events):
        summary = {
            "status": "STOP_FIXED_OPPORTUNITY_BURDEN_COVERAGE_GATE",
            "pair_audit": pair_audit,
            "map_audit": map_audit,
            "coverage": coverage.to_dict(orient="records"),
            "gates": {"complete_primary_ge1500": gate_n, "nonzero_burden_ge200": gate_events},
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return

    analyses: dict[str, pd.DataFrame] = {}
    association_frames = []
    for rule in ["ART_priority", "NIBP_only", "NIBP_no_ART"]:
        operation = metrics.loc[metrics.rule.eq(rule) & metrics.complete_six_bins].drop(columns=["rule"])
        frame = pair.merge(operation, on="LOG_ID", how="inner", validate="one_to_one")
        analyses[rule] = frame
        result = association_analysis(frame)
        result.insert(0, "measurement_rule", rule)
        association_frames.append(result)
    relative_nibp_analysis = build_relative_nibp_sustained(pair, keys)
    relative_nibp_analysis[
        ["LOG_ID", "relative_low_bins", "relative_low_at_least_10min",
         "relative_any_low", "relative_deficit_burden_pct_min"]
    ].to_csv(RESTRICTED_RELATIVE_METRICS, index=False, compression="gzip")
    protect_file(RESTRICTED_RELATIVE_METRICS)
    relative_result = association_analysis(
        relative_nibp_analysis,
        binary_column="relative_low_at_least_10min",
        burden_column="relative_deficit_burden_pct_min",
        binary_label="At least 10 fixed-bin minutes below 80% of pre-hypnotic MAP",
        burden_label="log(1 + relative deficit burden below 80% baseline, percentage-point*min)",
        burden_scale="ratio of geometric mean (1 + relative deficit burden)",
    )
    relative_result.insert(0, "measurement_rule", "NIBP_relative_80pct_baseline")
    association_frames.append(relative_result)
    analysis = analyses["ART_priority"]
    associations = pd.concat(association_frames, ignore_index=True)
    associations.to_csv(OUT / "association_results.csv", index=False)
    bands = prior_band_table(analysis)
    bands.to_csv(OUT / "risk_by_prior_response_band.csv", index=False)
    bootstrap = bootstrap_band_burden(analysis, reps=500)
    burden_ci = (
        bootstrap.groupby("prior_drop_band", observed=True).mean_deficit_burden
        .quantile([0.025, 0.975]).unstack(level=1).reset_index()
        .rename(columns={0.025: "mean_burden_ci_low", 0.975: "mean_burden_ci_high"})
    )
    burden_ci.to_csv(OUT / "mean_burden_cluster_bootstrap_ci.csv", index=False)
    make_figure(bands, bootstrap, len(analysis), int(analysis.low_at_least_10min.sum()))

    max_abs_smd = float(selection.standardized_mean_difference.abs().max())
    art = coverage.loc[coverage.rule.eq("ART_only")].iloc[0]
    nibp = coverage.loc[coverage.rule.eq("NIBP_only")].iloc[0]
    nibp_no_art_cov = coverage.loc[coverage.rule.eq("NIBP_no_ART")].iloc[0]
    art_only_sufficient = bool(
        art.operations_complete_six_bins >= 300 and art.at_least_10_low_minutes_events >= 50
    )
    nibp_sensitivities_sufficient = bool(
        nibp.operations_complete_six_bins >= 300
        and nibp.at_least_10_low_minutes_events >= 50
        and nibp_no_art_cov.operations_complete_six_bins >= 300
        and nibp_no_art_cov.at_least_10_low_minutes_events >= 50
    )
    primary_results = associations.loc[associations.measurement_rule.eq("ART_priority")]
    nibp_results = associations.loc[associations.measurement_rule.eq("NIBP_only")]
    no_art_results = associations.loc[associations.measurement_rule.eq("NIBP_no_ART")]
    relative_results = associations.loc[
        associations.measurement_rule.eq("NIBP_relative_80pct_baseline")
    ]

    def all_effects_positive(frame: pd.DataFrame) -> bool:
        ratios = frame.loc[~frame.scale.str.contains("risk difference")]
        differences = frame.loc[frame.scale.str.contains("risk difference")]
        return bool((ratios.ci_low > 1).all() and (differences.ci_low > 0).all())

    if max_abs_smd > 0.50:
        decision = "SUPPLEMENTARY_ONLY_OBSERVATION_PROCESS_LIMITED"
    elif (
        nibp_sensitivities_sufficient
        and all_effects_positive(primary_results)
        and all_effects_positive(nibp_results)
        and all_effects_positive(no_art_results)
    ):
        decision = "KEEP_AS_MOVER_CLINICAL_BURDEN_DEEPENING_ART_TRANSPORT_UNRESOLVED"
    else:
        decision = "DO_NOT_PROMOTE_BURDEN_ASSOCIATION"
    summary = {
        "status": decision,
        "question": "Does prior medication-timed relative MAP decline mark fixed-opportunity current 30-minute hypotension burden?",
        "pair_audit": pair_audit,
        "map_audit": map_audit,
        "coverage": coverage.to_dict(orient="records"),
        "gates": {
            "complete_primary_ge1500": gate_n,
            "nonzero_burden_ge200": gate_events,
            "max_abs_included_excluded_smd": max_abs_smd,
            "nibp_only_sensitivity_sufficient": nibp_sensitivities_sufficient,
            "art_only_sensitivity_sufficient": art_only_sufficient,
        },
        "primary_cohort": {
            "pairs": int(len(analysis)),
            "patients": int(analysis.patient_id.nunique()),
            "any_low": int(analysis.any_low.sum()),
            "at_least_10_low_minutes": int(analysis.low_at_least_10min.sum()),
            "median_low_minutes": float(analysis.low_minutes.median()),
            "mean_deficit_burden": float(analysis.deficit_burden_mmhg_min.mean()),
        },
        "relative_nibp_sustained_cohort": {
            "pairs": int(len(relative_nibp_analysis)),
            "patients": int(relative_nibp_analysis.patient_id.nunique()),
            "any_below_80pct_baseline": int(relative_nibp_analysis.relative_any_low.sum()),
            "at_least_10min_below_80pct_baseline": int(
                relative_nibp_analysis.relative_low_at_least_10min.sum()
            ),
            "association_direction_consistent": bool(
                (relative_results.loc[~relative_results.scale.str.contains("risk difference"), "ci_low"] > 1).all()
                and (relative_results.loc[relative_results.scale.str.contains("risk difference"), "ci_low"] > 0).all()
            ),
        },
        "associations": associations.to_dict(orient="records"),
        "claim_boundary": (
            "Fixed-bin recorded burden under ART-priority measurement, replicated in complete NIBP-only and "
            "NIBP-without-any-ART cohorts if their gates pass. It is not continuous waveform burden, does not "
            "establish organ injury or treatment benefit, and cannot be transported to ART-only cases because "
            "the complete ART-only cohort is too small."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
