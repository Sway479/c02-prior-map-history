#!/usr/bin/env python3
"""Deepen C02 positive results without adding black-box model complexity.

Questions:
1. Which component of the prior response carries information?
2. Does continuous history improve fixed-capacity risk selection in MOVER?
3. Does an older anaesthetic add beyond the immediate prior anaesthetic?
4. Can simple local recalibration repair fixed INSPIRE probabilities?

Only aggregate results are written. Restricted MOVER identifiers and rows stay
inside the in-memory analysis.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from run_mover_c02_external_validation import (
    COMMON_FEATURES,
    EXPANDED_FEATURES,
    calibration,
    canonicalize,
    make_pipeline,
)
from c02_cluster_logit import fit_clustered_logit


from c02_runtime import private_workspace_root


ROOT = private_workspace_root()
BASE = ROOT / "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild"
INSPIRE = BASE / "first_two_sampling_bridge/first_two_pair_cohort.csv.gz"
MOVER = ROOT / "data/restricted/mover/extracted/mover_c02_cleaned_pair_cohort.csv.gz"
FIXED_MODELS = BASE / "cross_database_minimal_bridge"
OUT = BASE / "deepening_v1"
SEED = 20260813
BOOTSTRAP_REPS = 1000


MOVER_BASE = EXPANDED_FEATURES["M1_prior_context_and_alert"]
MOVER_MODELS = {
    "M1_binary_alert_context": MOVER_BASE,
    "M1_plus_prior_level": MOVER_BASE + ["prior_first_map"],
    "M1_plus_prior_change": MOVER_BASE + ["prior_first2_change"],
    "M2_level_plus_change": MOVER_BASE + ["prior_first_map", "prior_first2_change"],
    "M2_level_plus_relative_change": MOVER_BASE + [
        "prior_first_map", "prior_first2_percent_change"
    ],
    "M2_plus_nonlinearity": MOVER_BASE + [
        "prior_first_map", "prior_first2_change", "prior_first_map_sq", "prior_first2_change_sq"
    ],
    "M2_plus_interaction": MOVER_BASE + [
        "prior_first_map", "prior_first2_change", "prior_level_x_change"
    ],
}

INSPIRE_BASE = [
    "age_years", "bmi_kg_m2", "asa_numeric", "sex_common", "interval_log1p",
    "prior_first2_any_low",
]
INSPIRE_MODELS = {
    "M1_binary_alert_context": INSPIRE_BASE,
    "M1_plus_prior_level": INSPIRE_BASE + ["prior_first_map"],
    "M1_plus_prior_change": INSPIRE_BASE + ["prior_first2_change"],
    "M2_level_plus_change": INSPIRE_BASE + ["prior_first_map", "prior_first2_change"],
    "M2_level_plus_relative_change": INSPIRE_BASE + [
        "prior_first_map", "prior_first2_percent_change"
    ],
    "M2_plus_nonlinearity": INSPIRE_BASE + [
        "prior_first_map", "prior_first2_change", "prior_first_map_sq", "prior_first2_change_sq"
    ],
    "M2_plus_interaction": INSPIRE_BASE + [
        "prior_first_map", "prior_first2_change", "prior_level_x_change"
    ],
}


def add_response_terms(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    # Constants are fixed for numerical conditioning only; folds do not learn them.
    level = pd.to_numeric(d["prior_first_map"], errors="coerce")
    change = pd.to_numeric(d["prior_first2_change"], errors="coerce")
    d["prior_first_map_sq"] = ((level - 95.0) / 10.0) ** 2
    d["prior_first2_change_sq"] = (change / 10.0) ** 2
    d["prior_level_x_change"] = ((level - 95.0) / 10.0) * (change / 10.0)
    d["prior_first2_percent_change"] = 100.0 * change / level.where(level > 0)
    return d


def canonicalize_inspire(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["age_years"] = pd.to_numeric(d["age"], errors="coerce")
    d["bmi_kg_m2"] = pd.to_numeric(d["bmi"], errors="coerce")
    d["asa_numeric"] = pd.to_numeric(d["asa"], errors="coerce")
    d["sex_common"] = d["sex"].astype("string").str.upper().replace(
        {"MALE": "M", "FEMALE": "F", "1": "M", "2": "F"}
    )
    d.loc[~d["sex_common"].isin(["M", "F"]), "sex_common"] = "<MISSING>"
    d["interval_log1p"] = np.log1p(pd.to_numeric(d["interval_days"], errors="coerce").clip(lower=0))
    d["prior_first_map"] = pd.to_numeric(d["prior_first2_map_0"], errors="coerce")
    d["prior_first2_change"] = (
        pd.to_numeric(d["prior_first2_map_1"], errors="coerce") - d["prior_first_map"]
    )
    d["prior_first2_any_low"] = (
        d[["prior_first2_map_0", "prior_first2_map_1"]].min(axis=1) < 65
    ).astype(int)
    d["target"] = pd.to_numeric(d["target_any_low"], errors="coerce").astype(int)
    # Match MOVER's recorded General-to-General target population.
    d = d.loc[
        d["antype"].astype("string").str.strip().eq("General")
        & d["prior_antype"].astype("string").str.strip().eq("General")
    ].copy()
    return add_response_terms(d)


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
        "mean_prediction": float(np.mean(prediction)),
    }


def oof_predictions(
    d: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    specs: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    predictions = {name: np.full(len(d), np.nan) for name in specs}
    splits = list(GroupKFold(n_splits=5).split(d, y, groups=groups))
    for train, test in splits:
        for name, features in specs.items():
            model = make_pipeline(features)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model.fit(d.iloc[train][features], y[train])
                predictions[name][test] = model.predict_proba(d.iloc[test][features])[:, 1]
    if any(not np.all(np.isfinite(p)) for p in predictions.values()):
        raise RuntimeError("non-finite OOF prediction")
    return predictions, splits


def model_table(
    centre: str,
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    reference: str,
) -> pd.DataFrame:
    reference_metric = metric(y, predictions[reference])
    rows = []
    for name, prediction in predictions.items():
        current = metric(y, prediction)
        rows.append(
            {
                "centre": centre,
                "model": name,
                "n": len(y),
                "events": int(y.sum()),
                **current,
                "delta_auroc_vs_M1": current["auroc"] - reference_metric["auroc"],
                "delta_average_precision_vs_M1": current["average_precision"] - reference_metric["average_precision"],
                "brier_improvement_vs_M1": reference_metric["brier"] - current["brier"],
                "log_loss_improvement_vs_M1": reference_metric["log_loss"] - current["log_loss"],
            }
        )
    return pd.DataFrame(rows)


def group_bootstrap_contrasts(
    centre: str,
    y: np.ndarray,
    groups: np.ndarray,
    predictions: dict[str, np.ndarray],
    reference: str,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_groups = pd.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED + (0 if centre == "INSPIRE" else 10000))
    rows = []
    for rep in range(reps):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([lookup[group] for group in sampled])
        yy = y[index]
        if np.unique(yy).size < 2:
            continue
        base = metric(yy, predictions[reference][index])
        for name, prediction in predictions.items():
            if name == reference:
                continue
            current = metric(yy, prediction[index])
            rows.append(
                {
                    "centre": centre,
                    "rep": rep,
                    "model": name,
                    "delta_auroc_vs_M1": current["auroc"] - base["auroc"],
                    "delta_average_precision_vs_M1": current["average_precision"] - base["average_precision"],
                    "brier_improvement_vs_M1": base["brier"] - current["brier"],
                    "log_loss_improvement_vs_M1": base["log_loss"] - current["log_loss"],
                }
            )
    boot = pd.DataFrame(rows)
    summary_rows = []
    for (current_centre, name), frame in boot.groupby(["centre", "model"], sort=False):
        row = {"centre": current_centre, "model": name, "bootstrap_reps": frame["rep"].nunique()}
        for column in [
            "delta_auroc_vs_M1", "delta_average_precision_vs_M1",
            "brier_improvement_vs_M1", "log_loss_improvement_vs_M1",
        ]:
            row[column + "_ci_low"] = float(frame[column].quantile(.025))
            row[column + "_ci_high"] = float(frame[column].quantile(.975))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), boot


def group_bootstrap_pairwise(
    centre: str,
    y: np.ndarray,
    groups: np.ndarray,
    predictions: dict[str, np.ndarray],
    comparisons: list[tuple[str, str]],
    reps: int = BOOTSTRAP_REPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Directly test the few mechanistically important nested comparisons."""
    unique_groups = pd.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED + (50000 if centre == "INSPIRE" else 60000))
    rows = []
    for rep in range(reps):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([lookup[group] for group in sampled])
        yy = y[index]
        if np.unique(yy).size < 2:
            continue
        for candidate, reference in comparisons:
            current = metric(yy, predictions[candidate][index])
            base = metric(yy, predictions[reference][index])
            rows.append(
                {
                    "centre": centre,
                    "rep": rep,
                    "candidate": candidate,
                    "reference": reference,
                    "delta_auroc": current["auroc"] - base["auroc"],
                    "delta_average_precision": current["average_precision"] - base["average_precision"],
                    "brier_improvement": base["brier"] - current["brier"],
                    "log_loss_improvement": base["log_loss"] - current["log_loss"],
                }
            )
    boot = pd.DataFrame(rows)
    summaries = []
    for keys, frame in boot.groupby(["centre", "candidate", "reference"], sort=False):
        row = dict(zip(["centre", "candidate", "reference"], keys))
        row["bootstrap_reps"] = int(frame["rep"].nunique())
        for column in [
            "delta_auroc", "delta_average_precision", "brier_improvement", "log_loss_improvement"
        ]:
            row[column + "_point"] = float(
                metric(y, predictions[keys[1]])[column.replace("delta_", "").replace("_improvement", "")]
                - metric(y, predictions[keys[2]])[column.replace("delta_", "").replace("_improvement", "")]
            ) if column in {"delta_auroc", "delta_average_precision"} else float(
                metric(y, predictions[keys[2]])[column.replace("_improvement", "")]
                - metric(y, predictions[keys[1]])[column.replace("_improvement", "")]
            )
            row[column + "_ci_low"] = float(frame[column].quantile(.025))
            row[column + "_ci_high"] = float(frame[column].quantile(.975))
        summaries.append(row)
    return pd.DataFrame(summaries), boot


def fixed_capacity(
    d: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    capacities = [.05, .10, .20, .30]

    def one(capacity: float, index: np.ndarray) -> dict:
        yy, p1, p2 = y[index], m1[index], m2[index]
        k = max(1, int(math.ceil(capacity * len(index))))
        top1 = np.argsort(-p1, kind="stable")[:k]
        top2 = np.argsort(-p2, kind="stable")[:k]
        set1, set2 = set(top1.tolist()), set(top2.tolist())
        events_total = int(yy.sum())
        events1 = int(yy[top1].sum())
        events2 = int(yy[top2].sum())
        entered = np.array(sorted(set2 - set1), dtype=int)
        exited = np.array(sorted(set1 - set2), dtype=int)
        return {
            "capacity": capacity,
            "n": len(index),
            "selected_n": k,
            "events_total": events_total,
            "M1_events_captured": events1,
            "M2_events_captured": events2,
            "capture_M1": events1 / events_total,
            "capture_M2": events2 / events_total,
            "capture_improvement": (events2 - events1) / events_total,
            "ppv_M1": events1 / k,
            "ppv_M2": events2 / k,
            "ppv_improvement": (events2 - events1) / k,
            "selected_overlap": len(set1 & set2) / k,
            "entered_n": len(entered),
            "exited_n": len(exited),
            "entered_events": int(yy[entered].sum()) if len(entered) else 0,
            "exited_events": int(yy[exited].sum()) if len(exited) else 0,
        }

    point = pd.DataFrame([one(capacity, np.arange(len(y))) for capacity in capacities])
    unique_groups = pd.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED + 20000)
    rows = []
    for rep in range(BOOTSTRAP_REPS):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([lookup[group] for group in sampled])
        if y[index].sum() == 0:
            continue
        for capacity in capacities:
            rows.append({"rep": rep, **one(capacity, index)})
    boot = pd.DataFrame(rows)
    for column in ["capture_improvement", "ppv_improvement", "selected_overlap"]:
        intervals = boot.groupby("capacity")[column].quantile([.025, .975]).unstack()
        point[column + "_ci_low"] = point["capacity"].map(intervals[.025])
        point[column + "_ci_high"] = point["capacity"].map(intervals[.975])
    return point, boot


def reclassification_profiles(
    d: pd.DataFrame,
    y: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
) -> pd.DataFrame:
    """Describe, without identifiers, which patients continuous history moves."""
    rows = []
    for capacity in [.05, .10, .20, .30]:
        k = max(1, int(math.ceil(capacity * len(d))))
        selected_m1 = np.zeros(len(d), dtype=bool)
        selected_m2 = np.zeros(len(d), dtype=bool)
        selected_m1[np.argsort(-m1, kind="stable")[:k]] = True
        selected_m2[np.argsort(-m2, kind="stable")[:k]] = True
        memberships = {
            "entered_with_continuous_history": ~selected_m1 & selected_m2,
            "exited_with_continuous_history": selected_m1 & ~selected_m2,
            "retained_high_risk": selected_m1 & selected_m2,
            "not_selected": ~selected_m1 & ~selected_m2,
        }
        for label, mask in memberships.items():
            frame = d.loc[mask]
            rows.append(
                {
                    "capacity": capacity,
                    "group": label,
                    "n": int(mask.sum()),
                    "events": int(y[mask].sum()),
                    "event_rate": float(y[mask].mean()) if mask.any() else math.nan,
                    "prior_binary_alert_rate": float(frame["prior_first2_any_low"].mean()) if len(frame) else math.nan,
                    "prior_first_map_mean": float(frame["prior_first_map"].mean()) if len(frame) else math.nan,
                    "prior_first2_change_mean": float(frame["prior_first2_change"].mean()) if len(frame) else math.nan,
                    "current_ASA_mean": float(frame["asa_numeric"].mean()) if len(frame) else math.nan,
                    "interval_days_median": float(np.expm1(frame["interval_log1p"]).median()) if len(frame) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def clinical_risk_surface(
    centre: str,
    d: pd.DataFrame,
    y: np.ndarray,
) -> pd.DataFrame:
    """A directly readable two-dimensional risk surface for prior response."""
    level = pd.cut(
        d["prior_first_map"],
        bins=[-np.inf, 80, 90, 100, 110, np.inf],
        right=False,
        labels=["<80", "80–89", "90–99", "100–109", "≥110"],
    )
    change = pd.cut(
        d["prior_first2_change"],
        bins=[-np.inf, -10, -5, 5, np.inf],
        right=False,
        labels=["≤−10", "−9 to −5", "−4 to +4", "≥+5"],
    )
    work = pd.DataFrame({"level_band": level, "change_band": change, "target": y})
    rows = []
    for (change_band, level_band), frame in work.groupby(
        ["change_band", "level_band"], observed=False
    ):
        rows.append(
            {
                "centre": centre,
                "prior_change_band": str(change_band),
                "prior_level_band": str(level_band),
                "n": int(len(frame)),
                "events": int(frame["target"].sum()),
                "event_rate": float(frame["target"].mean()) if len(frame) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def harmonized_cluster_associations(
    centre: str,
    d: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    """Two-centre, common-covariate associations with patient-cluster SEs."""
    common = pd.DataFrame(
        {
            "age_per_10y": pd.to_numeric(d["age_years"], errors="coerce") / 10,
            "bmi_per_5": pd.to_numeric(d["bmi_kg_m2"], errors="coerce") / 5,
            "asa_class": pd.to_numeric(d["asa_numeric"], errors="coerce"),
            "male": d["sex_common"].astype(str).str.upper().eq("M").astype(float),
            "log1p_interval_days": pd.to_numeric(d["interval_log1p"], errors="coerce"),
            "prior_binary_alert": pd.to_numeric(d["prior_first2_any_low"], errors="coerce"),
            "prior_first_MAP_per_10mmHg": pd.to_numeric(d["prior_first_map"], errors="coerce") / 10,
            "prior_MAP_change_per_10mmHg": pd.to_numeric(d["prior_first2_change"], errors="coerce") / 10,
            "prior_percent_change_per_10pp": pd.to_numeric(
                d["prior_first2_percent_change"], errors="coerce"
            ) / 10,
        }
    )
    for column in common:
        common[column] = common[column].fillna(common[column].median())

    specifications = {
        "absolute_change_model": [
            "age_per_10y", "bmi_per_5", "asa_class", "male", "log1p_interval_days",
            "prior_binary_alert", "prior_first_MAP_per_10mmHg", "prior_MAP_change_per_10mmHg",
        ],
        "relative_change_model": [
            "age_per_10y", "bmi_per_5", "asa_class", "male", "log1p_interval_days",
            "prior_binary_alert", "prior_first_MAP_per_10mmHg", "prior_percent_change_per_10pp",
        ],
    }
    rows = []
    for specification, columns in specifications.items():
        fitted = fit_clustered_logit(common, y, groups, columns)
        fitted = fitted.set_index("term")
        for term in [
            "prior_binary_alert",
            "prior_first_MAP_per_10mmHg",
            "prior_MAP_change_per_10mmHg",
            "prior_percent_change_per_10pp",
        ]:
            if term not in fitted.index:
                continue
            result = fitted.loc[term]
            rows.append(
                {
                    "centre": centre,
                    "specification": specification,
                    "term": term,
                    "odds_ratio": float(result.odds_ratio),
                    "ci_low": float(result.ci_low),
                    "ci_high": float(result.ci_high),
                    "p_value_cluster_robust": float(result.p_value),
                    "n": int(len(y)),
                    "events": int(y.sum()),
                    "patients": int(result.clusters),
                    "fit_iterations": int(result.fit_iterations),
                    "gradient_max_abs": float(result.gradient_max_abs),
                }
            )
    return pd.DataFrame(rows)


def build_history_chains_mover(d: pd.DataFrame) -> pd.DataFrame:
    lookup = d[["patient_id", "LOG_ID", "prior_first_map", "prior_first2_change"]].rename(
        columns={
            "LOG_ID": "prior_LOG_ID",
            "prior_first_map": "older_first_map",
            "prior_first2_change": "older_first2_change",
        }
    )
    if lookup.duplicated(["patient_id", "prior_LOG_ID"]).any():
        raise RuntimeError("non-unique MOVER history lookup")
    chain = d.merge(lookup, on=["patient_id", "prior_LOG_ID"], how="inner", validate="many_to_one")
    chain["older_first2_any_low"] = (chain["older_first_map"] < 65).astype(int)
    chain["two_history_mean_level"] = (chain["prior_first_map"] + chain["older_first_map"]) / 2
    chain["two_history_mean_change"] = (
        chain["prior_first2_change"] + chain["older_first2_change"]
    ) / 2
    return chain


def build_history_chains_inspire(d: pd.DataFrame) -> pd.DataFrame:
    lookup = d[[
        "subject_id", "current_anstart_time", "prior_first_map", "prior_first2_change"
    ]].rename(
        columns={
            "current_anstart_time": "prior_anstart_time",
            "prior_first_map": "older_first_map",
            "prior_first2_change": "older_first2_change",
        }
    )
    if lookup.duplicated(["subject_id", "prior_anstart_time"]).any():
        raise RuntimeError("non-unique INSPIRE history lookup")
    chain = d.merge(lookup, on=["subject_id", "prior_anstart_time"], how="inner", validate="many_to_one")
    chain["older_first2_any_low"] = (chain["older_first_map"] < 65).astype(int)
    chain["two_history_mean_level"] = (chain["prior_first_map"] + chain["older_first_map"]) / 2
    chain["two_history_mean_change"] = (
        chain["prior_first2_change"] + chain["older_first2_change"]
    ) / 2
    return chain


def history_depth(
    centre: str,
    d: pd.DataFrame,
    y_column: str,
    group_column: str,
    base_features: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    specs = {
        "H0_context_alert": base_features,
        "H1_immediate_prior": base_features + ["prior_first_map", "prior_first2_change"],
        "H1_older_prior": base_features + ["older_first_map", "older_first2_change"],
        "H2_immediate_plus_older": base_features + [
            "prior_first_map", "prior_first2_change", "older_first_map", "older_first2_change"
        ],
        "H2_two_history_mean": base_features + ["two_history_mean_level", "two_history_mean_change"],
    }
    y = d[y_column].to_numpy(int)
    groups = d[group_column].to_numpy()
    predictions, _ = oof_predictions(d, y, groups, specs)
    table = model_table(centre, y, predictions, "H0_context_alert")
    immediate = metric(y, predictions["H1_immediate_prior"])
    for row in table.index:
        current = metric(y, predictions[table.loc[row, "model"]])
        table.loc[row, "delta_auroc_vs_immediate"] = current["auroc"] - immediate["auroc"]
        table.loc[row, "brier_improvement_vs_immediate"] = immediate["brier"] - current["brier"]
        table.loc[row, "log_loss_improvement_vs_immediate"] = immediate["log_loss"] - current["log_loss"]
    return table, predictions


def history_bootstrap(
    centre: str,
    d: pd.DataFrame,
    y_column: str,
    group_column: str,
    predictions: dict[str, np.ndarray],
    reps: int = BOOTSTRAP_REPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = d[y_column].to_numpy(int)
    groups = d[group_column].to_numpy()
    unique_groups = pd.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED + (30000 if centre == "INSPIRE" else 40000))
    candidates = [
        "H1_older_prior", "H2_immediate_plus_older", "H2_two_history_mean"
    ]
    rows = []
    for rep in range(reps):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([lookup[group] for group in sampled])
        yy = y[index]
        if np.unique(yy).size < 2:
            continue
        immediate = metric(yy, predictions["H1_immediate_prior"][index])
        for candidate in candidates:
            current = metric(yy, predictions[candidate][index])
            rows.append(
                {
                    "centre": centre,
                    "rep": rep,
                    "candidate": candidate,
                    "delta_auroc_vs_immediate": current["auroc"] - immediate["auroc"],
                    "delta_average_precision_vs_immediate": (
                        current["average_precision"] - immediate["average_precision"]
                    ),
                    "brier_improvement_vs_immediate": immediate["brier"] - current["brier"],
                    "log_loss_improvement_vs_immediate": (
                        immediate["log_loss"] - current["log_loss"]
                    ),
                }
            )
    boot = pd.DataFrame(rows)
    summary_rows = []
    for (current_centre, candidate), frame in boot.groupby(["centre", "candidate"], sort=False):
        row = {
            "centre": current_centre,
            "candidate": candidate,
            "bootstrap_reps": frame["rep"].nunique(),
        }
        for column in [
            "delta_auroc_vs_immediate", "delta_average_precision_vs_immediate",
            "brier_improvement_vs_immediate", "log_loss_improvement_vs_immediate",
        ]:
            row[column + "_ci_low"] = float(frame[column].quantile(.025))
            row[column + "_ci_high"] = float(frame[column].quantile(.975))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), boot


def recalibration_audit(
    d: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m1_model = joblib.load(FIXED_MODELS / "inspire_fitted_M1_binary_prior_alert.joblib")
    m2_model = joblib.load(FIXED_MODELS / "inspire_fitted_M2_continuous_prior_response.joblib")
    p1_raw = m1_model.predict_proba(d[COMMON_FEATURES["M1_binary_prior_alert"]])[:, 1]
    p2_raw = m2_model.predict_proba(d[COMMON_FEATURES["M2_continuous_prior_response"]])[:, 1]
    outputs = {}
    for name, raw in [("M1", p1_raw), ("M2", p2_raw)]:
        x = np.log(np.clip(raw, 1e-8, 1 - 1e-8) / np.clip(1 - raw, 1e-8, 1))
        intercept_only = np.full(len(d), np.nan)
        intercept_slope = np.full(len(d), np.nan)
        for train, test in GroupKFold(n_splits=5).split(d, y, groups=groups):
            # Intercept-only: fixed slope=1, estimate a local offset on outer training patients.
            # Fit the calibration offset with the fixed raw logit as an offset,
            # instead of matching mean probability (which is only approximate).
            def offset_score(value: float) -> float:
                probability = 1 / (1 + np.exp(-np.clip(x[train] + value, -35, 35)))
                return float(np.sum(y[train] - probability))

            low, high = -12.0, 12.0
            for _ in range(80):
                middle = (low + high) / 2
                if offset_score(middle) > 0:
                    low = middle
                else:
                    high = middle
            offset = (low + high) / 2
            intercept_only[test] = 1 / (1 + np.exp(-np.clip(x[test] + offset, -35, 35)))

            slope_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
            slope_model.fit(x[train, None], y[train])
            intercept_slope[test] = slope_model.predict_proba(x[test, None])[:, 1]
        outputs[name + "_raw"] = raw
        outputs[name + "_intercept"] = intercept_only
        outputs[name + "_intercept_slope"] = intercept_slope
    rows = []
    for name, prediction in outputs.items():
        calibration_intercept, calibration_slope = calibration(y, prediction)
        rows.append(
            {
                "model": name,
                **metric(y, prediction),
                "calibration_intercept": calibration_intercept,
                "calibration_slope": calibration_slope,
            }
        )
    table = pd.DataFrame(rows)
    contrasts = []
    for variant in ["raw", "intercept", "intercept_slope"]:
        m1 = table.loc[table.model.eq("M1_" + variant)].iloc[0]
        m2 = table.loc[table.model.eq("M2_" + variant)].iloc[0]
        contrasts.append(
            {
                "recalibration": variant,
                "delta_auroc_M2_vs_M1": m2.auroc - m1.auroc,
                "brier_improvement_M2_vs_M1": m1.brier - m2.brier,
                "log_loss_improvement_M2_vs_M1": m1.log_loss - m2.log_loss,
            }
        )
    bin_rows = []
    for name in ["M2_raw", "M2_intercept", "M2_intercept_slope"]:
        prediction = outputs[name]
        # Equal-count bins give a stable visual at the low MOVER event rate.
        rank = pd.Series(prediction).rank(method="first")
        bins = pd.qcut(rank, q=10, labels=False)
        for bin_index in range(10):
            mask = bins.to_numpy() == bin_index
            bin_rows.append(
                {
                    "model": name,
                    "bin": bin_index + 1,
                    "n": int(mask.sum()),
                    "events": int(y[mask].sum()),
                    "mean_prediction": float(prediction[mask].mean()),
                    "observed_rate": float(y[mask].mean()),
                }
            )
    return table, pd.DataFrame(contrasts), pd.DataFrame(bin_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mover = add_response_terms(canonicalize(pd.read_csv(MOVER, low_memory=False)))
    mover_y = mover["target_any_low_first2"].to_numpy(int)
    mover_groups = mover["patient_id"].to_numpy()
    inspire = canonicalize_inspire(pd.read_csv(INSPIRE, low_memory=False))
    inspire_y = inspire["target"].to_numpy(int)
    inspire_groups = inspire["subject_id"].to_numpy()

    mover_predictions, _ = oof_predictions(mover, mover_y, mover_groups, MOVER_MODELS)
    inspire_predictions, _ = oof_predictions(inspire, inspire_y, inspire_groups, INSPIRE_MODELS)
    decomposition = pd.concat(
        [
            model_table("MOVER", mover_y, mover_predictions, "M1_binary_alert_context"),
            model_table("INSPIRE", inspire_y, inspire_predictions, "M1_binary_alert_context"),
        ],
        ignore_index=True,
    )
    decomposition.to_csv(OUT / "signal_decomposition_metrics.csv", index=False)
    associations = pd.concat(
        [
            harmonized_cluster_associations(
                "MOVER", mover, mover_y, mover_groups
            ),
            harmonized_cluster_associations(
                "INSPIRE", inspire, inspire_y, inspire_groups
            ),
        ],
        ignore_index=True,
    )
    associations.to_csv(OUT / "two_centre_harmonized_associations.csv", index=False)
    mover_ci, mover_boot = group_bootstrap_contrasts(
        "MOVER", mover_y, mover_groups, mover_predictions, "M1_binary_alert_context"
    )
    inspire_ci, inspire_boot = group_bootstrap_contrasts(
        "INSPIRE", inspire_y, inspire_groups, inspire_predictions, "M1_binary_alert_context"
    )
    decomposition_ci = pd.concat([mover_ci, inspire_ci], ignore_index=True)
    decomposition_ci.to_csv(OUT / "signal_decomposition_bootstrap_ci.csv", index=False)
    pd.concat([mover_boot, inspire_boot], ignore_index=True).to_csv(
        OUT / "signal_decomposition_bootstrap_replicates.csv.gz", index=False, compression="gzip"
    )
    pairwise_comparisons = [
        ("M2_level_plus_change", "M1_plus_prior_level"),
        ("M2_level_plus_relative_change", "M2_level_plus_change"),
        ("M2_plus_nonlinearity", "M2_level_plus_change"),
        ("M2_plus_interaction", "M2_level_plus_change"),
    ]
    mover_pairwise_ci, mover_pairwise_boot = group_bootstrap_pairwise(
        "MOVER", mover_y, mover_groups, mover_predictions, pairwise_comparisons
    )
    inspire_pairwise_ci, inspire_pairwise_boot = group_bootstrap_pairwise(
        "INSPIRE", inspire_y, inspire_groups, inspire_predictions, pairwise_comparisons
    )
    pairwise_ci = pd.concat([mover_pairwise_ci, inspire_pairwise_ci], ignore_index=True)
    pairwise_ci.to_csv(OUT / "signal_decomposition_pairwise_ci.csv", index=False)
    pd.concat([mover_pairwise_boot, inspire_pairwise_boot], ignore_index=True).to_csv(
        OUT / "signal_decomposition_pairwise_bootstrap.csv.gz", index=False, compression="gzip"
    )

    capacity, capacity_boot = fixed_capacity(
        mover,
        mover_y,
        mover_groups,
        mover_predictions["M1_binary_alert_context"],
        mover_predictions["M2_level_plus_change"],
    )
    capacity.to_csv(OUT / "mover_fixed_capacity_reclassification.csv", index=False)
    capacity_boot.to_csv(
        OUT / "mover_fixed_capacity_bootstrap.csv.gz", index=False, compression="gzip"
    )
    profiles = reclassification_profiles(
        mover,
        mover_y,
        mover_predictions["M1_binary_alert_context"],
        mover_predictions["M2_level_plus_change"],
    )
    profiles.to_csv(OUT / "mover_reclassification_profiles.csv", index=False)
    risk_surface = pd.concat(
        [
            clinical_risk_surface("INSPIRE", inspire, inspire_y),
            clinical_risk_surface("MOVER", mover, mover_y),
        ],
        ignore_index=True,
    )
    risk_surface.to_csv(OUT / "two_centre_prior_response_risk_surface.csv", index=False)

    mover_chains = build_history_chains_mover(mover)
    inspire_chains = build_history_chains_inspire(inspire)
    mover_history, mover_history_predictions = history_depth(
        "MOVER",
        mover_chains,
        "target_any_low_first2",
        "patient_id",
        MOVER_BASE,
    )
    inspire_history, inspire_history_predictions = history_depth(
        "INSPIRE",
        inspire_chains,
        "target",
        "subject_id",
        INSPIRE_BASE,
    )
    history = pd.concat([mover_history, inspire_history], ignore_index=True)
    history.to_csv(OUT / "history_depth_metrics.csv", index=False)
    mover_history_ci, mover_history_boot = history_bootstrap(
        "MOVER", mover_chains, "target_any_low_first2", "patient_id", mover_history_predictions
    )
    inspire_history_ci, inspire_history_boot = history_bootstrap(
        "INSPIRE", inspire_chains, "target", "subject_id", inspire_history_predictions
    )
    pd.concat([mover_history_ci, inspire_history_ci], ignore_index=True).to_csv(
        OUT / "history_depth_bootstrap_ci.csv", index=False
    )
    pd.concat([mover_history_boot, inspire_history_boot], ignore_index=True).to_csv(
        OUT / "history_depth_bootstrap_replicates.csv.gz", index=False, compression="gzip"
    )

    recalibration, recalibration_contrasts, recalibration_bins = recalibration_audit(
        mover, mover_y, mover_groups
    )
    recalibration.to_csv(OUT / "fixed_model_recalibration_metrics.csv", index=False)
    recalibration_contrasts.to_csv(OUT / "fixed_model_recalibration_contrasts.csv", index=False)
    recalibration_bins.to_csv(OUT / "fixed_model_recalibration_bins.csv", index=False)

    # Integrated paper-facing figure.
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.3))
    labels = {
        "M1_plus_prior_level": "Prior level only",
        "M1_plus_prior_change": "Prior change only",
        "M2_level_plus_change": "Level + change",
        "M2_level_plus_relative_change": "Level + % change",
        "M2_plus_interaction": "+ level×change",
    }
    plot = decomposition.loc[decomposition.model.ne("M1_binary_alert_context")].copy()
    plot["label"] = plot.model.map(labels)
    positions = np.arange(len(labels))
    width = .35
    for offset, (centre, color) in zip([-.18, .18], [("INSPIRE", "#2F6B9A"), ("MOVER", "#2A9D8F")]):
        frame = plot.loc[plot.centre.eq(centre)].set_index("model").loc[list(labels)].reset_index()
        axes[0, 0].bar(positions + offset, frame.delta_auroc_vs_M1, width, label=centre, color=color)
    axes[0, 0].axhline(0, color="#222", lw=1)
    axes[0, 0].set_xticks(positions, list(labels.values()), rotation=18, ha="right")
    axes[0, 0].set_ylabel("AUROC increment vs binary alert/context")
    axes[0, 0].set_title("A  Which part of prior response matters?")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].errorbar(
        capacity.capacity * 100,
        capacity.capture_improvement * 100,
        yerr=np.vstack([
            (capacity.capture_improvement - capacity.capture_improvement_ci_low) * 100,
            (capacity.capture_improvement_ci_high - capacity.capture_improvement) * 100,
        ]),
        marker="o", capsize=3, label="Event capture",
    )
    axes[0, 1].errorbar(
        capacity.capacity * 100,
        capacity.ppv_improvement * 100,
        yerr=np.vstack([
            (capacity.ppv_improvement - capacity.ppv_improvement_ci_low) * 100,
            (capacity.ppv_improvement_ci_high - capacity.ppv_improvement) * 100,
        ]),
        marker="s", capsize=3, label="PPV",
    )
    axes[0, 1].axhline(0, color="#222", lw=1)
    axes[0, 1].set_xlabel("Patients selected as high risk (%)")
    axes[0, 1].set_ylabel("Absolute improvement (percentage points)")
    axes[0, 1].set_title("B  Fixed-capacity clinical reclassification")
    axes[0, 1].legend(frameon=False)

    history_labels = {
        "H1_immediate_prior": "Immediate prior",
        "H1_older_prior": "Older prior",
        "H2_immediate_plus_older": "Immediate + older",
        "H2_two_history_mean": "Two-history mean",
    }
    history_plot = history.loc[history.model.isin(history_labels)].copy()
    hpositions = np.arange(len(history_labels))
    for offset, (centre, color) in zip([-.18, .18], [("INSPIRE", "#2F6B9A"), ("MOVER", "#2A9D8F")]):
        frame = history_plot.loc[history_plot.centre.eq(centre)].set_index("model").loc[list(history_labels)].reset_index()
        axes[1, 0].bar(hpositions + offset, frame.delta_auroc_vs_M1, width, label=centre, color=color)
    axes[1, 0].axhline(0, color="#222", lw=1)
    axes[1, 0].set_xticks(hpositions, list(history_labels.values()), rotation=18, ha="right")
    axes[1, 0].set_ylabel("AUROC increment vs context/alert")
    axes[1, 0].set_title(
        f"C  Deeper history (exploratory: {len(mover_chains)} MOVER chains/{int(mover_chains.target_any_low_first2.sum())} events)"
    )

    for model, label, color in [
        ("M2_raw", "Raw fixed INSPIRE model", "#D95F02"),
        ("M2_intercept", "Cross-fitted intercept recalibration", "#1B9E77"),
    ]:
        frame = recalibration_bins.loc[recalibration_bins.model.eq(model)].sort_values("bin")
        axes[1, 1].plot(
            frame.mean_prediction, frame.observed_rate, marker="o", label=label, color=color
        )
    limit = max(
        .08,
        float(recalibration_bins[["mean_prediction", "observed_rate"]].max().max()) * 1.08,
    )
    axes[1, 1].plot([0, limit], [0, limit], ls="--", color="#555", lw=1, label="Ideal")
    axes[1, 1].set_xlim(0, limit)
    axes[1, 1].set_ylim(0, limit)
    axes[1, 1].set_xlabel("Mean predicted risk by decile")
    axes[1, 1].set_ylabel("Observed event rate")
    axes[1, 1].set_title("D  Transport calibration before and after local intercept repair")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("C02 deepening: signal source, risk reclassification, history depth, and transport repair", y=.995)
    fig.tight_layout(rect=[0, 0, 1, .975])
    fig.savefig(OUT / "fig_c02_deepening_v1.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "fig_c02_deepening_v1.svg", bbox_inches="tight")
    plt.close(fig)

    # Clinical figure: the joint prior-response pattern, not a model score.
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)
    level_order = ["<80", "80–89", "90–99", "100–109", "≥110"]
    change_order = ["≤−10", "−9 to −5", "−4 to +4", "≥+5"]
    vmax = float(risk_surface.event_rate.max())
    image = None
    for ax, centre in zip(axes, ["INSPIRE", "MOVER"]):
        frame = risk_surface.loc[risk_surface.centre.eq(centre)]
        rates = frame.pivot(index="prior_change_band", columns="prior_level_band", values="event_rate").loc[
            change_order, level_order
        ]
        counts = frame.pivot(index="prior_change_band", columns="prior_level_band", values="n").loc[
            change_order, level_order
        ]
        image = ax.imshow(rates.to_numpy() * 100, cmap="YlOrRd", vmin=0, vmax=vmax * 100, aspect="auto")
        for row in range(len(change_order)):
            for column in range(len(level_order)):
                rate = rates.iloc[row, column] * 100
                n = int(counts.iloc[row, column])
                text_color = "white" if rate > vmax * 100 * .57 else "#222222"
                ax.text(column, row, f"{rate:.1f}%\n(n={n})", ha="center", va="center", fontsize=9, color=text_color)
        ax.set_xticks(range(len(level_order)), level_order)
        ax.set_yticks(range(len(change_order)), change_order)
        ax.set_xlabel("Prior first MAP (mmHg)")
        ax.set_ylabel("Prior first-to-second MAP change (mmHg)")
        ax.set_title(f"{centre}: current early MAP<65 risk")
    fig.colorbar(image, ax=axes, shrink=.85, label="Observed event rate (%)")
    fig.suptitle("Two-centre joint risk gradient from the immediately prior anaesthetic", fontsize=14)
    fig.savefig(OUT / "fig_c02_two_centre_risk_surface.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "fig_c02_two_centre_risk_surface.svg", bbox_inches="tight")
    plt.close(fig)

    mover_decomp = decomposition.set_index(["centre", "model"])
    inspire_decomp = decomposition.set_index(["centre", "model"])
    mover_h = history.set_index(["centre", "model"])
    inspire_h = history.set_index(["centre", "model"])
    summary = {
        "status": "C02_DEEPENING_V1_COMPLETED",
        "cohorts": {
            "INSPIRE": {"pairs": len(inspire), "patients": int(pd.Series(inspire_groups).nunique()), "events": int(inspire_y.sum())},
            "MOVER": {"pairs": len(mover), "patients": int(pd.Series(mover_groups).nunique()), "events": int(mover_y.sum())},
            "INSPIRE_three_case_chains": {"pairs": len(inspire_chains), "patients": int(inspire_chains.subject_id.nunique()), "events": int(inspire_chains.target.sum())},
            "MOVER_three_case_chains": {"pairs": len(mover_chains), "patients": int(mover_chains.patient_id.nunique()), "events": int(mover_chains.target_any_low_first2.sum())},
        },
        "signal_decomposition": {
            model: {
                "INSPIRE_delta_auroc": float(inspire_decomp.loc[("INSPIRE", model), "delta_auroc_vs_M1"]),
                "MOVER_delta_auroc": float(mover_decomp.loc[("MOVER", model), "delta_auroc_vs_M1"]),
            }
            for model in labels
        },
        "fixed_capacity": capacity.to_dict(orient="records"),
        "clinical_risk_surface": risk_surface.to_dict(orient="records"),
        "signal_pairwise_comparisons": pairwise_ci.to_dict(orient="records"),
        "harmonized_associations": associations.to_dict(orient="records"),
        "history_depth": {
            model: {
                "INSPIRE_delta_auroc_vs_context": float(inspire_h.loc[("INSPIRE", model), "delta_auroc_vs_M1"]),
                "MOVER_delta_auroc_vs_context": float(mover_h.loc[("MOVER", model), "delta_auroc_vs_M1"]),
                "INSPIRE_delta_auroc_vs_immediate": float(inspire_h.loc[("INSPIRE", model), "delta_auroc_vs_immediate"]),
                "MOVER_delta_auroc_vs_immediate": float(mover_h.loc[("MOVER", model), "delta_auroc_vs_immediate"]),
            }
            for model in history_labels
        },
        "recalibration": recalibration.to_dict(orient="records"),
        "claim_boundary": (
            "Deepening describes the source and use of patient-history information. It does not establish a stable phenotype, "
            "treatment target, sustained hypotension benefit, or organ-outcome benefit."
        ),
        "patient_identifiers_in_outputs": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
