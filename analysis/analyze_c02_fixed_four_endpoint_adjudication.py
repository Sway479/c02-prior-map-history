#!/usr/bin/env python3
"""Adjudicate the clinically strongest fixed-first-four C02 endpoint.

This analysis is allowed to overturn the existing consecutive-pair endpoint.
It compares any-low, at-least-two-low, and consecutive-low summaries, validates
them against events strictly after the fourth NIBP reading, and tests the
at-least-two endpoint in INSPIRE and MOVER with the same grouped model ladder.
Only aggregate report artifacts are written.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from analyze_c02_repeated_alert_comparator import oof_predictions
from analyze_c02_repeated_alert_comparator import all_relevant_inspire_operations
from analyze_c02_two_centre_repeated_low import (
    INSPIRE_EXPANDED_M1,
    MOVER_EXPANDED_M1,
    build_inspire_first4,
    expanded_oof,
    inspire_features,
    metric,
)
from analyze_mover_c02_early_vasopressor_action import valid_bolus_rows
from run_mover_c02_external_validation import canonicalize as canonicalize_mover


from c02_runtime import private_workspace_root


ROOT = private_workspace_root()
BASE = ROOT / "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild"
INSPIRE_PAIR = BASE / "first_two_sampling_bridge/first_two_pair_cohort.csv.gz"
MOVER_PAIR = ROOT / "data/restricted/mover/extracted/mover_c02_cleaned_pair_cohort.csv.gz"
MOVER_MAP = ROOT / "data/restricted/mover/extracted/mover_cleaned_early_map.csv.gz"
MOVER_MAR = ROOT / "data/restricted/mover/extracted/mover_early_vasopressor_mar.csv.gz"
OUT = BASE / "clinical_endpoint_upgrade/fixed_four_adjudication"

MOVER_ARTIFACT_FILTERS = {
    "all_discrete_ART": {"drop_first": 0},
    "drop_first_2_ART_records": {"drop_first": 2},
    "drop_first_3_ART_records": {"drop_first": 3},
}


def grouped_oof(
    d: pd.DataFrame, y: np.ndarray, groups: np.ndarray, specs: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    return oof_predictions(d, y, groups, specs)


def performance(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
    }


def increments(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict[str, float]:
    first = performance(y, p1)
    second = performance(y, p2)
    return {
        "delta_auroc": second["auroc"] - first["auroc"],
        "delta_average_precision": second["average_precision"] - first["average_precision"],
        "brier_improvement": first["brier"] - second["brier"],
        "log_loss_improvement": first["log_loss"] - second["log_loss"],
    }


def cluster_bootstrap(
    y: np.ndarray, groups: np.ndarray, p1: np.ndarray, p2: np.ndarray,
    label: str, seed: int, reps: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = pd.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique}
    point = increments(y, p1, p2)
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        sampled = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([lookup[group] for group in sampled])
        if np.unique(y[index]).size < 2:
            continue
        rows.append({"analysis": label, "rep": rep, **increments(y[index], p1[index], p2[index])})
    boot = pd.DataFrame(rows)
    summary = []
    for name, value in point.items():
        summary.append(
            {
                "analysis": label, "metric": name, "point": float(value),
                "ci_low": float(boot[name].quantile(.025)),
                "ci_high": float(boot[name].quantile(.975)),
                "bootstrap_reps": int(len(boot)),
            }
        )
    return pd.DataFrame(summary), boot


def build_mover_operation_table(
    pair: pd.DataFrame, bolus: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ids = set(pair["LOG_ID"].astype(str)) | set(pair["prior_LOG_ID"].astype(str))
    maps = pd.read_csv(MOVER_MAP, dtype={"LOG_ID": str}, low_memory=False)
    maps = maps.loc[
        maps["LOG_ID"].isin(ids)
        & maps["RECORD_TYPE"].eq("INTRA-OP")
        & maps["relative_min"].between(0, 30, inclusive="both")
        & maps["value"].between(20, 200, inclusive="both")
    ].copy()
    maps["time"] = pd.to_datetime(maps["RECORDED_TIME"], errors="coerce")
    maps = maps.loc[maps["time"].notna()]
    key = (
        maps.groupby(["LOG_ID", "modality_hint", "time"], as_index=False, observed=True)
        .agg(
            relative_min=("relative_min", "min"),
            distinct_values=("value", "nunique"),
            value=("value", "first"),
        )
    )
    conflicts = int(key["distinct_values"].gt(1).sum())
    key = key.loc[key["distinct_values"].eq(1)].sort_values(
        ["LOG_ID", "modality_hint", "relative_min", "time"]
    )
    nibp = key.loc[key["modality_hint"].eq("NIBP")].copy()
    art = key.loc[key["modality_hint"].eq("ART")].copy()
    bolus_lookup = {
        str(log_id): frame["relative_min"].to_numpy(float)
        for log_id, frame in bolus.loc[bolus["LOG_ID"].isin(ids)].groupby("LOG_ID", observed=True)
    }
    rows = []
    for log_id, frame in nibp.groupby("LOG_ID", sort=False, observed=True):
        frame = frame.sort_values(["relative_min", "time"])
        first4 = frame.head(4)
        if len(first4) < 4:
            continue
        values = first4["value"].to_numpy(float)
        times = first4["relative_min"].to_numpy(float)
        low = values < 65
        gap = np.diff(times)
        landmark = float(times[-1])
        post = frame.loc[
            frame["relative_min"].gt(landmark)
            & frame["relative_min"].le(landmark + 10)
        ]
        action_times = bolus_lookup.get(str(log_id), np.array([], dtype=float))
        post_actions = action_times[(action_times > landmark) & (action_times <= landmark + 10)]
        rows.append(
            {
                "LOG_ID": str(log_id),
                "first4_any_low": int(low.any()),
                "first4_two_low": int(low.sum() >= 2),
                "first4_consecutive_low": int(
                    (low[:-1] & low[1:] & (gap >= 2) & (gap <= 15)).any()
                ),
                "first4_low_count": int(low.sum()),
                "first4_nadir": float(values.min()),
                "landmark_min": landmark,
                "post10_n_nibp": int(len(post)),
                "post10_span_min": (
                    float(post["relative_min"].max() - post["relative_min"].min())
                    if len(post) >= 2 else 0.0
                ),
                "post10_any_low": int((post["value"] < 65).any()) if len(post) else np.nan,
                "post10_two_low": int((post["value"] < 65).sum() >= 2) if len(post) >= 2 else np.nan,
                "post10_first2_both_low": (
                    int((post.head(2)["value"] < 65).all()) if len(post) >= 2 else np.nan
                ),
                "post10_any_bolus": int(len(post_actions) >= 1),
                "post10_two_bolus": int(len(post_actions) >= 2),
                **{f"first4_map_{index}": float(values[index]) for index in range(4)},
                **{f"first4_rel_{index}": float(times[index]) for index in range(4)},
            }
        )
    operation = pd.DataFrame(rows)
    audit = {
        "map_rows": int(len(maps)),
        "same_time_modality_keys": int(len(key) + conflicts),
        "conflicting_same_time_modality_keys_excluded": conflicts,
        "operations_with_first4_NIBP": int(len(operation)),
        "operations_with_any_ART": int(art["LOG_ID"].nunique()),
    }
    return operation, art, audit


def prepare_two_centre(
    mover_operation: pd.DataFrame, pair: pd.DataFrame, bolus: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Rebuild fixed-four records for *both* current and prior operations from
    # raw INSPIRE vitals. Looking up the prior only among operations already
    # present as current rows would incorrectly discard most valid pairs.
    inspire_pair = pd.read_csv(INSPIRE_PAIR, low_memory=False)
    inspire_pair = inspire_pair.loc[
        inspire_pair["antype"].astype("string").str.strip().eq("General")
        & inspire_pair["prior_antype"].astype("string").str.strip().eq("General")
    ].copy()
    linked, timing, link_audit = all_relevant_inspire_operations(inspire_pair)
    first4, first4_audit = build_inspire_first4(timing)
    current_i = first4[["op_id", "target_two_low_first4"]]
    prior_i = first4[["op_id", "target_two_low_first4"]].rename(
        columns={"op_id": "prior_op_id", "target_two_low_first4": "prior_two_low_first4"}
    )
    inspire = linked.merge(current_i, on="op_id", how="inner", validate="one_to_one")
    inspire = inspire.merge(prior_i, on="prior_op_id", how="inner", validate="many_to_one")
    inspire = inspire_features(inspire)

    current_m = mover_operation.rename(columns={
        "first4_two_low": "target_two_low_first4"
    })
    prior_m = mover_operation[["LOG_ID", "first4_two_low"]].rename(columns={
        "LOG_ID": "prior_LOG_ID", "first4_two_low": "prior_two_low_first4"
    })
    mover = pair.merge(current_m, on="LOG_ID", how="inner", validate="one_to_one")
    mover = mover.merge(prior_m, on="prior_LOG_ID", how="inner", validate="many_to_one")
    prior_bolus = (
        bolus.loc[bolus["LOG_ID"].isin(set(mover["prior_LOG_ID"]))]
        .groupby("LOG_ID", observed=True).size().rename("prior_bolus_count").reset_index()
        .rename(columns={"LOG_ID": "prior_LOG_ID"})
    )
    mover = mover.merge(prior_bolus, on="prior_LOG_ID", how="left", validate="many_to_one")
    mover["prior_bolus_count"] = mover["prior_bolus_count"].fillna(0)
    mover["prior_any_bolus"] = mover["prior_bolus_count"].ge(1).astype(int)
    mover["prior_two_bolus"] = mover["prior_bolus_count"].ge(2).astype(int)
    return inspire, mover, {"link": link_audit, "first4": first4_audit}


def temporal_validity(operation: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    d = operation.merge(pair[["LOG_ID", "patient_id"]], on="LOG_ID", how="inner")
    d = d.loc[d["landmark_min"].le(20)].copy()
    rows = []
    outcomes = {
        "post10_any_bolus": np.ones(len(d), dtype=bool),
        "post10_two_bolus": np.ones(len(d), dtype=bool),
        "post10_any_low": d["post10_n_nibp"].ge(2).to_numpy(),
        "post10_two_low": d["post10_n_nibp"].ge(2).to_numpy(),
        "post10_first2_both_low": d["post10_n_nibp"].ge(2).to_numpy(),
    }
    predictors = ["first4_any_low", "first4_two_low", "first4_consecutive_low"]
    for outcome, eligible in outcomes.items():
        q = d.loc[eligible].copy()
        for predictor in predictors:
            for level, frame in q.groupby(predictor, observed=True):
                rows.append(
                    {
                        "outcome": outcome, "predictor": predictor,
                        "predictor_level": int(level), "n": int(len(frame)),
                        "events": int(frame[outcome].sum()),
                        "risk": float(frame[outcome].mean()),
                        "patients": int(frame["patient_id"].nunique()),
                    }
                )
        for restriction, frame in [
            ("all", q),
            ("among_any_low", q.loc[q["first4_any_low"].eq(1)]),
            ("among_two_low", q.loc[q["first4_two_low"].eq(1)]),
        ]:
            for predictor in ["first4_two_low", "first4_consecutive_low"]:
                if frame[predictor].nunique() < 2:
                    continue
                risk = frame.groupby(predictor, observed=True)[outcome].mean()
                rows.append(
                    {
                        "outcome": outcome,
                        "predictor": predictor + "_risk_difference",
                        "predictor_level": restriction,
                        "n": int(len(frame)),
                        "events": int(frame[outcome].sum()),
                        "risk": float(risk.loc[1] - risk.loc[0]),
                        "patients": int(frame["patient_id"].nunique()),
                    }
                )
    return pd.DataFrame(rows)


def adjusted_temporal_validity(
    operation: pd.DataFrame, pair: pd.DataFrame, bolus: pd.DataFrame
) -> pd.DataFrame:
    """Does fixed-four count/adjacency predict *future* records beyond severity?"""
    columns = [
        "LOG_ID", "patient_id", "age_years", "bmi_kg_m2", "asa_numeric",
        "sex_common", "patient_class_common",
    ]
    d = operation.merge(pair[columns], on="LOG_ID", how="inner", validate="one_to_one")
    d = d.loc[d.landmark_min.le(20) & d.first4_any_low.eq(1)].copy()
    action_lookup = {
        str(log_id): frame.relative_min.to_numpy(float)
        for log_id, frame in bolus.groupby("LOG_ID", observed=True)
    }
    d["pre_landmark_bolus"] = [
        int((action_lookup.get(str(log_id), np.array([], dtype=float)) <= landmark).sum())
        for log_id, landmark in zip(d.LOG_ID, d.landmark_min)
    ]
    d["fourth_MAP"] = d["first4_map_3"]
    specifications = {
        "case_landmark_adjusted": [
            "first4_two_low", "first4_consecutive_low", "age_per_10y", "bmi_per_5",
            "asa_numeric", "male", "inpatient", "fourth_MAP_per_10",
            "landmark_per_10", "pre_landmark_bolus",
        ],
        "severity_adjusted": [
            "first4_two_low", "first4_consecutive_low", "age_per_10y", "bmi_per_5",
            "asa_numeric", "male", "inpatient", "fourth_MAP_per_10",
            "first4_nadir_per_10", "landmark_per_10", "pre_landmark_bolus",
        ],
    }
    design_frame = pd.DataFrame(
        {
            "first4_two_low": d.first4_two_low.astype(float),
            "first4_consecutive_low": d.first4_consecutive_low.astype(float),
            "age_per_10y": d.age_years / 10,
            "bmi_per_5": d.bmi_kg_m2 / 5,
            "asa_numeric": d.asa_numeric,
            "male": d.sex_common.astype(str).str.upper().eq("M").astype(float),
            "inpatient": d.patient_class_common.astype(str).eq("Inpatient").astype(float),
            "fourth_MAP_per_10": d.fourth_MAP / 10,
            "first4_nadir_per_10": d.first4_nadir / 10,
            "landmark_per_10": d.landmark_min / 10,
            "pre_landmark_bolus": d.pre_landmark_bolus.astype(float),
        }
    )
    for column in design_frame:
        design_frame[column] = pd.to_numeric(design_frame[column], errors="coerce")
        design_frame[column] = design_frame[column].fillna(design_frame[column].median())

    outcomes = {
        "post10_any_bolus": np.ones(len(d), dtype=bool),
        "post10_two_low": d.post10_n_nibp.ge(2).to_numpy(),
        "post10_first2_both_low": d.post10_n_nibp.ge(2).to_numpy(),
    }
    rows = []
    for outcome, eligible in outcomes.items():
        q_index = np.flatnonzero(eligible)
        y = d.iloc[q_index][outcome].to_numpy(int)
        groups = d.iloc[q_index].patient_id.to_numpy()
        for specification, terms in specifications.items():
            x = design_frame.iloc[q_index][terms]
            names = ["intercept", *terms]
            matrix = np.column_stack([np.ones(len(x)), x.to_numpy(float)])

            def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
                eta = np.einsum("ij,j->i", matrix, beta, optimize=False)
                probability = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
                gradient = np.einsum(
                    "ij,i->j", matrix, probability - y, optimize=False
                )
                return float(np.sum(np.logaddexp(0, eta) - y * eta)), gradient

            fit = minimize(
                lambda beta: objective(beta)[0], np.zeros(matrix.shape[1]),
                jac=lambda beta: objective(beta)[1], method="BFGS",
                options={"maxiter": 2500, "gtol": 1e-8},
            )
            eta = np.einsum("ij,j->i", matrix, fit.x, optimize=False)
            probability = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
            weight = probability * (1 - probability)
            hessian = np.einsum(
                "ni,n,nj->ij", matrix, weight, matrix, optimize=False
            )
            bread = np.linalg.pinv(hessian)
            score = matrix * (y - probability)[:, None]
            score_frame = pd.DataFrame(score)
            score_frame.insert(0, "group", groups)
            cluster = score_frame.groupby("group", sort=False).sum().to_numpy()
            meat = np.einsum("ni,nj->ij", cluster, cluster, optimize=False)
            n, k = matrix.shape
            correction = (len(cluster) / (len(cluster) - 1)) * ((n - 1) / (n - k))
            covariance = correction * np.einsum(
                "ij,jk,kl->il", bread, meat, bread, optimize=False
            )
            se = np.sqrt(np.clip(np.diag(covariance), 0, None))
            for term in ["first4_two_low", "first4_consecutive_low"]:
                position = names.index(term)
                estimate = float(fit.x[position])
                standard_error = float(se[position])
                rows.append(
                    {
                        "outcome": outcome, "specification": specification,
                        "term": term, "odds_ratio": math.exp(estimate),
                        "ci_low": math.exp(estimate - 1.959963984540054 * standard_error),
                        "ci_high": math.exp(estimate + 1.959963984540054 * standard_error),
                        "p_value_cluster_robust": float(
                            2 * norm.sf(abs(estimate / standard_error))
                        ),
                        "n": int(len(y)), "events": int(y.sum()),
                        "patients": int(pd.Series(groups).nunique()),
                        "interpretation": "post-landmark adjusted association, not causal",
                    }
                )
    return pd.DataFrame(rows)


def art_audit(
    operation: pd.DataFrame, art: pd.DataFrame, pair: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = operation.merge(pair[["LOG_ID", "patient_id"]], on="LOG_ID", how="inner")
    art_groups = {str(log_id): frame.sort_values("relative_min") for log_id, frame in art.groupby("LOG_ID")}
    burden_rows = []
    concurrent_rows = []
    for rule, specification in MOVER_ARTIFACT_FILTERS.items():
        drop_first = specification["drop_first"]
        for _, row in d.iterrows():
            aa = art_groups.get(str(row["LOG_ID"]))
            if aa is None:
                continue
            aa = aa.iloc[drop_first:].copy()
            if len(aa) >= 5:
                times = aa["relative_min"].to_numpy(float)
                values = aa["value"].to_numpy(float)
                gaps = np.diff(times)
                if (times[-1] - times[0]) >= 5 and gaps.max(initial=0) <= 2:
                    duration = gaps
                    burden_rows.append(
                        {
                            "rule": rule, "LOG_ID": row["LOG_ID"],
                            "first4_two_low": int(row["first4_two_low"]),
                            "first4_consecutive_low": int(row["first4_consecutive_low"]),
                            "art_records": int(len(aa)),
                            "art_span_min": float(times[-1] - times[0]),
                            "art_any_low": int((values < 65).any()),
                            "art_two_low": int((values < 65).sum() >= 2),
                            "art_low_minutes_on_observed_support": float(duration[values[:-1] < 65].sum()),
                            "art_AUT_on_observed_support": float(
                                (np.maximum(65 - values[:-1], 0) * duration).sum()
                            ),
                        }
                    )
            # Pair first-four NIBP to nearest retained ART within 2 min.
            nibp = pd.DataFrame(
                {
                    "relative_min": [row[f"first4_rel_{i}"] for i in range(4)],
                    "nibp": [row[f"first4_map_{i}"] for i in range(4)],
                }
            ).sort_values("relative_min")
            paired = pd.merge_asof(
                nibp,
                aa[["relative_min", "value"]].sort_values("relative_min"),
                on="relative_min", direction="nearest", tolerance=2,
            ).dropna()
            if len(paired) >= 2:
                concurrent_rows.append(
                    {
                        "rule": rule, "LOG_ID": row["LOG_ID"],
                        "first4_two_low": int(row["first4_two_low"]),
                        "first4_consecutive_low": int(row["first4_consecutive_low"]),
                        "paired_points": int(len(paired)),
                        "nibp_minus_art_bias": float((paired["nibp"] - paired["value"]).mean()),
                        "absolute_error": float((paired["nibp"] - paired["value"]).abs().mean()),
                        "threshold_agreement": float(
                            ((paired["nibp"] < 65) == (paired["value"] < 65)).mean()
                        ),
                    }
                )
    return pd.DataFrame(burden_rows), pd.DataFrame(concurrent_rows)


def association(
    d: pd.DataFrame, outcome: str, group_column: str, centre: str
) -> pd.DataFrame:
    x = pd.DataFrame(
        {
            "age_per_10y": d["age_years"] / 10,
            "bmi_per_5": d["bmi_kg_m2"] / 5,
            "asa": d["asa_numeric"],
            "male": d["sex_common"].astype(str).str.upper().eq("M").astype(float),
            "interval_log": d["interval_log1p"],
            "prior_first2_alert": d["prior_first2_any_low"],
            "prior_two_low_alert": d["prior_two_low_first4"],
            "prior_first_MAP_per_10": d["prior_first_map"] / 10,
            "prior_change_per_10": d["prior_first2_change"] / 10,
        }
    )
    if centre == "MOVER":
        x["prior_any_bolus"] = d["prior_any_bolus"]
        x["prior_two_bolus"] = d["prior_two_bolus"]
    for column in x:
        x[column] = pd.to_numeric(x[column], errors="coerce")
        x[column] = x[column].fillna(x[column].median())
    names = ["intercept", *x.columns]
    matrix = np.column_stack([np.ones(len(x)), x.to_numpy(float)])
    y = d[outcome].to_numpy(int)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = matrix @ beta
        probability = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
        return float(np.sum(np.logaddexp(0, eta) - y * eta)), matrix.T @ (probability - y)

    fit = minimize(
        lambda beta: objective(beta)[0], np.zeros(matrix.shape[1]),
        jac=lambda beta: objective(beta)[1], method="BFGS",
        options={"maxiter": 2500, "gtol": 1e-8},
    )
    eta = matrix @ fit.x
    probability = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
    weight = probability * (1 - probability)
    bread_inv = np.linalg.pinv(matrix.T @ (matrix * weight[:, None]))
    score = matrix * (y - probability)[:, None]
    score_frame = pd.DataFrame(score)
    score_frame.insert(0, "group", d[group_column].to_numpy())
    cluster = score_frame.groupby("group", sort=False).sum().to_numpy()
    n, k = matrix.shape
    correction = (len(cluster) / (len(cluster) - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread_inv @ (cluster.T @ cluster) @ bread_inv
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    rows = []
    for term in names[1:]:
        position = names.index(term)
        estimate = float(fit.x[position])
        standard_error = float(se[position])
        rows.append(
            {
                "centre": centre, "term": term,
                "odds_ratio": math.exp(estimate),
                "ci_low": math.exp(estimate - 1.959963984540054 * standard_error),
                "ci_high": math.exp(estimate + 1.959963984540054 * standard_error),
                "p_value_cluster_robust": float(2 * norm.sf(abs(estimate / standard_error))),
                "n": int(len(d)), "events": int(y.sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pair = canonicalize_mover(
        pd.read_csv(MOVER_PAIR, dtype={"LOG_ID": str, "prior_LOG_ID": str}, low_memory=False)
    )
    raw_mar = pd.read_csv(MOVER_MAR, dtype={"LOG_ID": str}, low_memory=False)
    bolus = valid_bolus_rows(raw_mar)
    operation, art, map_audit = build_mover_operation_table(pair, bolus)
    inspire, mover, inspire_audit = prepare_two_centre(operation, pair, bolus)

    model_rows = []
    increment_frames = []
    bootstrap_frames = []
    association_frames = []
    for centre, d, group, m1_base in [
        ("INSPIRE", inspire, "subject_id", INSPIRE_EXPANDED_M1),
        ("MOVER", mover, "patient_id", MOVER_EXPANDED_M1),
    ]:
        # Replace the old prior first-two-only alert with both the readable
        # first-two alert and the prior fixed-four >=2 alert. MOVER additionally
        # includes actual prior treatment history.
        base = [column for column in m1_base if column != "prior_first2_any_low"]
        m1 = base + ["prior_first2_any_low", "prior_two_low_first4"]
        if centre == "MOVER":
            m1 += ["prior_any_bolus", "prior_two_bolus"]
        m2 = m1 + ["prior_first_map", "prior_first2_change"]
        specs = {"M1_readable_history": m1, "M2_continuous_MAP": m2}
        y = d["target_two_low_first4"].to_numpy(int)
        groups = d[group].to_numpy()
        prediction = grouped_oof(d, y, groups, specs)
        for name, values in prediction.items():
            model_rows.append({"centre": centre, "model": name, **metric(y, values)})
        ci, boot = cluster_bootstrap(
            y, groups, prediction["M1_readable_history"], prediction["M2_continuous_MAP"],
            label=f"{centre}_two_low_M2_vs_M1", seed=20260830 if centre == "INSPIRE" else 20260831,
        )
        ci.insert(0, "centre", centre)
        increment_frames.append(ci)
        bootstrap_frames.append(boot)
        association_frames.append(association(d, "target_two_low_first4", group, centre))
    models = pd.DataFrame(model_rows)
    increment_table = pd.concat(increment_frames, ignore_index=True)
    associations = pd.concat(association_frames, ignore_index=True)
    models.to_csv(OUT / "two_centre_two_low_model_metrics.csv", index=False)
    increment_table.to_csv(OUT / "two_centre_two_low_increment_ci.csv", index=False)
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(
        OUT / "two_centre_two_low_bootstrap.csv.gz", index=False, compression="gzip"
    )
    associations.to_csv(OUT / "two_centre_two_low_associations.csv", index=False)

    validity = temporal_validity(operation, pair)
    validity.to_csv(OUT / "post_landmark_temporal_validity.csv", index=False)
    adjusted_validity = adjusted_temporal_validity(operation, pair, bolus)
    adjusted_validity.to_csv(OUT / "post_landmark_adjusted_associations.csv", index=False)
    burden, concurrent = art_audit(operation, art, pair)
    burden_summary = (
        burden.groupby(["rule", "first4_two_low"], as_index=False, observed=True)
        .agg(
            n=("LOG_ID", "size"),
            art_any_low_rate=("art_any_low", "mean"),
            art_two_low_rate=("art_two_low", "mean"),
            art_low_minutes_median=("art_low_minutes_on_observed_support", "median"),
            art_low_minutes_mean=("art_low_minutes_on_observed_support", "mean"),
            art_AUT_median=("art_AUT_on_observed_support", "median"),
        )
    )
    concurrent_summary = (
        concurrent.groupby(["rule", "first4_two_low"], as_index=False, observed=True)
        .agg(
            n=("LOG_ID", "size"), paired_points=("paired_points", "sum"),
            bias_median=("nibp_minus_art_bias", "median"),
            MAE_median=("absolute_error", "median"),
            threshold_agreement_weighted=(
                "threshold_agreement", lambda x: float(
                    np.average(x, weights=concurrent.loc[x.index, "paired_points"])
                )
            ),
        )
    )
    burden_summary.to_csv(OUT / "discrete_ART_burden_summary.csv", index=False)
    concurrent_summary.to_csv(OUT / "concurrent_NIBP_ART_summary.csv", index=False)

    # One adjudication figure: model replication, future low NIBP, future action,
    # and the explicit negative ART gate.
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    pivot = models.pivot(index="centre", columns="model", values="auroc").loc[["INSPIRE", "MOVER"]]
    x = np.arange(2)
    axes[0, 0].bar(x - .18, pivot["M1_readable_history"], .36, label="Readable BP/treatment history", color="#DDA15E")
    axes[0, 0].bar(x + .18, pivot["M2_continuous_MAP"], .36, label="+ continuous prior MAP", color="#2A9D8F")
    axes[0, 0].set_xticks(x, ["INSPIRE", "MOVER"])
    axes[0, 0].set_ylim(.5, max(.7, pivot.max().max() + .03))
    axes[0, 0].set_ylabel("Patient-grouped OOF AUROC")
    axes[0, 0].set_title("A  Fixed first four: at least two MAP<65")
    axes[0, 0].legend(frameon=False, fontsize=9)

    def risk_for(outcome: str, predictor: str, restriction: str | None = None) -> pd.DataFrame:
        q = validity.loc[
            validity["outcome"].eq(outcome) & validity["predictor"].eq(predictor)
        ].copy()
        if restriction is not None:
            q = q.loc[q["predictor_level"].astype(str).eq(restriction)]
        return q

    # Absolute risks by three mutually interpretable fixed-four definitions.
    low_rows = []
    for predictor, label in [
        ("first4_any_low", "≥1 low"),
        ("first4_two_low", "≥2 low"),
        ("first4_consecutive_low", "consecutive low"),
    ]:
        q = validity.loc[
            validity.outcome.eq("post10_two_low")
            & validity.predictor.eq(predictor)
            & validity.predictor_level.eq(1)
        ].iloc[0]
        low_rows.append((label, 100 * q.risk, int(q.n)))
    axes[0, 1].bar([row[0] for row in low_rows], [row[1] for row in low_rows], color=["#A8DADC", "#457B9D", "#1D3557"])
    axes[0, 1].set_ylabel("Subsequent 10-min ≥2 low NIBP risk (%)")
    axes[0, 1].set_title("B  Temporal physiologic validity")

    action_rows = []
    for predictor, label in [
        ("first4_any_low", "≥1 low"),
        ("first4_two_low", "≥2 low"),
        ("first4_consecutive_low", "consecutive low"),
    ]:
        q = validity.loc[
            validity.outcome.eq("post10_any_bolus")
            & validity.predictor.eq(predictor)
            & validity.predictor_level.eq(1)
        ].iloc[0]
        action_rows.append((label, 100 * q.risk, int(q.n)))
    axes[1, 0].bar([row[0] for row in action_rows], [row[1] for row in action_rows], color=["#F4A261", "#E76F51", "#9D4E45"])
    axes[1, 0].set_ylabel("Subsequent 10-min IV bolus risk (%)")
    axes[1, 0].set_title("C  Temporal treatment-demand validity")

    art_plot = concurrent_summary.loc[concurrent_summary.rule.eq("all_discrete_ART")]
    axes[1, 1].bar(
        ["<2 low NIBP", "≥2 low NIBP"],
        art_plot.set_index("first4_two_low").loc[[0, 1], "threshold_agreement_weighted"],
        color="#999999",
    )
    axes[1, 1].axhline(.8, color="#C44E52", ls="--", label="predefined credibility reference")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Concurrent NIBP–ART threshold agreement")
    axes[1, 1].set_title("D  Discrete ART is not a gold standard here")
    axes[1, 1].legend(frameon=False, fontsize=9)
    fig.suptitle("Adjudicating the fixed-first-four early low-MAP construct", y=.995)
    fig.text(
        .5, .005,
        "Post-landmark outcomes use only the 10 min after the fourth NIBP value and require landmark ≤20 min. "
        "All analyses are secondary and post hoc.",
        ha="center", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=[0, .035, 1, .98])
    fig.savefig(OUT / "fig_fixed_four_endpoint_adjudication.png", dpi=230, bbox_inches="tight")
    fig.savefig(OUT / "fig_fixed_four_endpoint_adjudication.svg", bbox_inches="tight")
    plt.close(fig)

    # Decision gates: two-centre positive AUROC increment; future physiologic and
    # action risk at least 5pp higher for >=2 vs exactly one low; adjacency must
    # not be promoted unless it adds at least 5pp beyond nonconsecutive >=2.
    auc_rows = increment_table.loc[increment_table.metric.eq("delta_auroc")]
    two_centre_positive = bool((auc_rows.ci_low > 0).all())
    def rd(outcome: str, predictor: str, restriction: str) -> float:
        row = validity.loc[
            validity.outcome.eq(outcome)
            & validity.predictor.eq(predictor + "_risk_difference")
            & validity.predictor_level.astype(str).eq(restriction)
        ]
        return float(row.risk.iloc[0]) if len(row) else math.nan
    two_low_future_low_rd = rd("post10_two_low", "first4_two_low", "among_any_low")
    two_low_future_action_rd = rd("post10_any_bolus", "first4_two_low", "among_any_low")
    adjacency_future_low_rd = rd("post10_two_low", "first4_consecutive_low", "among_two_low")
    adjacency_future_action_rd = rd("post10_any_bolus", "first4_consecutive_low", "among_two_low")
    art_credibility = bool(
        concurrent["threshold_agreement"].mean() >= .8
        and burden.loc[burden.first4_two_low.eq(1), "LOG_ID"].nunique() >= 50
    )
    severity_two_low = adjusted_validity.loc[
        adjusted_validity.specification.eq("severity_adjusted")
        & adjusted_validity.term.eq("first4_two_low")
    ]
    independent_warning = bool(
        (severity_two_low.ci_low > 1).any() or (severity_two_low.ci_high < 1).any()
    )
    endpoint_decision = (
        "KEEP_AT_LEAST_TWO_LOW_AS_BURDEN_SENSITIVITY_NOT_INDEPENDENT_WARNING"
        if two_centre_positive
        and two_low_future_low_rd >= .05
        and two_low_future_action_rd >= .05
        and not (
            adjacency_future_low_rd >= .05 and adjacency_future_action_rd >= .05
        )
        and not independent_warning
        else "DO_NOT_REPLACE_ENDPOINT"
    )
    summary = {
        "status": endpoint_decision,
        "cohorts": {
            "INSPIRE": {
                "pairs": int(len(inspire)), "patients": int(inspire.subject_id.nunique()),
                "events": int(inspire.target_two_low_first4.sum()),
            },
            "MOVER": {
                "pairs": int(len(mover)), "patients": int(mover.patient_id.nunique()),
                "events": int(mover.target_two_low_first4.sum()),
            },
        },
        "map_audit": map_audit,
        "inspire_rebuild_audit": inspire_audit,
        "two_centre_model_metrics": models.to_dict(orient="records"),
        "two_centre_increment_ci": increment_table.to_dict(orient="records"),
        "temporal_validity_gates": {
            "at_least_two_vs_any_low_future_two_low_risk_difference": two_low_future_low_rd,
            "at_least_two_vs_any_low_future_bolus_risk_difference": two_low_future_action_rd,
            "adjacency_beyond_two_low_future_two_low_risk_difference": adjacency_future_low_rd,
            "adjacency_beyond_two_low_future_bolus_risk_difference": adjacency_future_action_rd,
        },
        "post_landmark_adjusted_associations": adjusted_validity.to_dict(orient="records"),
        "discrete_ART_gate": {
            "credible_as_gold_standard": art_credibility,
            "reason": (
                "concurrent threshold agreement below 0.80 and too few fixed-four low events "
                "with stable dense ART support"
            ),
        },
        "claim_boundary": (
            "Fixed-opportunity count of early low NIBP values with post-landmark temporal validity. "
            "Not continuous hypotension duration, causal susceptibility, or treatment benefit."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
