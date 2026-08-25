#!/usr/bin/env python3
"""Two-centre C02 analysis for fixed-first-four repeated early low NIBP.

INSPIRE and MOVER use the same outcome: among the first four conflict-free NIBP
MAP values in anaesthesia-start +0..30 min, at least one consecutive pair is
below 65 mmHg and separated by 2-15 minutes.  The result is a repeated-record
proxy, not continuous waveform duration.  Restricted row-level outputs remain
local; report outputs are aggregate only.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_c02_cross_database_minimal_bridge import FEATURES as COMMON_FEATURES
from run_c02_cross_database_minimal_bridge import make_pipeline as make_inspire_pipeline
from run_mover_c02_external_validation import calibration
from run_mover_c02_external_validation import canonicalize as canonicalize_mover
from run_mover_c02_external_validation import EXPANDED_FEATURES
from analyze_c02_deepening_v1 import fixed_capacity


from c02_runtime import private_workspace_root, protect_file, secure_directory


ROOT = private_workspace_root()
DATA = ROOT / "INSPIRE_v1.4.2"
BASE = ROOT / "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild"
INSPIRE_PAIR = BASE / "first_two_sampling_bridge/first_two_pair_cohort.csv.gz"
MOVER_PAIR = ROOT / "data/restricted/mover/extracted/mover_c02_cleaned_pair_cohort.csv.gz"
MOVER_MAPS = ROOT / "data/restricted/mover/extracted/mover_cleaned_early_map.csv.gz"
MOVER_UPGRADE = BASE / "clinical_endpoint_upgrade/repeated_low"
OUT = BASE / "clinical_endpoint_upgrade/two_centre_repeated_low"
RESTRICTED_INSPIRE = ROOT / "data/restricted/derived/inspire_c02_first4_repeated_low.csv.gz"

INSPIRE_EXPANDED_M1 = [
    "age_years", "bmi_kg_m2", "asa_numeric", "sex_common", "emop",
    "department", "procedure3", "interval_log1p", "prior_asa", "prior_emop",
    "prior_department", "prior_procedure3", "prior_first2_any_low",
]
INSPIRE_EXPANDED_M2 = INSPIRE_EXPANDED_M1 + ["prior_first_map", "prior_first2_change"]
MOVER_EXPANDED_M1 = EXPANDED_FEATURES["M1_prior_context_and_alert"]
MOVER_EXPANDED_M2 = EXPANDED_FEATURES["M2_continuous_prior_response"]


def generic_pipeline(d: pd.DataFrame, features: list[str]) -> Pipeline:
    categorical = [
        column for column in features
        if pd.api.types.is_object_dtype(d[column])
        or isinstance(d[column].dtype, pd.StringDtype)
    ]
    numeric = [column for column in features if column not in categorical]
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                ]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(C=.1, solver="lbfgs", max_iter=3000)),
    ])


def expanded_oof(
    d: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    m1_features: list[str],
    m2_features: list[str],
) -> dict[str, np.ndarray]:
    specs = {"M1_expanded": m1_features, "M2_expanded_continuous": m2_features}
    predictions = {name: np.full(len(d), np.nan) for name in specs}
    for train, test in GroupKFold(n_splits=5).split(d, y, groups=groups):
        for name, features in specs.items():
            fitted = generic_pipeline(d, features)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fitted.fit(d.iloc[train][features], y[train])
                predictions[name][test] = fitted.predict_proba(d.iloc[test][features])[:, 1]
    if any(not np.all(np.isfinite(values)) for values in predictions.values()):
        raise RuntimeError("non-finite expanded OOF predictions")
    return predictions


def inspire_features(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["age_years"] = pd.to_numeric(d["age"], errors="coerce")
    d["bmi_kg_m2"] = pd.to_numeric(d["bmi"], errors="coerce")
    d["asa_numeric"] = pd.to_numeric(d["asa"], errors="coerce")
    d["sex_common"] = d["sex"].astype("string").str.upper().replace(
        {"MALE": "M", "FEMALE": "F", "1": "M", "2": "F"}
    )
    d.loc[~d.sex_common.isin(["M", "F"]), "sex_common"] = pd.NA
    d["interval_log1p"] = np.log1p(pd.to_numeric(d.interval_days, errors="coerce").clip(lower=0))
    d["prior_first_map"] = pd.to_numeric(d.prior_first2_map_0, errors="coerce")
    d["prior_first2_change"] = (
        pd.to_numeric(d.prior_first2_map_1, errors="coerce") - d.prior_first_map
    )
    d["prior_first2_any_low"] = (
        d[["prior_first2_map_0", "prior_first2_map_1"]].min(axis=1) < 65
    ).astype(int)
    return d


def build_inspire_first4(pair: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ids = set(pd.to_numeric(pair.op_id, errors="coerce").dropna().astype(int))
    timing = pair[["op_id", "subject_id", "current_anstart_time"]].drop_duplicates("op_id")
    timing = timing.rename(columns={"current_anstart_time": "anstart_time"})
    timing["op_id"] = pd.to_numeric(timing.op_id, errors="coerce").astype("Int64")
    timing["subject_id"] = pd.to_numeric(timing.subject_id, errors="coerce")
    frames = []
    raw_nibp = 0
    for chunk in pd.read_csv(
        DATA / "vitals.csv.gz",
        usecols=["op_id", "subject_id", "chart_time", "item_name", "value"],
        chunksize=1_000_000,
    ):
        x = chunk.loc[chunk.item_name.eq("nibp_mbp")].copy()
        if x.empty:
            continue
        x["op_id"] = pd.to_numeric(x.op_id, errors="coerce").astype("Int64")
        x = x.loc[x.op_id.isin(ids)]
        if x.empty:
            continue
        raw_nibp += len(x)
        x = x.merge(timing, on=["op_id", "subject_id"], how="inner", validate="many_to_one")
        x["relative_min"] = pd.to_numeric(x.chart_time, errors="coerce") - x.anstart_time
        x["value"] = pd.to_numeric(x.value, errors="coerce")
        x = x.loc[
            x.relative_min.between(0, 30, inclusive="both")
            & x.value.between(20, 200, inclusive="both")
        ]
        if not x.empty:
            frames.append(x[["op_id", "chart_time", "relative_min", "value"]])
    raw = pd.concat(frames, ignore_index=True)
    key = (
        raw.groupby(["op_id", "chart_time"], as_index=False, observed=True)
        .agg(relative_min=("relative_min", "min"), records=("value", "size"),
             distinct_values=("value", "nunique"), value=("value", "first"))
    )
    conflicts = int(key.distinct_values.gt(1).sum())
    key = key.loc[key.distinct_values.eq(1)].sort_values(["op_id", "relative_min", "chart_time"])
    rows = []
    for op_id, frame in key.groupby("op_id", sort=False, observed=True):
        first4 = frame.head(4)
        if len(first4) < 4:
            continue
        values = first4.value.to_numpy(float)
        times = first4.relative_min.to_numpy(float)
        low = values < 65
        gap = np.diff(times)
        event_2_15 = low[:-1] & low[1:] & (gap >= 2) & (gap <= 15)
        event_3_10 = low[:-1] & low[1:] & (gap >= 3) & (gap <= 10)
        rows.append(
            {
                "op_id": int(op_id),
                "target_repeated_low_first4_gap_2_15": int(event_2_15.any()),
                "target_repeated_low_first4_gap_3_10": int(event_3_10.any()),
                "target_two_low_first4": int(low.sum() >= 2),
                **{f"current_first4_map_{i}": float(values[i]) for i in range(4)},
                **{f"current_first4_rel_{i}": float(times[i]) for i in range(4)},
            }
        )
    out = pd.DataFrame(rows)
    audit = {
        "raw_nibp_rows_for_current_operations": int(raw_nibp),
        "rows_0_30": int(len(raw)),
        "same_time_keys": int(len(key) + conflicts),
        "conflicting_same_time_keys_excluded": conflicts,
        "operations_with_four_observations": int(len(out)),
    }
    return out, audit


def metric(y: np.ndarray, prediction: np.ndarray) -> dict:
    intercept, slope = calibration(y, prediction)
    return {
        "n": int(len(y)), "events": int(y.sum()), "event_rate": float(y.mean()),
        "mean_prediction": float(prediction.mean()),
        "auroc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
        "calibration_intercept": intercept, "calibration_slope": slope,
    }


def oof(d: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    specs = {
        "M1_binary_prior_alert": COMMON_FEATURES["M1_binary_prior_alert"],
        "M2_continuous_prior_response": COMMON_FEATURES["M2_continuous_prior_response"],
    }
    predictions = {name: np.full(len(d), np.nan) for name in specs}
    for train, test in GroupKFold(n_splits=5).split(d, y, groups=groups):
        for name, features in specs.items():
            model = make_inspire_pipeline(d, features)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model.fit(d.iloc[train][features], y[train])
                predictions[name][test] = model.predict_proba(d.iloc[test][features])[:, 1]
    if any(not np.all(np.isfinite(p)) for p in predictions.values()):
        raise RuntimeError("non-finite INSPIRE OOF prediction")
    return predictions


def bootstrap_increment(
    centre: str, y: np.ndarray, groups: np.ndarray, p1: np.ndarray, p2: np.ndarray,
    reps: int = 1000,
) -> pd.DataFrame:
    unique = pd.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(20260815 if centre == "INSPIRE" else 20260816)
    rows = []
    for rep in range(reps):
        sample = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([lookup[group] for group in sample])
        yy = y[index]
        if np.unique(yy).size < 2:
            continue
        rows.append(
            {
                "centre": centre,
                "rep": rep,
                "delta_auroc": roc_auc_score(yy, p2[index]) - roc_auc_score(yy, p1[index]),
                "delta_average_precision": average_precision_score(yy, p2[index]) - average_precision_score(yy, p1[index]),
                "brier_improvement": brier_score_loss(yy, p1[index]) - brier_score_loss(yy, p2[index]),
                "log_loss_improvement": log_loss(yy, p1[index], labels=[0, 1]) - log_loss(yy, p2[index], labels=[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def summary_ci(boot: pd.DataFrame, centre: str, point: dict) -> pd.DataFrame:
    rows = []
    for name, value in point.items():
        rows.append(
            {
                "centre": centre, "metric": name, "point": value,
                "ci_low": float(boot[name].quantile(.025)),
                "ci_high": float(boot[name].quantile(.975)),
                "bootstrap_reps": int(len(boot)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    secure_directory(OUT)
    secure_directory(RESTRICTED_INSPIRE.parent)
    raw_pair = pd.read_csv(INSPIRE_PAIR, low_memory=False)
    raw_pair = raw_pair.loc[
        raw_pair.antype.astype("string").str.strip().eq("General")
        & raw_pair.prior_antype.astype("string").str.strip().eq("General")
    ].copy()
    first4, audit = build_inspire_first4(raw_pair)
    inspire = inspire_features(raw_pair.merge(first4, on="op_id", how="inner", validate="one_to_one"))
    inspire.to_csv(RESTRICTED_INSPIRE, index=False, compression="gzip")
    protect_file(RESTRICTED_INSPIRE)

    endpoint = "target_repeated_low_first4_gap_2_15"
    y = inspire[endpoint].astype(int).to_numpy()
    groups = inspire.subject_id.to_numpy()
    predictions = oof(inspire, y, groups)
    model_rows = []
    for name, prediction in predictions.items():
        model_rows.append({"centre": "INSPIRE", "model": name, **metric(y, prediction)})
    point = {
        "delta_auroc": roc_auc_score(y, predictions["M2_continuous_prior_response"]) - roc_auc_score(y, predictions["M1_binary_prior_alert"]),
        "delta_average_precision": average_precision_score(y, predictions["M2_continuous_prior_response"]) - average_precision_score(y, predictions["M1_binary_prior_alert"]),
        "brier_improvement": brier_score_loss(y, predictions["M1_binary_prior_alert"]) - brier_score_loss(y, predictions["M2_continuous_prior_response"]),
        "log_loss_improvement": log_loss(y, predictions["M1_binary_prior_alert"], labels=[0, 1]) - log_loss(y, predictions["M2_continuous_prior_response"], labels=[0, 1]),
    }
    inspire_boot = bootstrap_increment(
        "INSPIRE", y, groups,
        predictions["M1_binary_prior_alert"], predictions["M2_continuous_prior_response"]
    )
    inspire_ci = summary_ci(inspire_boot, "INSPIRE", point)

    # Fit fixed INSPIRE pipelines for unchanged transport to the existing MOVER cohort.
    for name, features in {
        "M1_binary_prior_alert": COMMON_FEATURES["M1_binary_prior_alert"],
        "M2_continuous_prior_response": COMMON_FEATURES["M2_continuous_prior_response"],
    }.items():
        fitted = make_inspire_pipeline(inspire, features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fitted.fit(inspire[features], y)
        joblib.dump(fitted, OUT / f"inspire_fixed_{name}.joblib")

    mover_pair = canonicalize_mover(
        pd.read_csv(MOVER_PAIR, dtype={"LOG_ID": str}, low_memory=False)
    )
    # Recover the fixed-first-four outcome at operation level from the already
    # constructed restricted outcome table by matching eligible LOG_IDs.
    mover_map = pd.read_csv(
        MOVER_MAPS,
        usecols=[
            "LOG_ID", "RECORDED_TIME", "relative_min", "value", "RECORD_TYPE",
            "FLO_MEAS_NAME", "FLO_DISPLAY_NAME", "modality_hint",
        ],
        dtype={"LOG_ID": str},
        low_memory=False,
    )
    mover_ids = set(mover_pair.LOG_ID.astype(str))
    mover_map = mover_map.loc[
        mover_map.LOG_ID.astype(str).isin(mover_ids)
        & mover_map.RECORD_TYPE.eq("INTRA-OP")
        & mover_map.FLO_MEAS_NAME.eq("UC ANE R BLOOD PRESSURE - MAP")
        & mover_map.FLO_DISPLAY_NAME.eq("NIBP - MAP")
        & mover_map.modality_hint.eq("NIBP")
    ].copy()
    mover_map["time"] = pd.to_datetime(mover_map.RECORDED_TIME, errors="coerce")
    mover_map["relative_min"] = pd.to_numeric(mover_map.relative_min, errors="coerce")
    mover_map["value"] = pd.to_numeric(mover_map.value, errors="coerce")
    mover_map = mover_map.loc[
        mover_map.relative_min.between(0, 30, inclusive="both")
        & mover_map.value.between(20, 200, inclusive="both")
        & mover_map.time.notna()
    ]
    mover_key = (
        mover_map.groupby(["LOG_ID", "time"], as_index=False, observed=True)
        .agg(relative_min=("relative_min", "min"), distinct_values=("value", "nunique"),
             value=("value", "first"))
    )
    mover_key = mover_key.loc[mover_key.distinct_values.eq(1)].sort_values(["LOG_ID", "relative_min", "time"])
    mover_outcome_rows = []
    for log_id, frame in mover_key.groupby("LOG_ID", sort=False, observed=True):
        first4_frame = frame.head(4)
        if len(first4_frame) < 4:
            continue
        values = first4_frame.value.to_numpy(float)
        times = first4_frame.relative_min.to_numpy(float)
        low = values < 65
        gap = np.diff(times)
        mover_outcome_rows.append(
            {
                "LOG_ID": str(log_id),
                endpoint: int((low[:-1] & low[1:] & (gap >= 2) & (gap <= 15)).any()),
                **{f"current_first4_rel_{i}": float(times[i]) for i in range(4)},
            }
        )
    mover_outcomes = pd.DataFrame(mover_outcome_rows)
    mover_fixed = mover_pair.merge(mover_outcomes, on="LOG_ID", how="inner", validate="one_to_one")
    mover_y = mover_fixed[endpoint].astype(int).to_numpy()
    fixed_rows = []
    fixed_predictions: dict[str, np.ndarray] = {}
    for name, features in {
        "M1_binary_prior_alert": COMMON_FEATURES["M1_binary_prior_alert"],
        "M2_continuous_prior_response": COMMON_FEATURES["M2_continuous_prior_response"],
    }.items():
        fitted = joblib.load(OUT / f"inspire_fixed_{name}.joblib")
        prediction = fitted.predict_proba(mover_fixed[features])[:, 1]
        fixed_predictions[name] = prediction
        fixed_rows.append(
            {"analysis": "fixed_INSPIRE_to_MOVER", "model": name, **metric(mover_y, prediction)}
        )
    fixed_point = {
        "delta_auroc": roc_auc_score(mover_y, fixed_predictions["M2_continuous_prior_response"]) - roc_auc_score(mover_y, fixed_predictions["M1_binary_prior_alert"]),
        "delta_average_precision": average_precision_score(mover_y, fixed_predictions["M2_continuous_prior_response"]) - average_precision_score(mover_y, fixed_predictions["M1_binary_prior_alert"]),
        "brier_improvement": brier_score_loss(mover_y, fixed_predictions["M1_binary_prior_alert"]) - brier_score_loss(mover_y, fixed_predictions["M2_continuous_prior_response"]),
        "log_loss_improvement": log_loss(mover_y, fixed_predictions["M1_binary_prior_alert"], labels=[0, 1]) - log_loss(mover_y, fixed_predictions["M2_continuous_prior_response"], labels=[0, 1]),
    }
    fixed_boot = bootstrap_increment(
        "FIXED_INSPIRE_TO_MOVER", mover_y, mover_fixed.patient_id.to_numpy(),
        fixed_predictions["M1_binary_prior_alert"], fixed_predictions["M2_continuous_prior_response"]
    )
    fixed_ci = summary_ci(fixed_boot, "FIXED_INSPIRE_TO_MOVER", fixed_point)
    pd.DataFrame(fixed_rows).to_csv(OUT / "fixed_model_transport_metrics.csv", index=False)
    fixed_ci.to_csv(OUT / "fixed_model_transport_increment_ci.csv", index=False)
    fixed_boot.to_csv(OUT / "fixed_model_transport_bootstrap.csv.gz", index=False, compression="gzip")

    # Timing support is an important construct check because the two centres
    # differ in recording cadence even under the same fixed-four opportunity.
    timing_rows = []
    for centre, frame in [("INSPIRE", inspire), ("MOVER", mover_fixed)]:
        for index in range(4):
            values = pd.to_numeric(frame[f"current_first4_rel_{index}"], errors="coerce")
            timing_rows.append(
                {
                    "centre": centre, "measurement_order": index + 1,
                    "n": int(values.notna().sum()), "median_min": float(values.median()),
                    "q1_min": float(values.quantile(.25)), "q3_min": float(values.quantile(.75)),
                    "p90_min": float(values.quantile(.90)),
                }
            )
    pd.DataFrame(timing_rows).to_csv(OUT / "first4_timing_support.csv", index=False)
    mover_outcome = pd.read_csv(MOVER_UPGRADE / "model_metrics.csv")
    mover_metrics = mover_outcome.loc[
        mover_outcome.endpoint.eq("consecutive_low_first4_gap_2_15")
    ].copy()
    mover_metrics.insert(0, "centre", "MOVER")
    model_table = pd.concat([pd.DataFrame(model_rows), mover_metrics], ignore_index=True)

    mover_increment = pd.read_csv(MOVER_UPGRADE / "increment_cluster_bootstrap.csv")
    mover_ci = mover_increment.loc[
        mover_increment.endpoint.eq("consecutive_low_first4_gap_2_15")
    ].rename(columns={"bootstrap_reps": "bootstrap_reps"})[
        ["metric", "point", "ci_low", "ci_high", "bootstrap_reps"]
    ]
    mover_ci.insert(0, "centre", "MOVER")
    ci_table = pd.concat([inspire_ci, mover_ci], ignore_index=True)
    model_table.to_csv(OUT / "two_centre_model_metrics.csv", index=False)
    ci_table.to_csv(OUT / "two_centre_increment_ci.csv", index=False)
    inspire_boot.to_csv(OUT / "inspire_increment_bootstrap.csv.gz", index=False, compression="gzip")

    capacity, _ = fixed_capacity(
        inspire, y, groups,
        predictions["M1_binary_prior_alert"], predictions["M2_continuous_prior_response"]
    )
    capacity.insert(0, "centre", "INSPIRE")
    mover_capacity = pd.read_csv(MOVER_UPGRADE / "fixed_capacity_reclassification.csv")
    mover_capacity = mover_capacity.loc[
        mover_capacity.endpoint.eq("consecutive_low_first4_gap_2_15")
    ].drop(columns="endpoint")
    mover_capacity.insert(0, "centre", "MOVER")
    pd.concat([capacity, mover_capacity], ignore_index=True).to_csv(
        OUT / "two_centre_fixed_capacity.csv", index=False
    )

    # Centre-specific expanded context: same scientific M1->M2 contrast, with
    # all available current/prior case context in each centre.
    expanded_rows = []
    expanded_ci_frames = []
    for centre, frame, yy, gg, m1_features, m2_features in [
        (
            "INSPIRE", inspire, y, groups,
            INSPIRE_EXPANDED_M1, INSPIRE_EXPANDED_M2,
        ),
        (
            "MOVER", mover_fixed, mover_y, mover_fixed.patient_id.to_numpy(),
            MOVER_EXPANDED_M1, MOVER_EXPANDED_M2,
        ),
    ]:
        prediction = expanded_oof(frame, yy, gg, m1_features, m2_features)
        for name, values in prediction.items():
            expanded_rows.append({"centre": centre, "model": name, **metric(yy, values)})
        boot = bootstrap_increment(
            centre + "_EXPANDED", yy, gg,
            prediction["M1_expanded"], prediction["M2_expanded_continuous"]
        )
        point_expanded = {
            "delta_auroc": roc_auc_score(yy, prediction["M2_expanded_continuous"]) - roc_auc_score(yy, prediction["M1_expanded"]),
            "delta_average_precision": average_precision_score(yy, prediction["M2_expanded_continuous"]) - average_precision_score(yy, prediction["M1_expanded"]),
            "brier_improvement": brier_score_loss(yy, prediction["M1_expanded"]) - brier_score_loss(yy, prediction["M2_expanded_continuous"]),
            "log_loss_improvement": log_loss(yy, prediction["M1_expanded"], labels=[0, 1]) - log_loss(yy, prediction["M2_expanded_continuous"], labels=[0, 1]),
        }
        current_ci = summary_ci(boot, centre, point_expanded)
        expanded_ci_frames.append(current_ci)
    expanded_metrics = pd.DataFrame(expanded_rows)
    expanded_ci = pd.concat(expanded_ci_frames, ignore_index=True)
    expanded_metrics.to_csv(OUT / "two_centre_expanded_model_metrics.csv", index=False)
    expanded_ci.to_csv(OUT / "two_centre_expanded_increment_ci.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    pivot = expanded_metrics.pivot(index="centre", columns="model", values="auroc").loc[["INSPIRE", "MOVER"]]
    x = np.arange(2)
    axes[0].bar(x - .18, pivot["M1_expanded"], .36, label="M1 expanded context", color="#9E9AC8")
    axes[0].bar(x + .18, pivot["M2_expanded_continuous"], .36, label="M2 + continuous history", color="#756BB1")
    axes[0].set_xticks(x, ["INSPIRE", "MOVER"])
    axes[0].set_ylim(.5, max(.7, float(pivot.max().max()) + .03))
    axes[0].set_ylabel("Patient-grouped OOF AUROC")
    axes[0].set_title("Fixed-first-four repeated-low endpoint")
    axes[0].legend(frameon=False)

    auroc_ci = expanded_ci.loc[expanded_ci.metric.eq("delta_auroc")].set_index("centre").loc[["INSPIRE", "MOVER"]]
    axes[1].errorbar(
        x, auroc_ci.point,
        yerr=np.vstack([auroc_ci.point - auroc_ci.ci_low, auroc_ci.ci_high - auroc_ci.point]),
        fmt="o", capsize=4, color="#2A9D8F",
    )
    axes[1].axhline(0, color="#333", lw=1)
    axes[1].set_xticks(x, ["INSPIRE", "MOVER"])
    axes[1].set_ylabel("M2 − M1 AUROC")
    axes[1].set_title("Increment replicated across centres")
    fig.tight_layout()
    fig.savefig(OUT / "fig_two_centre_repeated_low.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "fig_two_centre_repeated_low.svg", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "status": (
            "GO_TWO_CENTRE_FIXED_OPPORTUNITY_REPEATED_LOW"
            if point["delta_auroc"] > 0 and float(inspire_ci.loc[inspire_ci.metric.eq("delta_auroc"), "ci_low"].iloc[0]) > 0
            and float(mover_ci.loc[mover_ci.metric.eq("delta_auroc"), "ci_low"].iloc[0]) > 0
            else "CONDITIONAL_REPEATED_LOW_NOT_REPLICATED"
        ),
        "endpoint": (
            "among first four conflict-free NIBP MAP values in 0-30 min, at least one "
            "consecutive pair <65 separated by 2-15 min"
        ),
        "construct": "fixed-opportunity repeated-record proxy; not continuous waveform duration",
        "inspire_audit": audit,
        "inspire_pairs": int(len(inspire)),
        "inspire_patients": int(inspire.subject_id.nunique()),
        "inspire_events": int(y.sum()),
        "mover_pairs": 7410,
        "mover_patients": 5154,
        "mover_events": 164,
        "metrics": model_table.to_dict(orient="records"),
        "increments": ci_table.to_dict(orient="records"),
        "expanded_metrics": expanded_metrics.to_dict(orient="records"),
        "expanded_increments": expanded_ci.to_dict(orient="records"),
        "fixed_model_transport_metrics": fixed_rows,
        "fixed_model_transport_increments": fixed_ci.to_dict(orient="records"),
        "claim_boundary": (
            "Two-centre feature replication for repeated early low NIBP records. "
            "No continuous hypotension burden, treatment, organ-outcome, or deployment claim."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
