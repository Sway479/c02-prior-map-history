#!/usr/bin/env python3
"""Does continuous prior response add beyond prior repeated-low alerts?

Both centres require four conflict-free NIBP MAP observations in the current
and immediately prior anaesthetic.  The current endpoint and prior alert use
the same fixed-first-four repeated-low definition.  Continuous terms are then
tested beyond both the prior first-two-any-low alert and the prior repeated-low
alert.  Aggregate outputs only are written.
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

from analyze_c02_deepening_v1 import fixed_capacity
from analyze_c02_two_centre_repeated_low import build_inspire_first4, inspire_features, metric
from run_mover_c02_external_validation import (
    EXPANDED_FEATURES, calibration, canonicalize as canonicalize_mover, make_pipeline,
)


from c02_runtime import private_workspace_root


ROOT = private_workspace_root()
BASE = ROOT / "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild"
INSPIRE_PAIR = BASE / "first_two_sampling_bridge/first_two_pair_cohort.csv.gz"
OPS = ROOT / "INSPIRE_v1.4.2/operations.csv.gz"
MOVER_PAIR = ROOT / "data/restricted/mover/extracted/mover_c02_cleaned_pair_cohort.csv.gz"
MOVER_MAPS = ROOT / "data/restricted/mover/extracted/mover_cleaned_early_map.csv.gz"
OUT = BASE / "clinical_endpoint_upgrade/repeated_alert_comparator"


COMMON_BASE = ["age_years", "bmi_kg_m2", "asa_numeric", "sex_common", "interval_log1p"]
COMMON_SPECS = {
    "M1_first2_binary": COMMON_BASE + ["prior_first2_any_low"],
    "M1_repeated_binary": COMMON_BASE + ["prior_repeated_low_first4"],
    "M1_both_binary": COMMON_BASE + ["prior_first2_any_low", "prior_repeated_low_first4"],
    "M2_both_binary_continuous": COMMON_BASE + [
        "prior_first2_any_low", "prior_repeated_low_first4",
        "prior_first_map", "prior_first2_change",
    ],
}

INSPIRE_EXPANDED_BASE = [
    "age_years", "bmi_kg_m2", "asa_numeric", "sex_common", "emop",
    "department", "procedure3", "interval_log1p", "prior_asa", "prior_emop",
    "prior_department", "prior_procedure3",
]
INSPIRE_EXPANDED_SPECS = {
    "M1_expanded_both_binary": INSPIRE_EXPANDED_BASE + [
        "prior_first2_any_low", "prior_repeated_low_first4"
    ],
    "M2_expanded_both_binary_continuous": INSPIRE_EXPANDED_BASE + [
        "prior_first2_any_low", "prior_repeated_low_first4",
        "prior_first_map", "prior_first2_change",
    ],
}
MOVER_EXPANDED_BASE = [
    column for column in EXPANDED_FEATURES["M1_prior_context_and_alert"]
    if column != "prior_first2_any_low"
]
MOVER_EXPANDED_SPECS = {
    "M1_expanded_both_binary": MOVER_EXPANDED_BASE + [
        "prior_first2_any_low", "prior_repeated_low_first4"
    ],
    "M2_expanded_both_binary_continuous": MOVER_EXPANDED_BASE + [
        "prior_first2_any_low", "prior_repeated_low_first4",
        "prior_first_map", "prior_first2_change",
    ],
}


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


def all_relevant_inspire_operations(pair: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ops = pd.read_csv(OPS, usecols=["op_id", "subject_id", "anstart_time"])
    for column in ["op_id", "subject_id", "anstart_time"]:
        ops[column] = pd.to_numeric(ops[column], errors="coerce")
    duplicate_timing = ops.duplicated(["subject_id", "anstart_time"], keep=False)
    conflicting_timing = int(
        ops.loc[duplicate_timing].groupby(["subject_id", "anstart_time"]).op_id.nunique().gt(1).sum()
    )
    safe_ops = ops.loc[~ops.duplicated(["subject_id", "anstart_time"], keep=False)].copy()
    prior = pair[["subject_id", "prior_anstart_time"]].merge(
        safe_ops,
        left_on=["subject_id", "prior_anstart_time"],
        right_on=["subject_id", "anstart_time"],
        how="left",
        validate="many_to_one",
    )
    prior_map = prior[["subject_id", "prior_anstart_time", "op_id"]].drop_duplicates(
        ["subject_id", "prior_anstart_time"]
    ).rename(columns={"op_id": "prior_op_id"})
    paired = pair.merge(
        prior_map, on=["subject_id", "prior_anstart_time"], how="left", validate="many_to_one"
    )
    current_timing = paired[["op_id", "subject_id", "current_anstart_time"]].rename(
        columns={"current_anstart_time": "current_anstart_time"}
    )
    prior_timing = paired[["prior_op_id", "subject_id", "prior_anstart_time"]].dropna().rename(
        columns={"prior_op_id": "op_id", "prior_anstart_time": "current_anstart_time"}
    )
    prior_timing["op_id"] = prior_timing.op_id.astype(int)
    timing = pd.concat([current_timing, prior_timing], ignore_index=True).drop_duplicates("op_id")
    audit = {
        "operation_timing_conflict_keys_excluded": conflicting_timing,
        "pairs_before_prior_op_link": int(len(pair)),
        "pairs_with_prior_op_link": int(paired.prior_op_id.notna().sum()),
        "relevant_unique_operations": int(timing.op_id.nunique()),
    }
    return paired, timing, audit


def build_mover_operation_outcomes(pair: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ids = set(pair.LOG_ID.astype(str)) | set(pair.prior_LOG_ID.astype(str))
    usecols = [
        "LOG_ID", "RECORDED_TIME", "relative_min", "value", "RECORD_TYPE",
        "FLO_MEAS_NAME", "FLO_DISPLAY_NAME", "modality_hint",
    ]
    maps = pd.read_csv(MOVER_MAPS, usecols=usecols, dtype={"LOG_ID": str}, low_memory=False)
    maps = maps.loc[
        maps.LOG_ID.astype(str).isin(ids)
        & maps.RECORD_TYPE.eq("INTRA-OP")
        & maps.FLO_MEAS_NAME.eq("UC ANE R BLOOD PRESSURE - MAP")
        & maps.FLO_DISPLAY_NAME.eq("NIBP - MAP")
        & maps.modality_hint.eq("NIBP")
    ].copy()
    maps["time"] = pd.to_datetime(maps.RECORDED_TIME, errors="coerce")
    maps["relative_min"] = pd.to_numeric(maps.relative_min, errors="coerce")
    maps["value"] = pd.to_numeric(maps.value, errors="coerce")
    maps = maps.loc[
        maps.relative_min.between(0, 30, inclusive="both")
        & maps.value.between(20, 200, inclusive="both") & maps.time.notna()
    ]
    key = (
        maps.groupby(["LOG_ID", "time"], as_index=False, observed=True)
        .agg(relative_min=("relative_min", "min"), distinct_values=("value", "nunique"),
             value=("value", "first"))
    )
    conflicts = int(key.distinct_values.gt(1).sum())
    key = key.loc[key.distinct_values.eq(1)].sort_values(["LOG_ID", "relative_min", "time"])
    rows = []
    for log_id, frame in key.groupby("LOG_ID", sort=False, observed=True):
        first4 = frame.head(4)
        if len(first4) < 4:
            continue
        values = first4.value.to_numpy(float)
        times = first4.relative_min.to_numpy(float)
        low = values < 65
        gap = np.diff(times)
        rows.append(
            {
                "LOG_ID": str(log_id),
                "repeated_low_first4": int((low[:-1] & low[1:] & (gap >= 2) & (gap <= 15)).any()),
            }
        )
    return pd.DataFrame(rows), {
        "relevant_map_rows": int(len(maps)), "same_time_conflicts_excluded": conflicts,
        "operations_with_first4": int(len(rows)),
    }


def oof_predictions(
    d: pd.DataFrame, y: np.ndarray, groups: np.ndarray, specs: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    predictions = {name: np.full(len(d), np.nan) for name in specs}
    for train, test in GroupKFold(n_splits=5).split(d, y, groups=groups):
        for name, features in specs.items():
            fitted = generic_pipeline(d, features)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fitted.fit(d.iloc[train][features], y[train])
                predictions[name][test] = fitted.predict_proba(d.iloc[test][features])[:, 1]
    if any(not np.all(np.isfinite(p)) for p in predictions.values()):
        raise RuntimeError("non-finite OOF predictions")
    return predictions


def contrast(
    centre: str, y: np.ndarray, groups: np.ndarray, p1: np.ndarray, p2: np.ndarray,
    reps: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = pd.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(20260817 if centre == "INSPIRE" else 20260818)

    def values(index: np.ndarray) -> dict:
        yy = y[index]
        return {
            "delta_auroc": roc_auc_score(yy, p2[index]) - roc_auc_score(yy, p1[index]),
            "delta_average_precision": average_precision_score(yy, p2[index]) - average_precision_score(yy, p1[index]),
            "brier_improvement": brier_score_loss(yy, p1[index]) - brier_score_loss(yy, p2[index]),
            "log_loss_improvement": log_loss(yy, p1[index], labels=[0, 1]) - log_loss(yy, p2[index], labels=[0, 1]),
        }

    point = values(np.arange(len(y)))
    rows = []
    for rep in range(reps):
        sample = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([lookup[group] for group in sample])
        if np.unique(y[index]).size < 2:
            continue
        rows.append({"centre": centre, "rep": rep, **values(index)})
    boot = pd.DataFrame(rows)
    summaries = []
    for name, value in point.items():
        summaries.append({
            "centre": centre, "metric": name, "point": value,
            "ci_low": float(boot[name].quantile(.025)),
            "ci_high": float(boot[name].quantile(.975)),
            "bootstrap_reps": int(len(boot)),
        })
    return pd.DataFrame(summaries), boot


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    inspire_pair = pd.read_csv(INSPIRE_PAIR, low_memory=False)
    inspire_pair = inspire_pair.loc[
        inspire_pair.antype.astype("string").str.strip().eq("General")
        & inspire_pair.prior_antype.astype("string").str.strip().eq("General")
    ].copy()
    linked, timing, inspire_link_audit = all_relevant_inspire_operations(inspire_pair)
    first4, inspire_first4_audit = build_inspire_first4(timing)
    current = first4[["op_id", "target_repeated_low_first4_gap_2_15"]].rename(
        columns={"target_repeated_low_first4_gap_2_15": "target_repeated_low_first4"}
    )
    prior = first4[["op_id", "target_repeated_low_first4_gap_2_15"]].rename(
        columns={"op_id": "prior_op_id", "target_repeated_low_first4_gap_2_15": "prior_repeated_low_first4"}
    )
    inspire = linked.merge(current, on="op_id", how="inner", validate="one_to_one").merge(
        prior, on="prior_op_id", how="inner", validate="many_to_one"
    )
    inspire = inspire_features(inspire)

    mover_pair = canonicalize_mover(pd.read_csv(MOVER_PAIR, dtype={"LOG_ID": str}, low_memory=False))
    mover_outcome, mover_audit = build_mover_operation_outcomes(mover_pair)
    current_m = mover_outcome.rename(columns={
        "repeated_low_first4": "target_repeated_low_first4"
    })
    prior_m = mover_outcome.rename(columns={
        "LOG_ID": "prior_LOG_ID", "repeated_low_first4": "prior_repeated_low_first4"
    })
    mover = mover_pair.merge(current_m, on="LOG_ID", how="inner", validate="one_to_one").merge(
        prior_m, on="prior_LOG_ID", how="inner", validate="many_to_one"
    )

    all_metrics = []
    all_ci = []
    all_boot = []
    all_capacity = []
    predictions_by_centre = {}
    for centre, d, target, group in [
        ("INSPIRE", inspire, "target_repeated_low_first4", "subject_id"),
        ("MOVER", mover, "target_repeated_low_first4", "patient_id"),
    ]:
        y = d[target].astype(int).to_numpy()
        groups = d[group].to_numpy()
        prediction = oof_predictions(d, y, groups, COMMON_SPECS)
        predictions_by_centre[centre] = prediction
        for name, values in prediction.items():
            all_metrics.append({"centre": centre, "model": name, **metric(y, values)})
        ci, boot = contrast(
            centre, y, groups,
            prediction["M1_both_binary"], prediction["M2_both_binary_continuous"]
        )
        all_ci.append(ci)
        all_boot.append(boot)
        capacity, _ = fixed_capacity(
            d, y, groups,
            prediction["M1_both_binary"], prediction["M2_both_binary_continuous"]
        )
        capacity.insert(0, "centre", centre)
        all_capacity.append(capacity)

    metrics = pd.DataFrame(all_metrics)
    ci_table = pd.concat(all_ci, ignore_index=True)
    capacity_table = pd.concat(all_capacity, ignore_index=True)
    metrics.to_csv(OUT / "two_centre_model_metrics.csv", index=False)
    ci_table.to_csv(OUT / "two_centre_continuous_increment_ci.csv", index=False)
    pd.concat(all_boot, ignore_index=True).to_csv(
        OUT / "two_centre_continuous_increment_bootstrap.csv.gz", index=False, compression="gzip"
    )
    capacity_table.to_csv(OUT / "two_centre_fixed_capacity.csv", index=False)

    # Conservative, centre-specific case-mix baselines. These are not fixed
    # cross-centre models; they test whether the feature increment survives
    # richer local current/prior surgical context.
    expanded_metrics = []
    expanded_ci_frames = []
    expanded_capacity_frames = []
    for centre, d, specs, target, group in [
        ("INSPIRE", inspire, INSPIRE_EXPANDED_SPECS, "target_repeated_low_first4", "subject_id"),
        ("MOVER", mover, MOVER_EXPANDED_SPECS, "target_repeated_low_first4", "patient_id"),
    ]:
        yy = d[target].astype(int).to_numpy()
        gg = d[group].to_numpy()
        pred = oof_predictions(d, yy, gg, specs)
        m1_name = "M1_expanded_both_binary"
        m2_name = "M2_expanded_both_binary_continuous"
        for name, values in pred.items():
            expanded_metrics.append({"centre": centre, "model": name, **metric(yy, values)})
        current_ci, _ = contrast(centre + "_EXPANDED", yy, gg, pred[m1_name], pred[m2_name])
        current_ci["centre"] = centre
        expanded_ci_frames.append(current_ci)
        current_capacity, _ = fixed_capacity(d, yy, gg, pred[m1_name], pred[m2_name])
        current_capacity.insert(0, "centre", centre)
        expanded_capacity_frames.append(current_capacity)
    expanded_metrics_frame = pd.DataFrame(expanded_metrics)
    expanded_ci_table = pd.concat(expanded_ci_frames, ignore_index=True)
    expanded_capacity_table = pd.concat(expanded_capacity_frames, ignore_index=True)
    expanded_metrics_frame.to_csv(OUT / "two_centre_expanded_model_metrics.csv", index=False)
    expanded_ci_table.to_csv(OUT / "two_centre_expanded_increment_ci.csv", index=False)
    expanded_capacity_table.to_csv(OUT / "two_centre_expanded_fixed_capacity.csv", index=False)

    # Train the exact INSPIRE common models and apply unchanged to MOVER.
    fixed_rows = []
    fixed_predictions = {}
    for name in ["M1_both_binary", "M2_both_binary_continuous"]:
        features = COMMON_SPECS[name]
        fitted = generic_pipeline(inspire, features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fitted.fit(inspire[features], inspire.target_repeated_low_first4.astype(int))
        joblib.dump(fitted, OUT / f"inspire_fixed_{name}.joblib")
        prediction = fitted.predict_proba(mover[features])[:, 1]
        fixed_predictions[name] = prediction
        fixed_rows.append({"model": name, **metric(mover.target_repeated_low_first4.to_numpy(int), prediction)})
    fixed_y = mover.target_repeated_low_first4.to_numpy(int)
    fixed_ci, fixed_boot = contrast(
        "FIXED_INSPIRE_TO_MOVER", fixed_y, mover.patient_id.to_numpy(),
        fixed_predictions["M1_both_binary"], fixed_predictions["M2_both_binary_continuous"]
    )
    pd.DataFrame(fixed_rows).to_csv(OUT / "fixed_model_transport_metrics.csv", index=False)
    fixed_ci.to_csv(OUT / "fixed_model_transport_increment_ci.csv", index=False)

    # Main figure: binary-alert ladder and continuous increment.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    order = ["M1_first2_binary", "M1_repeated_binary", "M1_both_binary", "M2_both_binary_continuous"]
    labels = ["First-two\nalert", "Repeated-low\nalert", "Both binary\nalerts", "+ continuous\nresponse"]
    x = np.arange(len(order))
    for offset, (centre, color) in zip([-.18, .18], [("INSPIRE", "#2F6B9A"), ("MOVER", "#2A9D8F")]):
        frame = metrics.loc[metrics.centre.eq(centre)].set_index("model").loc[order]
        axes[0].bar(x + offset, frame.auroc, .36, label=centre, color=color)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(.5, max(.7, metrics.auroc.max() + .03))
    axes[0].set_ylabel("Patient-grouped OOF AUROC")
    axes[0].set_title("Continuous response beyond readable prior alerts")
    axes[0].legend(frameon=False)
    # Use the conservative centre-specific expanded baseline for the headline
    # interval panel; the left ladder remains the common-feature alert contrast.
    auc = expanded_ci_table.loc[
        expanded_ci_table.metric.eq("delta_auroc")
    ].set_index("centre").loc[["INSPIRE", "MOVER"]]
    axes[1].errorbar(
        [0, 1], auc.point,
        yerr=np.vstack([auc.point - auc.ci_low, auc.ci_high - auc.point]),
        fmt="o", capsize=4, color="#7A5195",
    )
    axes[1].axhline(0, color="#333", lw=1)
    axes[1].set_xticks([0, 1], ["INSPIRE", "MOVER"])
    axes[1].set_ylabel("AUROC increment: continuous vs both alerts")
    axes[1].set_title("Expanded case-context comparison")
    fig.suptitle(
        "Fixed-first-four repeated low-NIBP proxy: continuous prior response beyond binary history",
        y=1.02,
    )
    fig.text(
        .5, -.015,
        "Outcome: among the current first four NIBP MAP values in 0–30 min, a consecutive pair <65 mmHg separated by 2–15 min. Error bars are patient-cluster 95% CIs.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=[0, .04, 1, .98])
    fig.savefig(OUT / "fig_repeated_alert_comparator.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "fig_repeated_alert_comparator.svg", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "status": (
            "GO_CONTINUOUS_BEYOND_BOTH_PRIOR_ALERTS"
            if all(
                ci_table.loc[(ci_table.centre.eq(c)) & ci_table.metric.eq("delta_auroc"), "ci_low"].iloc[0] > 0
                for c in ["INSPIRE", "MOVER"]
            ) else "CONDITIONAL_CONTINUOUS_ALERT_COMPARATOR"
        ),
        "outcome": "current fixed-first-four repeated low NIBP proxy",
        "primary_comparison": "both prior binary alerts versus both alerts plus prior first MAP and absolute first-to-second change",
        "cohorts": {
            "INSPIRE": {"pairs": int(len(inspire)), "patients": int(inspire.subject_id.nunique()), "events": int(inspire.target_repeated_low_first4.sum())},
            "MOVER": {"pairs": int(len(mover)), "patients": int(mover.patient_id.nunique()), "events": int(mover.target_repeated_low_first4.sum())},
        },
        "inspire_link_audit": inspire_link_audit,
        "inspire_first4_audit": inspire_first4_audit,
        "mover_first4_audit": mover_audit,
        "metrics": metrics.to_dict(orient="records"),
        "increments": ci_table.to_dict(orient="records"),
        "expanded_metrics": expanded_metrics_frame.to_dict(orient="records"),
        "expanded_increments": expanded_ci_table.to_dict(orient="records"),
        "fixed_transport_metrics": fixed_rows,
        "fixed_transport_increment": fixed_ci.to_dict(orient="records"),
        "claim_boundary": (
            "Continuous prior response adds ranking information beyond two binary prior-event summaries. "
            "This remains a repeated-record proxy and does not establish treatment or organ benefit."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
