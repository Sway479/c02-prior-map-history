#!/usr/bin/env python3
"""Minimal external validation of C02 in MOVER EPIC.

The cohort and endpoint are constructed upstream without model inspection. This
script evaluates (1) the frozen INSPIRE common-feature models unchanged in
MOVER, and (2) fixed-C patient-grouped MOVER logistic models, including a more
conservative case-mix baseline. It writes aggregate results only.
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from c02_cluster_logit import fit_clustered_logit
from c02_runtime import private_workspace_root


ROOT = private_workspace_root()
COHORT = ROOT / "data/restricted/mover/extracted/mover_c02_cleaned_pair_cohort.csv.gz"
INSPIRE_MODELS = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild/"
    "cross_database_minimal_bridge"
)
OUT = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild/"
    "mover_external_validation"
)

COMMON_FEATURES = {
    "M0_current_common": [
        "age_years", "bmi_kg_m2", "asa_numeric", "sex_common", "interval_log1p"
    ],
    "M1_binary_prior_alert": [
        "age_years", "bmi_kg_m2", "asa_numeric", "sex_common", "interval_log1p",
        "prior_first2_any_low",
    ],
    "M2_continuous_prior_response": [
        "age_years", "bmi_kg_m2", "asa_numeric", "sex_common", "interval_log1p",
        "prior_first2_any_low", "prior_first_map", "prior_first2_change",
    ],
}

EXPANDED_FEATURES = {
    "M0_current_case_mix": [
        "age_years", "bmi_kg_m2", "asa_numeric", "sex_common",
        "patient_class_common", "procedure_common",
    ],
    "M1_prior_context_and_alert": [
        "age_years", "bmi_kg_m2", "asa_numeric", "sex_common",
        "patient_class_common", "procedure_common", "interval_log1p",
        "prior_asa_numeric", "prior_patient_class_common", "prior_procedure_common",
        "prior_first2_any_low",
    ],
    "M2_continuous_prior_response": [
        "age_years", "bmi_kg_m2", "asa_numeric", "sex_common",
        "patient_class_common", "procedure_common", "interval_log1p",
        "prior_asa_numeric", "prior_patient_class_common", "prior_procedure_common",
        "prior_first2_any_low", "prior_first_map", "prior_first2_change",
    ],
}


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    numeric = [
        "age_years", "bmi_kg_m2", "asa_numeric", "prior_asa_numeric",
        "interval_log1p", "prior_first2_any_low", "prior_first_map",
        "prior_first2_change", "target_any_low_first2",
    ]
    for column in numeric:
        d[column] = pd.to_numeric(d[column], errors="coerce")
    for column in [
        "sex_common", "patient_class_common", "procedure_common",
        "prior_patient_class_common", "prior_procedure_common",
    ]:
        d[column] = d[column].astype("string").fillna("<MISSING>").astype(str)
    d["sex_common"] = d["sex_common"].str.upper()
    d.loc[~d["sex_common"].isin(["M", "F"]), "sex_common"] = "<MISSING>"
    return d


def make_pipeline(features: list[str]) -> Pipeline:
    categorical = [
        column for column in features
        if column in {
            "sex_common", "patient_class_common", "procedure_common",
            "prior_patient_class_common", "prior_procedure_common",
        }
    ]
    numeric = [column for column in features if column not in categorical]
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", prep),
            ("model", LogisticRegression(C=0.1, max_iter=3000, solver="lbfgs")),
        ]
    )


def calibration(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(prediction, float), 1e-8, 1 - 1e-8)
    x = np.log(p / (1 - p))
    y = np.asarray(y, float)

    def objective(beta: np.ndarray) -> float:
        z = beta[0] + beta[1] * x
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    fitted = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    if not fitted.success and not np.all(np.isfinite(fitted.x)):
        return math.nan, math.nan
    return float(fitted.x[0]), float(fitted.x[1])


def metric_row(y: np.ndarray, prediction: np.ndarray) -> dict:
    prediction = np.asarray(prediction, float)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("non-finite prediction")
    intercept, slope = calibration(y, prediction)
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "mean_prediction": float(prediction.mean()),
        "auroc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def local_oof(
    d: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_sets: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    predictions = {name: np.full(len(d), np.nan) for name in feature_sets}
    folds = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(splitter.split(d, y, groups=groups)):
        folds.append(
            {
                "fold": int(fold),
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "test_events": int(y[test].sum()),
                "test_patients": int(pd.Series(groups[test]).nunique()),
            }
        )
        for name, features in feature_sets.items():
            model = make_pipeline(features)
            model.fit(d.iloc[train][features], y[train])
            predictions[name][test] = model.predict_proba(d.iloc[test][features])[:, 1]
    if any(not np.all(np.isfinite(values)) for values in predictions.values()):
        raise RuntimeError("OOF predictions incomplete or non-finite")
    return predictions, folds


def subject_bootstrap_increment(
    y: np.ndarray,
    groups: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    analysis: str,
    reps: int = 1000,
    seed: int = 20260813,
) -> tuple[list[dict], pd.DataFrame]:
    unique_groups = pd.unique(groups)
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)

    def increments(index: np.ndarray) -> dict:
        yy = y[index]
        p1 = m1[index]
        p2 = m2[index]
        return {
            "delta_auroc": roc_auc_score(yy, p2) - roc_auc_score(yy, p1),
            "delta_average_precision": (
                average_precision_score(yy, p2) - average_precision_score(yy, p1)
            ),
            "brier_improvement": brier_score_loss(yy, p1) - brier_score_loss(yy, p2),
            "log_loss_improvement": (
                log_loss(yy, p1, labels=[0, 1]) - log_loss(yy, p2, labels=[0, 1])
            ),
        }

    point = increments(np.arange(len(y)))
    bootstrap = []
    for rep in range(reps):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([group_rows[group] for group in sampled])
        if np.unique(y[index]).size < 2:
            continue
        bootstrap.append({"rep": rep, **increments(index)})
    boot = pd.DataFrame(bootstrap)
    summary = []
    for metric, value in point.items():
        summary.append(
            {
                "analysis": analysis,
                "metric": metric,
                "point": float(value),
                "ci_low": float(boot[metric].quantile(0.025)),
                "ci_high": float(boot[metric].quantile(0.975)),
                "bootstrap_reps": int(len(boot)),
            }
        )
    boot.insert(0, "analysis", analysis)
    return summary, boot


def wilson(events: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = events / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def risk_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alert, frame in d.groupby("prior_first2_any_low", observed=True):
        events = int(frame["target_any_low_first2"].sum())
        low, high = wilson(events, len(frame))
        rows.append(
            {
                "stratifier": "prior_binary_alert",
                "level": "prior any MAP<65" if int(alert) == 1 else "prior no MAP<65",
                "n": int(len(frame)),
                "events": events,
                "event_rate": events / len(frame),
                "ci_low": low,
                "ci_high": high,
            }
        )
    bands = pd.cut(
        d["prior_first_map"],
        bins=[20, 65, 75, 85, 201],
        right=False,
        labels=["<65", "65-74", "75-84", ">=85"],
    )
    for level, frame in d.groupby(bands, observed=False):
        events = int(frame["target_any_low_first2"].sum())
        low, high = wilson(events, len(frame))
        rows.append(
            {
                "stratifier": "prior_first_MAP_mmHg",
                "level": str(level),
                "n": int(len(frame)),
                "events": events,
                "event_rate": events / len(frame) if len(frame) else math.nan,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def interpretable_cluster_model(d: pd.DataFrame) -> list[dict]:
    x = pd.DataFrame(
        {
            "age_per_10y": d["age_years"] / 10,
            "bmi_per_5": d["bmi_kg_m2"] / 5,
            "current_ASA": d["asa_numeric"],
            "prior_ASA": d["prior_asa_numeric"],
            "log1p_interval_days": d["interval_log1p"],
            "male": d["sex_common"].eq("M").astype(float),
            "current_inpatient": d["patient_class_common"].eq("Inpatient").astype(float),
            "prior_binary_alert": d["prior_first2_any_low"].astype(float),
            "prior_first_MAP_per_10mmHg": d["prior_first_map"] / 10,
            "prior_MAP_change_per_10mmHg": d["prior_first2_change"] / 10,
        }
    )
    for column in x.columns:
        x[column] = x[column].fillna(x[column].median())
    y = d["target_any_low_first2"].to_numpy(float)
    fitted = fit_clustered_logit(
        x,
        y,
        d["patient_id"].to_numpy(),
        list(x.columns),
    ).set_index("term")

    rows = []
    for term in [
        "prior_binary_alert", "prior_first_MAP_per_10mmHg", "prior_MAP_change_per_10mmHg"
    ]:
        result = fitted.loc[term]
        rows.append(
            {
                "term": term,
                "odds_ratio": float(result.odds_ratio),
                "ci_low": float(result.ci_low),
                "ci_high": float(result.ci_high),
                "p_value_cluster_robust": float(result.p_value),
                "interpretation": "association adjusted for listed covariates; not causal",
                "n": int(result.n),
                "events": int(result.events),
                "patients": int(result.clusters),
                "fit_iterations": int(result.fit_iterations),
                "gradient_max_abs": float(result.gradient_max_abs),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = canonicalize(pd.read_csv(COHORT, low_memory=False))
    y = d["target_any_low_first2"].to_numpy(int)
    groups = d["patient_id"].to_numpy()
    if len(d) != 7721 or int(y.sum()) != 240:
        raise RuntimeError("frozen MOVER cohort count drift")

    fixed_predictions = {}
    fixed_rows = []
    for name, features in COMMON_FEATURES.items():
        model = joblib.load(INSPIRE_MODELS / f"inspire_fitted_{name}.joblib")
        prediction = model.predict_proba(d[features])[:, 1]
        fixed_predictions[name] = prediction
        fixed_rows.append({"analysis": "fixed_INSPIRE_to_MOVER", "model": name, **metric_row(y, prediction)})

    common_predictions, common_folds = local_oof(d, y, groups, COMMON_FEATURES)
    expanded_predictions, expanded_folds = local_oof(d, y, groups, EXPANDED_FEATURES)
    local_rows = []
    for name, prediction in common_predictions.items():
        local_rows.append({"analysis": "MOVER_grouped_OOF_common", "model": name, **metric_row(y, prediction)})
    for name, prediction in expanded_predictions.items():
        local_rows.append({"analysis": "MOVER_grouped_OOF_expanded", "model": name, **metric_row(y, prediction)})

    increment_rows = []
    bootstrap_frames = []
    comparisons = [
        (
            "fixed_INSPIRE_to_MOVER_M2_vs_M1",
            fixed_predictions["M1_binary_prior_alert"],
            fixed_predictions["M2_continuous_prior_response"],
            20260813,
        ),
        (
            "MOVER_grouped_OOF_common_M2_vs_M1",
            common_predictions["M1_binary_prior_alert"],
            common_predictions["M2_continuous_prior_response"],
            20260814,
        ),
        (
            "MOVER_grouped_OOF_expanded_M2_vs_M1",
            expanded_predictions["M1_prior_context_and_alert"],
            expanded_predictions["M2_continuous_prior_response"],
            20260815,
        ),
    ]
    for analysis, m1, m2, seed in comparisons:
        summary, bootstrap = subject_bootstrap_increment(y, groups, m1, m2, analysis, seed=seed)
        increment_rows.extend(summary)
        bootstrap_frames.append(bootstrap)

    fixed_metrics = pd.DataFrame(fixed_rows)
    local_metrics = pd.DataFrame(local_rows)
    increments = pd.DataFrame(increment_rows)
    risks = risk_table(d)
    association = pd.DataFrame(interpretable_cluster_model(d))
    fixed_metrics.to_csv(OUT / "fixed_model_transport_metrics.csv", index=False)
    local_metrics.to_csv(OUT / "mover_grouped_oof_metrics.csv", index=False)
    increments.to_csv(OUT / "m2_vs_m1_increment_cluster_bootstrap.csv", index=False)
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(
        OUT / "m2_vs_m1_increment_bootstrap_replicates.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(common_folds).assign(analysis="common").to_csv(OUT / "common_folds.csv", index=False)
    pd.DataFrame(expanded_folds).assign(analysis="expanded").to_csv(OUT / "expanded_folds.csv", index=False)
    risks.to_csv(OUT / "clinical_risk_by_prior_response.csv", index=False)
    association.to_csv(OUT / "cluster_robust_logistic_association.csv", index=False)

    # One clinically oriented result figure.
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    alert = risks.loc[risks["stratifier"].eq("prior_binary_alert")].copy()
    axes[0].bar(alert["level"], alert["event_rate"] * 100, color=["#8da0cb", "#fc8d62"])
    axes[0].errorbar(
        np.arange(len(alert)), alert["event_rate"] * 100,
        yerr=[(alert["event_rate"] - alert["ci_low"]) * 100, (alert["ci_high"] - alert["event_rate"]) * 100],
        fmt="none", ecolor="black", capsize=3,
    )
    axes[0].set_ylabel("Current early low-MAP risk (%)")
    axes[0].set_title("A  Same-patient recurrence")
    axes[0].tick_params(axis="x", rotation=15)

    bands = risks.loc[risks["stratifier"].eq("prior_first_MAP_mmHg")].copy()
    axes[1].plot(bands["level"], bands["event_rate"] * 100, marker="o", color="#1b9e77")
    axes[1].fill_between(
        np.arange(len(bands)), bands["ci_low"].to_numpy(float) * 100,
        bands["ci_high"].to_numpy(float) * 100, color="#1b9e77", alpha=0.18,
    )
    axes[1].set_ylabel("Current early low-MAP risk (%)")
    axes[1].set_xlabel("First MAP in prior anaesthetic (mmHg)")
    axes[1].set_title("B  Continuous prior response")

    plot_rows = []
    for analysis, label in [
        ("fixed_INSPIRE_to_MOVER_M2_vs_M1", "Fixed INSPIRE→MOVER"),
        ("MOVER_grouped_OOF_common_M2_vs_M1", "MOVER common baseline"),
        ("MOVER_grouped_OOF_expanded_M2_vs_M1", "MOVER expanded baseline"),
    ]:
        row = increments.loc[
            increments["analysis"].eq(analysis) & increments["metric"].eq("delta_auroc")
        ].iloc[0]
        plot_rows.append((label, row["point"], row["ci_low"], row["ci_high"]))
    yy = np.arange(len(plot_rows))
    point = np.array([row[1] for row in plot_rows])
    low = np.array([row[2] for row in plot_rows])
    high = np.array([row[3] for row in plot_rows])
    axes[2].errorbar(point, yy, xerr=[point - low, high - point], fmt="o", color="#7570b3", capsize=3)
    axes[2].axvline(0, color="black", lw=1, ls="--")
    axes[2].set_yticks(yy, [row[0] for row in plot_rows])
    axes[2].set_xlabel("AUROC increment over binary alert")
    axes[2].set_title("C  External incremental information")
    axes[2].invert_yaxis()
    fig.suptitle("MOVER external validation of prior-anaesthetic MAP response", y=1.03)
    fig.text(
        0.5,
        0.985,
        "7,721 adjacent general-anaesthetic pairs; 5,297 patients; 240 current early low-MAP events",
        ha="center",
        va="top",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_mover_c02_external_validation.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig_mover_c02_external_validation.svg", bbox_inches="tight")
    plt.close(fig)

    primary_increment = increments.loc[
        increments["analysis"].eq("MOVER_grouped_OOF_expanded_M2_vs_M1")
    ].set_index("metric")
    pass_direction = bool(
        primary_increment.loc["delta_auroc", "point"] > 0
        and primary_increment.loc["brier_improvement", "point"] > 0
        and primary_increment.loc["log_loss_improvement", "point"] > 0
    )
    summary = {
        "status": (
            "EXTERNAL_INCREMENT_DIRECTION_RETAINED" if pass_direction
            else "EXTERNAL_INCREMENT_NOT_RETAINED_STOP_COMPLEX_RESCUE"
        ),
        "cohort": {
            "pairs": int(len(d)),
            "patients": int(d["patient_id"].nunique()),
            "primary_events": int(y.sum()),
            "primary_event_rate": float(y.mean()),
            "secondary_both_low_events": int(d["target_both_low_first2"].sum()),
            "secondary_not_modelled_due_to_low_events": True,
            "first_measurement_median_min": float(d["current_nibp_rel_0"].median()),
            "second_measurement_median_min": float(d["current_nibp_rel_1"].median()),
        },
        "fixed_model_transport_metrics": fixed_metrics.to_dict(orient="records"),
        "mover_grouped_oof_metrics": local_metrics.to_dict(orient="records"),
        "increment_cluster_bootstrap": increments.to_dict(orient="records"),
        "cluster_robust_association": association.to_dict(orient="records"),
        "primary_decision_rule": (
            "direction retained only if expanded-baseline M2 vs M1 has positive AUROC, "
            "Brier and log-loss increments; confidence intervals quantify uncertainty"
        ),
        "claim_boundary": (
            "External replication or non-replication of incremental predictive information. "
            "No causal, treatment-benefit, organ-protection, or deployment claim."
        ),
        "patient_level_outputs_in_report_directory": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
