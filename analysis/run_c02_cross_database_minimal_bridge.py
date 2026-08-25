#!/usr/bin/env python3
"""INSPIRE bridge for the fixed MOVER-compatible C02 feature sequence.

This deliberately uses only variables that can be recreated in MOVER EPIC:
age, sex, BMI, ASA, interval since the immediately prior anaesthetic, a binary
first-two-MAP alert, and two continuous prior-response terms. It estimates the
internal bridge signal and persists the fitted preprocessing/model objects only
after the feature sequence is fixed; it does not inspect MOVER outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from c02_runtime import private_workspace_root

ROOT = private_workspace_root()
INPUT = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild/"
    "first_two_sampling_bridge/first_two_pair_cohort.csv.gz"
)
OUT = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild/"
    "cross_database_minimal_bridge"
)


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["age_years"] = pd.to_numeric(d["age"], errors="coerce")
    d["bmi_kg_m2"] = pd.to_numeric(d["bmi"], errors="coerce")
    d["asa_numeric"] = pd.to_numeric(d["asa"], errors="coerce")
    d["sex_common"] = (
        d["sex"].astype("string").str.strip().str.upper().replace(
            {"MALE": "M", "FEMALE": "F", "1": "M", "2": "F"}
        )
    )
    d.loc[~d["sex_common"].isin(["M", "F"]), "sex_common"] = pd.NA
    d["interval_log1p"] = np.log1p(pd.to_numeric(d["interval_days"], errors="coerce").clip(lower=0))
    prior0 = pd.to_numeric(d["prior_first2_map_0"], errors="coerce")
    prior1 = pd.to_numeric(d["prior_first2_map_1"], errors="coerce")
    d["prior_first2_any_low"] = (pd.concat([prior0, prior1], axis=1).min(axis=1) < 65).astype(int)
    d["prior_first_map"] = prior0
    d["prior_first2_change"] = prior1 - prior0
    d["target_any_low_first2"] = d["target_any_low"].astype(int)
    # Match the external MOVER target population exactly. The previous bridge
    # pooled other recorded anaesthetic types in INSPIRE, which made the fixed
    # transport comparison needlessly heterogeneous.
    d = d.loc[
        d["antype"].astype("string").str.strip().eq("General")
        & d["prior_antype"].astype("string").str.strip().eq("General")
    ].copy()
    return d


FEATURES = {
    "M0_current_common": ["age_years", "bmi_kg_m2", "asa_numeric", "sex_common", "interval_log1p"],
    "M1_binary_prior_alert": [
        "age_years", "bmi_kg_m2", "asa_numeric", "sex_common", "interval_log1p",
        "prior_first2_any_low",
    ],
    "M2_continuous_prior_response": [
        "age_years", "bmi_kg_m2", "asa_numeric", "sex_common", "interval_log1p",
        "prior_first2_any_low", "prior_first_map", "prior_first2_change",
    ],
}


def make_pipeline(frame: pd.DataFrame, features: list[str]) -> Pipeline:
    categorical = [column for column in features if column == "sex_common"]
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
            ("model", LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")),
        ]
    )


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = canonicalize(pd.read_csv(INPUT, low_memory=False))
    y = d["target_any_low_first2"].to_numpy(int)
    groups = d["subject_id"].to_numpy()
    predictions = {model: np.full(len(d), np.nan) for model in FEATURES}
    fold_rows = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(splitter.split(d, y, groups=groups)):
        for model_name, feature_names in FEATURES.items():
            fitted = make_pipeline(d, feature_names)
            fitted.fit(d.iloc[train][feature_names], y[train])
            predictions[model_name][test] = fitted.predict_proba(d.iloc[test][feature_names])[:, 1]
        fold_rows.append(
            {
                "fold": fold,
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "test_events": int(y[test].sum()),
                "test_patients": int(pd.Series(groups[test]).nunique()),
            }
        )

    result_rows = []
    for model_name, prediction in predictions.items():
        row = {"model": model_name, **metrics(y, prediction)}
        result_rows.append(row)
    results = pd.DataFrame(result_rows)
    base = results.loc[results["model"].eq("M0_current_common")].iloc[0]
    alert = results.loc[results["model"].eq("M1_binary_prior_alert")].iloc[0]
    results["delta_auroc_vs_M0"] = results["auroc"] - base["auroc"]
    results["brier_improvement_vs_M0"] = base["brier"] - results["brier"]
    results["log_loss_improvement_vs_M0"] = base["log_loss"] - results["log_loss"]
    results["delta_auroc_vs_M1"] = results["auroc"] - alert["auroc"]
    results["brier_improvement_vs_M1"] = alert["brier"] - results["brier"]
    results["log_loss_improvement_vs_M1"] = alert["log_loss"] - results["log_loss"]
    results.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "folds.csv", index=False)

    oof = d[["op_id", "subject_id", "target_any_low_first2"]].copy()
    for model_name, prediction in predictions.items():
        oof["pred_" + model_name] = prediction
    oof.to_csv(OUT / "oof_predictions.csv.gz", index=False)

    # Fit the exact full-INSPIRE pipelines that may later be applied, unchanged,
    # to MOVER after its independent cohort has been constructed.
    for model_name, feature_names in FEATURES.items():
        fitted = make_pipeline(d, feature_names)
        fitted.fit(d[feature_names], y)
        joblib.dump(fitted, OUT / f"inspire_fitted_{model_name}.joblib")

    completeness = {
        column: float(d[column].notna().mean())
        for column in sorted(set(column for values in FEATURES.values() for column in values))
    }
    summary = {
        "status": "INSPIRE_CROSS_DATABASE_MINIMAL_BRIDGE",
        "target_population": "recorded General-to-General adjacent anaesthetics",
        "cohort_n": int(len(d)),
        "patients": int(d["subject_id"].nunique()),
        "events": int(y.sum()),
        "feature_sets": FEATURES,
        "feature_completeness": completeness,
        "metrics": results.to_dict(orient="records"),
        "external_model_status": "FITTED_ON_INSPIRE_NOT_APPLIED_TO_MOVER",
        "claim_boundary": (
            "This establishes an INSPIRE bridge under a MOVER-compatible feature definition; "
            "it is not external validation or clinical utility evidence."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    labels = ["Current common", "+ Binary prior alert", "+ Continuous prior response"]
    colors = ["#8da0cb", "#66c2a5", "#fc8d62"]
    bars = ax.bar(labels, results["auroc"], color=colors)
    ax.set_ylim(max(0.5, results["auroc"].min() - 0.04), results["auroc"].max() + 0.025)
    ax.set_ylabel("Patient-grouped OOF AUROC")
    ax.set_title("INSPIRE bridge using only MOVER-compatible variables")
    for bar, value in zip(bars, results["auroc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_c02_cross_database_minimal_bridge.png", dpi=220)
    fig.savefig(OUT / "fig_c02_cross_database_minimal_bridge.svg")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
