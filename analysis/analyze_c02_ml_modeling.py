#!/usr/bin/env python3
"""Machine-learning modelling attempt for C02.

Question: does adding machine-learning model classes improve the strictly
out-of-fold performance of the eight-variable prior-response specification,
and does the continuous prior-response increment persist beyond logistic
regression?

Design (pre-specified to match the manuscript's primary specification):

- Centres: INSPIRE and MOVER, identical eight-feature specification.
- Outcome: current-anaesthetic early MAP<65 mmHg among the first two NIBP
  observations (target_any_low).
- Outer evaluation: patient-grouped five-fold GroupKFold, identical to the
  manuscript's local OOF protocol.
- Hyperparameter selection: nested patient-grouped three-fold inner CV on the
  outer training fold only, selected by mean inner AUROC.
- Models: frozen manuscript logistic (C=0.1), tuned logistic, logistic with
  spline terms on the two continuous prior-response features, random forest,
  sklearn gradient boosting, XGBoost, LightGBM, and a post-hoc mean ensemble
  of spline-logistic + XGBoost + LightGBM OOF probabilities.
- Transport probe: the model class selected by the full INSPIRE inner CV is
  fitted once on all INSPIRE pairs and applied unchanged to MOVER.
- Paired uncertainty: 1,000 patient-cluster bootstrap replicates of metric
  differences against the frozen manuscript logistic model.

Only aggregate outputs are written. Restricted MOVER identifiers stay
in-memory.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

# NumPy 2.0 linked to macOS Accelerate can emit repeated floating-point
# warnings from finite scikit-learn matrix products. Prediction checks below
# remain fail-closed, so suppress only this narrow, known warning signature.
warnings.filterwarnings(
    "ignore",
    message=r"(divide by zero|overflow|invalid value) encountered in matmul",
    category=RuntimeWarning,
    module=r"(sklearn\..*|numpy\.linalg\._linalg)",
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from run_mover_c02_external_validation import calibration, canonicalize


from c02_runtime import private_workspace_root


ROOT = private_workspace_root()
BASE = ROOT / "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild"
INSPIRE = BASE / "first_two_sampling_bridge/first_two_pair_cohort.csv.gz"
MOVER = ROOT / "data/restricted/mover/extracted/mover_c02_cleaned_pair_cohort.csv.gz"
OUT = BASE / "machine_learning_attempt"

SEED = 20260822
BOOTSTRAP_REPS = 1000

FEATURES = [
    "age_years",
    "bmi_kg_m2",
    "asa_numeric",
    "sex_common",
    "interval_log1p",
    "prior_first2_any_low",
    "prior_first_map",
    "prior_first2_change",
]
SPLINE_FEATURES = ["prior_first_map", "prior_first2_change"]
CATEGORICAL = ["sex_common"]
NUMERIC = [column for column in FEATURES if column not in CATEGORICAL]

LOGIT_C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
RF_GRID = [
    {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": weight}
    for depth in [8, 16, None]
    for leaf in [25, 50]
    for weight in [None, "balanced_subsample"]
]
GBT_GRID = [
    {"n_estimators": trees, "learning_rate": lr, "max_depth": depth}
    for trees in [200, 400]
    for lr in [0.02, 0.05]
    for depth in [2, 3]
]
XGB_GRID = [
    {"n_estimators": trees, "learning_rate": lr, "max_depth": depth}
    for trees in [200, 400]
    for lr in [0.02, 0.05]
    for depth in [2, 3]
]
LGBM_GRID = [
    {"n_estimators": trees, "learning_rate": lr, "num_leaves": leaves, "min_child_samples": samples}
    for trees in [200, 400]
    for lr in [0.02, 0.05]
    for leaves in [7, 15]
    for samples in [50, 100]
]


def canonicalize_inspire(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["age_years"] = pd.to_numeric(d["age"], errors="coerce")
    d["bmi_kg_m2"] = pd.to_numeric(d["bmi"], errors="coerce")
    d["asa_numeric"] = pd.to_numeric(d["asa"], errors="coerce")
    d["sex_common"] = d["sex"].astype("string").str.upper().replace(
        {"MALE": "M", "FEMALE": "F", "1": "M", "2": "F"}
    )
    d.loc[~d["sex_common"].isin(["M", "F"]), "sex_common"] = "<MISSING>"
    d["interval_log1p"] = np.log1p(
        pd.to_numeric(d["interval_days"], errors="coerce").clip(lower=0)
    )
    d["prior_first_map"] = pd.to_numeric(d["prior_first2_map_0"], errors="coerce")
    d["prior_first2_change"] = (
        pd.to_numeric(d["prior_first2_map_1"], errors="coerce") - d["prior_first_map"]
    )
    d["prior_first2_any_low"] = (
        d[["prior_first2_map_0", "prior_first2_map_1"]].min(axis=1) < 65
    ).astype(int)
    d["target"] = pd.to_numeric(d["target_any_low"], errors="coerce").astype(int)
    d = d.loc[
        d["antype"].astype("string").str.strip().eq("General")
        & d["prior_antype"].astype("string").str.strip().eq("General")
    ].copy()
    for column in CATEGORICAL:
        d[column] = d[column].astype("string").fillna("<MISSING>").astype(str)
    return d


def preprocess(scaled: bool, splines: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scaled:
        numeric_steps.append(("scale", StandardScaler()))
    if splines:
        numeric_steps.append(
            (
                "spline",
                SplineTransformer(n_knots=4, degree=3, include_bias=False),
            )
        )
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
    )


def build_model(name: str, params: dict):
    if name == "logit_ref":
        return Pipeline(
            [
                ("preprocess", preprocess(scaled=True, splines=False)),
                ("model", LogisticRegression(C=0.1, max_iter=3000)),
            ]
        )
    if name == "logit_tuned":
        return Pipeline(
            [
                ("preprocess", preprocess(scaled=True, splines=False)),
                ("model", LogisticRegression(C=params["C"], max_iter=3000)),
            ]
        )
    if name == "logit_spline":
        return Pipeline(
            [
                ("preprocess", preprocess(scaled=True, splines=True)),
                ("model", LogisticRegression(C=params["C"], max_iter=3000)),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("preprocess", preprocess(scaled=False, splines=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=params["max_depth"],
                        min_samples_leaf=params["min_samples_leaf"],
                        class_weight=params["class_weight"],
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "gradient_boosting":
        return Pipeline(
            [
                ("preprocess", preprocess(scaled=False, splines=False)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=params["n_estimators"],
                        learning_rate=params["learning_rate"],
                        max_depth=params["max_depth"],
                        subsample=0.9,
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if name == "xgboost":
        return Pipeline(
            [
                ("preprocess", preprocess(scaled=False, splines=False)),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        tree_method="hist",
                        n_estimators=params["n_estimators"],
                        learning_rate=params["learning_rate"],
                        max_depth=params["max_depth"],
                        reg_lambda=1.0,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "lightgbm":
        return Pipeline(
            [
                ("preprocess", preprocess(scaled=False, splines=False)),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=params["n_estimators"],
                        learning_rate=params["learning_rate"],
                        num_leaves=params["num_leaves"],
                        min_child_samples=params["min_child_samples"],
                        random_state=SEED,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )
    raise KeyError(name)


GRIDS = {
    "logit_tuned": LOGIT_C_GRID,
    "logit_spline": LOGIT_C_GRID,
    "random_forest": RF_GRID,
    "gradient_boosting": GBT_GRID,
    "xgboost": XGB_GRID,
    "lightgbm": LGBM_GRID,
}


def checked_probability(
    prediction: np.ndarray, *, expected_n: int, label: str
) -> np.ndarray:
    """Require complete finite probabilities before scoring or caching."""
    values = np.asarray(prediction, dtype=float)
    if values.shape != (expected_n,):
        raise RuntimeError(
            f"{label} prediction shape mismatch: {values.shape}; expected={(expected_n,)}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} produced missing or non-finite probabilities")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError(f"{label} produced probabilities outside [0, 1]")
    return values


def inner_select(
    d: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    name: str,
) -> tuple[dict, float]:
    grid = GRIDS[name]
    splitter = GroupKFold(n_splits=3)
    best_params: dict = {}
    best_score = -np.inf
    for candidate in grid:
        params = {"C": candidate} if not isinstance(candidate, dict) else candidate
        scores = []
        for train, valid in splitter.split(d, y, groups=groups):
            model = build_model(name, params)
            model.fit(d.iloc[train][FEATURES], y[train])
            prob = checked_probability(
                model.predict_proba(d.iloc[valid][FEATURES])[:, 1],
                expected_n=len(valid),
                label=f"{name} inner fold",
            )
            scores.append(roc_auc_score(y[valid], prob))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    return best_params, best_score


def metric_row(y: np.ndarray, prediction: np.ndarray) -> dict:
    prediction = np.asarray(prediction, float)
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


def nested_oof(
    centre: str,
    d: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    selection_log: list[dict],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for name in ["logit_ref"] + list(GRIDS):
        oof = np.full(len(d), np.nan)
        for train, test in GroupKFold(n_splits=5).split(d, y, groups=groups):
            if name == "logit_ref":
                params = {"C": 0.1}
                inner_score = float("nan")
            else:
                params, inner_score = inner_select(
                    d.iloc[train].reset_index(drop=True), y[train], groups[train], name
                )
            selection_log.append(
                {
                    "centre": centre,
                    "model": name,
                    "params": json.dumps(params),
                    "inner_auroc": inner_score,
                }
            )
            model = build_model(name, params)
            model.fit(d.iloc[train][FEATURES], y[train])
            oof[test] = checked_probability(
                model.predict_proba(d.iloc[test][FEATURES])[:, 1],
                expected_n=len(test),
                label=f"{centre} {name} outer fold",
            )
        predictions[name] = checked_probability(
            oof, expected_n=len(d), label=f"{centre} {name} OOF"
        )
    ensemble = np.mean(
        np.stack(
            [
                predictions["logit_spline"],
                predictions["xgboost"],
                predictions["lightgbm"],
            ]
        ),
        axis=0,
    )
    predictions["ensemble_avg"] = checked_probability(
        ensemble, expected_n=len(d), label=f"{centre} ensemble OOF"
    )
    return predictions


def paired_bootstrap_ci(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    reps: int,
) -> tuple[dict, dict[str, list[float]]]:
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(SEED)
    deltas: dict[str, list[float]] = {
        "delta_auroc": [],
        "delta_average_precision": [],
        "brier_improvement": [],
        "log_loss_improvement": [],
    }
    y_by_group = {g: y[groups == g] for g in unique_groups}
    a_by_group = {g: pred_a[groups == g] for g in unique_groups}
    b_by_group = {g: pred_b[groups == g] for g in unique_groups}
    n_groups = len(unique_groups)
    for _ in range(reps):
        sample = rng.choice(unique_groups, size=n_groups, replace=True)
        ys = np.concatenate([y_by_group[g] for g in sample])
        a_s = np.concatenate([a_by_group[g] for g in sample])
        b_s = np.concatenate([b_by_group[g] for g in sample])
        deltas["delta_auroc"].append(roc_auc_score(ys, b_s) - roc_auc_score(ys, a_s))
        deltas["delta_average_precision"].append(
            average_precision_score(ys, b_s) - average_precision_score(ys, a_s)
        )
        deltas["brier_improvement"].append(brier_score_loss(ys, a_s) - brier_score_loss(ys, b_s))
        deltas["log_loss_improvement"].append(log_loss(ys, a_s) - log_loss(ys, b_s))
    summary = {
        metric: {
            "point": float(np.mean(values)),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        }
        for metric, values in deltas.items()
    }
    return summary, deltas


def percentile_p_value(values: list[float]) -> float:
    arr = np.asarray(values, float)
    p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
    return float(min(1.0, p))


def decision_curve_metrics(y: np.ndarray, prediction: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    prevalence = float(y.mean())
    for threshold in thresholds:
        positive = prediction >= threshold
        tp = float(np.sum(positive & (y == 1)))
        fp = float(np.sum(positive & (y == 0)))
        net_benefit = tp / len(y) - (fp / len(y)) * threshold / (1 - threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "positives": int(positive.sum()),
                "sensitivity": float(tp / max(1, y.sum())),
                "net_benefit": float(net_benefit),
            }
        )
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    rows.append({"threshold": float("nan"), "positives": int(len(y)), "sensitivity": 1.0, "net_benefit": float("nan")})
    frame = pd.DataFrame(rows)
    return frame, treat_all


def roc_curve_points(y: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y, prediction)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def calibration_bins(y: np.ndarray, prediction: np.ndarray, q: int = 10) -> pd.DataFrame:
    rank = pd.Series(prediction).rank(method="first")
    bins = pd.qcut(rank, q=q, labels=False)
    rows = []
    for bin_index in range(q):
        mask = bins.to_numpy() == bin_index
        rows.append(
            {
                "bin": bin_index + 1,
                "n": int(mask.sum()),
                "events": int(y[mask].sum()),
                "mean_prediction": float(prediction[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def transport_probe(
    inspire_frame: pd.DataFrame,
    inspire_y: np.ndarray,
    mover_frame: pd.DataFrame,
    mover_y: np.ndarray,
    selected_model: str,
) -> dict:
    params, inner_score = inner_select(
        inspire_frame.reset_index(drop=True), inspire_y, inspire_frame["subject_id"].to_numpy(), selected_model
    )
    model = build_model(selected_model, params)
    model.fit(inspire_frame[FEATURES], inspire_y)
    prob = checked_probability(
        model.predict_proba(mover_frame[FEATURES])[:, 1],
        expected_n=len(mover_frame),
        label=f"{selected_model} INSPIRE-to-MOVER transport",
    )
    row = metric_row(mover_y, prob)
    row["model"] = f"{selected_model}_INSPIRE_to_MOVER_fixed"
    row["params"] = json.dumps(params)
    row["inner_auroc"] = inner_score
    return row, prob


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    selection_log: list[dict] = []

    inspire = canonicalize_inspire(pd.read_csv(INSPIRE, low_memory=False))
    inspire_y = inspire["target"].to_numpy(int)
    inspire_groups = inspire["subject_id"].to_numpy()
    mover = canonicalize(pd.read_csv(MOVER, low_memory=False))
    mover_y = mover["target_any_low_first2"].to_numpy(int)
    mover_groups = mover["patient_id"].to_numpy()

    centres = {
        "INSPIRE": (inspire, inspire_y, inspire_groups),
        "MOVER": (mover, mover_y, mover_groups),
    }

    all_predictions: dict[str, dict[str, np.ndarray]] = {}
    metric_rows: list[dict] = []
    for centre, (frame, y, groups) in centres.items():
        cache_path = OUT / f"cache_{centre}.npz"
        # Never trust a cache from an earlier cohort, environment, or protocol.
        # Recompute every nested-CV prediction, then retain the new cache only
        # as a private audit artifact for this completed run.
        predictions = nested_oof(centre, frame, y, groups, selection_log)
        np.savez(cache_path, **predictions)
        all_predictions[centre] = predictions
        for name, prob in predictions.items():
            row = metric_row(y, prob)
            row.update({"centre": centre, "model": name})
            metric_rows.append(row)
            print(f"[{centre}] {name}: AUROC {row['auroc']:.4f} AP {row['average_precision']:.4f} "
                  f"Brier {row['brier']:.5f} logloss {row['log_loss']:.4f} "
                  f"cal {row['calibration_intercept']:.3f}/{row['calibration_slope']:.3f}", flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "ml_model_metrics.csv", index=False)
    pd.DataFrame(selection_log).to_csv(OUT / "ml_inner_selection_log.csv", index=False)

    reference = "logit_ref"
    pairwise_rows = []
    replicate_rows = []
    for centre, (frame, y, groups) in centres.items():
        predictions = all_predictions[centre]
        for name in predictions:
            if name == reference:
                continue
            ci, replicates = paired_bootstrap_ci(
                y, predictions[reference], predictions[name], groups, BOOTSTRAP_REPS
            )
            for metric, values in ci.items():
                pairwise_rows.append(
                    {
                        "centre": centre,
                        "model": name,
                        "metric": metric,
                        "point": values["point"],
                        "ci_low": values["ci_low"],
                        "ci_high": values["ci_high"],
                        "p_value": percentile_p_value(replicates[metric]),
                    }
                )
            replicate_frame = pd.DataFrame(replicates)
            replicate_frame["centre"] = centre
            replicate_frame["model"] = name
            replicate_rows.append(replicate_frame)
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(OUT / "ml_pairwise_increments.csv", index=False)
    pd.concat(replicate_rows, ignore_index=True).to_csv(
        OUT / "ml_pairwise_bootstrap_replicates.csv.gz", index=False
    )

    best_model = (
        metrics.loc[metrics.model.ne(reference)]
        .sort_values("auroc", ascending=False)
        .groupby("centre", sort=False)
        .head(1)
        .set_index("centre")["model"]
        .to_dict()
    )
    transport_row, transport_prob = transport_probe(
        inspire, inspire_y, mover, mover_y, best_model["INSPIRE"]
    )
    pd.DataFrame([transport_row]).to_csv(OUT / "ml_transport_probe.csv", index=False)

    thresholds = np.array([0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30])
    decision_rows = []
    for centre, (frame, y, groups) in centres.items():
        predictions = all_predictions[centre]
        for name in [reference, best_model[centre]]:
            curve, treat_all = decision_curve_metrics(y, predictions[name], thresholds)
            curve["centre"] = centre
            curve["model"] = name
            decision_rows.append(curve)
        treat_all_frame = pd.DataFrame({"threshold": thresholds, "net_benefit_treat_all": treat_all})
        treat_all_frame["centre"] = centre
        decision_rows.append(treat_all_frame.assign(model="treat_all"))
    pd.concat(decision_rows, ignore_index=True).to_csv(OUT / "ml_decision_curve.csv", index=False)

    roc_frames = []
    calibration_frames = []
    for centre, (frame, y, groups) in centres.items():
        predictions = all_predictions[centre]
        for name in [reference, best_model[centre], "ensemble_avg"]:
            roc = roc_curve_points(y, predictions[name])
            roc["centre"] = centre
            roc["model"] = name
            roc_frames.append(roc)
            bins = calibration_bins(y, predictions[name])
            bins["centre"] = centre
            bins["model"] = name
            calibration_frames.append(bins)
    pd.concat(roc_frames, ignore_index=True).to_csv(OUT / "ml_roc_curves.csv", index=False)
    pd.concat(calibration_frames, ignore_index=True).to_csv(OUT / "ml_calibration_bins.csv", index=False)

    make_figure(metrics, pairwise, best_model)

    summary = {
        "design": {
            "outer_cv": "patient-grouped GroupKFold n_splits=5",
            "inner_cv": "patient-grouped GroupKFold n_splits=3, selected by mean inner AUROC",
            "features": FEATURES,
            "outcome": "current first-two NIBP MAP<65 mmHg",
            "bootstrap": BOOTSTRAP_REPS,
            "seed": SEED,
        },
        "cohorts": {
            "INSPIRE": {"pairs": int(len(inspire_y)), "events": int(inspire_y.sum())},
            "MOVER": {"pairs": int(len(mover_y)), "events": int(mover_y.sum())},
        },
        "best_model": best_model,
        "transport_probe": transport_row,
        "wall_time_seconds": time.time() - started,
        "claims": [
            "Nested patient-grouped CV only; no test-fold tuning.",
            "Machine learning may improve local OOF performance but does not convert INSPIRE training into zero-cost external deployment.",
            "AP and probability metrics must be interpreted against each centre's event rate.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Done in", round(time.time() - started, 1), "seconds")


def make_figure(metrics: pd.DataFrame, pairwise: pd.DataFrame, best_model: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    colours = {"INSPIRE": "#3A7CA5", "MOVER": "#D95F59"}
    reference = "logit_ref"
    model_labels = {
        "logit_ref": "Fixed logistic",
        "logit_tuned": "Tuned logistic",
        "logit_spline": "Spline logistic",
        "random_forest": "Random forest",
        "gradient_boosting": "Gradient boosting",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "ensemble_avg": "Mean ensemble",
    }

    ax = axes[0, 0]
    model_order = list(metrics.model.unique())
    width = 0.38
    x = np.arange(len(model_order))
    for offset, centre in zip([-width / 2, width / 2], ["INSPIRE", "MOVER"]):
        colour = colours[centre]
        part = metrics.loc[metrics.centre.eq(centre)].set_index("model").reindex(model_order)
        ax.bar(x + offset, part.auroc, width=width, color=colour, alpha=0.85, label=centre)
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels[name] for name in model_order], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Patient-grouped OOF AUROC")
    ax.set_ylim(0.5, max(0.75, metrics.auroc.max() + 0.03))
    ax.legend(fontsize=8)
    ax.set_title("A  Nested out-of-fold AUROC", loc="left", fontsize=10, fontweight="bold")

    ax = axes[0, 1]
    roc = pd.read_csv(OUT / "ml_roc_curves.csv")
    for centre, colour in colours.items():
        for model, ls in [(reference, "--"), (best_model[centre], "-")]:
            part = roc.loc[roc.centre.eq(centre) & roc.model.eq(model)].sort_values("fpr")
            if part.empty:
                continue
            ax.plot(part.fpr, part.tpr, color=colour, ls=ls, lw=1.6,
                    label=f"{centre}: {model_labels[model]}")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.7)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("B  OOF ROC: manuscript logistic vs best ML", loc="left", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    cal = pd.read_csv(OUT / "ml_calibration_bins.csv")
    for centre, colour in colours.items():
        for model, marker in [(reference, "o"), (best_model[centre], "s")]:
            part = cal.loc[cal.centre.eq(centre) & cal.model.eq(model)]
            if part.empty:
                continue
            ax.plot(part.mean_prediction, part.observed_rate, color=colour, marker=marker,
                    ms=4, lw=1.2, label=f"{centre}: {model_labels[model]}")
    calibration_limit = 0.30
    ax.plot([0, calibration_limit], [0, calibration_limit], color="grey", lw=0.7)
    ax.set_xlim(0, calibration_limit)
    ax.set_ylim(0, calibration_limit)
    ax.set_xlabel("Mean predicted risk by decile")
    ax.set_ylabel("Observed event rate")
    ax.set_title("C  Calibration deciles", loc="left", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    delta = pairwise.loc[pairwise.metric.eq("delta_auroc")]
    labels = []
    y_positions = []
    for index, (centre, colour) in enumerate(colours.items()):
        part = delta.loc[delta.centre.eq(centre)].sort_values("point")
        for row_index, (_, row) in enumerate(part.iterrows()):
            y = index * (len(part) + 1) + row_index
            labels.append(f"{model_labels[row.model]} ({centre})")
            y_positions.append(y)
            ax.plot([row.ci_low, row.ci_high], [y, y], color=colour, lw=1.4)
            ax.plot(row.point, y, "o", color=colour, ms=4)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("ΔAUROC vs manuscript logistic (patient-cluster bootstrap 95% CI)")
    ax.set_title("D  AUROC increment vs logistic reference", loc="left", fontsize=10, fontweight="bold")

    fig.suptitle("Model-form robustness with the same eight pre-anaesthetic variables", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for suffix in ["png", "svg"]:
        fig.savefig(OUT / f"fig_c02_ml_attempt.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
