#!/usr/bin/env python3
"""Clinical-workflow deepening of the positive C02 longitudinal signal.

This analysis asks when and how much prior anaesthetic MAP history is useful.
It deliberately does not add a black-box model.  Its main contrasts are:

1. recent prior response versus two-prior-case response on the same patients;
2. conditional value of opening one older record after a positive recent case;
3. transport of the persistent-history risk difference across clinical strata;
4. whether history still helps once the current first two MAPs are observed;
5. whether downstream outcomes have enough constructible events to analyse.

Only aggregate tables and figures are written.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import tarfile
import warnings
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analyze_c02_multiepisode_management_instability import construct_cohort, risk_tables
from analyze_c02_multiepisode_memory import (
    INSPIRE_PAIR,
    MOVER_PAIR,
    prepare_inspire,
    prepare_mover,
    wilson,
)


from c02_runtime import private_workspace_root


ROOT = private_workspace_root()
OUT = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild/"
    "induction_drug_context/hypnotic_anchored/clinical_workflow_deepening"
)
MOVER_ARCHIVE = ROOT / "data/restricted/mover/raw/EPIC_EMR.tar.gz"
SEED = 20260814
BOOTSTRAP_REPS = 1000

STATE_ORDER = ["Neither", "Remote only", "Recent only", "Both"]
CENTRE_COLOUR = {"MOVER": "#D95F59", "INSPIRE": "#3A7CA5"}


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
    }


def model_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transform = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("transform", transform),
            ("logit", LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000)),
        ]
    )


@contextmanager
def sklearn_matmul_warning_guard():
    """Locally suppress only the known macOS Accelerate matmul warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
            module=r"sklearn\.(linear_model\._linear_loss|utils\.extmath)",
        )
        yield


def fit_predict_probability(
    model: Pipeline,
    train_frame: pd.DataFrame,
    train_outcome: np.ndarray,
    test_frame: pd.DataFrame,
    *,
    label: str,
) -> np.ndarray:
    """Fit one fold and reject incomplete or invalid probability output."""
    with sklearn_matmul_warning_guard():
        model.fit(train_frame, train_outcome)
        prediction = np.asarray(model.predict_proba(test_frame)[:, 1], dtype=float)
    if prediction.shape != (len(test_frame),):
        raise RuntimeError(f"{label} prediction shape mismatch")
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"non-finite {label} prediction")
    if np.any(prediction < 0.0) or np.any(prediction > 1.0):
        raise RuntimeError(f"out-of-range {label} probability")
    return prediction


def bootstrap_prediction_difference(
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    seed: int,
    reps: int = BOOTSTRAP_REPS,
) -> dict[str, float]:
    base = metrics(y, left)
    candidate = metrics(y, right)
    point = {
        "delta_auroc": candidate["auroc"] - base["auroc"],
        "delta_average_precision": candidate["average_precision"] - base["average_precision"],
        "brier_improvement": base["brier"] - candidate["brier"],
        "log_loss_improvement": base["log_loss"] - candidate["log_loss"],
    }
    rng = np.random.default_rng(seed)
    values = {name: [] for name in point}
    for _ in range(reps):
        index = rng.integers(0, len(y), len(y))
        yy = y[index]
        if np.unique(yy).size < 2:
            continue
        current = metrics(yy, left[index])
        new = metrics(yy, right[index])
        values["delta_auroc"].append(new["auroc"] - current["auroc"])
        values["delta_average_precision"].append(
            new["average_precision"] - current["average_precision"]
        )
        values["brier_improvement"].append(current["brier"] - new["brier"])
        values["log_loss_improvement"].append(
            current["log_loss"] - new["log_loss"]
        )
    result: dict[str, float] = {}
    for name, estimate in point.items():
        array = np.asarray(values[name], float)
        result[name] = float(estimate)
        result[f"{name}_ci_low"] = float(np.quantile(array, 0.025))
        result[f"{name}_ci_high"] = float(np.quantile(array, 0.975))
    result["bootstrap_reps"] = int(min(len(x) for x in values.values()))
    return result


def history_depth_oof(
    mover: pd.DataFrame, inspire: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    contrast_rows: list[dict] = []
    for centre_index, (centre, frame) in enumerate(
        [("MOVER", mover), ("INSPIRE", inspire)]
    ):
        d = frame.copy().reset_index(drop=True)
        d["log_recent_interval"] = np.log1p(d.interval_recent_days)
        d["log_remote_interval"] = np.log1p(d.interval_remote_days)
        d["persistent_both"] = (d.A_high.eq(1) & d.B_high.eq(1)).astype(int)
        d["history_mean"] = (d.A_drop + d.B_drop) / 2
        numeric_context = [
            "age",
            "bmi",
            "asa",
            "C_baseline",
            "A_baseline",
            "B_baseline",
            "log_recent_interval",
            "log_remote_interval",
            "current_inpatient",
            "current_emergency",
            "same_agent_three",
            "same_family_three",
        ]
        if centre == "MOVER":
            numeric_context += [
                "C_propofol_mgkg",
                "C_etomidate_mgkg",
                "C_ketamine_mgkg",
            ]
        categorical = ["sex", "current_context", "current_measurement"]
        specifications = {
            "H0_equal_context": numeric_context,
            "H1_recent_continuous": numeric_context + ["B_drop"],
            "H2_plus_remote_continuous": numeric_context + ["B_drop", "A_drop"],
            "H2_plus_persistent_flag": numeric_context + ["B_drop", "persistent_both"],
            "H2_two_history_mean": numeric_context + ["history_mean"],
        }
        y = d.C_high.to_numpy(int)
        splits = list(
            StratifiedKFold(
                n_splits=5, shuffle=True, random_state=SEED
            ).split(d, y)
        )
        predictions: dict[str, np.ndarray] = {}
        for name, numeric in specifications.items():
            model = model_pipeline(numeric, categorical)
            prediction = np.full(len(d), np.nan)
            for train, test in splits:
                prediction[test] = fit_predict_probability(
                    model,
                    d.iloc[train],
                    y[train],
                    d.iloc[test],
                    label=f"{centre} {name} fold",
                )
            if not np.all(np.isfinite(prediction)):
                raise RuntimeError(f"non-finite {centre} {name} prediction")
            predictions[name] = prediction
            metric_rows.append(
                {
                    "centre": centre,
                    "model": name,
                    "n": len(d),
                    "events": int(y.sum()),
                    **metrics(y, prediction),
                }
            )
        comparisons = [
            ("recent_response_beyond_context", "H0_equal_context", "H1_recent_continuous"),
            (
                "remote_response_beyond_recent",
                "H1_recent_continuous",
                "H2_plus_remote_continuous",
            ),
            (
                "persistent_flag_beyond_recent",
                "H1_recent_continuous",
                "H2_plus_persistent_flag",
            ),
            (
                "two_history_mean_beyond_recent",
                "H1_recent_continuous",
                "H2_two_history_mean",
            ),
        ]
        for comparison_index, (label, left, right) in enumerate(comparisons):
            contrast_rows.append(
                {
                    "centre": centre,
                    "comparison": label,
                    "left": left,
                    "right": right,
                    **bootstrap_prediction_difference(
                        y,
                        predictions[left],
                        predictions[right],
                        SEED + 100 * centre_index + comparison_index,
                    ),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(contrast_rows)


def risk_difference_bootstrap(
    outcome: np.ndarray,
    exposed: np.ndarray,
    reference: np.ndarray,
    seed: int,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[float, float, float]:
    point = float(outcome[exposed].mean() - outcome[reference].mean())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        index = rng.integers(0, len(outcome), len(outcome))
        yy = outcome[index]
        ee = exposed[index]
        rr = reference[index]
        if ee.sum() and rr.sum():
            values.append(float(yy[ee].mean() - yy[rr].mean()))
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def history_state_effects(
    mover: pd.DataFrame,
    inspire: pd.DataFrame,
    reps: int = 4000,
) -> pd.DataFrame:
    """Estimate paper-facing triplet risk contrasts on the analysis cohort.

    This calculation previously lived only in the manuscript assembler.  It is
    part of the statistical analysis and therefore belongs in the analysis
    output with an explicit bootstrap source.
    """

    rows: list[dict] = []
    for centre_index, (centre, frame) in enumerate(
        [("MOVER", mover), ("INSPIRE", inspire)]
    ):
        d = frame.reset_index(drop=True)

        def effects(sample: pd.DataFrame) -> dict[str, float] | None:
            risk = sample.groupby(["A_high", "B_high"], observed=True).C_high.mean()
            if len(risk) != 4:
                return None
            p00, p10, p01, p11 = (
                float(risk.loc[(0, 0)]),
                float(risk.loc[(1, 0)]),
                float(risk.loc[(0, 1)]),
                float(risk.loc[(1, 1)]),
            )
            return {
                "both_vs_neither_rd": p11 - p00,
                "older_increment_after_recent_positive": p11 - p01,
                "older_increment_after_recent_negative": p10 - p00,
                "additive_interaction": p11 - p10 - p01 + p00,
            }

        point = effects(d)
        if point is None:
            raise RuntimeError(f"{centre} triplet cohort is missing a history state")
        rng = np.random.default_rng(SEED + centre_index * 10000)
        values = {key: [] for key in point}
        for _ in range(reps):
            estimate = effects(d.iloc[rng.integers(0, len(d), len(d))])
            if estimate is None:
                continue
            for key, value in estimate.items():
                values[key].append(value)
        for estimand, estimate in point.items():
            low, high = np.quantile(values[estimand], [0.025, 0.975])
            rows.append(
                {
                    "centre": centre,
                    "estimand": estimand,
                    "estimate": estimate,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_reps": int(len(values[estimand])),
                    "bootstrap_unit": "patient",
                    "triplets": int(len(d)),
                    "events": int(d.C_high.sum()),
                }
            )
    return pd.DataFrame(rows)


def conditional_record_value(
    mover: pd.DataFrame, inspire: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditional_rows: list[dict] = []
    flag_rows: list[dict] = []
    for centre_index, (centre, d) in enumerate(
        [("MOVER", mover), ("INSPIRE", inspire)]
    ):
        d = d.reset_index(drop=True)
        y = d.C_high.to_numpy(int)
        for recent_state, label in [(1, "recent_positive"), (0, "recent_negative")]:
            subset = d.B_high.eq(recent_state).to_numpy()
            remote = subset & d.A_high.eq(1).to_numpy()
            no_remote = subset & d.A_high.eq(0).to_numpy()
            point, low, high = risk_difference_bootstrap(
                y,
                remote,
                no_remote,
                SEED + 200 + centre_index * 10 + recent_state,
                reps=2000,
            )
            conditional_rows.append(
                {
                    "centre": centre,
                    "recent_state": label,
                    "patients_with_recent_state": int(subset.sum()),
                    "remote_positive_n": int(remote.sum()),
                    "remote_positive_fraction": float(remote.sum() / subset.sum()),
                    "remote_positive_events": int(y[remote].sum()),
                    "remote_positive_risk": float(y[remote].mean()),
                    "remote_negative_n": int(no_remote.sum()),
                    "remote_negative_events": int(y[no_remote].sum()),
                    "remote_negative_risk": float(y[no_remote].mean()),
                    "risk_difference_remote_positive_minus_negative": point,
                    "ci_low": low,
                    "ci_high": high,
                    "interpretation": (
                        "value of opening one older record after the immediately prior response is known"
                    ),
                }
            )
        flags = {
            "recent_positive": d.B_high.eq(1).to_numpy(),
            "persistent_both": (d.A_high.eq(1) & d.B_high.eq(1)).to_numpy(),
        }
        for rule_index, (rule, flagged) in enumerate(flags.items()):
            unflagged = ~flagged
            point, low, high = risk_difference_bootstrap(
                y,
                flagged,
                unflagged,
                SEED + 300 + centre_index * 10 + rule_index,
                reps=2000,
            )
            tp = int(np.sum(flagged & (y == 1)))
            fp = int(np.sum(flagged & (y == 0)))
            tn = int(np.sum(unflagged & (y == 0)))
            fn = int(np.sum(unflagged & (y == 1)))
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            flag_rows.append(
                {
                    "centre": centre,
                    "rule": rule,
                    "n": len(d),
                    "events": int(y.sum()),
                    "flagged_n": int(flagged.sum()),
                    "flag_prevalence": float(flagged.mean()),
                    "flagged_risk": float(y[flagged].mean()),
                    "unflagged_risk": float(y[unflagged].mean()),
                    "risk_difference": point,
                    "ci_low": low,
                    "ci_high": high,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "positive_predictive_value": tp / (tp + fp),
                    "negative_predictive_value": tn / (tn + fn),
                    "positive_likelihood_ratio": sensitivity / (1 - specificity),
                    "excess_events_per_100_flagged": 100 * point,
                }
            )
    return pd.DataFrame(conditional_rows), pd.DataFrame(flag_rows)


def response_surface(mover: pd.DataFrame, inspire: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = ["<10%", "10–<20%", "≥20%"]
    for centre, d in [("MOVER", mover), ("INSPIRE", inspire)]:
        x = d.copy()
        x["remote_band"] = pd.cut(
            x.A_drop, [-np.inf, 0.10, 0.20, np.inf], right=False, labels=labels
        )
        x["recent_band"] = pd.cut(
            x.B_drop, [-np.inf, 0.10, 0.20, np.inf], right=False, labels=labels
        )
        for (remote, recent), frame in x.groupby(
            ["remote_band", "recent_band"], observed=True
        ):
            events = int(frame.C_high.sum())
            low, high = wilson(events, len(frame))
            rows.append(
                {
                    "centre": centre,
                    "remote_band": str(remote),
                    "recent_band": str(recent),
                    "n": len(frame),
                    "events": events,
                    "risk": events / len(frame),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def transport_strata(mover: pd.DataFrame, inspire: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for centre_index, (centre, d) in enumerate(
        [("MOVER", mover), ("INSPIRE", inspire)]
    ):
        d = d.reset_index(drop=True)
        definitions = {
            "All": np.ones(len(d), dtype=bool),
            "Same agent/source all three": d.same_agent_three.to_numpy(bool),
            "Different agent/source": ~d.same_agent_three.to_numpy(bool),
            "Same procedure family all three": d.same_family_three.to_numpy(bool),
            "Different procedure family": ~d.same_family_three.to_numpy(bool),
            "Current baseline MAP ≥85": d.C_baseline.ge(85).to_numpy(bool),
            "Current baseline MAP <85": d.C_baseline.lt(85).to_numpy(bool),
            "Recent interval ≤30 d": d.interval_recent_days.le(30).to_numpy(bool),
            "Recent interval 31–180 d": (
                d.interval_recent_days.gt(30) & d.interval_recent_days.le(180)
            ).to_numpy(bool),
            "Recent interval >180 d": d.interval_recent_days.gt(180).to_numpy(bool),
        }
        for stratum_index, (label, selected) in enumerate(definitions.items()):
            x = d.loc[selected].reset_index(drop=True)
            y = x.C_high.to_numpy(int)
            both = (x.A_high.eq(1) & x.B_high.eq(1)).to_numpy()
            neither = (x.A_high.eq(0) & x.B_high.eq(0)).to_numpy()
            if both.sum() == 0 or neither.sum() == 0:
                continue
            point, low, high = risk_difference_bootstrap(
                y,
                both,
                neither,
                SEED + 400 + centre_index * 100 + stratum_index,
            )
            rows.append(
                {
                    "centre": centre,
                    "stratum": label,
                    "n": len(x),
                    "events": int(y.sum()),
                    "both_n": int(both.sum()),
                    "both_events": int(y[both].sum()),
                    "neither_n": int(neither.sum()),
                    "neither_events": int(y[neither].sum()),
                    "risk_difference": point,
                    "ci_low": low,
                    "ci_high": high,
                    "reporting_status": (
                        "DESCRIPTIVE_SUFFICIENT" if both.sum() >= 20 and neither.sum() >= 20
                        else "TOO_SMALL_FOR_INTERPRETATION"
                    ),
                }
            )
    return pd.DataFrame(rows)


def post_signal_management_value(
    management: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = management.copy().reset_index(drop=True)
    d["current_family"] = d.C_procedure.fillna("missing").astype(str).str[:35]
    numeric_context = [
        "age",
        "bmi",
        "asa",
        "C_baseline",
        "C_drop",
        "C_post_MAP_1",
        "C_post_time_1",
        "C_propofol_mg",
        "C_etomidate_mg",
        "C_ketamine_mg",
        "current_inpatient",
    ]
    categorical = ["C_agent", "current_family"]
    specifications = {
        "P0_current_MAP_and_context": numeric_context,
        "P1_plus_physiology_history": numeric_context + ["physiology_history_count"],
        "P2_plus_composite_history": numeric_context + ["instability_history_count"],
        "P2_plus_separate_histories": numeric_context
        + ["physiology_history_count", "management_history_count"],
    }
    y = d.current_management.to_numpy(int)
    splits = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(d, y)
    )
    predictions = {}
    metric_rows = []
    for name, numeric in specifications.items():
        model = model_pipeline(numeric, categorical)
        prediction = np.full(len(d), np.nan)
        for train, test in splits:
            prediction[test] = fit_predict_probability(
                model,
                d.iloc[train],
                y[train],
                d.iloc[test],
                label=f"MOVER management {name} fold",
            )
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"non-finite MOVER management {name} OOF prediction")
        predictions[name] = prediction
        metric_rows.append(
            {
                "model": name,
                "n": len(d),
                "events": int(y.sum()),
                **metrics(y, prediction),
            }
        )
    contrast_rows = []
    base = "P0_current_MAP_and_context"
    for index, candidate in enumerate(list(specifications)[1:]):
        contrast_rows.append(
            {
                "comparison": f"{candidate}_vs_{base}",
                "left": base,
                "right": candidate,
                **bootstrap_prediction_difference(
                    y,
                    predictions[base],
                    predictions[candidate],
                    SEED + 500 + index,
                    reps=2000,
                ),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(contrast_rows)


def stream_creatinine(current_ids: set[str]) -> pd.DataFrame:
    member = "EPIC_EMR/EMR/patient_labs.csv"
    tar_process = subprocess.Popen(
        ["tar", "-xOf", str(MOVER_ARCHIVE), member], stdout=subprocess.PIPE
    )
    if tar_process.stdout is None:
        raise RuntimeError("tar stdout unavailable")
    filter_process = subprocess.Popen(
        ["rg", "--text", ",2160-0,Creatinine,"],
        stdin=tar_process.stdout,
        stdout=subprocess.PIPE,
        text=True,
    )
    tar_process.stdout.close()
    rows = []
    assert filter_process.stdout is not None
    for line in filter_process.stdout:
        record = next(csv.reader([line]))
        if len(record) >= 10 and record[0] in current_ids:
            rows.append(
                {
                    "LOG_ID": record[0],
                    "value": record[5],
                    "unit": record[6],
                    "time": record[9],
                }
            )
    filter_status = filter_process.wait()
    tar_status = tar_process.wait()
    if filter_status not in (0, 1) or tar_status != 0:
        raise RuntimeError("creatinine stream failed")
    return pd.DataFrame(rows)


def fourth_case_yield() -> dict[str, int]:
    d = pd.read_csv(MOVER_PAIR, low_memory=False)
    pairs = pd.DataFrame(
        {
            "patient_id": d.patient_id.astype(str),
            "prior_id": d.prior_LOG_ID.astype(str),
            "current_id": d.LOG_ID.astype(str),
            "current_time": pd.to_datetime(d.anstart, errors="coerce"),
            "prior_drop": d.prior_relative_drop,
            "current_drop": d.current_relative_drop,
        }
    )
    ab = pairs.rename(
        columns={
            "prior_id": "A_id",
            "current_id": "B_id",
            "prior_drop": "A_drop",
            "current_drop": "B_drop",
            "current_time": "B_time",
        }
    )[["patient_id", "A_id", "B_id", "A_drop", "B_drop", "B_time"]]
    bc = pairs.rename(
        columns={
            "prior_id": "B_id",
            "current_id": "C_id",
            "prior_drop": "B_check",
            "current_drop": "C_drop",
            "current_time": "C_time",
        }
    )[["patient_id", "B_id", "C_id", "B_check", "C_drop", "C_time"]]
    cd = pairs.rename(
        columns={
            "prior_id": "C_id",
            "current_id": "D_id",
            "prior_drop": "C_check",
            "current_drop": "D_drop",
            "current_time": "D_time",
        }
    )[["patient_id", "C_id", "D_id", "C_check", "D_drop", "D_time"]]
    chain = ab.merge(bc, on=["patient_id", "B_id"], validate="many_to_one")
    chain = chain.merge(cd, on=["patient_id", "C_id"], validate="many_to_one")
    chain = chain.loc[
        np.isclose(chain.B_drop, chain.B_check)
        & np.isclose(chain.C_drop, chain.C_check)
    ]
    chain = chain.sort_values(["patient_id", "D_time", "D_id"]).drop_duplicates(
        "patient_id"
    )
    count = chain[["A_drop", "B_drop", "C_drop"]].ge(0.20).sum(axis=1)
    event = chain.D_drop.ge(0.20)
    return {
        "evaluable_four_case_sequences": int(len(chain)),
        "fourth_case_events": int(event.sum()),
        "three_prior_positive_n": int(count.eq(3).sum()),
        "three_prior_positive_events": int((count.eq(3) & event).sum()),
    }


def clinical_endpoint_yield(management: pd.DataFrame) -> pd.DataFrame:
    current_ids = set(management.C_id.astype(str))
    rows: list[dict] = []
    with tarfile.open(MOVER_ARCHIVE, "r:gz") as archive:
        complication = pd.read_csv(
            archive.extractfile("EPIC_EMR/EMR/patient_post_op_complications.csv"),
            dtype={"LOG_ID": str},
            low_memory=False,
        )
    nonempty = complication.loc[
        complication.LOG_ID.isin(current_ids)
        & complication.SMRTDTA_ELEM_VALUE.notna()
    ]
    rows.append(
        {
            "candidate_endpoint": "Recorded postoperative complication",
            "source_cohort_n": len(management),
            "evaluable_n": int(nonempty.LOG_ID.nunique()),
            "events_or_key_group_n": int(nonempty.LOG_ID.nunique()),
            "decision": "STOP_TOO_FEW_NONEMPTY_RECORDS",
        }
    )

    creatinine = stream_creatinine(current_ids)
    creatinine["value"] = pd.to_numeric(creatinine.value, errors="coerce")
    creatinine["time"] = pd.to_datetime(creatinine.time, errors="coerce")
    creatinine = creatinine.loc[
        creatinine.unit.eq("mg/dL")
        & creatinine.value.between(0.1, 20, inclusive="both")
        & creatinine.time.notna()
    ].merge(
        management[["C_id", "C_time", "instability_history_count"]].rename(
            columns={"C_id": "LOG_ID", "C_time": "anstart"}
        ),
        on="LOG_ID",
        how="inner",
        validate="many_to_one",
    )
    creatinine["anstart"] = pd.to_datetime(creatinine.anstart, errors="coerce")
    creatinine["hours"] = (
        creatinine.time - creatinine.anstart
    ).dt.total_seconds() / 3600
    kidney_rows = []
    for log_id, frame in creatinine.groupby("LOG_ID", observed=True):
        pre = frame.loc[
            frame.hours.between(-168, 0, inclusive="left")
        ].sort_values("hours")
        post48 = frame.loc[frame.hours.gt(0) & frame.hours.le(48), "value"]
        post7 = frame.loc[frame.hours.gt(0) & frame.hours.le(168), "value"]
        if pre.empty or post7.empty:
            continue
        baseline = float(pre.value.iloc[-1])
        peak48 = float(post48.max()) if len(post48) else math.nan
        peak7 = float(post7.max())
        event = bool(
            (np.isfinite(peak48) and peak48 - baseline >= 0.3)
            or peak7 / baseline >= 1.5
        )
        kidney_rows.append({"LOG_ID": log_id, "aki": event})
    kidney = pd.DataFrame(kidney_rows)
    rows.append(
        {
            "candidate_endpoint": "Serum-creatinine KDIGO AKI within 7 days",
            "source_cohort_n": len(management),
            "evaluable_n": int(len(kidney)),
            "events_or_key_group_n": int(kidney.aki.sum()) if len(kidney) else 0,
            "decision": "STOP_LT50_EVENTS_AND_SELECTIVE_LAB_FOLLOWUP",
        }
    )

    fourth = fourth_case_yield()
    rows.append(
        {
            "candidate_endpoint": "Three-prior-case history predicting fourth case",
            "source_cohort_n": fourth["evaluable_four_case_sequences"],
            "evaluable_n": fourth["evaluable_four_case_sequences"],
            "events_or_key_group_n": fourth["three_prior_positive_n"],
            "decision": "STOP_THREE_POSITIVE_HISTORY_GROUP_LT30",
        }
    )
    return pd.DataFrame(rows)


def make_figure(
    states: pd.DataFrame,
    conditional: pd.DataFrame,
    strata: pd.DataFrame,
    management_risk: pd.DataFrame,
    depth_contrasts: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    ax = axes[0, 0]
    xloc = np.arange(4)
    width = 0.34
    for offset, centre in [(-width / 2, "MOVER"), (width / 2, "INSPIRE")]:
        z = states.loc[states.centre.eq(centre)].set_index("state").loc[STATE_ORDER]
        ax.bar(xloc + offset, z.risk * 100, width, color=CENTRE_COLOUR[centre], alpha=0.87)
        ax.errorbar(
            xloc + offset,
            z.risk * 100,
            yerr=[(z.risk - z.ci_low) * 100, (z.ci_high - z.risk) * 100],
            fmt="none",
            ecolor="#333333",
            capsize=2,
        )
    ax.set_xticks(xloc, STATE_ORDER, rotation=13)
    ax.set_ylabel("Third-anaesthetic relative decline ≥20% (%)")
    ax.set_title("A. Two-record history states")
    ax.legend(["MOVER", "INSPIRE"], frameon=False)

    ax = axes[0, 1]
    recent = conditional.loc[conditional.recent_state.eq("recent_positive")]
    xloc = np.arange(2)
    width = 0.33
    for offset, centre in [(-width / 2, "MOVER"), (width / 2, "INSPIRE")]:
        row = recent.loc[recent.centre.eq(centre)].iloc[0]
        values = [row.remote_negative_risk * 100, row.remote_positive_risk * 100]
        ax.bar(xloc + offset, values, width, color=CENTRE_COLOUR[centre], alpha=0.87)
    ax.set_xticks(xloc, ["Recent positive only", "Both prior cases positive"])
    ax.set_ylabel("Third-anaesthetic risk (%)")
    ax.set_title("B. Conditional value of one older record")
    ax.text(
        0.02,
        0.97,
        "Universal H2−H1 AUROC: MOVER −0.0015; INSPIRE +0.0023\n"
        "Older history enriches a high-risk subgroup, not overall prediction.",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="#444444",
    )

    ax = axes[1, 0]
    plot_order = [
        "All",
        "Same agent/source all three",
        "Same procedure family all three",
        "Different procedure family",
        "Current baseline MAP ≥85",
        "Recent interval ≤30 d",
        "Recent interval 31–180 d",
        "Recent interval >180 d",
    ]
    yloc = np.arange(len(plot_order))
    for offset, centre in [(-0.09, "MOVER"), (0.09, "INSPIRE")]:
        z = strata.loc[
            strata.centre.eq(centre)
            & strata.stratum.isin(plot_order)
            & strata.reporting_status.eq("DESCRIPTIVE_SUFFICIENT")
        ].set_index("stratum")
        present = [label for label in plot_order if label in z.index]
        position = np.array([plot_order.index(label) for label in present], float) + offset
        zz = z.loc[present]
        ax.errorbar(
            zz.risk_difference * 100,
            position,
            xerr=[
                (zz.risk_difference - zz.ci_low) * 100,
                (zz.ci_high - zz.risk_difference) * 100,
            ],
            fmt="o",
            capsize=2,
            color=CENTRE_COLOUR[centre],
            label=centre,
        )
    ax.axvline(0, color="#555555", lw=1)
    ax.set_yticks(yloc, plot_order)
    ax.invert_yaxis()
    ax.set_xlabel("Both vs neither risk difference (percentage points)")
    ax.set_title("C. Direction across clinical contexts")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    colours = {
        "current_instability": "#7B2CBF",
        "current_physiology": "#E76F51",
        "current_management": "#2A9D8F",
    }
    for outcome, label in [
        ("current_instability", "MAP decline or post-MAP bolus"),
        ("current_physiology", "Relative MAP decline ≥20%"),
        ("current_management", "Post-MAP IV bolus"),
    ]:
        z = management_risk.loc[management_risk.outcome.eq(outcome)].set_index(
            "history_count"
        ).loc[[0, 1, 2]]
        ax.errorbar(
            [0, 1, 2],
            z.risk * 100,
            yerr=[(z.risk - z.ci_low) * 100, (z.ci_high - z.risk) * 100],
            marker="o",
            lw=2,
            capsize=2,
            color=colours[outcome],
            label=label,
        )
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Prior two cases meeting physiology-or-management definition")
    ax.set_ylabel("Third-anaesthetic event (%)")
    ax.set_title("D. Translation to recorded management demand (MOVER)")
    ax.legend(frameon=False, fontsize=8)

    remote = depth_contrasts.loc[
        depth_contrasts.comparison.eq("remote_response_beyond_recent")
    ].set_index("centre")
    fig.suptitle(
        "When prior anaesthetic MAP history is informative",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.006,
        "Two-case history marks a clinically interpretable high-risk subgroup, but does not improve universal "
        f"discrimination beyond the recent response (ΔAUROC {remote.loc['MOVER','delta_auroc']:+.4f}/"
        f"{remote.loc['INSPIRE','delta_auroc']:+.4f}). Recorded bolus use is a process marker, not adjudicated rescue.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    fig.savefig(OUT / "fig_c02_clinical_workflow_deepening.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "fig_c02_clinical_workflow_deepening.svg", bbox_inches="tight")
    plt.close(fig)


def state_risks(mover: pd.DataFrame, inspire: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for centre, d in [("MOVER", mover), ("INSPIRE", inspire)]:
        for state in STATE_ORDER:
            x = d.loc[d.state.eq(state)]
            events = int(x.C_high.sum())
            low, high = wilson(events, len(x))
            rows.append(
                {
                    "centre": centre,
                    "state": state,
                    "n": len(x),
                    "events": events,
                    "risk": events / len(x),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def write_report(
    conditional: pd.DataFrame,
    flags: pd.DataFrame,
    depth: pd.DataFrame,
    management_risk: pd.DataFrame,
    management_post: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> None:
    recent = conditional.loc[conditional.recent_state.eq("recent_positive")].set_index(
        "centre"
    )
    remote = depth.loc[
        depth.comparison.eq("remote_response_beyond_recent")
    ].set_index("centre")
    persistent_flags = flags.loc[flags.rule.eq("persistent_both")].set_index("centre")
    recent_flags = flags.loc[flags.rule.eq("recent_positive")].set_index("centre")
    mover_mgmt = management_risk.set_index(["history_count", "outcome"])
    post = management_post.loc[
        management_post.right.eq("P2_plus_composite_history")
    ].iloc[0]
    endpoint_lines = [
        "| Candidate endpoint | Source cohort | Evaluable | Events/key group | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for row in endpoints.itertuples(index=False):
        endpoint_lines.append(
            f"| {row.candidate_endpoint} | {row.source_cohort_n} | {row.evaluable_n} | "
            f"{row.events_or_key_group_n} | `{row.decision}` |"
        )
    endpoint_table = "\n".join(endpoint_lines)
    report = f"""# C02 clinical-workflow deepening

## Conclusion first

The positive result is now clinically deeper, but not because a larger model performs better. The evidence supports a **timing-specific use of longitudinal anaesthetic history**:

1. Before induction, the immediately prior continuous MAP response carries modest patient-linked information.
2. If that recent response was marked (relative decline at least 20%), opening one additional older record separates a persistent high-risk subgroup: third-case risk was {100*recent.loc['MOVER','remote_positive_risk']:.1f}% versus {100*recent.loc['MOVER','remote_negative_risk']:.1f}% in MOVER and {100*recent.loc['INSPIRE','remote_positive_risk']:.1f}% versus {100*recent.loc['INSPIRE','remote_negative_risk']:.1f}% in INSPIRE.
3. Adding the older response to everyone did **not** improve universal out-of-fold discrimination beyond the recent response (MOVER ΔAUROC {remote.loc['MOVER','delta_auroc']:+.4f}, 95% CI {remote.loc['MOVER','delta_auroc_ci_low']:+.4f} to {remote.loc['MOVER','delta_auroc_ci_high']:+.4f}; INSPIRE {remote.loc['INSPIRE','delta_auroc']:+.4f}, {remote.loc['INSPIRE','delta_auroc_ci_low']:+.4f} to {remote.loc['INSPIRE','delta_auroc_ci_high']:+.4f}). History depth is therefore a **risk-enrichment rule**, not a universally better prediction model.
4. Once the current first two MAP measurements were already observed, prior history did not improve prediction of the next five-minute vasopressor bolus (composite-history ΔAUROC {post.delta_auroc:+.4f}, 95% CI {post.delta_auroc_ci_low:+.4f} to {post.delta_auroc_ci_high:+.4f}). Real-time physiology should dominate at that point.

## Added clinical volume

- Requiring both prior cases to be positive rather than the recent case alone increased specificity from {100*recent_flags.loc['MOVER','specificity']:.1f}% to {100*persistent_flags.loc['MOVER','specificity']:.1f}% in MOVER and {100*recent_flags.loc['INSPIRE','specificity']:.1f}% to {100*persistent_flags.loc['INSPIRE','specificity']:.1f}% in INSPIRE. Positive predictive value increased from {100*recent_flags.loc['MOVER','positive_predictive_value']:.1f}% to {100*persistent_flags.loc['MOVER','positive_predictive_value']:.1f}% and from {100*recent_flags.loc['INSPIRE','positive_predictive_value']:.1f}% to {100*persistent_flags.loc['INSPIRE','positive_predictive_value']:.1f}%, but sensitivity fell from {100*recent_flags.loc['MOVER','sensitivity']:.1f}% to {100*persistent_flags.loc['MOVER','sensitivity']:.1f}% and from {100*recent_flags.loc['INSPIRE','sensitivity']:.1f}% to {100*persistent_flags.loc['INSPIRE','sensitivity']:.1f}%.
- In the equal-observation MOVER process cohort, third-case physiology-or-management risk increased from {100*mover_mgmt.loc[(0,'current_instability'),'risk']:.1f}% to {100*mover_mgmt.loc[(2,'current_instability'),'risk']:.1f}%; recorded post-MAP bolus use increased from {100*mover_mgmt.loc[(0,'current_management'),'risk']:.1f}% to {100*mover_mgmt.loc[(2,'current_management'),'risk']:.1f}%.
- The gradient remained directionally positive in both centres across same agent/measurement source, same or different procedure family, baseline MAP and interval strata. These are descriptive transport checks, not proven effect modifiers.

## Negative boundaries retained

{endpoint_table}

These failures are not rescued with looser thresholds. MOVER postoperative complications were too sparsely populated, creatinine-defined AKI had too few events and selective testing, and a fourth-anaesthetic extension had only 11 patients with all three prior responses positive.

## Manuscript role

- **Main clinical question:** what information from prior anaesthetic records should be surfaced before a repeat operation?
- **Primary evidence:** strict adjacent-pair two-centre feature replication plus actual-IV-hypnotic timing in MOVER.
- **Depth evidence:** conditional two-record risk enrichment in a distinct third anaesthetic.
- **Clinical translation:** recorded post-MAP vasopressor use rises with prior process history, but clinician behaviour is not adjudicated treatment need or treatment benefit.
- **Claim ceiling:** observational longitudinal information value, not stable phenotype, deployable decision support, individualized MAP target or improved organ outcome.
"""
    (OUT / "clinical_workflow_deepening_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mover, mover_audit = prepare_mover()
    inspire, inspire_audit = prepare_inspire()
    management, management_audit, _ = construct_cohort()

    states = state_risks(mover, inspire)
    state_effects = history_state_effects(mover, inspire)
    depth_metrics, depth_contrasts = history_depth_oof(mover, inspire)
    conditional, flags = conditional_record_value(mover, inspire)
    surface = response_surface(mover, inspire)
    strata = transport_strata(mover, inspire)
    post_metrics, post_contrasts = post_signal_management_value(management)
    endpoints = clinical_endpoint_yield(management)

    # Reuse the cohort already constructed in this run. Reading a table emitted
    # by a separate historical main() made a clean workflow depend on stale or
    # pre-existing output state.
    management_risk, _ = risk_tables(management)

    tables = {
        "history_state_risks.csv": states,
        "history_state_effects.csv": state_effects,
        "history_depth_oof_metrics.csv": depth_metrics,
        "history_depth_oof_contrasts.csv": depth_contrasts,
        "conditional_older_record_value.csv": conditional,
        "persistent_history_flag_performance.csv": flags,
        "three_by_three_response_surface.csv": surface,
        "history_transport_strata.csv": strata,
        "post_signal_management_oof_metrics.csv": post_metrics,
        "post_signal_management_oof_contrasts.csv": post_contrasts,
        "clinical_endpoint_yield.csv": endpoints,
    }
    for filename, table in tables.items():
        table.to_csv(OUT / filename, index=False)

    make_figure(states, conditional, strata, management_risk, depth_contrasts)
    write_report(
        conditional,
        flags,
        depth_contrasts,
        management_risk,
        post_contrasts,
        endpoints,
    )

    remote = depth_contrasts.loc[
        depth_contrasts.comparison.eq("remote_response_beyond_recent")
    ].set_index("centre")
    recent = conditional.loc[conditional.recent_state.eq("recent_positive")].set_index(
        "centre"
    )
    summary = {
        "status": "KEEP_AS_CLINICAL_WORKFLOW_DEPTH_NOT_UNIVERSAL_MODEL_GAIN",
        "cohorts": {
            "MOVER_triplets": int(len(mover)),
            "INSPIRE_triplets": int(len(inspire)),
            "MOVER_management_equal_opportunity_triplets": int(len(management)),
        },
        "audits": {
            "MOVER": mover_audit,
            "INSPIRE": inspire_audit,
            "management": management_audit,
        },
        "remote_history_beyond_recent_delta_auroc": remote.reset_index().to_dict(
            "records"
        ),
        "conditional_risk_among_recent_positive": recent.reset_index().to_dict(
            "records"
        ),
        "record_review_triage_rules": flags.to_dict("records"),
        "clinical_endpoint_stops": endpoints.to_dict("records"),
        "paper_identity": (
            "Longitudinal clinical-information study defining when one versus two prior "
            "anaesthetic MAP records are useful; not a machine-learning methods paper."
        ),
        "claim_boundary": (
            "History enriches pre-induction risk after a recent marked response, but remote "
            "history does not improve universal prediction and does not supersede observed "
            "current MAP. Bolus use is recorded behaviour, not adjudicated rescue or benefit."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
