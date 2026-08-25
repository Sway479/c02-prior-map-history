#!/usr/bin/env python3
"""Deepen C02 with an administered early vasopressor action endpoint in MOVER.

The medication construct is deliberately narrow: an INTRA-OP MAR action of
"Given" for intravenous phenylephrine or ephedrine, with plausible dose units
and values, during anaesthesia-start +0..30 min.  This is a recorded treatment
action, not proof that every dose was rescue for hypotension.  A stricter
process endpoint additionally requires a conflict-free NIBP MAP <65 followed
by the bolus within five minutes.

Only aggregate report outputs are written.  Restricted row-level source data
remain under data/restricted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from run_mover_c02_external_validation import (
    EXPANDED_FEATURES,
    canonicalize,
    local_oof,
    metric_row,
    subject_bootstrap_increment,
    wilson,
)


from c02_runtime import private_workspace_root


ROOT = private_workspace_root()
BASE = ROOT / "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild"
PAIR = ROOT / "data/restricted/mover/extracted/mover_c02_cleaned_pair_cohort.csv.gz"
MAPS = ROOT / "data/restricted/mover/extracted/mover_cleaned_early_map.csv.gz"
MAR = ROOT / "data/restricted/mover/extracted/mover_early_vasopressor_mar.csv.gz"
OUT = BASE / "clinical_endpoint_upgrade/vasopressor_action"

NIBP_MEAS = "UC ANE R BLOOD PRESSURE - MAP"
NIBP_DISPLAY = "NIBP - MAP"

M0_CASE_CONTEXT = [
    feature
    for feature in EXPANDED_FEATURES["M1_prior_context_and_alert"]
    if feature != "prior_first2_any_low"
]
M1_BP_HISTORY = M0_CASE_CONTEXT + ["prior_first2_any_low"]
M1_READABLE_HISTORY = M1_BP_HISTORY + [
    "prior_any_bolus_0_30",
    "prior_repeated_bolus_0_30",
    "prior_low_to_bolus_5",
]
M2_CONTINUOUS_RESPONSE = M1_READABLE_HISTORY + [
    "prior_first_map",
    "prior_first2_change",
]

OUTCOMES = [
    "current_low_to_bolus_5",
    "current_any_bolus_0_30",
    "current_repeated_bolus_0_30",
    "current_any_bolus_0_15",
    "current_low_to_bolus_10",
]


def valid_bolus_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return narrow, actually administered routine IV vasopressor boluses."""
    d = frame.copy()
    d["ADMIN_SIG"] = pd.to_numeric(d["ADMIN_SIG"], errors="coerce")
    d["relative_min"] = pd.to_numeric(d["relative_min"], errors="coerce")
    route = d["MED_ROUTE_NM"].fillna("").astype(str)
    base = (
        d["RECORD_TYPE"].eq("INTRA-OP")
        & d["MAR_ACTION_NM"].eq("Given")
        & route.str.fullmatch(r"IntraVENOUS(?: Push)?", case=False)
        & d["drug_class"].isin(["phenylephrine", "ephedrine"])
        & d["relative_min"].between(0, 30, inclusive="both")
        & d["ADMIN_SIG"].gt(0)
    )
    plausible = (
        d["drug_class"].eq("phenylephrine")
        & d["DOSE_UNIT_NM"].eq("mcg")
        & d["ADMIN_SIG"].between(10, 500, inclusive="both")
    ) | (
        d["drug_class"].eq("ephedrine")
        & d["DOSE_UNIT_NM"].eq("mg")
        & d["ADMIN_SIG"].between(1, 50, inclusive="both")
    )
    return d.loc[base & plausible].copy()


def conflict_free_low_times(operation_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    usecols = [
        "LOG_ID", "RECORDED_TIME", "relative_min", "value", "RECORD_TYPE",
        "FLO_MEAS_NAME", "FLO_DISPLAY_NAME", "modality_hint",
    ]
    d = pd.read_csv(MAPS, usecols=usecols, dtype={"LOG_ID": str}, low_memory=False)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d["relative_min"] = pd.to_numeric(d["relative_min"], errors="coerce")
    d["time"] = pd.to_datetime(d["RECORDED_TIME"], errors="coerce")
    d = d.loc[
        d["LOG_ID"].isin(operation_ids)
        & d["RECORD_TYPE"].eq("INTRA-OP")
        & d["FLO_MEAS_NAME"].eq(NIBP_MEAS)
        & d["FLO_DISPLAY_NAME"].eq(NIBP_DISPLAY)
        & d["modality_hint"].eq("NIBP")
        & d["relative_min"].between(0, 30, inclusive="both")
        & d["value"].between(20, 200, inclusive="both")
        & d["time"].notna()
    ].copy()
    key = (
        d.groupby(["LOG_ID", "time"], as_index=False, observed=True)
        .agg(
            relative_min=("relative_min", "min"),
            distinct_values=("value", "nunique"),
            value=("value", "first"),
        )
    )
    conflicts = int(key["distinct_values"].gt(1).sum())
    key = key.loc[key["distinct_values"].eq(1)].copy()
    low = key.loc[key["value"].lt(65), ["LOG_ID", "relative_min"]].copy()
    return low, {
        "valid_NIBP_rows_0_30": int(len(d)),
        "same_time_keys": int(len(key) + conflicts),
        "conflicting_same_time_keys_excluded": conflicts,
        "operations_with_any_low_NIBP": int(low["LOG_ID"].nunique()),
    }


def operation_features(
    operation_ids: set[str], bolus: pd.DataFrame, low: pd.DataFrame
) -> pd.DataFrame:
    base = pd.DataFrame({"LOG_ID": sorted(operation_ids)})
    action = (
        bolus.loc[bolus["LOG_ID"].isin(operation_ids)]
        .groupby("LOG_ID", as_index=False, observed=True)
        .agg(
            n_bolus_0_30=("LOG_ID", "size"),
            first_bolus_min=("relative_min", "min"),
            last_bolus_min=("relative_min", "max"),
            any_phenylephrine=("drug_class", lambda x: int((x == "phenylephrine").any())),
            any_ephedrine=("drug_class", lambda x: int((x == "ephedrine").any())),
        )
    )
    early = (
        bolus.loc[bolus["LOG_ID"].isin(operation_ids) & bolus["relative_min"].between(0, 15)]
        .groupby("LOG_ID", observed=True).size().rename("n_bolus_0_15").reset_index()
    )
    linked = bolus.loc[bolus["LOG_ID"].isin(operation_ids), ["LOG_ID", "relative_min"]].merge(
        low.loc[low["LOG_ID"].isin(operation_ids)], on="LOG_ID", suffixes=("_bolus", "_low")
    )
    if len(linked):
        linked["lag_min"] = linked["relative_min_bolus"] - linked["relative_min_low"]
        response = (
            linked.groupby("LOG_ID", as_index=False, observed=True)
            .agg(
                low_to_bolus_5=("lag_min", lambda x: int(x.between(0, 5).any())),
                low_to_bolus_10=("lag_min", lambda x: int(x.between(0, 10).any())),
                low_to_bolus_minus2_5=("lag_min", lambda x: int(x.between(-2, 5).any())),
            )
        )
    else:
        response = pd.DataFrame(
            columns=["LOG_ID", "low_to_bolus_5", "low_to_bolus_10", "low_to_bolus_minus2_5"]
        )
    out = base.merge(action, on="LOG_ID", how="left", validate="one_to_one")
    out = out.merge(early, on="LOG_ID", how="left", validate="one_to_one")
    out = out.merge(response, on="LOG_ID", how="left", validate="one_to_one")
    for column in [
        "n_bolus_0_30", "n_bolus_0_15", "any_phenylephrine", "any_ephedrine",
        "low_to_bolus_5", "low_to_bolus_10", "low_to_bolus_minus2_5",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    out["any_bolus_0_30"] = out["n_bolus_0_30"].ge(1).astype(int)
    out["repeated_bolus_0_30"] = out["n_bolus_0_30"].ge(2).astype(int)
    out["any_bolus_0_15"] = out["n_bolus_0_15"].ge(1).astype(int)
    return out


def interpretable_association(d: pd.DataFrame, outcome: str) -> pd.DataFrame:
    x = pd.DataFrame(
        {
            "age_per_10y": d["age_years"] / 10,
            "bmi_per_5": d["bmi_kg_m2"] / 5,
            "current_ASA": d["asa_numeric"],
            "prior_ASA": d["prior_asa_numeric"],
            "log1p_interval_days": d["interval_log1p"],
            "male": d["sex_common"].eq("M").astype(float),
            "current_inpatient": d["patient_class_common"].eq("Inpatient").astype(float),
            "prior_first2_low_alert": d["prior_first2_any_low"].astype(float),
            "prior_any_bolus": d["prior_any_bolus_0_30"].astype(float),
            "prior_low_to_bolus_5": d["prior_low_to_bolus_5"].astype(float),
            "prior_first_MAP_per_10mmHg": d["prior_first_map"] / 10,
            "prior_MAP_change_per_10mmHg": d["prior_first2_change"] / 10,
        }
    )
    for column in x.columns:
        x[column] = pd.to_numeric(x[column], errors="coerce")
        x[column] = x[column].fillna(x[column].median())
    names = ["const", *x.columns]
    design = np.column_stack([np.ones(len(x)), x.to_numpy(float)])
    y = d[outcome].to_numpy(float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        probability = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
        loss = float(np.sum(np.logaddexp(0, eta) - y * eta))
        return loss, design.T @ (probability - y)

    fit = minimize(
        lambda b: objective(b)[0], np.zeros(design.shape[1]),
        jac=lambda b: objective(b)[1], method="BFGS",
        options={"maxiter": 2500, "gtol": 1e-8},
    )
    if not np.all(np.isfinite(fit.x)):
        raise RuntimeError("non-finite interpretable association fit")
    beta = fit.x
    eta = design @ beta
    probability = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
    weight = probability * (1 - probability)
    bread_inverse = np.linalg.pinv(design.T @ (design * weight[:, None]))
    score = design * (y - probability)[:, None]
    score_frame = pd.DataFrame(score)
    score_frame.insert(0, "patient_id", d["patient_id"].to_numpy())
    cluster_score = score_frame.groupby("patient_id", sort=False).sum().to_numpy()
    meat = cluster_score.T @ cluster_score
    clusters = len(cluster_score)
    n, k = design.shape
    correction = (clusters / (clusters - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread_inverse @ meat @ bread_inverse
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    rows = []
    for term in [
        "prior_first2_low_alert", "prior_any_bolus", "prior_low_to_bolus_5",
        "prior_first_MAP_per_10mmHg", "prior_MAP_change_per_10mmHg",
    ]:
        j = names.index(term)
        estimate = float(beta[j])
        standard_error = float(se[j])
        z = estimate / standard_error
        rows.append(
            {
                "outcome": outcome,
                "term": term,
                "odds_ratio": math.exp(estimate),
                "ci_low": math.exp(estimate - 1.959963984540054 * standard_error),
                "ci_high": math.exp(estimate + 1.959963984540054 * standard_error),
                "p_value_cluster_robust": float(2 * norm.sf(abs(z))),
                "interpretation": "adjusted association, not causal effect",
            }
        )
    return pd.DataFrame(rows)


def risk_strata(d: pd.DataFrame, outcome: str) -> pd.DataFrame:
    band = pd.cut(
        d["prior_first_map"], bins=[20, 65, 75, 85, 201], right=False,
        labels=["<65", "65-74", "75-84", ">=85"],
    )
    rows = []
    for (level, prior_action), frame in d.groupby(
        [band, "prior_any_bolus_0_30"], observed=False
    ):
        events = int(frame[outcome].sum())
        low, high = wilson(events, len(frame))
        rows.append(
            {
                "prior_first_MAP_band": str(level),
                "prior_any_bolus": int(prior_action),
                "n": int(len(frame)),
                "events": events,
                "event_rate": events / len(frame) if len(frame) else math.nan,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pair = canonicalize(pd.read_csv(PAIR, dtype={"LOG_ID": str, "prior_LOG_ID": str}, low_memory=False))
    if len(pair) != 7721 or pair["patient_id"].nunique() != 5297:
        raise RuntimeError("MOVER pair cohort drift")
    operation_ids = set(pair["LOG_ID"].astype(str)) | set(pair["prior_LOG_ID"].astype(str))
    raw_mar = pd.read_csv(MAR, dtype={"LOG_ID": str}, low_memory=False)
    bolus = valid_bolus_rows(raw_mar)
    low, map_audit = conflict_free_low_times(operation_ids)
    features = operation_features(operation_ids, bolus, low)

    current = features.add_prefix("current_").rename(columns={"current_LOG_ID": "LOG_ID"})
    prior = features.add_prefix("prior_").rename(columns={"prior_LOG_ID": "prior_LOG_ID"})
    d = pair.merge(current, on="LOG_ID", how="left", validate="many_to_one")
    d = d.merge(prior, on="prior_LOG_ID", how="left", validate="many_to_one")
    action_columns = [
        "any_bolus_0_30", "repeated_bolus_0_30", "any_bolus_0_15",
        "low_to_bolus_5", "low_to_bolus_10", "low_to_bolus_minus2_5",
    ]
    for prefix in ["current", "prior"]:
        for column in action_columns:
            name = f"{prefix}_{column}"
            d[name] = pd.to_numeric(d[name], errors="coerce").fillna(0).astype(int)

    y_columns = {
        "current_low_to_bolus_5": "low MAP followed by bolus within 0-5 min",
        "current_any_bolus_0_30": "any routine IV vasopressor bolus in 0-30 min",
        "current_repeated_bolus_0_30": "at least two routine IV vasopressor boluses in 0-30 min",
        "current_any_bolus_0_15": "any routine IV vasopressor bolus in 0-15 min",
        "current_low_to_bolus_10": "low MAP followed by bolus within 0-10 min",
    }
    event_rows = []
    for outcome, label in y_columns.items():
        event_rows.append(
            {
                "outcome": outcome,
                "label": label,
                "pairs": int(len(d)),
                "patients": int(d["patient_id"].nunique()),
                "events": int(d[outcome].sum()),
                "event_rate": float(d[outcome].mean()),
            }
        )
    event_flow = pd.DataFrame(event_rows)
    event_flow.to_csv(OUT / "event_flow.csv", index=False)

    feature_sets = {
        "M0_case_context": M0_CASE_CONTEXT,
        "M1_BP_alert": M1_BP_HISTORY,
        "M1_readable_BP_and_action_history": M1_READABLE_HISTORY,
        "M2_plus_continuous_prior_MAP_response": M2_CONTINUOUS_RESPONSE,
    }
    metric_rows = []
    increment_rows = []
    bootstrap_frames = []
    prediction_store: dict[str, dict[str, np.ndarray]] = {}
    groups = d["patient_id"].to_numpy()
    for outcome_index, outcome in enumerate(OUTCOMES):
        y = d[outcome].to_numpy(int)
        prediction, _ = local_oof(d, y, groups, feature_sets)
        prediction_store[outcome] = prediction
        for model, values in prediction.items():
            metric_rows.append(
                {"outcome": outcome, "model": model, **metric_row(y, values)}
            )
        for comparison, left, right, seed_offset in [
            (
                "M1_BP_alert_vs_M0",
                "M0_case_context", "M1_BP_alert", 0,
            ),
            (
                "M1A_action_history_vs_BP_alert",
                "M1_BP_alert", "M1_readable_BP_and_action_history", 50,
            ),
            (
                "M2_vs_M1_continuous_response",
                "M1_readable_BP_and_action_history",
                "M2_plus_continuous_prior_MAP_response", 100,
            ),
        ]:
            summary, bootstrap = subject_bootstrap_increment(
                y, groups, prediction[left], prediction[right],
                analysis=f"{outcome}_{comparison}", reps=1000,
                seed=20260815 + outcome_index + seed_offset,
            )
            for row in summary:
                row.update({"outcome": outcome, "comparison": comparison})
            bootstrap.insert(0, "outcome", outcome)
            bootstrap.insert(1, "comparison", comparison)
            increment_rows.extend(summary)
            bootstrap_frames.append(bootstrap)
    metrics = pd.DataFrame(metric_rows)
    increments = pd.DataFrame(increment_rows)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    increments.to_csv(OUT / "increment_cluster_bootstrap.csv", index=False)
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(
        OUT / "increment_bootstrap_replicates.csv.gz", index=False, compression="gzip"
    )

    primary = "current_low_to_bolus_5"
    association = interpretable_association(d, primary)
    association.to_csv(OUT / "cluster_robust_association.csv", index=False)
    strata = risk_strata(d, primary)
    strata.to_csv(OUT / "primary_risk_strata.csv", index=False)

    # How much the physiologic and treatment constructs overlap in the current case.
    overlap_rows = []
    repeated_path = BASE / "clinical_endpoint_upgrade/two_centre_repeated_low/summary.json"
    # Reconstruct current repeated-low on the same operation IDs using the first four NIBP values.
    map_full = pd.read_csv(
        MAPS,
        usecols=[
            "LOG_ID", "RECORDED_TIME", "relative_min", "value", "RECORD_TYPE",
            "FLO_MEAS_NAME", "FLO_DISPLAY_NAME", "modality_hint",
        ],
        dtype={"LOG_ID": str}, low_memory=False,
    )
    map_full["relative_min"] = pd.to_numeric(map_full["relative_min"], errors="coerce")
    map_full["value"] = pd.to_numeric(map_full["value"], errors="coerce")
    map_full["time"] = pd.to_datetime(map_full["RECORDED_TIME"], errors="coerce")
    map_full = map_full.loc[
        map_full["LOG_ID"].isin(set(d["LOG_ID"]))
        & map_full["RECORD_TYPE"].eq("INTRA-OP")
        & map_full["FLO_MEAS_NAME"].eq(NIBP_MEAS)
        & map_full["FLO_DISPLAY_NAME"].eq(NIBP_DISPLAY)
        & map_full["modality_hint"].eq("NIBP")
        & map_full["relative_min"].between(0, 30)
        & map_full["value"].between(20, 200)
        & map_full["time"].notna()
    ]
    map_key = (
        map_full.groupby(["LOG_ID", "time"], as_index=False, observed=True)
        .agg(relative_min=("relative_min", "min"), distinct=("value", "nunique"), value=("value", "first"))
    )
    map_key = map_key.loc[map_key["distinct"].eq(1)].sort_values(["LOG_ID", "relative_min", "time"])
    repeated = {}
    for log_id, frame in map_key.groupby("LOG_ID", observed=True, sort=False):
        first4 = frame.head(4)
        if len(first4) < 4:
            continue
        value = first4["value"].to_numpy(float)
        time = first4["relative_min"].to_numpy(float)
        low4 = value < 65
        gap = np.diff(time)
        repeated[str(log_id)] = int((low4[:-1] & low4[1:] & (gap >= 2) & (gap <= 15)).any())
    overlap = d[["LOG_ID", primary, "current_any_bolus_0_30"]].copy()
    overlap["repeated_low_first4"] = overlap["LOG_ID"].map(repeated)
    overlap = overlap.loc[overlap["repeated_low_first4"].notna()].copy()
    for repeated_value, frame in overlap.groupby("repeated_low_first4", observed=True):
        overlap_rows.append(
            {
                "repeated_low_first4": int(repeated_value),
                "n": int(len(frame)),
                "low_to_bolus_5_events": int(frame[primary].sum()),
                "low_to_bolus_5_rate": float(frame[primary].mean()),
                "any_bolus_0_30_events": int(frame["current_any_bolus_0_30"].sum()),
                "any_bolus_0_30_rate": float(frame["current_any_bolus_0_30"].mean()),
            }
        )
    overlap_table = pd.DataFrame(overlap_rows)
    overlap_table.to_csv(OUT / "physiology_action_overlap.csv", index=False)

    # One paper-oriented four-panel figure.
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0))
    plot_flow = event_flow.set_index("outcome").loc[OUTCOMES]
    short_labels = ["Low→bolus\n≤5 min", "Any bolus\n0–30 min", "≥2 boluses\n0–30 min", "Any bolus\n0–15 min", "Low→bolus\n≤10 min"]
    axes[0, 0].bar(short_labels, plot_flow["event_rate"] * 100, color="#3A86A8")
    axes[0, 0].set_ylabel("Pairs with endpoint (%)")
    axes[0, 0].set_title("A  Clinically ordered action endpoints")
    axes[0, 0].tick_params(axis="x", labelrotation=12)

    primary_metrics = metrics.loc[metrics["outcome"].eq(primary)].set_index("model")
    model_order = list(feature_sets)
    axes[0, 1].bar(
        ["M0\ncase context", "M1\n+ BP alert", "M1A\n+ action history", "M2\n+ continuous MAP"],
        primary_metrics.loc[model_order, "auroc"],
        color=["#B8C0CC", "#9CB8A5", "#DDA15E", "#2A9D8F"],
    )
    axes[0, 1].set_ylim(max(.5, primary_metrics.auroc.min() - .04), primary_metrics.auroc.max() + .025)
    axes[0, 1].set_ylabel("Patient-grouped OOF AUROC")
    axes[0, 1].set_title("B  Low MAP followed by bolus within 5 min")

    ci = increments.loc[
        increments["comparison"].eq("M2_vs_M1_continuous_response")
        & increments["metric"].eq("delta_auroc")
    ].set_index("outcome").loc[OUTCOMES]
    point = ci["point"].to_numpy(float)
    low_ci = ci["ci_low"].to_numpy(float)
    high_ci = ci["ci_high"].to_numpy(float)
    yy = np.arange(len(ci))
    axes[1, 0].errorbar(
        point, yy, xerr=[point - low_ci, high_ci - point],
        fmt="o", color="#8E5EA2", capsize=3,
    )
    axes[1, 0].axvline(0, color="black", lw=1)
    axes[1, 0].set_yticks(yy, short_labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("AUROC increment: continuous MAP beyond readable history")
    axes[1, 0].set_title("C  Increment across action definitions")

    heat = strata.pivot(index="prior_first_MAP_band", columns="prior_any_bolus", values="event_rate")
    heat = heat.reindex(["<65", "65-74", "75-84", ">=85"])[[0, 1]] * 100
    image = axes[1, 1].imshow(heat.to_numpy(float), cmap="YlOrRd", aspect="auto")
    axes[1, 1].set_xticks([0, 1], ["No prior bolus", "Prior bolus"])
    axes[1, 1].set_yticks(np.arange(4), heat.index)
    axes[1, 1].set_ylabel("Prior first MAP (mmHg)")
    axes[1, 1].set_title("D  Current low→bolus risk (%)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            axes[1, 1].text(j, i, f"{value:.1f}" if np.isfinite(value) else "NA", ha="center", va="center")
    fig.colorbar(image, ax=axes[1, 1], label="Risk (%)", fraction=.046, pad=.04)
    fig.suptitle("Prior-anaesthetic MAP response and next-case early vasopressor treatment demand", y=.995)
    fig.text(
        .5, .005,
        "Routine bolus = recorded INTRA-OP IV phenylephrine/ephedrine Given, plausible unit/dose. "
        "This is an administered-action proxy, not proof of treatment indication or benefit.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=[0, .035, 1, .98])
    fig.savefig(OUT / "fig_vasopressor_action_deepening.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig_vasopressor_action_deepening.svg", bbox_inches="tight")
    plt.close(fig)

    primary_ci = increments.loc[
        increments["outcome"].eq(primary)
        & increments["comparison"].eq("M2_vs_M1_continuous_response")
    ].set_index("metric")
    readable_ci = increments.loc[
        increments["outcome"].eq(primary)
        & increments["comparison"].eq("M1A_action_history_vs_BP_alert")
    ].set_index("metric")
    treatment_history_positive = bool(
        readable_ci.loc["delta_auroc", "ci_low"] > 0
        and readable_ci.loc["brier_improvement", "ci_low"] > 0
        and readable_ci.loc["log_loss_improvement", "ci_low"] > 0
    )
    continuous_null = bool(
        primary_ci.loc["delta_auroc", "ci_low"] <= 0
        and primary_ci.loc["delta_auroc", "ci_high"] >= 0
    )
    summary = {
        "status": (
            "KEEP_NEGATIVE_MECHANISTIC_COMPARATOR_ACTION_HISTORY_DOMINATES"
            if treatment_history_positive and continuous_null
            else "ACTION_ENDPOINT_PATTERN_NOT_CLEANLY_SEPARATED"
        ),
        "construct": {
            "main_action": (
                "INTRA-OP MAR_ACTION=Given; IV/IV Push; phenylephrine 10-500 mcg "
                "or ephedrine 1-50 mg; anaesthesia-start +0..30 min"
            ),
            "primary_endpoint": "conflict-free NIBP MAP<65 followed by main action within 0-5 min",
            "claim_boundary": (
                "recorded treatment-demand/process proxy; not proof of hypotension indication, "
                "treatment benefit, causality, or organ-outcome improvement"
            ),
        },
        "source_audit": {
            "pair_rows": int(len(d)),
            "patients": int(d["patient_id"].nunique()),
            "candidate_MAR_rows": int(len(raw_mar)),
            "valid_routine_IV_bolus_rows_all_pair_operations": int(len(bolus.loc[bolus.LOG_ID.isin(operation_ids)])),
            **map_audit,
        },
        "event_flow": event_flow.to_dict(orient="records"),
        "model_metrics": metrics.to_dict(orient="records"),
        "increment_cluster_bootstrap": increments.to_dict(orient="records"),
        "cluster_robust_association": association.to_dict(orient="records"),
        "physiology_action_overlap": overlap_table.to_dict(orient="records"),
        "patient_level_outputs_in_report_directory": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
