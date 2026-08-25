#!/usr/bin/env python3
"""Multi-anaesthetic physiology-or-management history in MOVER.

The post hoc process construct is deliberately narrow and time ordered:
  * physiology: relative MAP decline >=20% from the pre-hypnotic value to the
    nadir of the first two post-hypnotic NIBP MAP values;
  * management: a recorded INTRA-OP/Given/routine IV phenylephrine or ephedrine
    bolus strictly after the second selected MAP and within five minutes.

All three operations must have their second selected MAP no later than ten
minutes after the actual IV hypnotic, giving every operation the same five
minute management-observation opportunity. Only the first eligible triplet per
patient is used. The construct is a recorded haemodynamic-instability or
management event, not adjudicated rescue, treatment effect, or organ injury.
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analyze_c02_hypnotic_anchored_reproducibility import operation_hypnotics
from analyze_c02_multiepisode_memory import prepare_mover, procedure_family, wilson
from analyze_mover_c02_early_vasopressor_action import valid_bolus_rows


from c02_runtime import private_workspace_root


ROOT = private_workspace_root()
PAIR = ROOT / "data/restricted/mover/extracted/mover_c02_relative_hypnotic_pair.csv.gz"
MAP = ROOT / "data/restricted/mover/extracted/mover_cleaned_early_map.csv.gz"
HYPNOTIC_MAR = ROOT / "data/restricted/mover/extracted/mover_early_anesthetic_mar.csv.gz"
VASOPRESSOR_MAR = ROOT / "data/restricted/mover/extracted/mover_early_vasopressor_mar.csv.gz"
OUT = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild/"
    "induction_drug_context/hypnotic_anchored/multiepisode_management_instability"
)
SEED = 20261181

HISTORY_STATE = {(0, 0): "Neither", (1, 0): "Remote only", (0, 1): "Recent only", (1, 1): "Both"}
STATE_ORDER = ["Neither", "Remote only", "Recent only", "Both"]


def reconstruct_selected_map_times(operation_ids: set[str], anchors: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    maps = pd.read_csv(MAP, dtype={"LOG_ID": str}, low_memory=False)
    maps = maps.loc[
        maps.LOG_ID.isin(operation_ids)
        & maps.RECORD_TYPE.eq("INTRA-OP")
        & maps.modality_hint.eq("NIBP")
    ].copy()
    maps["relative_min"] = pd.to_numeric(maps.relative_min, errors="coerce")
    maps["value"] = pd.to_numeric(maps.value, errors="coerce")
    maps["time"] = pd.to_datetime(maps.RECORDED_TIME, errors="coerce")
    maps = maps.loc[
        maps.relative_min.between(0, 30, inclusive="both")
        & maps.value.between(20, 200, inclusive="both")
        & maps.time.notna()
    ]
    grouped = maps.groupby(["LOG_ID", "time"], as_index=False, observed=True).agg(
        relative_min=("relative_min", "min"), value=("value", "first"),
        distinct_values=("value", "nunique"), raw_records=("value", "size"),
    )
    conflicts = int(grouped.distinct_values.gt(1).sum())
    grouped = grouped.loc[grouped.distinct_values.eq(1)].merge(
        anchors, on="LOG_ID", how="inner", validate="many_to_one"
    )
    grouped["anchor_delta_min"] = grouped.relative_min-grouped.anchor_rel
    rows = []
    for log_id, frame in grouped.groupby("LOG_ID", sort=False, observed=True):
        pre = frame.loc[
            frame.anchor_delta_min.between(-10, 0, inclusive="left")
        ].sort_values(["anchor_delta_min", "time"]).tail(1)
        post = frame.loc[
            frame.anchor_delta_min.between(0, 15, inclusive="right")
        ].sort_values(["anchor_delta_min", "time"]).head(2)
        if len(pre) != 1 or len(post) != 2:
            continue
        baseline = float(pre.value.iloc[0])
        values = post.value.to_numpy(float)
        times = post.anchor_delta_min.to_numpy(float)
        rows.append({
            "LOG_ID": str(log_id), "baseline_MAP": baseline,
            "post_MAP_0": values[0], "post_MAP_1": values[1],
            "post_time_0": times[0], "post_time_1": times[1],
            "relative_drop": (baseline-float(values.min()))/baseline,
            "relative_drop_20": int((baseline-float(values.min()))/baseline >= .20),
            "absolute_low": int(np.any(values < 65)),
        })
    output = pd.DataFrame(rows)
    return output, {
        "raw_NIBP_rows": int(len(maps)),
        "same_time_keys": int(len(grouped)+conflicts),
        "conflicting_same_time_keys_excluded": conflicts,
        "operations_with_baseline_and_two_post_MAP": int(len(output)),
    }


def construct_cohort() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    triplets, triplet_audit = prepare_mover()
    operation_ids = set(triplets.A_id)|set(triplets.B_id)|set(triplets.C_id)
    hypnotic = pd.read_csv(HYPNOTIC_MAR, dtype={"LOG_ID": str}, low_memory=False)
    anchors, anchor_audit = operation_hypnotics(hypnotic.loc[hypnotic.LOG_ID.isin(operation_ids)])
    operation, map_audit = reconstruct_selected_map_times(operation_ids, anchors)
    if set(operation.LOG_ID) != operation_ids:
        raise RuntimeError("not every MOVER triplet operation reconstructed")
    lookup = operation.set_index("LOG_ID")
    for occasion in "ABC":
        expected = triplets[f"{occasion}_drop"].to_numpy(float)
        reconstructed = triplets[f"{occasion}_id"].map(lookup.relative_drop).to_numpy(float)
        if np.max(np.abs(expected-reconstructed)) > 1e-12:
            raise RuntimeError(f"{occasion} response reconstruction failed")
        triplets[f"{occasion}_post_time_1"] = triplets[f"{occasion}_id"].map(lookup.post_time_1)
        triplets[f"{occasion}_post_MAP_1"] = triplets[f"{occasion}_id"].map(lookup.post_MAP_1)

    vaso = pd.read_csv(VASOPRESSOR_MAR, dtype={"LOG_ID": str}, low_memory=False)
    bolus = valid_bolus_rows(vaso.loc[vaso.LOG_ID.isin(operation_ids)]).copy()
    bolus["dose"] = pd.to_numeric(bolus.ADMIN_SIG, errors="coerce")
    before_dedup = len(bolus)
    exact = ["LOG_ID", "drug_class", "MED_ACTION_TIME", "dose", "DOSE_UNIT_NM", "MED_ROUTE_NM"]
    duplicate_rows = int(bolus.duplicated(exact, keep=False).sum())
    bolus = bolus.drop_duplicates(exact)
    bolus = bolus.merge(anchors[["LOG_ID", "anchor_rel"]], on="LOG_ID", validate="many_to_one")
    bolus = bolus.merge(
        operation[["LOG_ID", "post_time_0", "post_time_1", "post_MAP_1", "relative_drop"]],
        on="LOG_ID", validate="many_to_one",
    )
    bolus["after_anchor_min"] = bolus.relative_min-bolus.anchor_rel
    bolus["after_second_MAP_min"] = bolus.after_anchor_min-bolus.post_time_1
    post_second = bolus.loc[
        bolus.after_second_MAP_min.gt(0)&bolus.after_second_MAP_min.le(5)
    ].copy()
    action_operations = set(post_second.LOG_ID)

    eligible = triplets[[f"{occasion}_post_time_1" for occasion in "ABC"]].le(10).all(axis=1)
    d = triplets.loc[eligible].copy().reset_index(drop=True)
    anchor_lookup = anchors.set_index("LOG_ID")
    for occasion in "ABC":
        d[f"{occasion}_management"] = d[f"{occasion}_id"].isin(action_operations).astype(int)
        d[f"{occasion}_instability"] = (
            d[f"{occasion}_high"].astype(int)+d[f"{occasion}_management"]
        ).gt(0).astype(int)
        for drug in ["propofol", "etomidate", "ketamine"]:
            d[f"{occasion}_{drug}_mg"] = d[f"{occasion}_id"].map(
                anchor_lookup[f"anchor_{drug}_mg"]
            )
    d["physiology_history_count"] = d.A_high+d.B_high
    d["management_history_count"] = d.A_management+d.B_management
    d["instability_history_count"] = d.A_instability+d.B_instability
    d["current_instability"] = d.C_instability
    d["current_physiology"] = d.C_high
    d["current_management"] = d.C_management
    d["current_absolute_low"] = d.C_absolute_low.astype(int)
    d["history_state"] = [
        HISTORY_STATE[(a, b)] for a, b in zip(d.A_instability, d.B_instability)
    ]

    duration = pd.read_csv(PAIR, dtype={"LOG_ID": str, "prior_LOG_ID": str}, low_memory=False)
    current = duration[["LOG_ID", "anstart", "anstop"]].rename(
        columns={"anstart": "start", "anstop": "stop"}
    )
    prior = duration[["prior_LOG_ID", "prior_anstart", "prior_anstop"]].rename(
        columns={"prior_LOG_ID": "LOG_ID", "prior_anstart": "start", "prior_anstop": "stop"}
    )
    timing = pd.concat([current, prior], ignore_index=True).drop_duplicates()
    timing["start"] = pd.to_datetime(timing.start, errors="coerce")
    timing["stop"] = pd.to_datetime(timing.stop, errors="coerce")
    timing["duration_min"] = (timing.stop-timing.start).dt.total_seconds()/60
    timing_conflicts = int(timing.groupby("LOG_ID").duration_min.nunique().gt(1).sum())
    timing = timing.drop_duplicates("LOG_ID").merge(
        anchors[["LOG_ID", "anchor_rel"]], on="LOG_ID", how="inner", validate="one_to_one"
    )
    timing["support_after_anchor_min"] = timing.duration_min-timing.anchor_rel
    used_ids = set(d.A_id)|set(d.B_id)|set(d.C_id)
    support = timing.loc[timing.LOG_ID.isin(used_ids), "support_after_anchor_min"]
    if len(support) != 3*len(d) or support.min() < 15:
        raise RuntimeError("equal follow-up opportunity or anaesthesia support failed")

    post_second_primary = post_second.loc[post_second.LOG_ID.isin(used_ids)].copy()
    action_only = operation.loc[
        operation.LOG_ID.isin(used_ids)
        & operation.LOG_ID.isin(action_operations)
        & operation.relative_drop.lt(.20)
    ].copy()
    audit = {
        "triplet_source": triplet_audit,
        "anchor_audit": anchor_audit,
        "map_audit": map_audit,
        "candidate_triplets": int(len(triplets)),
        "equal_opportunity_triplets_post2_le10": int(len(d)),
        "coverage_fraction": float(len(d)/len(triplets)),
        "unique_patients": int(d.patient_id.nunique()),
        "all_three_operations_have_anchor_plus15_support": True,
        "minimum_support_after_anchor_min": float(support.min()),
        "timing_conflicting_LOG_IDs": timing_conflicts,
        "narrow_valid_bolus_rows_before_dedup": int(before_dedup),
        "exact_duplicate_bolus_rows": duplicate_rows,
        "post_second_MAP_5min_bolus_rows": int(len(post_second_primary)),
        "post_second_MAP_5min_bolus_operations": int(post_second_primary.LOG_ID.nunique()),
        "action_only_operations": int(len(action_only)),
        "action_only_post_MAP1_median": float(action_only.post_MAP_1.median()),
        "action_only_relative_drop_median": float(action_only.relative_drop.median()),
        "action_only_MAPlt80_or_dropge10_fraction": float(
            (action_only.post_MAP_1.lt(80)|action_only.relative_drop.ge(.10)).mean()
        ),
        "construct": (
            "relative MAP decline >=20% OR recorded routine IV phenylephrine/ephedrine Given "
            "strictly after second selected MAP and within 5 min; post2<=10 min in all A/B/C"
        ),
    }
    return d, audit, post_second_primary


def risk_tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_rows, state_rows = [], []
    for count in [0, 1, 2]:
        frame = d.loc[d.instability_history_count.eq(count)]
        for outcome in ["current_instability", "current_physiology", "current_management", "current_absolute_low"]:
            events = int(frame[outcome].sum()); low, high = wilson(events, len(frame))
            count_rows.append({
                "history_count": count, "outcome": outcome, "n": len(frame),
                "events": events, "risk": events/len(frame), "ci_low": low, "ci_high": high,
            })
    for state in STATE_ORDER:
        frame = d.loc[d.history_state.eq(state)]
        events = int(frame.current_instability.sum()); low, high = wilson(events, len(frame))
        state_rows.append({
            "history_state": state, "n": len(frame), "events": events,
            "risk": events/len(frame), "ci_low": low, "ci_high": high,
        })
    return pd.DataFrame(count_rows), pd.DataFrame(state_rows)


def design_frame(d: pd.DataFrame, history: str) -> pd.DataFrame:
    x = pd.DataFrame({
        "C_baseline_per10": d.C_baseline/10, "B_baseline_per10": d.B_baseline/10,
        "A_baseline_per10": d.A_baseline/10, "age_per10": d.age/10,
        "bmi_per5": d.bmi/5, "asa": d.asa,
        "log_recent_interval": np.log1p(d.interval_recent_days),
        "log_remote_interval": np.log1p(d.interval_remote_days),
        "male": d.sex.eq("M").astype(int), "current_inpatient": d.current_inpatient,
        "same_agent_three": d.same_agent_three.astype(int),
        "same_family_three": d.same_family_three.astype(int),
        "C_post_time_1": d.C_post_time_1,
    })
    for occasion in "ABC":
        x[f"{occasion}_propofol_mg"] = d[f"{occasion}_propofol_mg"]
        x[f"{occasion}_etomidate_mg"] = d[f"{occasion}_etomidate_mg"]
        x[f"{occasion}_ketamine_mg"] = d[f"{occasion}_ketamine_mg"]
    if history == "composite_count":
        x["history_count"] = d.instability_history_count.astype(float)
    elif history == "decomposed":
        x["physiology_history_count"] = d.physiology_history_count.astype(float)
        x["management_history_count"] = d.management_history_count.astype(float)
    elif history != "none":
        raise ValueError(history)
    categorical = pd.get_dummies(pd.DataFrame({
        "current_family": d.C_procedure.map(procedure_family),
        "A_family": d.A_procedure.map(procedure_family),
        "B_family": d.B_procedure.map(procedure_family),
        "current_agent": d.C_agent, "A_agent": d.A_agent, "B_agent": d.B_agent,
    }).fillna("missing").astype(str), drop_first=True, dtype=float)
    x = pd.concat([x, categorical], axis=1)
    x = x.apply(pd.to_numeric, errors="coerce")
    for column in x:
        median = x[column].median()
        x[column] = x[column].fillna(0 if not np.isfinite(median) else median)
        if column not in [
            "history_count", "physiology_history_count", "management_history_count"
        ]:
            sd = x[column].std(ddof=0)
            if sd > 0:
                x[column] = (x[column]-x[column].mean())/sd
    return x


def robust_logit(d: pd.DataFrame, outcome: str, history: str) -> tuple[pd.DataFrame, dict]:
    x = design_frame(d, history)
    X = np.column_stack([np.ones(len(x)), x.to_numpy(float)])
    names = ["intercept", *x.columns]
    y = d[outcome].to_numpy(float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.einsum("ij,j->i", X, beta)
        probability = 1/(1+np.exp(-np.clip(eta, -35, 35)))
        loss = float(np.sum(np.logaddexp(0, eta)-y*eta)+1e-6*np.sum(beta[1:]**2))
        gradient = np.einsum("ni,n->i", X, probability-y)
        gradient[1:] += 2e-6*beta[1:]
        return loss, gradient

    fit = minimize(lambda b: objective(b)[0], np.zeros(X.shape[1]),
                   jac=lambda b: objective(b)[1], method="BFGS",
                   options={"maxiter": 2000, "gtol": 1e-7})
    beta = fit.x
    eta = np.einsum("ij,j->i", X, beta)
    probability = 1/(1+np.exp(-np.clip(eta, -35, 35)))
    weight = probability*(1-probability)
    bread = np.einsum("ni,n,nj->ij", X, weight, X)+np.eye(X.shape[1])*1e-8
    inv = np.linalg.pinv(bread)
    residual = y-probability
    meat = np.einsum("ni,n,nj->ij", X, residual*residual, X)
    covariance = np.einsum("ij,jk,kl->il", inv, meat, inv)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    selected = [name for name in [
        "history_count", "physiology_history_count", "management_history_count"
    ] if name in names]
    rows = []
    for term in selected:
        index = names.index(term); b = float(beta[index]); s = float(se[index])
        rows.append({
            "outcome": outcome, "model": history, "term": term,
            "odds_ratio_per_additional_prior_event": math.exp(b),
            "ci_low": math.exp(b-1.96*s), "ci_high": math.exp(b+1.96*s),
            "p_value": float(2*norm.sf(abs(b/s))) if s > 0 else math.nan,
        })
    standardized = {}
    if history == "composite_count":
        for value in [0, 1, 2]:
            modified = d.copy(); modified["instability_history_count"] = value
            Xnew = np.column_stack([np.ones(len(d)), design_frame(modified, history).to_numpy(float)])
            p = 1/(1+np.exp(-np.clip(np.einsum("ij,j->i", Xnew, beta), -35, 35)))
            standardized[str(value)] = float(p.mean())
    return pd.DataFrame(rows), {
        "outcome": outcome, "model": history, "n": len(d), "events": int(y.sum()),
        "parameters": X.shape[1], "converged": bool(fit.success or np.all(np.isfinite(beta))),
        "max_abs_gradient": float(np.max(np.abs(objective(beta)[1]))),
        "standardized_risks": standardized,
    }


def oof_information_models(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_context = [
        "C_baseline", "B_baseline", "A_baseline", "age", "bmi", "asa",
        "interval_recent_days", "interval_remote_days", "C_post_time_1",
        "current_inpatient", "same_agent_three", "same_family_three",
        "A_propofol_mg", "A_etomidate_mg", "A_ketamine_mg",
        "B_propofol_mg", "B_etomidate_mg", "B_ketamine_mg",
        "C_propofol_mg", "C_etomidate_mg", "C_ketamine_mg",
    ]
    categorical_context = [
        "C_agent", "A_agent", "B_agent", "current_family", "A_family", "B_family"
    ]
    frame = d.copy()
    frame["current_family"] = frame.C_procedure.map(procedure_family)
    frame["A_family"] = frame.A_procedure.map(procedure_family)
    frame["B_family"] = frame.B_procedure.map(procedure_family)
    specs = {
        "M0_context": [],
        "M1_plus_physiology_history": ["physiology_history_count"],
        "M1_plus_management_history": ["management_history_count"],
        "M2_plus_both_histories": ["physiology_history_count", "management_history_count"],
        "M2_composite_history_count": ["instability_history_count"],
    }
    predictions, rows = {}, []
    for outcome in ["current_instability", "current_physiology", "current_management"]:
        y = frame[outcome].to_numpy(int)
        splits = list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(frame, y)
        )
        for name, history_features in specs.items():
            numeric = numeric_context+history_features
            preprocessor = ColumnTransformer([
                ("numeric", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), numeric),
                ("categorical", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), categorical_context),
            ])
            model = Pipeline([
                ("preprocess", preprocessor),
                ("logit", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
            ])
            prediction = np.full(len(frame), np.nan)
            for train, test in splits:
                model.fit(frame.iloc[train], y[train])
                prediction[test] = model.predict_proba(frame.iloc[test])[:, 1]
            predictions[(outcome, name)] = prediction
            rows.append({
                "outcome": outcome, "model": name, "n": len(y), "events": int(y.sum()),
                "auroc": float(roc_auc_score(y, prediction)),
                "average_precision": float(average_precision_score(y, prediction)),
                "brier": float(brier_score_loss(y, prediction)),
                "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
            })
    metrics = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED+1)
    boot_rows = []
    comparisons = [
        ("management_beyond_physiology", "M1_plus_physiology_history", "M2_plus_both_histories"),
        ("physiology_beyond_management", "M1_plus_management_history", "M2_plus_both_histories"),
        ("composite_beyond_context", "M0_context", "M2_composite_history_count"),
    ]
    for rep in range(1000):
        index = rng.integers(0, len(frame), len(frame))
        for outcome in ["current_instability", "current_physiology", "current_management"]:
            y = frame[outcome].to_numpy(int)[index]
            if np.unique(y).size < 2:
                continue
            for label, left, right in comparisons:
                p0 = predictions[(outcome, left)][index]; p1 = predictions[(outcome, right)][index]
                boot_rows.append({
                    "rep": rep, "outcome": outcome, "comparison": label,
                    "delta_auroc": roc_auc_score(y, p1)-roc_auc_score(y, p0),
                    "delta_average_precision": average_precision_score(y, p1)-average_precision_score(y, p0),
                    "brier_improvement": brier_score_loss(y, p0)-brier_score_loss(y, p1),
                    "log_loss_improvement": log_loss(y, p0, labels=[0, 1])-log_loss(y, p1, labels=[0, 1]),
                })
    boot = pd.DataFrame(boot_rows)
    summaries = []
    metric_lookup = metrics.set_index(["outcome", "model"])
    for outcome in ["current_instability", "current_physiology", "current_management"]:
        for label, left, right in comparisons:
            frame_boot = boot.loc[(boot.outcome.eq(outcome))&boot.comparison.eq(label)]
            row = {"outcome": outcome, "comparison": label, "left": left, "right": right, "reps": 1000}
            for metric in ["delta_auroc", "delta_average_precision", "brier_improvement", "log_loss_improvement"]:
                left_metric = {
                    "delta_auroc": "auroc", "delta_average_precision": "average_precision",
                    "brier_improvement": "brier", "log_loss_improvement": "log_loss",
                }[metric]
                if metric.startswith("delta_"):
                    point = metric_lookup.loc[(outcome, right), left_metric]-metric_lookup.loc[(outcome, left), left_metric]
                else:
                    point = metric_lookup.loc[(outcome, left), left_metric]-metric_lookup.loc[(outcome, right), left_metric]
                row[metric] = float(point)
                row[metric+"_ci_low"] = float(frame_boot[metric].quantile(.025))
                row[metric+"_ci_high"] = float(frame_boot[metric].quantile(.975))
            summaries.append(row)
    return metrics, pd.DataFrame(summaries)


def bootstrap_contrasts(d: pd.DataFrame, reps: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED+2)
    rows = []
    for outcome in ["current_instability", "current_physiology", "current_management", "current_absolute_low"]:
        y = d[outcome].to_numpy(int); history = d.instability_history_count.to_numpy(int)
        point = float(y[history==2].mean()-y[history==0].mean())
        values = []
        for _ in range(reps):
            index = rng.integers(0, len(d), len(d)); yy = y[index]; hh = history[index]
            if np.any(hh==0)&np.any(hh==2):
                values.append(float(yy[hh==2].mean()-yy[hh==0].mean()))
        rows.append({
            "outcome": outcome, "contrast": "two_vs_zero_prior_instability_events",
            "point": point, "ci_low": float(np.quantile(values, .025)),
            "ci_high": float(np.quantile(values, .975)), "reps": len(values),
        })
    return pd.DataFrame(rows)


def matched_stranger_null(d: pd.DataFrame, reps: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = d.copy().reset_index(drop=True)
    x["baseline_bin"] = pd.qcut(x.C_baseline.rank(method="first"), 5, labels=False)
    x["interval_bin"] = pd.cut(x.interval_recent_days, [-np.inf, 30, 180, np.inf], labels=False)
    x["asa_bin"] = pd.to_numeric(x.asa, errors="coerce").round().fillna(-1).astype(int)
    x["current_family"] = x.C_procedure.map(procedure_family)
    levels = [
        ["C_agent", "current_family", "baseline_bin", "interval_bin", "asa_bin"],
        ["C_agent", "current_family", "baseline_bin", "interval_bin"],
        ["C_agent", "baseline_bin", "interval_bin", "asa_bin"],
        ["C_agent", "baseline_bin", "interval_bin"], ["C_agent", "baseline_bin"],
        ["baseline_bin"], [],
    ]
    options, used = [], []
    for i, row in x.iterrows():
        for level_index, columns in enumerate(levels):
            mask = np.ones(len(x), dtype=bool)
            for column in columns:
                mask &= x[column].eq(row[column]).to_numpy()
            candidates = np.flatnonzero(mask&(np.arange(len(x))!=i))
            if len(candidates):
                options.append(candidates); used.append(level_index); break
    history = x.instability_history_count.to_numpy(int)
    outcome = x.current_instability.to_numpy(int)
    own = float(outcome[history==2].mean()-outcome[history==0].mean())
    rng = np.random.default_rng(SEED+3); values = []
    assignment_rows = []
    for rep in range(reps):
        donors = np.array([rng.choice(candidate) for candidate in options], int)
        donated = history[donors]
        value = float(outcome[donated==2].mean()-outcome[donated==0].mean())
        values.append(value)
        assignment_rows.append({"rep": rep, "risk_difference": value})
    summary = pd.DataFrame([{
        "metric": "two_vs_zero_prior_instability_events_RD", "own_value": own,
        "null_median": float(np.median(values)), "null_ci_low": float(np.quantile(values, .025)),
        "null_ci_high": float(np.quantile(values, .975)),
        "empirical_one_sided_p": float((1+np.sum(np.asarray(values)>=own))/(reps+1)),
        "reps": reps, "match_level_counts": json.dumps(pd.Series(used).value_counts().sort_index().to_dict()),
    }])
    return summary, pd.DataFrame(assignment_rows)


def sensitivity_table(d: pd.DataFrame, post_second: pd.DataFrame) -> pd.DataFrame:
    rows = []
    action_by_window = {}
    for window in [3, 5]:
        action_by_window[window] = set(post_second.loc[
            post_second.after_second_MAP_min.le(window), "LOG_ID"
        ])
    for label, subset, window in [
        ("primary_equal_opportunity_5min", np.ones(len(d), bool), 5),
        ("post_MAP_3min_window", np.ones(len(d), bool), 3),
        ("same_agent_all_three", d.same_agent_three.to_numpy(bool), 5),
        ("same_procedure_family_all_three", d.same_family_three.to_numpy(bool), 5),
        ("current_baseline_MAP_ge85", d.C_baseline.ge(85).to_numpy(bool), 5),
    ]:
        frame = d.loc[subset].copy()
        action_ids = action_by_window[window]
        for occasion in "ABC":
            frame[f"{occasion}_sens_action"] = frame[f"{occasion}_id"].isin(action_ids).astype(int)
            frame[f"{occasion}_sens_instability"] = (
                frame[f"{occasion}_high"]+frame[f"{occasion}_sens_action"]
            ).gt(0).astype(int)
        frame["sens_history_count"] = frame.A_sens_instability+frame.B_sens_instability
        for count in [0, 1, 2]:
            q = frame.loc[frame.sens_history_count.eq(count)]
            rows.append({
                "sensitivity": label, "history_count": count, "n": len(q),
                "current_composite_events": int(q.C_sens_instability.sum()),
                "current_composite_risk": float(q.C_sens_instability.mean()),
                "current_physiology_events": int(q.C_high.sum()),
                "current_physiology_risk": float(q.C_high.mean()),
                "current_management_events": int(q.C_sens_action.sum()),
                "current_management_risk": float(q.C_sens_action.mean()),
            })
    return pd.DataFrame(rows)


def make_figure(risks: pd.DataFrame, adjusted: dict, contrast: pd.DataFrame,
                stranger: pd.DataFrame, profile: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    colors = {"current_instability": "#7B2CBF", "current_physiology": "#E76F51",
              "current_management": "#2A9D8F"}
    ax = axes[0, 0]
    for outcome, label in [("current_instability", "Composite process"),
                           ("current_physiology", "Relative MAP decline ≥20%"),
                           ("current_management", "Post-MAP IV bolus")]:
        q = risks.loc[risks.outcome.eq(outcome)].set_index("history_count").loc[[0,1,2]]
        ax.errorbar([0,1,2], q.risk*100,
                    yerr=[(q.risk-q.ci_low)*100,(q.ci_high-q.risk)*100],
                    marker="o", capsize=3, lw=2, color=colors[outcome], label=label)
    ax.set_xticks([0,1,2]); ax.set_xlabel("Prior two anaesthetics meeting composite process definition")
    ax.set_ylabel("Third-anaesthetic event risk (%)"); ax.set_title("A. History-depth gradient")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    raw = risks.loc[risks.outcome.eq("current_instability")].set_index("history_count").loc[[0,1,2]].risk
    adj = pd.Series(adjusted, dtype=float).reindex(["0","1","2"])
    xloc = np.arange(3); width=.34
    ax.bar(xloc-width/2, raw*100, width, color="#B8A1CF", label="Raw")
    ax.bar(xloc+width/2, adj*100, width, color="#6A4C93", label="Context + current MAP adjusted")
    ax.set_xticks(xloc,["0","1","2"]); ax.set_ylabel("Third-anaesthetic composite risk (%)")
    ax.set_title("B. Standardized absolute risk"); ax.legend(frameon=False)

    ax = axes[1, 0]
    q = contrast.set_index("outcome").loc[["current_instability","current_physiology","current_management"]]
    yloc=np.arange(3)
    ax.errorbar(q.point*100, yloc,
                xerr=[(q.point-q.ci_low)*100,(q.ci_high-q.point)*100],
                fmt="o", color="#264653", capsize=3)
    ax.axvline(0,color="#555",lw=1)
    ax.set_yticks(yloc,["Composite process","Relative decline","Post-MAP bolus"])
    ax.set_xlabel("Two vs zero prior composite events: risk difference (pp)")
    ax.set_title("C. Component-specific persistence")

    ax = axes[1, 1]
    null = stranger.iloc[0]
    ax.errorbar(null.null_median*100,0,
                xerr=[[(null.null_median-null.null_ci_low)*100],[(null.null_ci_high-null.null_median)*100]],
                fmt="o",color="#457B9D",capsize=3,label="Matched-stranger null")
    ax.plot(null.own_value*100,1,"D",color="#D95F59",label="Own two-case history")
    ax.axvline(0,color="#555",lw=1);ax.set_yticks([0,1],["Matched stranger","Own history"])
    ax.set_xlabel("Two vs zero prior composite events: risk difference (pp)")
    ax.set_title("D. Patient-linkage negative control");ax.legend(frameon=False,fontsize=8)
    fig.suptitle("Multi-anaesthetic physiology-or-management history", fontweight="bold")
    fig.text(.5,.008,
             f"Action-only operations: median second MAP {profile['action_only_post_MAP1_median']:.0f} mmHg; "
             f"median relative decline {100*profile['action_only_relative_drop_median']:.1f}%",
             ha="center",fontsize=8,color="#444")
    fig.tight_layout(rect=[0,.025,1,.97])
    fig.savefig(OUT/"fig_multiepisode_management_instability.png",dpi=240,bbox_inches="tight")
    fig.savefig(OUT/"fig_multiepisode_management_instability.svg",bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d, audit, post_second = construct_cohort()
    risks, states = risk_tables(d)
    contrast = bootstrap_contrasts(d)
    robust_rows, diagnostics = [], {}
    for outcome in ["current_instability", "current_physiology", "current_management", "current_absolute_low"]:
        for history in ["composite_count", "decomposed"]:
            table, diagnostic = robust_logit(d, outcome, history)
            robust_rows.append(table); diagnostics[f"{outcome}_{history}"] = diagnostic
    robust = pd.concat(robust_rows, ignore_index=True)
    metrics, increments = oof_information_models(d)
    stranger, stranger_reps = matched_stranger_null(d)
    sensitivity = sensitivity_table(d, post_second)

    risks.to_csv(OUT/"risk_by_history_count.csv",index=False)
    states.to_csv(OUT/"risk_by_history_state.csv",index=False)
    contrast.to_csv(OUT/"bootstrap_risk_contrasts.csv",index=False)
    robust.to_csv(OUT/"adjusted_associations.csv",index=False)
    metrics.to_csv(OUT/"oof_model_metrics.csv",index=False)
    increments.to_csv(OUT/"oof_increment_bootstrap.csv",index=False)
    stranger.to_csv(OUT/"matched_stranger_summary.csv",index=False)
    stranger_reps.to_csv(OUT/"matched_stranger_replicates.csv.gz",index=False,compression="gzip")
    sensitivity.to_csv(OUT/"sensitivity_risks.csv",index=False)
    post_second.groupby(["drug_class"],observed=True).agg(
        rows=("LOG_ID","size"),operations=("LOG_ID","nunique"),
        median_dose=("dose","median"),min_dose=("dose","min"),max_dose=("dose","max"),
    ).reset_index().to_csv(OUT/"post_MAP_bolus_profile.csv",index=False)

    primary_diagnostic = diagnostics["current_instability_composite_count"]
    make_figure(risks, primary_diagnostic["standardized_risks"], contrast, stranger, audit)
    primary = contrast.loc[contrast.outcome.eq("current_instability")].iloc[0]
    main_increment = increments.loc[
        increments.outcome.eq("current_instability")
        & increments.comparison.eq("management_beyond_physiology")
    ].iloc[0]
    summary = {
        "status": "MULTIEPISODE_MANAGEMENT_INSTABILITY_COMPLETED",
        "cohort": {
            "triplets": int(len(d)), "patients": int(d.patient_id.nunique()),
            "current_composite_events": int(d.current_instability.sum()),
            "current_physiology_events": int(d.current_physiology.sum()),
            "current_management_events": int(d.current_management.sum()),
            "current_absolute_low_events": int(d.current_absolute_low.sum()),
        },
        "audit": audit,
        "primary_two_vs_zero_risk_difference": primary.to_dict(),
        "adjusted_standardized_risks": primary_diagnostic["standardized_risks"],
        "management_history_beyond_physiology_OOF": main_increment.to_dict(),
        "matched_stranger": stranger.iloc[0].to_dict(),
        "model_diagnostics": diagnostics,
        "decision": (
            "KEEP_AS_MOVER_ONLY_CLINICAL_PROCESS_DEPTH" if (
                primary.ci_low>0
                and stranger.iloc[0].own_value>stranger.iloc[0].null_ci_high
                and all(sensitivity.loc[
                    sensitivity.sensitivity.isin([
                        "post_MAP_3min_window","same_agent_all_three",
                        "same_procedure_family_all_three","current_baseline_MAP_ge85",
                    ])
                ].groupby("sensitivity").apply(
                    lambda z: z.sort_values("history_count").current_composite_risk.is_monotonic_increasing,
                    include_groups=False,
                ))
            ) else "SUPPLEMENT_ONLY_OR_STOP"
        ),
        "claim_boundary": (
            "Recorded relative MAP decline or subsequent routine IV bolus formed a repeated process marker. "
            "Bolus use is clinician behaviour/practice as well as patient need; it is not adjudicated rescue, "
            "treatment benefit, stable physiology, or an externally replicated endpoint."
        ),
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
