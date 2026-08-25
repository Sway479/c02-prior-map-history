#!/usr/bin/env python3
"""Three-anaesthetic history depth and recency analysis for C02.

MOVER is the primary, medication-timed construct. INSPIRE is a directional
replication based on its anaesthesia-start +15 minute ART-priority landmark.
Only the first eligible triplet per patient is used, so rows are independent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr

from mover_c02_procedure_family import procedure_family


from c02_runtime import private_workspace_root


ROOT = private_workspace_root()
MOVER_PAIR = ROOT / "data/restricted/mover/extracted/mover_c02_relative_hypnotic_pair.csv.gz"
INSPIRE_PAIR = ROOT / (
    "outputs/inspire_curiosity/candidate_gates/c02_redesign_v2/"
    "all_consecutive_index_eligible_pairs.csv.gz"
)
INSPIRE_EXCLUSION = ROOT / (
    "outputs/inspire_curiosity/candidate_gates/c02_redesign_v2/"
    "posthoc_restricted_exclusion_flags.csv.gz"
)
INSPIRE_TRIPLET_CONFLICT_AUDIT = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild/"
    "induction_drug_context/hypnotic_anchored/multiepisode_memory/"
    "inspire_triplet_source_conflict_audit.csv.gz"
)
OUT = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild/"
    "induction_drug_context/hypnotic_anchored/multiepisode_memory"
)
SEED = 20261141

STATE_ORDER = ["Neither", "Remote only", "Recent only", "Both"]
STATE_LABEL = {
    (0, 0): "Neither", (1, 0): "Remote only",
    (0, 1): "Recent only", (1, 1): "Both",
}


def wilson(events: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = events / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def prepare_mover() -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(MOVER_PAIR, low_memory=False)
    bc = pd.DataFrame({
        "patient_id": d.patient_id.astype(str), "B_id": d.prior_LOG_ID.astype(str),
        "C_id": d.LOG_ID.astype(str), "C_time": pd.to_datetime(d.anstart, errors="coerce"),
        "B_drop": d.prior_relative_drop, "C_drop": d.current_relative_drop,
        "B_baseline": d.prior_baseline_MAP, "C_baseline": d.current_baseline_MAP,
        "B_agent": d.prior_anchor_agent, "C_agent": d.current_anchor_agent,
        "B_procedure": d.prior_procedure_common, "C_procedure": d.procedure_common,
        "age": d.age_years, "bmi": d.bmi_kg_m2, "asa": d.asa_numeric,
        "sex": d.sex_common, "patient_class": d.patient_class_common,
        "interval_recent_days": d.interval_days,
        "C_propofol_mgkg": d.current_anchor_propofol_mg / d.weight_kg,
        "C_etomidate_mgkg": d.current_anchor_etomidate_mg / d.weight_kg,
        "C_ketamine_mgkg": d.current_anchor_ketamine_mg / d.weight_kg,
        "C_absolute_low": d.target_post_any_low.astype(int),
        "C_pre_gap": d.current_pre_gap_min, "C_post_gap": d.current_post_first_gap_min,
        "B_pre_gap": d.prior_pre_gap_min, "B_post_gap": d.prior_post_first_gap_min,
    })
    ab = pd.DataFrame({
        "patient_id": d.patient_id.astype(str), "A_id": d.prior_LOG_ID.astype(str),
        "B_id": d.LOG_ID.astype(str), "A_drop": d.prior_relative_drop,
        "B_drop_check": d.current_relative_drop, "A_baseline": d.prior_baseline_MAP,
        "B_baseline_check": d.current_baseline_MAP,
        "A_agent": d.prior_anchor_agent, "B_agent_check": d.current_anchor_agent,
        "A_procedure": d.prior_procedure_common, "B_procedure_check": d.procedure_common,
        "interval_remote_days": d.interval_days,
        "A_pre_gap": d.prior_pre_gap_min, "A_post_gap": d.prior_post_first_gap_min,
    })
    t = bc.merge(ab, on=["patient_id", "B_id"], validate="many_to_one")
    max_drop_error = float(np.max(np.abs(t.B_drop - t.B_drop_check)))
    max_baseline_error = float(np.max(np.abs(t.B_baseline - t.B_baseline_check)))
    if max_drop_error > 1e-12 or max_baseline_error > 1e-12:
        raise RuntimeError("MOVER middle-operation identity failed")
    t = t.sort_values(["patient_id", "C_time", "C_id"]).drop_duplicates("patient_id").copy()
    t["dataset"] = "MOVER"
    t["A_high"] = t.A_drop.ge(.20).astype(int)
    t["B_high"] = t.B_drop.ge(.20).astype(int)
    t["C_high"] = t.C_drop.ge(.20).astype(int)
    t["state"] = [STATE_LABEL[(a, b)] for a, b in zip(t.A_high, t.B_high)]
    t["history_count"] = t.A_high + t.B_high
    t["current_family"] = t.C_procedure.map(procedure_family)
    t["same_agent_three"] = (
        t.A_agent.eq(t.B_agent) & t.B_agent.eq(t.C_agent)
    )
    t["same_family_three"] = (
        t.A_procedure.map(procedure_family).eq(t.B_procedure.map(procedure_family))
        & t.B_procedure.map(procedure_family).eq(t.current_family)
    )
    t["current_context"] = t.current_family
    t["current_measurement"] = t.C_agent.fillna("missing")
    t["current_inpatient"] = t.patient_class.eq("Inpatient").astype(int)
    t["current_emergency"] = 0
    audit = {
        "adjacent_pairs": int(len(d)), "linked_triplets": int(len(bc.merge(ab, on=["patient_id", "B_id"]))),
        "first_triplets": int(len(t)), "patients": int(t.patient_id.nunique()),
        "middle_drop_identity_max_abs_error": max_drop_error,
        "middle_baseline_identity_max_abs_error": max_baseline_error,
    }
    return t, audit


def prepare_inspire(
    apply_triplet_source_conflict_audit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(INSPIRE_PAIR, low_memory=False)
    excluded = pd.read_csv(INSPIRE_EXCLUSION, usecols=["subject_id", "exclude_posthoc"])
    excluded_subjects = set(excluded.loc[excluded.exclude_posthoc, "subject_id"].astype(int))
    bc = pd.DataFrame({
        "patient_id": d.subject_id.astype(str), "B_id": d.prior_op_id.astype("Int64").astype(str),
        "C_id": d.op_id.astype("Int64").astype(str), "C_time": d.anstart_time,
        "B_drop": -d.prior_delta15 / d.prior_preop_formula_map,
        "C_drop": (d.preop_formula_map - d.best_art_priority_value) / d.preop_formula_map,
        "B_baseline": d.prior_preop_formula_map, "C_baseline": d.preop_formula_map,
        "B_agent": d.prior_best_art_priority_modality,
        "C_agent": d.best_art_priority_modality,
        "B_procedure": d.prior_procedure3, "C_procedure": d.procedure3,
        "age": d.age, "bmi": d.bmi, "asa": d.asa, "sex": d.sex,
        "patient_class": "not_available", "interval_recent_days": d.interval_days,
        "C_propofol_mgkg": np.nan, "C_etomidate_mgkg": np.nan,
        "C_ketamine_mgkg": np.nan, "C_absolute_low": d.outcome_low65.astype("Int64"),
        "current_department": d.department, "current_emop": d.emop,
        "C_pre_gap": np.nan, "C_post_gap": np.nan, "B_pre_gap": np.nan,
        "B_post_gap": np.nan,
    })
    ab = pd.DataFrame({
        "patient_id": d.subject_id.astype(str), "A_id": d.prior_op_id.astype("Int64").astype(str),
        "B_id": d.op_id.astype("Int64").astype(str),
        "A_drop": -d.prior_delta15 / d.prior_preop_formula_map,
        "B_drop_check": (d.preop_formula_map - d.best_art_priority_value) / d.preop_formula_map,
        "A_baseline": d.prior_preop_formula_map, "B_baseline_check": d.preop_formula_map,
        "A_agent": d.prior_best_art_priority_modality, "B_agent_check": d.best_art_priority_modality,
        "A_procedure": d.prior_procedure3, "B_procedure_check": d.procedure3,
        "interval_remote_days": d.interval_days,
        "A_pre_gap": np.nan, "A_post_gap": np.nan,
    })
    linked = bc.merge(ab, on=["patient_id", "B_id"], validate="many_to_one")
    valid_identity = linked[["B_drop", "B_drop_check", "B_baseline", "B_baseline_check"]].notna().all(axis=1)
    max_drop_error = float(np.max(np.abs(linked.loc[valid_identity, "B_drop"] - linked.loc[valid_identity, "B_drop_check"])))
    max_baseline_error = float(np.max(np.abs(linked.loc[valid_identity, "B_baseline"] - linked.loc[valid_identity, "B_baseline_check"])))
    if max_drop_error > 1e-12 or max_baseline_error > 1e-12:
        raise RuntimeError("INSPIRE middle-operation identity failed")
    t = linked.loc[
        linked[["A_drop", "B_drop", "C_drop"]].notna().all(axis=1)
        & ~linked.patient_id.astype(int).isin(excluded_subjects)
    ].copy()
    t = t.sort_values(["patient_id", "C_time", "C_id"]).drop_duplicates("patient_id").copy()
    source_conflict_subjects: set[int] = set()
    if apply_triplet_source_conflict_audit:
        if not INSPIRE_TRIPLET_CONFLICT_AUDIT.exists():
            raise RuntimeError(
                "Run audit_c02_multiepisode_source_conflicts.py before the main analysis"
            )
        source_audit = pd.read_csv(INSPIRE_TRIPLET_CONFLICT_AUDIT)
        source_conflict_subjects = set(
            source_audit.loc[
                source_audit.exclude_triplet_source_conflict.astype(bool), "subject_id"
            ].astype(int)
        )
        t = t.loc[~t.patient_id.astype(int).isin(source_conflict_subjects)].copy()
    t["dataset"] = "INSPIRE"
    t["A_high"] = t.A_drop.ge(.20).astype(int)
    t["B_high"] = t.B_drop.ge(.20).astype(int)
    t["C_high"] = t.C_drop.ge(.20).astype(int)
    t["state"] = [STATE_LABEL[(a, b)] for a, b in zip(t.A_high, t.B_high)]
    t["history_count"] = t.A_high + t.B_high
    t["current_family"] = t.C_procedure.fillna("missing").astype(str).str[:2]
    t["same_agent_three"] = t.A_agent.eq(t.B_agent) & t.B_agent.eq(t.C_agent)
    t["same_family_three"] = (
        t.A_procedure.fillna("missing").astype(str).str[:2].eq(
            t.B_procedure.fillna("missing").astype(str).str[:2]
        ) & t.B_procedure.fillna("missing").astype(str).str[:2].eq(t.current_family)
    )
    t["current_context"] = t.current_department.fillna("missing")
    t["current_measurement"] = t.C_agent.fillna("missing")
    t["current_inpatient"] = 0
    t["current_emergency"] = t.current_emop.fillna(0).astype(int)
    audit = {
        "adjacent_pairs": int(len(d)), "linked_triplets": int(len(linked)),
        "excluded_subjects_from_existing_correction_flags": int(len(excluded_subjects)),
        "excluded_triplets_for_new_source_conflict_audit": int(
            len(source_conflict_subjects)
        ),
        "first_valid_triplets": int(len(t)), "patients": int(t.patient_id.nunique()),
        "middle_drop_identity_max_abs_error": max_drop_error,
        "middle_baseline_identity_max_abs_error": max_baseline_error,
    }
    return t, audit


def risk_tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    states, counts, times, hidden = [], [], [], []
    for dataset, x in d.groupby("dataset", sort=False):
        for state in STATE_ORDER:
            z = x.loc[x.state.eq(state)]
            events = int(z.C_high.sum()); n = len(z); lo, hi = wilson(events, n)
            states.append({"dataset": dataset, "state": state, "n": n, "events": events,
                           "risk": events/n, "ci_low": lo, "ci_high": hi})
        for count in [0, 1, 2]:
            z = x.loc[x.history_count.eq(count)]
            events = int(z.C_high.sum()); n = len(z); lo, hi = wilson(events, n)
            counts.append({"dataset": dataset, "history_count": count, "n": n,
                           "events": events, "risk": events/n, "ci_low": lo, "ci_high": hi})
        bands = pd.cut(x.interval_recent_days, [-np.inf, 30, 180, np.inf],
                       labels=["≤30 days", "31–180 days", ">180 days"])
        for band in bands.cat.categories:
            for recent in [0, 1]:
                z = x.loc[bands.eq(band) & x.B_high.eq(recent)]
                events = int(z.C_high.sum()); n = len(z); lo, hi = wilson(events, n)
                times.append({"dataset": dataset, "interval_band": str(band),
                              "recent_high": recent, "n": n, "events": events,
                              "risk": events/n, "ci_low": lo, "ci_high": hi})
        z = x.loc[x.C_baseline.ge(85)].copy()
        for state in STATE_ORDER:
            q = z.loc[z.state.eq(state)]
            events = int(q.C_high.sum()); n = len(q); lo, hi = wilson(events, n)
            hidden.append({"dataset": dataset, "state": state, "n": n, "events": events,
                           "risk": events/n, "ci_low": lo, "ci_high": hi})
    return map(pd.DataFrame, (states, counts, times, hidden))


def bootstrap_contrasts(d: pd.DataFrame, reps: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for dataset_index, (dataset, x) in enumerate(d.groupby("dataset", sort=False)):
        x = x.reset_index(drop=True)

        def calculate(z: pd.DataFrame) -> dict[str, float]:
            risks = z.groupby(["A_high", "B_high"]).C_high.mean()
            p00 = risks.loc[(0, 0)]; p10 = risks.loc[(1, 0)]
            p01 = risks.loc[(0, 1)]; p11 = risks.loc[(1, 1)]
            result = {
                "persistent_vs_neither_rd": p11-p00,
                "persistent_vs_neither_rr": p11/p00,
                "recent_only_vs_neither_rd": p01-p00,
                "remote_only_vs_neither_rd": p10-p00,
                "remote_increment_when_recent_high_rd": p11-p01,
                "recent_increment_when_remote_high_rd": p11-p10,
                "additive_interaction": p11-p10-p01+p00,
            }
            bands = pd.cut(z.interval_recent_days, [-np.inf, 30, 180, np.inf], labels=False)
            for band, label in [(0, "le30"), (1, "31to180"), (2, "gt180")]:
                q = z.loc[bands.eq(band)]
                result[f"recent_rd_{label}"] = (
                    q.loc[q.B_high.eq(1), "C_high"].mean()
                    - q.loc[q.B_high.eq(0), "C_high"].mean()
                )
            return result

        point = calculate(x)
        boot = {name: [] for name in point}
        for _ in range(reps):
            sample = x.iloc[rng.integers(0, len(x), len(x))]
            if sample.groupby(["A_high", "B_high"]).ngroups < 4:
                continue
            values = calculate(sample)
            for name, value in values.items():
                if np.isfinite(value):
                    boot[name].append(value)
        for name, value in point.items():
            values = np.asarray(boot[name], float)
            rows.append({"dataset": dataset, "contrast": name, "point": value,
                         "ci_low": float(np.quantile(values, .025)),
                         "ci_high": float(np.quantile(values, .975)),
                         "bootstrap_reps": int(len(values))})
    return pd.DataFrame(rows)


def design_matrix(
    d: pd.DataFrame, count_model: bool = False,
    include_current_baseline: bool = True,
) -> tuple[np.ndarray, list[str]]:
    numeric = pd.DataFrame({
        "A_baseline_per10": d.A_baseline / 10,
        "B_baseline_per10": d.B_baseline / 10,
        "age_per10": d.age / 10, "bmi_per5": d.bmi / 5,
        "asa": d.asa, "log_recent_interval": np.log1p(d.interval_recent_days),
        "log_remote_interval": np.log1p(d.interval_remote_days),
        "current_inpatient": d.current_inpatient,
        "current_emergency": d.current_emergency,
        "same_agent_three": d.same_agent_three.astype(float),
        "same_family_three": d.same_family_three.astype(float),
    })
    if include_current_baseline:
        numeric["C_baseline_per10"] = d.C_baseline / 10
    if d.dataset.iloc[0] == "MOVER":
        numeric["propofol_mgkg"] = d.C_propofol_mgkg
        numeric["etomidate_mgkg"] = d.C_etomidate_mgkg
        numeric["ketamine_mgkg"] = d.C_ketamine_mgkg
        numeric["current_pre_gap"] = d.C_pre_gap
        numeric["current_post_gap"] = d.C_post_gap
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    for column in numeric:
        median = numeric[column].median()
        numeric[column] = numeric[column].fillna(0 if not np.isfinite(median) else median)
        sd = numeric[column].std(ddof=0)
        if sd > 0:
            numeric[column] = (numeric[column] - numeric[column].mean()) / sd
    categorical = pd.get_dummies(
        d[["sex", "current_context", "current_measurement"]].fillna("missing").astype(str),
        drop_first=True, dtype=float,
    )
    history = pd.DataFrame(index=d.index)
    if count_model:
        history["history_count"] = d.history_count.astype(float)
    else:
        history["remote_high"] = d.A_high.astype(float)
        history["recent_high"] = d.B_high.astype(float)
        history["remote_x_recent"] = d.A_high.astype(float) * d.B_high.astype(float)
    x = pd.concat([history, numeric, categorical], axis=1)
    X = np.column_stack([np.ones(len(x)), x.to_numpy(float)])
    return X, ["intercept", *x.columns]


def robust_logit(
    d: pd.DataFrame, count_model: bool = False,
    include_current_baseline: bool = True,
) -> tuple[pd.DataFrame, dict]:
    X, names = design_matrix(
        d, count_model=count_model, include_current_baseline=include_current_baseline
    )
    y = d.C_high.to_numpy(float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.einsum("ij,j->i", X, beta)
        probability = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
        loss = float(np.sum(np.logaddexp(0, eta) - y*eta) + 1e-6*np.sum(beta[1:]**2))
        gradient = np.einsum("ni,n->i", X, probability-y)
        gradient[1:] += 2e-6*beta[1:]
        return loss, gradient

    fit = minimize(lambda b: objective(b)[0], np.zeros(X.shape[1]),
                   jac=lambda b: objective(b)[1], method="BFGS",
                   options={"maxiter": 2000, "gtol": 1e-7})
    beta = fit.x
    eta = np.einsum("ij,j->i", X, beta)
    probability = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
    weight = probability*(1-probability)
    bread = np.einsum("ni,n,nj->ij", X, weight, X) + np.eye(X.shape[1])*1e-8
    inv = np.linalg.pinv(bread)
    residual = y-probability
    meat = np.einsum("ni,n,nj->ij", X, residual*residual, X)
    covariance = np.einsum("ij,jk,kl->il", inv, meat, inv)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    terms = ["history_count"] if count_model else ["remote_high", "recent_high", "remote_x_recent"]
    rows = []
    for term in terms:
        idx = names.index(term); b = float(beta[idx]); s = float(se[idx])
        rows.append({"term": term, "log_odds": b, "robust_se": s,
                     "odds_ratio": math.exp(b), "ci_low": math.exp(b-1.96*s),
                     "ci_high": math.exp(b+1.96*s),
                     "p_value": float(2*norm.sf(abs(b/s))) if s > 0 else math.nan})
    adjusted = {}
    if not count_model:
        for a, b in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            modified = d.copy(); modified["A_high"] = a; modified["B_high"] = b
            Xnew, new_names = design_matrix(
                modified, count_model=False,
                include_current_baseline=include_current_baseline,
            )
            if new_names != names:
                raise RuntimeError("adjusted prediction design mismatch")
            p = 1/(1+np.exp(-np.clip(np.einsum("ij,j->i", Xnew, beta), -35, 35)))
            adjusted[STATE_LABEL[(a, b)]] = float(p.mean())
    diagnostics = {
        "converged": bool(fit.success or np.all(np.isfinite(beta))),
        "n": int(len(d)), "events": int(y.sum()), "parameters": int(X.shape[1]),
        "max_abs_gradient": float(np.max(np.abs(objective(beta)[1]))),
        "adjusted_standardized_risks": adjusted,
    }
    return pd.DataFrame(rows), diagnostics


def attenuation_decomposition(d: pd.DataFrame, reps: int = 300) -> pd.DataFrame:
    """How much of the raw history gradient is absorbed by visible case context/current MAP?"""
    rng = np.random.default_rng(SEED + 60)

    def estimates(x: pd.DataFrame) -> dict[str, float]:
        raw = (
            x.loc[x.state.eq("Both"), "C_high"].mean()
            - x.loc[x.state.eq("Neither"), "C_high"].mean()
        )
        _, context = robust_logit(x, include_current_baseline=False)
        _, full = robust_logit(x, include_current_baseline=True)
        context_risk = context["adjusted_standardized_risks"]
        full_risk = full["adjusted_standardized_risks"]
        context_rd = context_risk["Both"] - context_risk["Neither"]
        full_rd = full_risk["Both"] - full_risk["Neither"]
        return {
            "raw_rd": float(raw), "context_adjusted_rd": float(context_rd),
            "plus_current_baseline_rd": float(full_rd),
            "attenuation_after_current_baseline": float(context_rd-full_rd),
            "fraction_context_rd_attenuated_by_current_baseline": (
                float((context_rd-full_rd)/context_rd) if context_rd != 0 else math.nan
            ),
        }

    rows = []
    for dataset, x in d.groupby("dataset", sort=False):
        x = x.reset_index(drop=True)
        point = estimates(x)
        boot = {key: [] for key in point}
        for _ in range(reps):
            sample = x.iloc[rng.integers(0, len(x), len(x))].reset_index(drop=True)
            if sample.groupby(["A_high", "B_high"]).ngroups < 4:
                continue
            values = estimates(sample)
            for key, value in values.items():
                if np.isfinite(value):
                    boot[key].append(value)
        for metric, value in point.items():
            values = np.asarray(boot[metric], float)
            rows.append({
                "dataset": dataset, "metric": metric, "point": value,
                "ci_low": float(np.quantile(values, .025)),
                "ci_high": float(np.quantile(values, .975)), "reps": len(values),
                "interpretation": "statistical attenuation, not causal mediation",
            })
    return pd.DataFrame(rows)


def sensitivity_tables(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, x in d.groupby("dataset", sort=False):
        subsets = {
            "all_first_triplets": np.ones(len(x), dtype=bool),
            "same_measurement_or_agent_all_three": x.same_agent_three.to_numpy(bool),
            "same_procedure_family_all_three": x.same_family_three.to_numpy(bool),
            "current_baseline_ge85": x.C_baseline.ge(85).to_numpy(bool),
        }
        for label, keep in subsets.items():
            z = x.loc[keep]
            for state in STATE_ORDER:
                q = z.loc[z.state.eq(state)]
                rows.append({"dataset": dataset, "sensitivity": label, "state": state,
                             "n": int(len(q)), "events": int(q.C_high.sum()),
                             "risk": float(q.C_high.mean()) if len(q) else math.nan})
    return pd.DataFrame(rows)


def threshold_sensitivity(d: pd.DataFrame, reps: int = 1000) -> pd.DataFrame:
    """Avoid making the history gradient depend on the prespecified 20% cut."""
    rng = np.random.default_rng(SEED + 20)
    rows = []
    for dataset, x in d.groupby("dataset", sort=False):
        x = x.reset_index(drop=True)
        for threshold in [.10, .15, .20, .25, .30]:
            a = x.A_drop.ge(threshold).to_numpy()
            b = x.B_drop.ge(threshold).to_numpy()
            c = x.C_drop.ge(threshold).to_numpy()
            neither = ~a & ~b; both = a & b
            point = float(c[both].mean() - c[neither].mean())
            boot = []
            for _ in range(reps):
                index = rng.integers(0, len(x), len(x))
                bb = both[index]; nn = neither[index]; cc = c[index]
                if bb.sum() and nn.sum():
                    boot.append(float(cc[bb].mean() - cc[nn].mean()))
            rows.append({
                "dataset": dataset, "decline_threshold": threshold,
                "both_n": int(both.sum()), "both_events": int(c[both].sum()),
                "both_risk": float(c[both].mean()),
                "neither_n": int(neither.sum()), "neither_events": int(c[neither].sum()),
                "neither_risk": float(c[neither].mean()), "risk_difference": point,
                "ci_low": float(np.quantile(boot, .025)),
                "ci_high": float(np.quantile(boot, .975)),
            })
    return pd.DataFrame(rows)


def endpoint_extension(d: pd.DataFrame, reps: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe whether relative-response history extends to recorded MAP<65."""
    rng = np.random.default_rng(SEED + 30)
    risks, contrasts = [], []
    for dataset, x in d.groupby("dataset", sort=False):
        x = x.loc[x.C_absolute_low.notna()].copy().reset_index(drop=True)
        x["C_absolute_low"] = x.C_absolute_low.astype(int)
        for state in STATE_ORDER:
            z = x.loc[x.state.eq(state)]
            n = len(z); events = int(z.C_absolute_low.sum()); lo, hi = wilson(events, n)
            risks.append({"dataset": dataset, "state": state, "n": n, "events": events,
                          "risk": events/n, "ci_low": lo, "ci_high": hi})
        neither = x.state.eq("Neither").to_numpy(); both = x.state.eq("Both").to_numpy()
        y = x.C_absolute_low.to_numpy(int)
        point = float(y[both].mean()-y[neither].mean())
        boot = []
        for _ in range(reps):
            index = rng.integers(0, len(x), len(x))
            bb = both[index]; nn = neither[index]; yy = y[index]
            if bb.sum() and nn.sum():
                boot.append(float(yy[bb].mean()-yy[nn].mean()))
        contrasts.append({"dataset": dataset, "contrast": "both_vs_neither_absolute_low_rd",
                          "point": point, "ci_low": float(np.quantile(boot, .025)),
                          "ci_high": float(np.quantile(boot, .975)), "reps": len(boot)})
    return pd.DataFrame(risks), pd.DataFrame(contrasts)


def continuous_history(d: pd.DataFrame, reps: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 40)
    correlations, quartiles = [], []
    for dataset, x in d.groupby("dataset", sort=False):
        x = x.copy().reset_index(drop=True)
        x["history_mean"] = (x.A_drop+x.B_drop)/2
        x["history_max"] = x[["A_drop", "B_drop"]].max(axis=1)
        for term in ["A_drop", "B_drop", "history_mean", "history_max"]:
            point = float(spearmanr(x[term], x.C_drop).statistic)
            boot = []
            for _ in range(reps):
                index = rng.integers(0, len(x), len(x))
                value = spearmanr(x[term].to_numpy()[index], x.C_drop.to_numpy()[index]).statistic
                if np.isfinite(value):
                    boot.append(float(value))
            correlations.append({"dataset": dataset, "history_term": term,
                                 "spearman": point, "ci_low": float(np.quantile(boot, .025)),
                                 "ci_high": float(np.quantile(boot, .975)), "reps": len(boot)})
        x["history_mean_quartile"] = pd.qcut(
            x.history_mean.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]
        )
        for quartile, z in x.groupby("history_mean_quartile", observed=True):
            n = len(z); events = int(z.C_high.sum()); lo, hi = wilson(events, n)
            quartiles.append({"dataset": dataset, "quartile": str(quartile), "n": n,
                              "events": events, "risk": events/n, "ci_low": lo, "ci_high": hi,
                              "history_mean_median": float(z.history_mean.median())})
    return pd.DataFrame(correlations), pd.DataFrame(quartiles)


def matched_stranger_null(d: pd.DataFrame, reps: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace own two-case history with a context-matched different patient's history."""
    rng = np.random.default_rng(SEED + 50)
    summaries, replicates = [], []
    for dataset, x in d.groupby("dataset", sort=False):
        x = x.copy().reset_index(drop=True)
        x["baseline_bin"] = pd.qcut(x.C_baseline.rank(method="first"), 5, labels=False)
        x["interval_bin"] = pd.cut(x.interval_recent_days, [-np.inf, 30, 180, np.inf], labels=False)
        x["asa_bin"] = pd.to_numeric(x.asa, errors="coerce").round().fillna(-1).astype(int)
        match_levels = [
            ["current_measurement", "current_context", "baseline_bin", "interval_bin", "asa_bin"],
            ["current_measurement", "current_context", "baseline_bin", "interval_bin"],
            ["current_measurement", "baseline_bin", "interval_bin", "asa_bin"],
            ["current_measurement", "baseline_bin", "interval_bin"],
            ["current_measurement", "baseline_bin"], ["baseline_bin"], [],
        ]
        donor_options, level_used = [], []
        for i, row in x.iterrows():
            chosen = None
            for level_index, columns in enumerate(match_levels):
                mask = np.ones(len(x), dtype=bool)
                for column in columns:
                    mask &= x[column].eq(row[column]).to_numpy()
                candidates = np.flatnonzero(mask & (np.arange(len(x)) != i))
                if len(candidates):
                    chosen = candidates; level_used.append(level_index); break
            if chosen is None:
                raise RuntimeError("matched-stranger donor unavailable")
            donor_options.append(chosen)
        history_mean = ((x.A_drop+x.B_drop)/2).to_numpy(float)
        history_both = (x.A_high.eq(1)&x.B_high.eq(1)).to_numpy()
        history_neither = (x.A_high.eq(0)&x.B_high.eq(0)).to_numpy()
        outcome_continuous = x.C_drop.to_numpy(float)
        outcome_binary = x.C_high.to_numpy(int)
        own_rho = float(spearmanr(history_mean, outcome_continuous).statistic)
        own_rd = float(outcome_binary[history_both].mean()-outcome_binary[history_neither].mean())
        null_rho, null_rd = [], []
        for rep in range(reps):
            donors = np.array([rng.choice(options) for options in donor_options], dtype=int)
            donor_mean = history_mean[donors]
            donor_both = history_both[donors]; donor_neither = history_neither[donors]
            rho = float(spearmanr(donor_mean, outcome_continuous).statistic)
            rd = float(outcome_binary[donor_both].mean()-outcome_binary[donor_neither].mean())
            null_rho.append(rho); null_rd.append(rd)
            replicates.extend([
                {"dataset": dataset, "rep": rep, "metric": "spearman_history_mean", "value": rho},
                {"dataset": dataset, "rep": rep, "metric": "both_vs_neither_rd", "value": rd},
            ])
        for metric, own, values in [
            ("spearman_history_mean", own_rho, null_rho),
            ("both_vs_neither_rd", own_rd, null_rd),
        ]:
            values = np.asarray(values, float)
            summaries.append({
                "dataset": dataset, "metric": metric, "own_value": own,
                "null_median": float(np.median(values)),
                "null_ci_low": float(np.quantile(values, .025)),
                "null_ci_high": float(np.quantile(values, .975)),
                "empirical_one_sided_p": float((1+np.sum(values >= own))/(len(values)+1)),
                "reps": len(values),
                "match_level_counts": json.dumps(pd.Series(level_used).value_counts().sort_index().to_dict()),
            })
    return pd.DataFrame(summaries), pd.DataFrame(replicates)


def make_supplement_figure(thresholds: pd.DataFrame, absolute: pd.DataFrame,
                           quartiles: pd.DataFrame, null_summary: pd.DataFrame) -> None:
    colors = {"MOVER": "#D95F59", "INSPIRE": "#3A7CA5"}
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
    ax = axes[0, 0]
    for dataset in ["MOVER", "INSPIRE"]:
        z = thresholds.loc[thresholds.dataset.eq(dataset)]
        ax.errorbar(z.decline_threshold*100, z.risk_difference*100,
                    yerr=[(z.risk_difference-z.ci_low)*100, (z.ci_high-z.risk_difference)*100],
                    fmt="o-", color=colors[dataset], capsize=3, label=dataset)
    ax.axhline(0, color="#555", lw=1); ax.set_xlabel("Decline threshold (%)")
    ax.set_ylabel("Both vs neither risk difference (pp)")
    ax.set_title("A. Threshold sensitivity"); ax.legend(frameon=False)

    ax = axes[0, 1]; xloc = np.arange(4); width = .34
    for offset, dataset in [(-width/2, "MOVER"), (width/2, "INSPIRE")]:
        z = absolute.loc[absolute.dataset.eq(dataset)].set_index("state").loc[STATE_ORDER]
        ax.bar(xloc+offset, z.risk*100, width, color=colors[dataset], alpha=.86, label=dataset)
    ax.set_xticks(xloc, STATE_ORDER, rotation=15); ax.set_ylabel("Recorded MAP<65 risk (%)")
    ax.set_title("B. Extension to an absolute endpoint"); ax.legend(frameon=False)

    ax = axes[1, 0]
    for dataset in ["MOVER", "INSPIRE"]:
        z = quartiles.loc[quartiles.dataset.eq(dataset)].set_index("quartile").loc[["Q1","Q2","Q3","Q4"]]
        ax.errorbar([1,2,3,4], z.risk*100,
                    yerr=[(z.risk-z.ci_low)*100, (z.ci_high-z.risk)*100],
                    fmt="o-", color=colors[dataset], capsize=3, label=dataset)
    ax.set_xticks([1,2,3,4], ["Q1","Q2","Q3","Q4"])
    ax.set_xlabel("Mean relative decline across prior two cases")
    ax.set_ylabel("Third-anaesthetic decline ≥20% risk (%)")
    ax.set_title("C. Continuous-history gradient"); ax.legend(frameon=False)

    ax = axes[1, 1]; yloc = np.arange(2)
    for offset, dataset in [(-.08, "MOVER"), (.08, "INSPIRE")]:
        z = null_summary.loc[
            null_summary.dataset.eq(dataset)&null_summary.metric.eq("both_vs_neither_rd")
        ].iloc[0]
        ax.errorbar(z.null_median*100, yloc[0]+offset,
                    xerr=[[ (z.null_median-z.null_ci_low)*100],[(z.null_ci_high-z.null_median)*100]],
                    fmt="o", color=colors[dataset], capsize=3)
        ax.plot(z.own_value*100, yloc[1]+offset, "D", color=colors[dataset], label=dataset)
    ax.axvline(0,color="#555",lw=1); ax.set_yticks(yloc,["Matched-stranger null","Own history"])
    ax.set_xlabel("Both vs neither risk difference (pp)")
    ax.set_title("D. Patient-linkage negative control"); ax.legend(frameon=False)
    fig.suptitle("Robustness of the multi-anaesthetic history gradient", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT/"fig_multiepisode_robustness.png",dpi=240,bbox_inches="tight")
    fig.savefig(OUT/"fig_multiepisode_robustness.svg",bbox_inches="tight"); plt.close(fig)


def make_figure(states: pd.DataFrame, counts: pd.DataFrame,
                times: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    colors = {"MOVER": "#D95F59", "INSPIRE": "#3A7CA5"}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    ax = axes[0, 0]
    x = np.arange(4); width = .34
    for offset, dataset in [(-width/2, "MOVER"), (width/2, "INSPIRE")]:
        z = states.loc[states.dataset.eq(dataset)].set_index("state").loc[STATE_ORDER]
        ax.bar(x+offset, z.risk*100, width, color=colors[dataset], alpha=.86, label=dataset)
        ax.errorbar(x+offset, z.risk*100,
                    yerr=[(z.risk-z.ci_low)*100, (z.ci_high-z.risk)*100],
                    fmt="none", ecolor="#333333", capsize=2, lw=1)
        for xx, risk, n in zip(x+offset, z.risk*100, z.n):
            ax.text(xx, risk+2.2, f"n={n}", ha="center", fontsize=7, rotation=45)
    ax.set_xticks(x, ["Neither", "Remote only", "Recent only", "Both"])
    ax.set_ylabel("Third-anaesthetic MAP decline ≥20% (%)")
    ax.set_title("A. Two-case history states")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for dataset in ["MOVER", "INSPIRE"]:
        z = counts.loc[counts.dataset.eq(dataset)].set_index("history_count").loc[[0, 1, 2]]
        ax.errorbar([0, 1, 2], z.risk*100,
                    yerr=[(z.risk-z.ci_low)*100, (z.ci_high-z.risk)*100],
                    fmt="o-", color=colors[dataset], capsize=3, lw=2, label=dataset)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Number of prior two anaesthetics with ≥20% decline")
    ax.set_ylabel("Third-anaesthetic risk (%)")
    ax.set_title("B. Cumulative history gradient")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    band_order = ["≤30 days", "31–180 days", ">180 days"]
    xx = np.arange(3)
    for offset, dataset in [(-.12, "MOVER"), (.12, "INSPIRE")]:
        z = contrasts.loc[
            contrasts.dataset.eq(dataset)
            & contrasts.contrast.isin(["recent_rd_le30", "recent_rd_31to180", "recent_rd_gt180"])
        ].set_index("contrast").loc[["recent_rd_le30", "recent_rd_31to180", "recent_rd_gt180"]]
        ax.errorbar(xx+offset, z.point*100,
                    yerr=[(z.point-z.ci_low)*100, (z.ci_high-z.point)*100],
                    fmt="o", color=colors[dataset], capsize=3, label=dataset)
    ax.axhline(0, color="#555555", lw=1)
    ax.set_xticks(xx, band_order)
    ax.set_ylabel("Risk difference: recent positive vs negative (pp)")
    ax.set_title("C. Recency of the immediately prior response")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    selected = ["persistent_vs_neither_rd", "remote_increment_when_recent_high_rd",
                "additive_interaction"]
    labels = ["Both vs neither", "Remote history beyond recent positive", "Additive interaction"]
    yy = np.arange(len(selected))
    for offset, dataset in [(-.08, "MOVER"), (.08, "INSPIRE")]:
        z = contrasts.loc[
            contrasts.dataset.eq(dataset) & contrasts.contrast.isin(selected)
        ].set_index("contrast").loc[selected]
        ax.errorbar(z.point*100, yy+offset,
                    xerr=[(z.point-z.ci_low)*100, (z.ci_high-z.point)*100],
                    fmt="o", color=colors[dataset], capsize=3, label=dataset)
    ax.axvline(0, color="#555555", lw=1)
    ax.set_yticks(yy, labels)
    ax.set_xlabel("Absolute risk contrast (percentage points)")
    ax.set_title("D. Persistence and accumulation")
    ax.legend(frameon=False)
    fig.suptitle("Depth and recency of prior anaesthetic MAP response", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_multiepisode_memory.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "fig_multiepisode_memory.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mover, mover_audit = prepare_mover()
    inspire, inspire_audit = prepare_inspire()
    combined = pd.concat([mover, inspire], ignore_index=True, sort=False)
    states, counts, times, hidden = risk_tables(combined)
    contrasts = bootstrap_contrasts(combined)
    sensitivity = sensitivity_tables(combined)
    thresholds = threshold_sensitivity(combined)
    absolute_risks, absolute_contrasts = endpoint_extension(combined)
    continuous_correlations, continuous_quartiles = continuous_history(combined)
    stranger_summary, stranger_replicates = matched_stranger_null(combined)
    attenuation = attenuation_decomposition(combined)
    adjusted_rows, diagnostics = [], {}
    for dataset, x in combined.groupby("dataset", sort=False):
        model, diagnostic = robust_logit(x, count_model=False)
        trend, trend_diagnostic = robust_logit(x, count_model=True)
        model.insert(0, "dataset", dataset); model.insert(1, "model", "state_model")
        trend.insert(0, "dataset", dataset); trend.insert(1, "model", "history_count_trend")
        adjusted_rows.extend([model, trend])
        diagnostics[dataset] = {"state_model": diagnostic, "count_model": trend_diagnostic}
    adjusted = pd.concat(adjusted_rows, ignore_index=True)

    states.to_csv(OUT / "risk_by_history_state.csv", index=False)
    counts.to_csv(OUT / "risk_by_history_count.csv", index=False)
    times.to_csv(OUT / "risk_by_recent_interval.csv", index=False)
    hidden.to_csv(OUT / "hidden_susceptibility_baseline_ge85.csv", index=False)
    contrasts.to_csv(OUT / "bootstrap_contrasts.csv", index=False)
    adjusted.to_csv(OUT / "adjusted_associations.csv", index=False)
    sensitivity.to_csv(OUT / "sensitivity_risks.csv", index=False)
    thresholds.to_csv(OUT / "threshold_sensitivity.csv", index=False)
    absolute_risks.to_csv(OUT / "absolute_endpoint_risks.csv", index=False)
    absolute_contrasts.to_csv(OUT / "absolute_endpoint_contrasts.csv", index=False)
    continuous_correlations.to_csv(OUT / "continuous_history_correlations.csv", index=False)
    continuous_quartiles.to_csv(OUT / "continuous_history_quartiles.csv", index=False)
    stranger_summary.to_csv(OUT / "matched_stranger_summary.csv", index=False)
    stranger_replicates.to_csv(OUT / "matched_stranger_replicates.csv.gz", index=False,
                               compression="gzip")
    attenuation.to_csv(OUT / "current_baseline_attenuation.csv", index=False)
    make_figure(states, counts, times, contrasts)
    make_supplement_figure(thresholds, absolute_risks, continuous_quartiles, stranger_summary)

    primary = contrasts.loc[contrasts.contrast.eq("persistent_vs_neither_rd")]
    trend = adjusted.loc[(adjusted.model.eq("history_count_trend")) & adjusted.term.eq("history_count")]
    summary = {
        "status": "MULTIEPISODE_MEMORY_COMPLETED",
        "constructs": {
            "MOVER": "actual-IV-hypnotic timed first two post-dose NIBP values",
            "INSPIRE": "anaesthesia-start +15-minute ART-priority released MAP landmark",
        },
        "audits": {"MOVER": mover_audit, "INSPIRE": inspire_audit},
        "persistent_vs_neither_absolute_risk_difference": primary.to_dict("records"),
        "adjusted_odds_ratio_per_additional_positive_history": trend.to_dict("records"),
        "absolute_endpoint_both_vs_neither": absolute_contrasts.to_dict("records"),
        "matched_stranger_negative_control": stranger_summary.to_dict("records"),
        "current_baseline_statistical_attenuation": attenuation.to_dict("records"),
        "model_diagnostics": diagnostics,
        "decision": (
            "KEEP_AS_MAIN_DEPTH_ANALYSIS" if (
                primary.point.gt(0).all()
                and states.loc[states.state.eq("Both")].risk.min()
                > states.loc[states.state.eq("Neither")].risk.max()
            ) else "SUPPLEMENT_ONLY_OR_STOP"
        ),
        "claim_boundary": (
            "Consistent two-centre history-depth gradient. MOVER is primary; INSPIRE is directional replication "
            "under a different landmark. This does not establish a stable phenotype, causal persistence, "
            "clinical utility, or evidence that clinicians reviewed prior records."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
