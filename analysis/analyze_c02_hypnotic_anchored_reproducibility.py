#!/usr/bin/env python3
"""Re-anchor C02 to actual IV hypnotic administration in MOVER.

This is a construct-correction analysis, not a new black-box model.  It asks
whether a patient's first two post-hypnotic NIBP MAP values in the immediately
prior general anaesthetic add information about the same post-hypnotic outcome
in the next general anaesthetic.  Only aggregate outputs leave the restricted
data directory.
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from analyze_c02_fixed_four_endpoint_adjudication import cluster_bootstrap, grouped_oof
from analyze_c02_repeated_alert_comparator import MOVER_EXPANDED_BASE
from c02_cluster_logit import fit_clustered_logit


from c02_runtime import private_workspace_root, protect_file, secure_directory


ROOT = private_workspace_root()
PAIR = ROOT / "data/restricted/mover/extracted/mover_c02_cleaned_pair_cohort.csv.gz"
MAP = ROOT / "data/restricted/mover/extracted/mover_cleaned_early_map.csv.gz"
MAR = ROOT / "data/restricted/mover/extracted/mover_early_anesthetic_mar.csv.gz"
RESTRICTED = ROOT / "data/restricted/mover/extracted/mover_c02_hypnotic_anchored_pair.csv.gz"
OUT = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild/"
    "induction_drug_context/hypnotic_anchored"
)

HYPNOTICS = {"propofol", "etomidate", "ketamine"}
PLAUSIBLE_MG = {
    "propofol": (10.0, 500.0),
    "etomidate": (2.0, 60.0),
    "ketamine": (1.0, 500.0),
}


def performance(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def operation_hypnotics(mar: pd.DataFrame, allowed_start: float = 0.0) -> tuple[pd.DataFrame, dict]:
    d = mar.loc[
        mar["drug_class"].isin(HYPNOTICS)
        & mar["RECORD_TYPE"].eq("INTRA-OP")
        & mar["MAR_ACTION_NM"].eq("Given")
        & mar["MED_ROUTE_NM"].eq("IntraVENOUS")
        & mar["DOSE_UNIT_NM"].eq("mg")
    ].copy()
    d["dose_mg"] = pd.to_numeric(d["ADMIN_SIG"], errors="coerce")
    d["relative_min"] = pd.to_numeric(d["relative_min"], errors="coerce")
    d = d.loc[d["relative_min"].between(allowed_start, 15, inclusive="both")]
    d = d.loc[d["dose_mg"].notna() & d["dose_mg"].gt(0)]
    before_plausibility = len(d)
    d = d.loc[
        [PLAUSIBLE_MG[k][0] <= x <= PLAUSIBLE_MG[k][1]
         for k, x in zip(d["drug_class"], d["dose_mg"])]
    ].copy()
    exact = ["LOG_ID", "drug_class", "MED_ACTION_TIME", "dose_mg"]
    exact_duplicate_rows = int(d.duplicated(exact, keep=False).sum())
    d = d.drop_duplicates(exact)
    d["action_time"] = pd.to_datetime(d["MED_ACTION_TIME"], errors="coerce")
    d = d.loc[d["action_time"].notna()]
    earliest = d.groupby("LOG_ID", observed=True)["relative_min"].min().rename("anchor_rel")
    at_anchor = d.merge(earliest, on="LOG_ID", how="inner")
    at_anchor = at_anchor.loc[np.isclose(at_anchor["relative_min"], at_anchor["anchor_rel"])]
    rows = []
    for log_id, frame in at_anchor.groupby("LOG_ID", sort=False, observed=True):
        agents = sorted(frame["drug_class"].unique())
        # Sum distinct administrations at the anchor; exact duplicate records were removed.
        dose = frame.groupby("drug_class", observed=True)["dose_mg"].sum().to_dict()
        rows.append(
            {
                "LOG_ID": str(log_id),
                "anchor_rel": float(frame["anchor_rel"].iloc[0]),
                "anchor_agent": agents[0] if len(agents) == 1 else "+".join(agents),
                "anchor_propofol_mg": float(dose.get("propofol", 0.0)),
                "anchor_etomidate_mg": float(dose.get("etomidate", 0.0)),
                "anchor_ketamine_mg": float(dose.get("ketamine", 0.0)),
                "anchor_agent_count": len(agents),
            }
        )
    return pd.DataFrame(rows), {
        "candidate_rows_before_plausibility": int(before_plausibility),
        "candidate_rows_after_plausibility": int(len(d)),
        "exact_duplicate_rows_before_collapse": exact_duplicate_rows,
        "operations_with_strict_hypnotic_anchor": int(len(rows)),
    }


def operation_post_anchor_map(
    maps: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    strictly_after_anchor: bool = False,
) -> tuple[pd.DataFrame, dict]:
    d = maps.loc[
        maps["RECORD_TYPE"].eq("INTRA-OP")
        & maps["modality_hint"].eq("NIBP")
    ].copy()
    d["relative_min"] = pd.to_numeric(d["relative_min"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d["time"] = pd.to_datetime(d["RECORDED_TIME"], errors="coerce")
    d = d.loc[d["relative_min"].between(0, 30, inclusive="both")]
    d = d.loc[d["value"].between(20, 200, inclusive="both") & d["time"].notna()]
    key = (
        d.groupby(["LOG_ID", "time"], as_index=False, observed=True)
        .agg(relative_min=("relative_min", "min"), value=("value", "first"),
             distinct_values=("value", "nunique"))
    )
    conflicts = int(key["distinct_values"].gt(1).sum())
    key = key.loc[key["distinct_values"].eq(1)].merge(anchors, on="LOG_ID", how="inner")
    key["after_anchor_min"] = key["relative_min"] - key["anchor_rel"]
    # A 15-minute window captures induction physiology without drifting into
    # maintenance.  The strict sensitivity removes observations recorded in
    # the administration minute by requiring a positive timestamp difference.
    lower = key["after_anchor_min"].gt(0) if strictly_after_anchor else key["after_anchor_min"].ge(0)
    key = key.loc[lower & key["after_anchor_min"].le(15)]
    rows = []
    for log_id, frame in key.groupby("LOG_ID", sort=False, observed=True):
        first = frame.sort_values(["after_anchor_min", "time"]).head(2)
        if len(first) < 2:
            continue
        values = first["value"].to_numpy(float)
        times = first["after_anchor_min"].to_numpy(float)
        anchor = first.iloc[0]
        rows.append(
            {
                "LOG_ID": str(log_id),
                "post_map_0": values[0], "post_map_1": values[1],
                "post_map_change": values[1] - values[0],
                "post_map_rel_0": times[0], "post_map_rel_1": times[1],
                "post_any_low": int(np.any(values < 65)),
                "post_both_low": int(np.all(values < 65)),
                "anchor_rel": float(anchor["anchor_rel"]),
                "anchor_agent": anchor["anchor_agent"],
                "anchor_propofol_mg": float(anchor["anchor_propofol_mg"]),
                "anchor_etomidate_mg": float(anchor["anchor_etomidate_mg"]),
                "anchor_ketamine_mg": float(anchor["anchor_ketamine_mg"]),
            }
        )
    return pd.DataFrame(rows), {
        "same_time_conflict_keys_excluded": conflicts,
        "strictly_after_anchor": strictly_after_anchor,
        "operations_with_anchor_and_two_post_anchor_nibp": int(len(rows)),
    }


def prepare_pair(
    anchor_map: pd.DataFrame,
    pair_source: Path | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach post-anchor features to an explicit or configured pair cohort."""
    if pair_source is None:
        pair = pd.read_csv(
            PAIR,
            dtype={"LOG_ID": str, "prior_LOG_ID": str},
            low_memory=False,
        )
    elif isinstance(pair_source, pd.DataFrame):
        pair = pair_source.copy()
    else:
        pair = pd.read_csv(
            pair_source,
            dtype={"LOG_ID": str, "prior_LOG_ID": str},
            low_memory=False,
        )
    current = anchor_map.add_prefix("current_").rename(columns={"current_LOG_ID": "LOG_ID"})
    prior = anchor_map.add_prefix("prior_post_").rename(columns={"prior_post_LOG_ID": "prior_LOG_ID"})
    d = pair.merge(current, on="LOG_ID", how="inner", validate="many_to_one")
    d = d.merge(prior, on="prior_LOG_ID", how="inner", validate="many_to_one")
    d["target_post_any_low"] = d["current_post_any_low"].astype(int)
    d["prior_post_any_low_alert"] = d["prior_post_post_any_low"].astype(int)
    d["prior_post_first_map"] = d["prior_post_post_map_0"]
    d["prior_post_change"] = d["prior_post_post_map_change"]
    d["current_anchor_dose_mg_per_kg"] = (
        d["current_anchor_propofol_mg"] + d["current_anchor_etomidate_mg"]
        + d["current_anchor_ketamine_mg"]
    ) / pd.to_numeric(d["weight_kg"], errors="coerce")
    # Prior weight is reconstructed from prior BMI and current height only when available.
    # It is not used: agent-specific absolute dose is safer than a synthetic prior weight.
    d["same_anchor_agent"] = (d["current_anchor_agent"] == d["prior_post_anchor_agent"]).astype(int)
    d["propofol_to_propofol"] = (
        d["current_anchor_agent"].eq("propofol")
        & d["prior_post_anchor_agent"].eq("propofol")
    ).astype(int)
    return d


def model_analysis(d: pd.DataFrame, label: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = MOVER_EXPANDED_BASE
    specs = {
        "M1_case_context_plus_binary_alert": base + ["prior_post_any_low_alert"],
        "M2_plus_continuous_post_hypnotic_response": base + [
            "prior_post_any_low_alert", "prior_post_first_map", "prior_post_change"
        ],
        # Explanatory sensitivity only: actual current drug choice is not pre-induction information.
        "M1D_plus_prior_and_current_drug_context": base + [
            "prior_post_any_low_alert", "prior_post_anchor_agent", "current_anchor_agent",
            "prior_post_anchor_propofol_mg", "prior_post_anchor_etomidate_mg",
            "prior_post_anchor_ketamine_mg", "current_anchor_propofol_mg",
            "current_anchor_etomidate_mg", "current_anchor_ketamine_mg",
        ],
        "M2D_drug_context_plus_continuous_response": base + [
            "prior_post_any_low_alert", "prior_post_anchor_agent", "current_anchor_agent",
            "prior_post_anchor_propofol_mg", "prior_post_anchor_etomidate_mg",
            "prior_post_anchor_ketamine_mg", "current_anchor_propofol_mg",
            "current_anchor_etomidate_mg", "current_anchor_ketamine_mg",
            "prior_post_first_map", "prior_post_change",
        ],
    }
    y = d["target_post_any_low"].to_numpy(int)
    groups = d["patient_id"].to_numpy()
    pred = grouped_oof(d, y, groups, specs)
    rows = []
    for name, p in pred.items():
        rows.append({"analysis": label, "model": name, "n": len(d), "events": int(y.sum()), **performance(y, p)})
    comparisons = [
        ("continuous_beyond_binary_and_case_context", "M1_case_context_plus_binary_alert", "M2_plus_continuous_post_hypnotic_response"),
        ("continuous_beyond_drug_context", "M1D_plus_prior_and_current_drug_context", "M2D_drug_context_plus_continuous_response"),
    ]
    summaries, boots = [], []
    for index, (comparison, left, right) in enumerate(comparisons):
        ci, boot = cluster_bootstrap(
            y, groups, pred[left], pred[right], f"{label}_{comparison}", seed + index, reps=1000
        )
        ci.insert(0, "cohort", label)
        ci.insert(1, "comparison", comparison)
        boot.insert(0, "cohort", label)
        boot.insert(1, "comparison", comparison)
        summaries.append(ci); boots.append(boot)
    return pd.DataFrame(rows), pd.concat(summaries, ignore_index=True), pd.concat(boots, ignore_index=True)


def clustered_logit(d: pd.DataFrame) -> pd.DataFrame:
    """GEE-like sandwich SE for interpretable association, not causal inference."""
    x = pd.DataFrame(
        {
            "prior_post_MAP_per_10": d["prior_post_first_map"] / 10,
            "prior_post_change_per_10": d["prior_post_change"] / 10,
            "prior_binary_alert": d["prior_post_any_low_alert"],
            "current_age_per_10": d["age_years"] / 10,
            "current_BMI_per_5": d["bmi_kg_m2"] / 5,
            "current_ASA": d["asa_numeric"],
            "interval_log1p": d["interval_log1p"],
            "prior_propofol_dose_per_100mg": d["prior_post_anchor_propofol_mg"] / 100,
            "prior_etomidate_any": d["prior_post_anchor_etomidate_mg"].gt(0).astype(int),
            "prior_ketamine_any": d["prior_post_anchor_ketamine_mg"].gt(0).astype(int),
            "current_propofol_dose_per_100mg": d["current_anchor_propofol_mg"] / 100,
            "current_etomidate_any": d["current_anchor_etomidate_mg"].gt(0).astype(int),
            "current_ketamine_any": d["current_anchor_ketamine_mg"].gt(0).astype(int),
        }
    )
    x = x.apply(pd.to_numeric, errors="coerce")
    y = d["target_post_any_low"].to_numpy(float)
    return fit_clustered_logit(
        x,
        y,
        d["patient_id"].to_numpy(),
        list(x.columns),
    )


def absolute_risk(d: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 65, 75, 85, np.inf]
    labels = ["<65", "65-74", "75-84", "≥85"]
    d = d.copy()
    d["prior_MAP_band"] = pd.cut(d["prior_post_first_map"], bins=bins, labels=labels, right=False)
    rows = []
    for band, frame in d.groupby("prior_MAP_band", observed=False):
        n = len(frame); events = int(frame["target_post_any_low"].sum())
        rate = events / n if n else math.nan
        se = math.sqrt(rate * (1 - rate) / n) if n else math.nan
        rows.append({"prior_MAP_band": str(band), "n": n, "events": events,
                     "event_rate": rate, "ci_low": max(0, rate - 1.96 * se),
                     "ci_high": min(1, rate + 1.96 * se)})
    return pd.DataFrame(rows)


def main() -> None:
    secure_directory(OUT)
    secure_directory(RESTRICTED.parent)
    pair = pd.read_csv(PAIR, dtype={"LOG_ID": str, "prior_LOG_ID": str}, low_memory=False)
    ids = set(pair["LOG_ID"]) | set(pair["prior_LOG_ID"])
    mar = pd.read_csv(MAR, dtype={"LOG_ID": str}, low_memory=False)
    mar = mar.loc[mar["LOG_ID"].isin(ids)]
    anchors, drug_audit = operation_hypnotics(mar)
    maps = pd.read_csv(MAP, dtype={"LOG_ID": str}, low_memory=False)
    maps = maps.loc[maps["LOG_ID"].isin(ids)]
    anchor_map, map_audit = operation_post_anchor_map(maps, anchors)
    d = prepare_pair(anchor_map)
    d.to_csv(RESTRICTED, index=False, compression="gzip")
    protect_file(RESTRICTED)

    analyses = [("all_strict_anchored_pairs", d, 20261001)]
    pp = d.loc[d["propofol_to_propofol"].eq(1)].copy()
    analyses.append(("propofol_to_propofol", pp, 20261011))
    model_frames, ci_frames, boot_frames = [], [], []
    for label, frame, seed in analyses:
        if len(frame) < 500 or frame["target_post_any_low"].sum() < 50:
            continue
        models, ci, boot = model_analysis(frame, label, seed)
        model_frames.append(models); ci_frames.append(ci); boot_frames.append(boot)
    models = pd.concat(model_frames, ignore_index=True)
    increments = pd.concat(ci_frames, ignore_index=True)
    pd.concat(boot_frames, ignore_index=True).to_csv(
        OUT / "increment_cluster_bootstrap.csv.gz", index=False, compression="gzip"
    )
    models.to_csv(OUT / "model_metrics.csv", index=False)
    increments.to_csv(OUT / "increment_ci.csv", index=False)
    association = clustered_logit(d)
    association.to_csv(OUT / "drug_adjusted_association.csv", index=False)
    risk = absolute_risk(d)
    risk.to_csv(OUT / "absolute_risk_by_prior_post_hypnotic_map.csv", index=False)

    coverage = pd.DataFrame(
        [
            {"stage": "original_adjacent_general_pairs", "n": len(pair)},
            {"stage": "current_and_prior_strict_hypnotic_anchor_plus_two_post_NIBP", "n": len(d)},
            {"stage": "propofol_to_propofol", "n": len(pp)},
        ]
    )
    coverage.to_csv(OUT / "cohort_flow.csv", index=False)
    agent = pd.crosstab(d["prior_post_anchor_agent"], d["current_anchor_agent"])
    agent.to_csv(OUT / "prior_current_anchor_agent_table.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
    x = np.arange(len(risk))
    y = risk["event_rate"].to_numpy(float) * 100
    axes[0].bar(x, y, color="#457B9D")
    axes[0].errorbar(x, y,
                     yerr=[y-risk["ci_low"].to_numpy()*100, risk["ci_high"].to_numpy()*100-y],
                     fmt="none", ecolor="black", capsize=3)
    axes[0].set_xticks(x, risk["prior_MAP_band"])
    axes[0].set_ylabel("Next-case post-hypnotic low MAP risk (%)")
    axes[0].set_title("A. Absolute risk gradient")

    plot = increments.loc[increments["metric"].eq("delta_auroc")].copy()
    axes[1].errorbar(plot["point"], np.arange(len(plot)),
                     xerr=[plot["point"]-plot["ci_low"], plot["ci_high"]-plot["point"]],
                     fmt="o", color="#E76F51", capsize=3)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_yticks(np.arange(len(plot)),
                      [f"{c}\n{a.replace('_',' ')}" for c,a in zip(plot.cohort, plot.comparison)],
                      fontsize=8)
    axes[1].set_xlabel("Increment in AUROC")
    axes[1].set_title("B. Increment beyond case/drug context")

    counts = d["prior_post_anchor_agent"].value_counts().head(5)
    axes[2].barh(counts.index[::-1], counts.values[::-1], color="#2A9D8F")
    axes[2].set_xlabel("Pairs")
    axes[2].set_title("C. Prior hypnotic anchor")
    fig.suptitle("C02 construct correction: actual hypnotic-anchored reproducibility", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hypnotic_anchored_reproducibility.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    key_ci = increments.loc[
        increments["metric"].eq("delta_auroc")
        & increments["cohort"].eq("all_strict_anchored_pairs")
    ].to_dict("records")
    summary = {
        "status": "COMPLETED_CONSTRUCT_CORRECTION",
        "original_pairs": int(len(pair)), "anchored_pairs": int(len(d)),
        "anchored_patients": int(d["patient_id"].nunique()),
        "events": int(d["target_post_any_low"].sum()),
        "propofol_to_propofol_pairs": int(len(pp)),
        "drug_audit": drug_audit, "map_audit": map_audit,
        "key_delta_auroc_rows": key_ci,
        "claim_boundary": (
            "Actual IV hypnotic anchored observational reproducibility. Drug-adjusted analyses are "
            "explanatory sensitivities; current actual drug is not a pre-induction deployable feature."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
