#!/usr/bin/env python3
"""Model C02 after excluding NIBP records in the hypnotic administration minute."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analyze_c02_fixed_four_endpoint_adjudication import cluster_bootstrap, grouped_oof
from analyze_c02_hypnotic_anchored_reproducibility import clustered_logit, performance
from analyze_c02_repeated_alert_comparator import MOVER_EXPANDED_BASE


from c02_runtime import private_workspace_root, secure_directory


ROOT = private_workspace_root()
PAIR = ROOT / "data/restricted/mover/extracted/mover_c02_hypnotic_strict_post_pair_sensitivity.csv.gz"
PATIENT_INFO = ROOT / "data/restricted/mover/extracted/EPIC_EMR/EMR/patient_information.csv"
OUT = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild/"
    "induction_drug_context/hypnotic_anchored/strict_postminute"
)


def main() -> None:
    secure_directory(OUT)
    d = pd.read_csv(PAIR, low_memory=False)
    info = pd.read_csv(
        PATIENT_INFO,
        usecols=["LOG_ID", "WEIGHT"], dtype={"LOG_ID": str}, low_memory=False,
    ).drop_duplicates()
    info["weight_kg_exact"] = pd.to_numeric(info["WEIGHT"], errors="coerce") / 35.274
    info = info.loc[info["weight_kg_exact"].between(20, 300, inclusive="both")]
    current_weight = info[["LOG_ID", "weight_kg_exact"]].rename(
        columns={"weight_kg_exact": "current_weight_kg_exact"}
    )
    prior_weight = info[["LOG_ID", "weight_kg_exact"]].rename(
        columns={"LOG_ID": "prior_LOG_ID", "weight_kg_exact": "prior_weight_kg_exact"}
    )
    d = d.merge(current_weight, on="LOG_ID", how="left", validate="many_to_one")
    d = d.merge(prior_weight, on="prior_LOG_ID", how="left", validate="many_to_one")
    for prefix, weight in [("cur", "current_weight_kg_exact"), ("pr", "prior_weight_kg_exact")]:
        d[f"{prefix}_propofol_mgkg"] = d[f"{prefix}_anchor_propofol_mg"] / d[weight]
        d[f"{prefix}_etomidate_mgkg"] = d[f"{prefix}_anchor_etomidate_mg"] / d[weight]
        d[f"{prefix}_ketamine_mgkg"] = d[f"{prefix}_anchor_ketamine_mg"] / d[weight]
    d["prior_alert"] = d["pr_anylow"].astype(int)
    d["prior_level"] = d["pr_map0"]
    d["prior_change"] = d["pr_change"]
    base = MOVER_EXPANDED_BASE
    drug = [
        "pr_anchor_agent", "cur_anchor_agent",
        "pr_anchor_propofol_mg", "pr_anchor_etomidate_mg", "pr_anchor_ketamine_mg",
        "cur_anchor_propofol_mg", "cur_anchor_etomidate_mg", "cur_anchor_ketamine_mg",
    ]
    drug_mgkg = [
        "pr_anchor_agent", "cur_anchor_agent",
        "pr_propofol_mgkg", "pr_etomidate_mgkg", "pr_ketamine_mgkg",
        "cur_propofol_mgkg", "cur_etomidate_mgkg", "cur_ketamine_mgkg",
    ]
    specs = {
        "M1_case_plus_binary_alert": base + ["prior_alert"],
        "M2_plus_continuous_response": base + ["prior_alert", "prior_level", "prior_change"],
        "M1D_case_binary_plus_drug_context": base + ["prior_alert"] + drug,
        "M2D_drug_context_plus_continuous": base + ["prior_alert"] + drug + ["prior_level", "prior_change"],
        "M1W_case_binary_plus_weight_normalized_drug": base + ["prior_alert"] + drug_mgkg,
        "M2W_weight_normalized_drug_plus_continuous": base + ["prior_alert"] + drug_mgkg + ["prior_level", "prior_change"],
    }
    y = d["cur_anylow"].to_numpy(int)
    groups = d["patient_id"].to_numpy()
    predictions = grouped_oof(d, y, groups, specs)
    metrics = pd.DataFrame([
        {"model": name, "n": len(d), "events": int(y.sum()), **performance(y, prediction)}
        for name, prediction in predictions.items()
    ])
    rows, boots = [], []
    for index, (label, left, right) in enumerate([
        ("continuous_beyond_case_and_binary", "M1_case_plus_binary_alert", "M2_plus_continuous_response"),
        ("continuous_beyond_case_binary_and_drug", "M1D_case_binary_plus_drug_context", "M2D_drug_context_plus_continuous"),
        ("continuous_beyond_weight_normalized_drug", "M1W_case_binary_plus_weight_normalized_drug", "M2W_weight_normalized_drug_plus_continuous"),
    ]):
        ci, boot = cluster_bootstrap(
            y, groups, predictions[left], predictions[right], label,
            20261041 + index, reps=1000,
        )
        ci.insert(0, "comparison", label); boot.insert(0, "comparison", label)
        rows.append(ci); boots.append(boot)
    increments = pd.concat(rows, ignore_index=True)
    # Fit the interpretable association on this exact strict-postminute cohort.
    # The earlier reporting map incorrectly paired association estimates from
    # the inclusive-anchor cohort with this cohort's sample/event counts.
    association_input = d.assign(
        prior_post_first_map=d["pr_map0"],
        prior_post_change=d["pr_change"],
        prior_post_any_low_alert=d["pr_anylow"].astype(int),
        prior_post_anchor_propofol_mg=d["pr_anchor_propofol_mg"],
        prior_post_anchor_etomidate_mg=d["pr_anchor_etomidate_mg"],
        prior_post_anchor_ketamine_mg=d["pr_anchor_ketamine_mg"],
        current_anchor_propofol_mg=d["cur_anchor_propofol_mg"],
        current_anchor_etomidate_mg=d["cur_anchor_etomidate_mg"],
        current_anchor_ketamine_mg=d["cur_anchor_ketamine_mg"],
        target_post_any_low=d["cur_anylow"].astype(int),
    )
    association = clustered_logit(association_input)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    increments.to_csv(OUT / "increment_ci.csv", index=False)
    association.to_csv(OUT / "drug_adjusted_association.csv", index=False)
    pd.concat(boots, ignore_index=True).to_csv(
        OUT / "bootstrap.csv.gz", index=False, compression="gzip"
    )
    summary = {
        "status": "STRICT_POSTMINUTE_SENSITIVITY_COMPLETED",
        "definition": "first two conflict-free NIBP MAP records 1-15 min after actual IV hypnotic administration",
        "pairs": len(d), "patients": int(d["patient_id"].nunique()),
        "events": int(y.sum()), "median_post_times_min": [float(d["cur_r0"].median()), float(d["cur_r1"].median())],
        "pairs_with_both_exact_weights": int(d[["current_weight_kg_exact", "prior_weight_kg_exact"]].notna().all(axis=1).sum()),
        "delta_auroc": increments.loc[increments.metric.eq("delta_auroc")].to_dict("records"),
        "association_source": "same strict-postminute cohort",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
