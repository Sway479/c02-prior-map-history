#!/usr/bin/env python3
"""Does prior hypnotic response add after current pre-hypnotic MAP is known?"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_c02_fixed_four_endpoint_adjudication import cluster_bootstrap, grouped_oof
from analyze_c02_hypnotic_anchored_reproducibility import performance
from analyze_c02_repeated_alert_comparator import MOVER_EXPANDED_BASE


from c02_runtime import private_workspace_root


ROOT = private_workspace_root()
PAIR = ROOT / "data/restricted/mover/extracted/mover_c02_relative_hypnotic_pair.csv.gz"
OUT = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild/"
    "induction_drug_context/hypnotic_anchored/current_baseline_increment"
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(PAIR, low_memory=False)
    current = MOVER_EXPANDED_BASE + [
        "current_baseline_MAP", "current_anchor_agent",
        "current_anchor_propofol_mg", "current_anchor_etomidate_mg", "current_anchor_ketamine_mg",
    ]
    prior_drug = [
        "prior_anchor_agent", "prior_anchor_propofol_mg",
        "prior_anchor_etomidate_mg", "prior_anchor_ketamine_mg",
    ]
    specs = {
        "C0_current_baseline_case_drug": current,
        "C1_plus_prior_binary_absolute_alert": current + ["prior_post_any_low"],
        "C2_plus_prior_absolute_response": current + prior_drug + [
            "prior_post_any_low", "prior_post_MAP_0", "prior_post_MAP_1"
        ],
        "C3_plus_prior_absolute_and_relative": current + prior_drug + [
            "prior_post_any_low", "prior_post_MAP_0", "prior_post_MAP_1",
            "prior_baseline_MAP", "prior_relative_drop"
        ],
        "R1_current_baseline_case_drug": current,
        "R2_plus_prior_relative_response": current + prior_drug + [
            "prior_baseline_MAP", "prior_relative_drop_20_alert", "prior_relative_drop"
        ],
    }
    comparisons = [
        ("history_binary_beyond_current_baseline", "C0_current_baseline_case_drug", "C1_plus_prior_binary_absolute_alert"),
        ("history_absolute_beyond_current_baseline_and_binary", "C1_plus_prior_binary_absolute_alert", "C2_plus_prior_absolute_response"),
        ("history_relative_beyond_current_baseline_and_absolute", "C2_plus_prior_absolute_response", "C3_plus_prior_absolute_and_relative"),
        ("total_absolute_history_beyond_current_baseline", "C0_current_baseline_case_drug", "C2_plus_prior_absolute_response"),
        ("total_absolute_relative_history_beyond_current_baseline", "C0_current_baseline_case_drug", "C3_plus_prior_absolute_and_relative"),
        ("history_relative_beyond_current_baseline", "R1_current_baseline_case_drug", "R2_plus_prior_relative_response"),
    ]
    metric_frames, ci_frames, boot_frames = [], [], []
    for outcome_index, outcome in enumerate(["target_post_any_low", "target_relative_drop_20"]):
        y = d[outcome].to_numpy(int)
        groups = d["patient_id"].to_numpy()
        predictions = grouped_oof(d, y, groups, specs)
        metric_frames.append(pd.DataFrame([
            {"outcome": outcome, "model": name, "n": len(d), "events": int(y.sum()), **performance(y, p)}
            for name, p in predictions.items()
        ]))
        for comparison_index, (label, left, right) in enumerate(comparisons):
            ci, boot = cluster_bootstrap(
                y, groups, predictions[left], predictions[right],
                f"{outcome}_{label}", 20261061 + 10*outcome_index + comparison_index, reps=1000,
            )
            ci.insert(0, "outcome", outcome); ci.insert(1, "comparison", label)
            boot.insert(0, "outcome", outcome); boot.insert(1, "comparison", label)
            ci_frames.append(ci); boot_frames.append(boot)
    metrics = pd.concat(metric_frames, ignore_index=True)
    increments = pd.concat(ci_frames, ignore_index=True)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    increments.to_csv(OUT / "increment_ci.csv", index=False)
    pd.concat(boot_frames, ignore_index=True).to_csv(
        OUT / "bootstrap.csv.gz", index=False, compression="gzip"
    )

    plot = increments.loc[increments.metric.eq("delta_auroc")].copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    y = np.arange(len(plot))
    ax.errorbar(plot.point, y, xerr=[plot.point-plot.ci_low, plot.ci_high-plot.point],
                fmt="o", color="#6A4C93", capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, [f"{o.replace('target_','')}\n{c.replace('_',' ')}" for o,c in zip(plot.outcome, plot.comparison)], fontsize=8)
    ax.set_xlabel("Increment in AUROC")
    ax.set_title("Prior-anaesthetic information after current pre-hypnotic MAP")
    fig.tight_layout()
    fig.savefig(OUT / "fig_current_baseline_increment.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "status": "CURRENT_BASELINE_INCREMENT_COMPLETED",
        "pairs": len(d), "patients": int(d.patient_id.nunique()),
        "absolute_low_events": int(d.target_post_any_low.sum()),
        "relative_drop_events": int(d.target_relative_drop_20.sum()),
        "delta_auroc": plot[["outcome", "comparison", "point", "ci_low", "ci_high"]].to_dict("records"),
        "claim_boundary": (
            "Current pre-hypnotic MAP and actual current drug context are included in both models. "
            "The contrast tests additional historical information, not pre-induction deployability of current drug dose."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
