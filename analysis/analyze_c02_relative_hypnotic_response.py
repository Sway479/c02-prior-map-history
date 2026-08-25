#!/usr/bin/env python3
"""Test whether within-case relative MAP drop recurs across adjacent anaesthetics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_c02_fixed_four_endpoint_adjudication import cluster_bootstrap, grouped_oof
from analyze_c02_hypnotic_anchored_reproducibility import (
    MAP, MAR, OUT as PARENT_OUT, PAIR, HYPNOTICS, operation_hypnotics, performance,
)
from analyze_c02_repeated_alert_comparator import MOVER_EXPANDED_BASE


from c02_runtime import private_workspace_root, protect_file, secure_directory


ROOT = private_workspace_root()
OUT = PARENT_OUT / "relative_response"
RESTRICTED = ROOT / "data/restricted/mover/extracted/mover_c02_relative_hypnotic_pair.csv.gz"


def build_operations(maps: pd.DataFrame, anchors: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = maps.loc[
        maps["RECORD_TYPE"].eq("INTRA-OP") & maps["modality_hint"].eq("NIBP")
    ].copy()
    d["relative_min"] = pd.to_numeric(d["relative_min"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d["time"] = pd.to_datetime(d["RECORDED_TIME"], errors="coerce")
    d = d.loc[d["relative_min"].between(0, 30, inclusive="both")]
    d = d.loc[d["value"].between(20, 200, inclusive="both") & d["time"].notna()]
    key = d.groupby(["LOG_ID", "time"], as_index=False, observed=True).agg(
        relative_min=("relative_min", "min"), value=("value", "first"),
        distinct_values=("value", "nunique")
    )
    conflicts = int(key["distinct_values"].gt(1).sum())
    key = key.loc[key["distinct_values"].eq(1)].merge(anchors, on="LOG_ID", how="inner")
    key["anchor_delta_min"] = key["relative_min"] - key["anchor_rel"]
    rows = []
    for log_id, frame in key.groupby("LOG_ID", sort=False, observed=True):
        pre = frame.loc[frame["anchor_delta_min"].between(-10, 0, inclusive="left")]
        pre = pre.sort_values(["anchor_delta_min", "time"]).tail(1)
        post = frame.loc[frame["anchor_delta_min"].between(0, 15, inclusive="right")]
        post = post.sort_values(["anchor_delta_min", "time"]).head(2)
        if len(pre) != 1 or len(post) != 2:
            continue
        baseline = float(pre["value"].iloc[0])
        values = post["value"].to_numpy(float)
        nadir = float(values.min())
        drop = baseline - nadir
        rows.append(
            {
                "LOG_ID": str(log_id), "baseline_MAP": baseline,
                "post_MAP_0": values[0], "post_MAP_1": values[1],
                "post_nadir": nadir, "absolute_drop": drop,
                "relative_drop": drop / baseline,
                "relative_drop_20": int(drop / baseline >= .20),
                "post_any_low": int(np.any(values < 65)),
                "pre_gap_min": -float(pre["anchor_delta_min"].iloc[0]),
                "post_first_gap_min": float(post["anchor_delta_min"].iloc[0]),
                "anchor_agent": frame["anchor_agent"].iloc[0],
                "anchor_propofol_mg": float(frame["anchor_propofol_mg"].iloc[0]),
                "anchor_etomidate_mg": float(frame["anchor_etomidate_mg"].iloc[0]),
                "anchor_ketamine_mg": float(frame["anchor_ketamine_mg"].iloc[0]),
            }
        )
    return pd.DataFrame(rows), {
        "same_time_conflicts_excluded": conflicts,
        "operations_with_prebaseline_and_two_post_maps": len(rows),
    }


def pair_operations(op: pd.DataFrame) -> pd.DataFrame:
    pair = pd.read_csv(PAIR, dtype={"LOG_ID": str, "prior_LOG_ID": str}, low_memory=False)
    current = op.add_prefix("current_").rename(columns={"current_LOG_ID": "LOG_ID"})
    prior = op.add_prefix("prior_").rename(columns={"prior_LOG_ID": "prior_LOG_ID"})
    d = pair.merge(current, on="LOG_ID", how="inner", validate="many_to_one")
    d = d.merge(prior, on="prior_LOG_ID", how="inner", validate="many_to_one")
    d["target_relative_drop_20"] = d["current_relative_drop_20"].astype(int)
    d["target_post_any_low"] = d["current_post_any_low"].astype(int)
    d["prior_relative_drop_20_alert"] = d["prior_relative_drop_20"].astype(int)
    return d


def run_models(d: pd.DataFrame, outcome: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = MOVER_EXPANDED_BASE
    drug = [
        "prior_anchor_agent", "current_anchor_agent",
        "prior_anchor_propofol_mg", "prior_anchor_etomidate_mg", "prior_anchor_ketamine_mg",
        "current_anchor_propofol_mg", "current_anchor_etomidate_mg", "current_anchor_ketamine_mg",
    ]
    specs = {
        "M1_case_drug_context": base + drug,
        "M2_plus_prior_absolute_MAP": base + drug + ["prior_post_MAP_0", "prior_post_MAP_1"],
        "M3_plus_prior_relative_drop": base + drug + [
            "prior_post_MAP_0", "prior_post_MAP_1", "prior_baseline_MAP", "prior_relative_drop"
        ],
        "R1_case_drug_context": base + drug,
        "R2_plus_prior_relative_response": base + drug + [
            "prior_baseline_MAP", "prior_relative_drop_20_alert", "prior_relative_drop"
        ],
    }
    y = d[outcome].to_numpy(int)
    groups = d["patient_id"].to_numpy()
    pred = grouped_oof(d, y, groups, specs)
    metrics = pd.DataFrame([
        {"outcome": outcome, "model": name, "n": len(d), "events": int(y.sum()), **performance(y, p)}
        for name, p in pred.items()
    ])
    comparisons = [
        ("absolute_post_MAP_beyond_context", "M1_case_drug_context", "M2_plus_prior_absolute_MAP"),
        ("relative_drop_beyond_absolute_post_MAP", "M2_plus_prior_absolute_MAP", "M3_plus_prior_relative_drop"),
        ("relative_response_beyond_context", "R1_case_drug_context", "R2_plus_prior_relative_response"),
    ]
    cis, boots = [], []
    for i, (label, left, right) in enumerate(comparisons):
        ci, boot = cluster_bootstrap(y, groups, pred[left], pred[right], f"{outcome}_{label}", seed+i, reps=1000)
        ci.insert(0, "outcome", outcome); ci.insert(1, "comparison", label)
        boot.insert(0, "outcome", outcome); boot.insert(1, "comparison", label)
        cis.append(ci); boots.append(boot)
    return metrics, pd.concat(cis, ignore_index=True), pd.concat(boots, ignore_index=True)


def risk_grid(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["prior_drop_band"] = pd.cut(
        x["prior_relative_drop"], [-np.inf, .10, .20, .30, np.inf],
        labels=["<10%", "10-19%", "20-29%", "≥30%"], right=False,
    )
    rows = []
    for band, frame in x.groupby("prior_drop_band", observed=False):
        for outcome in ["target_post_any_low", "target_relative_drop_20"]:
            rows.append({
                "prior_drop_band": str(band), "outcome": outcome, "n": len(frame),
                "events": int(frame[outcome].sum()), "event_rate": float(frame[outcome].mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    secure_directory(OUT)
    secure_directory(RESTRICTED.parent)
    pair = pd.read_csv(PAIR, dtype={"LOG_ID": str, "prior_LOG_ID": str}, low_memory=False)
    ids = set(pair["LOG_ID"]) | set(pair["prior_LOG_ID"])
    mar = pd.read_csv(MAR, dtype={"LOG_ID": str}, low_memory=False)
    anchors, drug_audit = operation_hypnotics(mar.loc[mar["LOG_ID"].isin(ids)])
    maps = pd.read_csv(MAP, dtype={"LOG_ID": str}, low_memory=False)
    operations, map_audit = build_operations(maps.loc[maps["LOG_ID"].isin(ids)], anchors)
    d = pair_operations(operations)
    d.to_csv(RESTRICTED, index=False, compression="gzip")
    protect_file(RESTRICTED)
    metrics, cis, boots = [], [], []
    for outcome, seed in [("target_post_any_low", 20261021), ("target_relative_drop_20", 20261031)]:
        m, c, b = run_models(d, outcome, seed)
        metrics.append(m); cis.append(c); boots.append(b)
    metrics = pd.concat(metrics, ignore_index=True)
    cis = pd.concat(cis, ignore_index=True)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    cis.to_csv(OUT / "increment_ci.csv", index=False)
    pd.concat(boots, ignore_index=True).to_csv(OUT / "bootstrap.csv.gz", index=False, compression="gzip")
    risks = risk_grid(d)
    risks.to_csv(OUT / "absolute_risk_by_prior_relative_drop.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for outcome, label, color in [
        ("target_post_any_low", "Post-hypnotic MAP <65", "#457B9D"),
        ("target_relative_drop_20", "Relative MAP drop ≥20%", "#E76F51"),
    ]:
        plot = risks.loc[risks["outcome"].eq(outcome)]
        axes[0].plot(plot["prior_drop_band"], plot["event_rate"]*100, marker="o", label=label, color=color)
    axes[0].set_ylabel("Next-case event rate (%)")
    axes[0].set_xlabel("Prior-case relative MAP drop")
    axes[0].set_title("A. Cross-case dose-response gradient")
    axes[0].legend(frameon=False)
    plot = cis.loc[cis["metric"].eq("delta_auroc")]
    y = np.arange(len(plot))
    axes[1].errorbar(plot["point"], y,
                     xerr=[plot["point"]-plot["ci_low"], plot["ci_high"]-plot["point"]],
                     fmt="o", color="#2A9D8F", capsize=3)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_yticks(y, [f"{o.replace('target_','')}\n{c.replace('_',' ')}" for o,c in zip(plot.outcome, plot.comparison)], fontsize=8)
    axes[1].set_xlabel("Increment in AUROC")
    axes[1].set_title("B. Relative response incremental value")
    fig.tight_layout()
    fig.savefig(OUT / "fig_relative_hypnotic_response.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "status": "COMPLETED_RELATIVE_RESPONSE_TEST",
        "pairs": len(d), "patients": int(d["patient_id"].nunique()),
        "post_low_events": int(d["target_post_any_low"].sum()),
        "relative_drop20_events": int(d["target_relative_drop_20"].sum()),
        "spearman_prior_current_relative_drop": float(d[["prior_relative_drop", "current_relative_drop"]].corr(method="spearman").iloc[0,1]),
        "drug_audit": drug_audit, "map_audit": map_audit,
        "claim_rule": "Relative drop is promoted only if it adds beyond absolute post-hypnotic MAP with positive proper-score improvements.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
