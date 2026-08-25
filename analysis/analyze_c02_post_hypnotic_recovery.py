#!/usr/bin/env python3
"""Analyse fixed-opportunity post-hypnotic recovery trajectories in MOVER.

The primary recovery cohort uses six complete NIBP-only five-minute bins after
actual recorded IV-hypnotic administration.  Aggregate outputs are written to
the paper workspace; operation-level trajectories remain under restricted data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from analyze_c02_post_hypnotic_burden import (
    build_design,
    clean_post_anchor_rows,
    fit_cluster_logistic,
    load_pair_with_anchor,
    wilson,
)


from c02_runtime import private_workspace_root, protect_file, secure_directory


ROOT = private_workspace_root()
BASE = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/"
    "c02_clean_rebuild"
)
BURDEN = BASE / "post_hypnotic_burden"
OUT = BASE / "post_hypnotic_recovery"
RESTRICTED = ROOT / (
    "data/restricted/mover/extracted/"
    "mover_c02_posthypnotic_recovery.csv.gz"
)
EXISTING_METRICS = ROOT / (
    "data/restricted/mover/extracted/"
    "mover_c02_posthypnotic_burden.csv.gz"
)

RULES = ["NIBP_only", "NIBP_no_ART", "ART_priority"]
BANDS = ["<10%", "10–19%", "20–29%", "≥30%"]
STATES = ["Never low", "Early only", "Delayed onset", "Persistent/recurrent"]


def complete_sequences(keys: pd.DataFrame, rule: str) -> pd.DataFrame:
    per_modality = (
        keys.groupby(
            ["LOG_ID", "bin", "modality_hint"],
            as_index=False,
            observed=True,
        )
        .agg(map_value=("value", "median"), records=("value", "size"))
    )
    if rule == "ART_priority":
        source = per_modality.copy()
        source["priority"] = source.modality_hint.map({"ART": 0, "NIBP": 1})
        source = (
            source.sort_values(["LOG_ID", "bin", "priority"])
            .drop_duplicates(["LOG_ID", "bin"], keep="first")
        )
    else:
        source = per_modality.loc[per_modality.modality_hint.eq("NIBP")].copy()
        if rule == "NIBP_no_ART":
            art_ids = set(
                keys.loc[keys.modality_hint.eq("ART"), "LOG_ID"].astype(str)
            )
            source = source.loc[~source.LOG_ID.astype(str).isin(art_ids)].copy()

    wide = source.pivot(index="LOG_ID", columns="bin", values="map_value")
    wide = wide.reindex(columns=range(6))
    wide = wide.loc[wide.notna().all(axis=1)].copy()
    wide.columns = [f"map_bin_{i}" for i in range(6)]
    wide = wide.reset_index()
    wide.insert(1, "rule", rule)

    values = wide[[f"map_bin_{i}" for i in range(6)]].to_numpy(float)
    low = values < 65
    early = low[:, :2].any(axis=1)
    later = low[:, 2:].any(axis=1)
    state = np.select(
        [~early & ~later, early & ~later, ~early & later, early & later],
        STATES,
        default="Invalid",
    )
    wide["trajectory_state"] = state
    wide["early_low"] = early.astype(int)
    wide["later_low"] = later.astype(int)
    wide["persistent_recurrent"] = (early & later).astype(int)
    wide["early_nadir_map"] = values[:, :2].min(axis=1)
    wide["late_nadir_map"] = values[:, 2:].min(axis=1)
    wide["total_low_bins"] = low.sum(axis=1)
    wide["last_low_bin"] = np.where(
        low.any(axis=1),
        5 - np.argmax(low[:, ::-1], axis=1),
        -1,
    )
    return wide


def add_prior_band(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["prior_drop_band"] = pd.cut(
        pd.to_numeric(out.prior_relative_drop, errors="coerce"),
        bins=[-np.inf, 0.10, 0.20, 0.30, np.inf],
        labels=BANDS,
        right=False,
    )
    return out


def composition_table(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = add_prior_band(frame)
    rows: list[dict] = []
    for band, group in d.groupby("prior_drop_band", observed=False):
        n = len(group)
        for state in STATES:
            count = int(group.trajectory_state.eq(state).sum())
            rows.append(
                {
                    "measurement_rule": rule,
                    "prior_drop_band": str(band),
                    "trajectory_state": state,
                    "n_band": int(n),
                    "events": count,
                    "proportion": float(count / n) if n else math.nan,
                }
            )
    return pd.DataFrame(rows)


def recovery_band_table(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = add_prior_band(frame.loc[frame.early_low.eq(1)].copy())
    rows: list[dict] = []
    for band, group in d.groupby("prior_drop_band", observed=False):
        n = len(group)
        events = int(group.persistent_recurrent.sum())
        lo, hi = wilson(events, n)
        rows.append(
            {
                "measurement_rule": rule,
                "prior_drop_band": str(band),
                "early_low_n": int(n),
                "persistent_recurrent_events": events,
                "persistent_recurrent_risk": float(events / n) if n else math.nan,
                "ci_low": lo,
                "ci_high": hi,
                "mean_early_nadir_map": float(group.early_nadir_map.mean()),
            }
        )
    return pd.DataFrame(rows)


def delayed_band_table(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = add_prior_band(frame.loc[frame.early_low.eq(0)].copy())
    rows: list[dict] = []
    for band, group in d.groupby("prior_drop_band", observed=False):
        n = len(group)
        events = int(group.later_low.sum())
        lo, hi = wilson(events, n)
        rows.append(
            {
                "measurement_rule": rule,
                "prior_drop_band": str(band),
                "early_normal_n": int(n),
                "delayed_onset_events": events,
                "delayed_onset_risk": float(events / n) if n else math.nan,
                "ci_low": lo,
                "ci_high": hi,
                "mean_early_nadir_map": float(group.early_nadir_map.mean()),
            }
        )
    return pd.DataFrame(rows)


def fit_trajectory_association(
    frame: pd.DataFrame,
    rule: str,
    include_early_nadir: bool,
    estimand: str,
) -> dict:
    if estimand == "persistence_after_early_low":
        d = frame.loc[frame.early_low.eq(1)].copy()
        y = d.persistent_recurrent.to_numpy(int)
    elif estimand == "delayed_onset_after_early_normotension":
        d = frame.loc[frame.early_low.eq(0)].copy()
        y = d.later_low.to_numpy(int)
    else:
        raise ValueError(f"unknown estimand: {estimand}")
    x, names, design_audit = build_design(d)
    if include_early_nadir:
        early = pd.to_numeric(d.early_nadir_map, errors="coerce") / 10.0
        early = early.fillna(float(early.median())).to_numpy(float)
        x = np.column_stack([x, early])
        names.append("early_nadir_per_10")
    exposure = names.index("prior_drop_per_10pp")
    groups = d.patient_id.astype(str).to_numpy()
    beta, covariance, fit_audit = fit_cluster_logistic(x, y, groups)
    coefficient = float(beta[exposure])
    se = math.sqrt(max(float(covariance[exposure, exposure]), 0.0))

    x_shifted = x.copy()
    x_shifted[:, exposure] += 1.0
    p0 = expit(np.einsum("ij,j->i", x, beta))
    p1 = expit(np.einsum("ij,j->i", x_shifted, beta))
    risk_difference = float(np.mean(p1 - p0))
    gradient = np.mean(
        x_shifted * (p1 * (1 - p1))[:, None]
        - x * (p0 * (1 - p0))[:, None],
        axis=0,
    )
    rd_se = float(np.sqrt(max(gradient @ covariance @ gradient, 0.0)))
    return {
        "measurement_rule": rule,
        "estimand": estimand,
        "adjustment": (
            "case context plus current early nadir"
            if include_early_nadir
            else "case context only"
        ),
        "effect_unit": "per 10-percentage-point larger prior relative MAP decline",
        "n": int(len(d)),
        "patients": int(d.patient_id.nunique()),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "odds_ratio": float(math.exp(coefficient)),
        "odds_ratio_ci_low": float(math.exp(coefficient - 1.96 * se)),
        "odds_ratio_ci_high": float(math.exp(coefficient + 1.96 * se)),
        "odds_ratio_p_value": float(2 * norm.sf(abs(coefficient / se))),
        "standardized_risk_difference_pp": float(100 * risk_difference),
        "risk_difference_ci_low_pp": float(100 * (risk_difference - 1.96 * rd_se)),
        "risk_difference_ci_high_pp": float(100 * (risk_difference + 1.96 * rd_se)),
        "fit_converged": bool(fit_audit["converged"]),
        "max_abs_gradient": float(fit_audit["max_abs_gradient"]),
        "design_columns": int(x.shape[1]),
        "median_imputation_fields": int(len(design_audit["median_imputation"])),
    }


def validate_against_existing(sequences: pd.DataFrame) -> dict:
    existing = pd.read_csv(
        EXISTING_METRICS,
        usecols=["LOG_ID", "rule", "complete_six_bins", "low_bins"],
        dtype={"LOG_ID": str},
    )
    existing = existing.loc[
        existing.complete_six_bins & existing.rule.isin(RULES)
    ].copy()
    merged = sequences.merge(
        existing[["LOG_ID", "rule", "low_bins"]],
        on=["LOG_ID", "rule"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    difference = (
        pd.to_numeric(merged.total_low_bins, errors="coerce")
        - pd.to_numeric(merged.low_bins, errors="coerce")
    ).abs()
    audit = {
        "rows_compared": int(len(merged)),
        "id_rule_join_all_both": bool(merged._merge.eq("both").all()),
        "max_abs_low_bin_difference": float(difference.max()),
        "state_exhaustive": bool(sequences.trajectory_state.isin(STATES).all()),
        "six_bins_complete": bool(
            sequences[[f"map_bin_{i}" for i in range(6)]].notna().all().all()
        ),
        "nibp_no_art_subset": bool(
            set(
                sequences.loc[sequences.rule.eq("NIBP_no_ART"), "LOG_ID"]
            ).issubset(
                set(sequences.loc[sequences.rule.eq("NIBP_only"), "LOG_ID"])
            )
        ),
    }
    if not (
        audit["id_rule_join_all_both"]
        and audit["max_abs_low_bin_difference"] == 0
        and audit["state_exhaustive"]
        and audit["six_bins_complete"]
        and audit["nibp_no_art_subset"]
    ):
        raise RuntimeError(f"trajectory validation failed: {audit}")
    return audit


def make_figure(
    composition: pd.DataFrame,
    recovery: pd.DataFrame,
    delayed: pd.DataFrame,
    n_complete: int,
    n_early_low: int,
    n_early_normal: int,
) -> None:
    comp = composition.loc[composition.measurement_rule.eq("NIBP_only")].copy()
    rec = recovery.loc[recovery.measurement_rule.eq("NIBP_only")].copy()
    delay = delayed.loc[delayed.measurement_rule.eq("NIBP_only")].copy()
    comp["prior_drop_band"] = pd.Categorical(
        comp.prior_drop_band, categories=BANDS, ordered=True
    )
    comp["trajectory_state"] = pd.Categorical(
        comp.trajectory_state, categories=STATES, ordered=True
    )
    comp = comp.sort_values(["prior_drop_band", "trajectory_state"])
    rec["prior_drop_band"] = pd.Categorical(
        rec.prior_drop_band, categories=BANDS, ordered=True
    )
    rec = rec.sort_values("prior_drop_band")
    delay["prior_drop_band"] = pd.Categorical(
        delay.prior_drop_band, categories=BANDS, ordered=True
    )
    delay = delay.sort_values("prior_drop_band")

    colours = {
        "Never low": "#3A7CA5",
        "Early only": "#D79B26",
        "Delayed onset": "#D96C4E",
        "Persistent/recurrent": "#B45A7A",
    }
    hatches = {
        "Never low": "",
        "Early only": "//",
        "Delayed onset": "..",
        "Persistent/recurrent": "xx",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.45))
    x = np.arange(len(BANDS))
    bottom = np.zeros(len(BANDS))
    for state in STATES:
        values = (
            comp.loc[comp.trajectory_state.eq(state)]
            .set_index("prior_drop_band")
            .reindex(BANDS).proportion.to_numpy(float)
        )
        axes[0].bar(
            x,
            values,
            bottom=bottom,
            label=state,
            color=colours[state],
            edgecolor="#374151",
            linewidth=0.55,
            hatch=hatches[state],
        )
        for xi, value, base in zip(x, values, bottom):
            if value >= 0.055:
                axes[0].text(
                    xi,
                    base + value / 2,
                    f"{100*value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if state in {"Never low", "Persistent/recurrent"} else "#111827",
                    fontweight="bold",
                )
        bottom += values
    axes[0].set_xticks(x, BANDS)
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("Current operations")
    axes[0].set_xlabel("Prior relative MAP decline")
    axes[0].set_title("Thirty-minute trajectory composition")
    handles, labels = axes[0].get_legend_handles_labels()

    rate = rec.persistent_recurrent_risk.to_numpy(float)
    lo = rec.ci_low.to_numpy(float)
    hi = rec.ci_high.to_numpy(float)
    axes[1].errorbar(
        x,
        rate,
        yerr=[rate - lo, hi - rate],
        fmt="o",
        color="#B45A7A",
        ecolor="#596273",
        capsize=4,
        markersize=7,
    )
    axes[1].set_xticks(x, BANDS)
    axes[1].set_ylim(0, min(1.0, max(0.70, float(np.nanmax(hi) + 0.10))))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].set_ylabel("Persistent/recurrent later low MAP")
    axes[1].set_xlabel("Prior relative MAP decline")
    axes[1].set_title("Among operations low in the first 10 minutes")
    for xi, row in enumerate(rec.itertuples(index=False)):
        axes[1].text(
            xi,
            row.ci_high + 0.020,
            f"{100*row.persistent_recurrent_risk:.1f}%\nn={row.early_low_n:,}",
            ha="center",
            fontsize=8.5,
        )

    delayed_rate = delay.delayed_onset_risk.to_numpy(float)
    delayed_lo = delay.ci_low.to_numpy(float)
    delayed_hi = delay.ci_high.to_numpy(float)
    axes[2].errorbar(
        x,
        delayed_rate,
        yerr=[delayed_rate - delayed_lo, delayed_hi - delayed_rate],
        fmt="s",
        color="#D96C4E",
        ecolor="#596273",
        capsize=4,
        markersize=7,
    )
    axes[2].set_xticks(x, BANDS)
    axes[2].set_ylim(0, min(1.0, max(0.45, float(np.nanmax(delayed_hi) + 0.10))))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[2].set_ylabel("Delayed-onset later low MAP")
    axes[2].set_xlabel("Prior relative MAP decline")
    axes[2].set_title("Among operations normal in the first 10 minutes")
    for xi, row in enumerate(delay.itertuples(index=False)):
        axes[2].text(
            xi,
            row.ci_high + 0.014,
            f"{100*row.delayed_onset_risk:.1f}%\nn={row.early_normal_n:,}",
            ha="center",
            fontsize=8.5,
        )
    for panel, ax in zip(["A", "B", "C"], axes):
        ax.grid(axis="y", color="#E1E5EA", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            -0.14,
            1.07,
            panel,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )
    fig.suptitle(
        "Post-hypnotic MAP trajectory states and recovery in MOVER",
        y=0.988,
    )
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8.5,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
    )
    fig.text(
        0.5,
        0.025,
        (
            f"Complete NIBP-only six-bin cohort: {n_complete:,} pairs; "
            f"{n_early_low:,} early-low and {n_early_normal:,} early-normal; "
            "early=0–10 min, later=10–30 min; no interpolation"
        ),
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0, 0.075, 1, 0.845])
    fig.savefig(OUT / "fig_c02_post_hypnotic_recovery.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "fig_c02_post_hypnotic_recovery.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    secure_directory(OUT)
    secure_directory(RESTRICTED.parent)
    pair, pair_audit = load_pair_with_anchor()
    keys, map_audit = clean_post_anchor_rows(pair)

    sequence_parts = [complete_sequences(keys, rule) for rule in RULES]
    sequences = pd.concat(sequence_parts, ignore_index=True)
    validation = validate_against_existing(sequences)
    sequences.to_csv(RESTRICTED, index=False, compression="gzip")
    protect_file(RESTRICTED)

    analysis_frames: dict[str, pd.DataFrame] = {}
    composition_parts = []
    recovery_parts = []
    delayed_parts = []
    association_rows = []
    flow_rows = []
    for rule in RULES:
        seq = sequences.loc[sequences.rule.eq(rule)].drop(columns="rule")
        frame = pair.merge(seq, on="LOG_ID", how="inner", validate="one_to_one")
        analysis_frames[rule] = frame
        composition_parts.append(composition_table(frame, rule))
        recovery_parts.append(recovery_band_table(frame, rule))
        delayed_parts.append(delayed_band_table(frame, rule))
        for estimand in [
            "persistence_after_early_low",
            "delayed_onset_after_early_normotension",
        ]:
            for include_early_nadir in [False, True]:
                association_rows.append(
                    fit_trajectory_association(
                        frame, rule, include_early_nadir, estimand
                    )
                )
        flow_rows.append(
            {
                "measurement_rule": rule,
                "complete_six_bin_pairs": int(len(frame)),
                "patients": int(frame.patient_id.nunique()),
                "early_low_pairs": int(frame.early_low.sum()),
                "early_normal_pairs": int(frame.early_low.eq(0).sum()),
                "persistent_recurrent_pairs": int(frame.persistent_recurrent.sum()),
                "early_only_pairs": int(frame.trajectory_state.eq("Early only").sum()),
                "delayed_onset_pairs": int(frame.trajectory_state.eq("Delayed onset").sum()),
                "never_low_pairs": int(frame.trajectory_state.eq("Never low").sum()),
            }
        )

    composition = pd.concat(composition_parts, ignore_index=True)
    recovery = pd.concat(recovery_parts, ignore_index=True)
    delayed = pd.concat(delayed_parts, ignore_index=True)
    associations = pd.DataFrame(association_rows)
    flow = pd.DataFrame(flow_rows)
    composition.to_csv(OUT / "trajectory_state_by_prior_band.csv", index=False)
    recovery.to_csv(OUT / "early_low_recovery_by_prior_band.csv", index=False)
    delayed.to_csv(OUT / "delayed_onset_by_prior_band.csv", index=False)
    associations.to_csv(OUT / "recovery_associations.csv", index=False)
    associations.to_csv(OUT / "trajectory_associations.csv", index=False)
    flow.to_csv(OUT / "coverage_and_events.csv", index=False)

    primary = flow.loc[flow.measurement_rule.eq("NIBP_only")].iloc[0]
    adjusted = associations.loc[
        associations.measurement_rule.eq("NIBP_only")
        & associations.estimand.eq("persistence_after_early_low")
        & associations.adjustment.eq("case context plus current early nadir")
    ].iloc[0]
    recovery_sensitivity_adjusted = associations.loc[
        associations.estimand.eq("persistence_after_early_low")
        & associations.adjustment.eq("case context plus current early nadir")
    ]
    delayed_adjusted = associations.loc[
        associations.measurement_rule.eq("NIBP_only")
        & associations.estimand.eq("delayed_onset_after_early_normotension")
        & associations.adjustment.eq("case context plus current early nadir")
    ].iloc[0]
    delayed_sensitivity_adjusted = associations.loc[
        associations.estimand.eq("delayed_onset_after_early_normotension")
        & associations.adjustment.eq("case context plus current early nadir")
    ]
    max_smd = float(
        pd.read_csv(BURDEN / "included_vs_excluded_smd.csv")
        .standardized_mean_difference.abs().max()
    )
    gates = {
        "recovery_primary_early_low_ge300": bool(primary.early_low_pairs >= 300),
        "recovery_primary_persistent_recurrent_ge100": bool(
            primary.persistent_recurrent_pairs >= 100
        ),
        "max_abs_observation_smd_le0_50": bool(max_smd <= 0.50),
        "recovery_primary_adjusted_ci_above_one": bool(
            adjusted.odds_ratio_ci_low > 1
        ),
        "recovery_adjusted_direction_agrees_all_rules": bool(
            recovery_sensitivity_adjusted.odds_ratio.gt(1).all()
        ),
        "delayed_primary_early_normal_ge1500": bool(
            primary.early_normal_pairs >= 1500
        ),
        "delayed_primary_events_ge300": bool(primary.delayed_onset_pairs >= 300),
        "delayed_primary_adjusted_ci_above_one": bool(
            delayed_adjusted.odds_ratio_ci_low > 1
        ),
        "delayed_adjusted_direction_agrees_all_rules": bool(
            delayed_sensitivity_adjusted.odds_ratio.gt(1).all()
        ),
        "all_fits_converged": bool(associations.fit_converged.all()),
    }
    recovery_pass = bool(
        gates["recovery_primary_early_low_ge300"]
        and gates["recovery_primary_persistent_recurrent_ge100"]
        and gates["max_abs_observation_smd_le0_50"]
        and gates["recovery_primary_adjusted_ci_above_one"]
        and gates["recovery_adjusted_direction_agrees_all_rules"]
        and gates["all_fits_converged"]
    )
    delayed_pass = bool(
        gates["delayed_primary_early_normal_ge1500"]
        and gates["delayed_primary_events_ge300"]
        and gates["max_abs_observation_smd_le0_50"]
        and gates["delayed_primary_adjusted_ci_above_one"]
        and gates["delayed_adjusted_direction_agrees_all_rules"]
        and gates["all_fits_converged"]
    )
    if recovery_pass:
        status = "KEEP_AS_MOVER_RECOVERY_TRAJECTORY_DEEPENING"
    elif delayed_pass:
        status = "KEEP_AS_MOVER_DELAYED_ONSET_LOCALIZATION_RECOVERY_NULL"
    elif gates["max_abs_observation_smd_le0_50"]:
        status = "STOP_TRAJECTORY_INCREMENT_NOT_RETAINED"
    else:
        status = "STOP_TRAJECTORY_OBSERVATION_GATE"

    make_figure(
        composition,
        recovery,
        delayed,
        n_complete=int(primary.complete_six_bin_pairs),
        n_early_low=int(primary.early_low_pairs),
        n_early_normal=int(primary.early_normal_pairs),
    )
    summary = {
        "status": status,
        "question": (
            "Does prior relative MAP decline mark persistence after early low MAP, "
            "or delayed-onset low MAP after an initially normal first 10 minutes?"
        ),
        "pair_audit": pair_audit,
        "map_audit": map_audit,
        "validation": validation,
        "max_abs_observation_smd": max_smd,
        "coverage_and_events": flow.to_dict(orient="records"),
        "trajectory_associations": associations.to_dict(orient="records"),
        "gates": gates,
        "claim_boundary": (
            "Fixed-bin recorded trajectory prognosis in a complete NIBP-observed population; "
            "not treatment response, waveform-continuous recovery, organ benefit or a causal phenotype."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
