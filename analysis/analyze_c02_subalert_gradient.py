#!/usr/bin/env python3
"""Explain why continuous prior MAP adds beyond a binary MAP<65 alert.

The analysis is restricted to prior anaesthetics whose first two NIBP MAP
records were both >=65 mmHg. Within this alert-negative population it reports
the next-anaesthetic endpoint across fixed first-MAP bands. It is descriptive
mechanism/clinical-interpretation work, not a new prediction model or a proposed
treatment threshold.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from analyze_c02_deepening_v1 import canonicalize_inspire
from run_mover_c02_external_validation import canonicalize, make_pipeline


from c02_runtime import private_workspace_root


ROOT = private_workspace_root()
BASE = ROOT / (
    "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild"
)
INSPIRE = BASE / "first_two_sampling_bridge/first_two_pair_cohort.csv.gz"
MOVER = ROOT / "data/restricted/mover/extracted/mover_c02_cleaned_pair_cohort.csv.gz"
OUT = BASE / "final_submission_core/subalert_gradient"
SEED = 20260814
REPS = 10000
BANDS = ["65–74", "75–84", "85–94", "≥95"]
COLOUR = {"INSPIRE": "#3A7CA5", "MOVER": "#D95F59"}


def wilson(events: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = events / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def load_data() -> pd.DataFrame:
    i = pd.read_csv(INSPIRE, low_memory=False)
    i = i.loc[
        i.antype.astype(str).str.strip().eq("General")
        & i.prior_antype.astype(str).str.strip().eq("General")
    ].copy()
    i = pd.DataFrame(
        {
            "centre": "INSPIRE",
            "patient_id": i.subject_id.astype(str),
            "outcome": pd.to_numeric(i.target_any_low, errors="coerce"),
            "prior_map_0": pd.to_numeric(i.prior_first2_map_0, errors="coerce"),
            "prior_map_1": pd.to_numeric(i.prior_first2_map_1, errors="coerce"),
        }
    )
    m = pd.read_csv(MOVER, low_memory=False)
    m = pd.DataFrame(
        {
            "centre": "MOVER",
            "patient_id": m.patient_id.astype(str),
            "outcome": pd.to_numeric(m.target_any_low_first2, errors="coerce"),
            "prior_map_0": pd.to_numeric(m.prior_first_map, errors="coerce"),
            "prior_map_1": pd.to_numeric(m.prior_first_map, errors="coerce")
            + pd.to_numeric(m.prior_first2_change, errors="coerce"),
        }
    )
    d = pd.concat([i, m], ignore_index=True)
    d = d.dropna(subset=["outcome", "prior_map_0", "prior_map_1"]).copy()
    d["outcome"] = d.outcome.astype(int)
    d["binary_prior_alert"] = d[["prior_map_0", "prior_map_1"]].min(axis=1).lt(65)
    d["prior_change"] = d.prior_map_1 - d.prior_map_0
    return d


def risk_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    no_alert = d.loc[~d.binary_prior_alert].copy()
    no_alert["prior_first_map_band"] = pd.cut(
        no_alert.prior_map_0,
        [65, 75, 85, 95, np.inf],
        right=False,
        labels=BANDS,
    )
    for centre, x in no_alert.groupby("centre", sort=False):
        for band in BANDS:
            z = x.loc[x.prior_first_map_band.eq(band)]
            events = int(z.outcome.sum())
            low, high = wilson(events, len(z))
            rows.append(
                {
                    "centre": centre,
                    "prior_binary_alert": "negative: both prior MAP values >=65",
                    "prior_first_map_band": band,
                    "pairs": int(len(z)),
                    "patients": int(z.patient_id.nunique()),
                    "events": events,
                    "risk": float(events / len(z)),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_contrasts(d: pd.DataFrame) -> pd.DataFrame:
    """Estimate contrasts with patient-cluster, rather than row, bootstrap.

    A patient may contribute more than one adjacent anaesthetic pair. Sampling
    individual pairs would therefore understate within-patient dependence while
    contradicting the manuscript's stated uncertainty unit. Each bootstrap draw
    samples patients with replacement and retains all of their eligible pairs.
    """
    rows = []
    for centre_index, (centre, x) in enumerate(d.groupby("centre", sort=False)):
        x = x.loc[~x.binary_prior_alert].reset_index(drop=True)
        definitions = {
            "first_MAP_65_84_vs_ge85": x.prior_map_0.lt(85),
            "first_MAP_65_74_vs_ge75": x.prior_map_0.lt(75),
            "second_minus_first_le_minus10_vs_other": x.prior_change.le(-10),
        }
        for contrast_index, (label, exposed_series) in enumerate(definitions.items()):
            exposed = exposed_series.to_numpy(bool)
            y = x.outcome.to_numpy(int)
            point = float(y[exposed].mean() - y[~exposed].mean())
            risk_ratio = float(y[exposed].mean() / y[~exposed].mean())
            cluster_rows = []
            for patient_id, index in x.groupby("patient_id", sort=False).indices.items():
                index = np.asarray(index, dtype=int)
                ee = exposed[index]
                yy = y[index]
                cluster_rows.append(
                    (
                        int(ee.sum()),
                        int(yy[ee].sum()),
                        int((~ee).sum()),
                        int(yy[~ee].sum()),
                    )
                )
            cluster_counts = np.asarray(cluster_rows, dtype=float)
            cluster_n = len(cluster_counts)
            rng = np.random.default_rng(
                SEED + centre_index * 1000 + contrast_index * 100
            )
            rd_values, rr_values = [], []
            for _ in range(REPS):
                totals = cluster_counts[
                    rng.integers(0, cluster_n, cluster_n)
                ].sum(axis=0)
                exposed_n, exposed_events, reference_n, reference_events = totals
                if exposed_n == 0 or reference_n == 0:
                    continue
                exposed_risk = exposed_events / exposed_n
                reference_risk = reference_events / reference_n
                if reference_risk <= 0:
                    continue
                rd_values.append(float(exposed_risk - reference_risk))
                rr_values.append(float(exposed_risk / reference_risk))
            rd_values = np.asarray(rd_values, dtype=float)
            rr_values = np.asarray(rr_values, dtype=float)
            rd_null = rd_values - point
            log_rr_point = math.log(risk_ratio)
            log_rr_null = np.log(rr_values) - log_rr_point
            rd_p_value = float(
                (np.sum(np.abs(rd_null) >= abs(point)) + 1)
                / (len(rd_values) + 1)
            )
            rr_p_value = float(
                (np.sum(np.abs(log_rr_null) >= abs(log_rr_point)) + 1)
                / (len(rr_values) + 1)
            )
            rows.append(
                {
                    "centre": centre,
                    "contrast": label,
                    "alert_negative_pairs": int(len(x)),
                    "exposed_n": int(exposed.sum()),
                    "exposed_events": int(y[exposed].sum()),
                    "exposed_risk": float(y[exposed].mean()),
                    "reference_n": int((~exposed).sum()),
                    "reference_events": int(y[~exposed].sum()),
                    "reference_risk": float(y[~exposed].mean()),
                    "risk_difference": point,
                    "rd_ci_low": float(np.quantile(rd_values, 0.025)),
                    "rd_ci_high": float(np.quantile(rd_values, 0.975)),
                    "rd_p_value": rd_p_value,
                    "risk_ratio": risk_ratio,
                    "rr_ci_low": float(np.quantile(rr_values, 0.025)),
                    "rr_ci_high": float(np.quantile(rr_values, 0.975)),
                    "rr_p_value": rr_p_value,
                    "bootstrap_reps": len(rd_values),
                    "bootstrap_unit": "patient",
                }
            )
    return pd.DataFrame(rows)


def triage_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for centre, x in d.groupby("centre", sort=False):
        y = x.outcome.astype(int).to_numpy()
        rules = {
            "binary_MAP_lt65_alert": x.binary_prior_alert.to_numpy(bool),
            "expanded_alert_MAP_lt65_or_first_MAP_lt85": (
                x.binary_prior_alert | x.prior_map_0.lt(85)
            ).to_numpy(bool),
        }
        for name, flag in rules.items():
            tp = int(np.sum(flag & (y == 1)))
            fp = int(np.sum(flag & (y == 0)))
            tn = int(np.sum((~flag) & (y == 0)))
            fn = int(np.sum((~flag) & (y == 1)))
            rows.append(
                {
                    "centre": centre,
                    "rule": name,
                    "pairs": int(len(x)),
                    "events": int(y.sum()),
                    "flagged_n": int(flag.sum()),
                    "flagged_fraction": float(flag.mean()),
                    "flagged_risk": float(y[flag].mean()),
                    "sensitivity": tp / (tp + fn),
                    "specificity": tn / (tn + fp),
                    "positive_predictive_value": tp / (tp + fp),
                    "negative_predictive_value": tn / (tn + fn),
                }
            )
    return pd.DataFrame(rows)


def prediction_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
    }


def alert_negative_increment() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Common-feature OOF decomposition within binary-alert-negative pairs."""
    raw_i = canonicalize_inspire(pd.read_csv(INSPIRE, low_memory=False))
    raw_m = canonicalize(pd.read_csv(MOVER, low_memory=False))
    common = ["age_years", "bmi_kg_m2", "asa_numeric", "sex_common", "interval_log1p"]
    configurations = [
        ("INSPIRE", raw_i, "subject_id", "target"),
        ("MOVER", raw_m, "patient_id", "target_any_low_first2"),
    ]
    specifications = {
        "context": common,
        "plus_prior_level": common + ["prior_first_map"],
        "plus_prior_change": common + ["prior_first2_change"],
        "plus_level_and_change": common + ["prior_first_map", "prior_first2_change"],
    }
    metric_rows, contrast_rows = [], []
    for centre_index, (centre, frame, group_column, outcome_column) in enumerate(configurations):
        x = frame.loc[frame.prior_first2_any_low.eq(0)].copy().reset_index(drop=True)
        y = x[outcome_column].astype(int).to_numpy()
        groups = x[group_column].astype(str).to_numpy()
        predictions = {name: np.full(len(x), np.nan) for name in specifications}
        splits = list(GroupKFold(5).split(x, y, groups))
        for train, test in splits:
            for name, features in specifications.items():
                model = make_pipeline(features)
                model.fit(x.iloc[train][features], y[train])
                predictions[name][test] = model.predict_proba(x.iloc[test][features])[:, 1]
        for name, prediction in predictions.items():
            metric_rows.append(
                {
                    "centre": centre,
                    "model": name,
                    "pairs": int(len(x)),
                    "patients": int(pd.Series(groups).nunique()),
                    "events": int(y.sum()),
                    **prediction_metrics(y, prediction),
                }
            )
        comparisons = [
            ("prior_level_beyond_context", "context", "plus_prior_level"),
            ("prior_change_beyond_context", "context", "plus_prior_change"),
            ("level_and_change_beyond_context", "context", "plus_level_and_change"),
            ("change_beyond_level", "plus_prior_level", "plus_level_and_change"),
        ]
        unique_groups = pd.unique(groups)
        lookup = {group: np.flatnonzero(groups == group) for group in unique_groups}
        rng = np.random.default_rng(SEED + 20000 + centre_index * 1000)
        for label, left, right in comparisons:
            base = prediction_metrics(y, predictions[left])
            candidate = prediction_metrics(y, predictions[right])
            point = {
                "delta_auroc": candidate["auroc"] - base["auroc"],
                "delta_average_precision": candidate["average_precision"]
                - base["average_precision"],
                "brier_improvement": base["brier"] - candidate["brier"],
                "log_loss_improvement": base["log_loss"] - candidate["log_loss"],
            }
            values = {metric: [] for metric in point}
            for _ in range(1000):
                sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
                index = np.concatenate([lookup[group] for group in sampled])
                yy = y[index]
                if np.unique(yy).size < 2:
                    continue
                b = prediction_metrics(yy, predictions[left][index])
                c = prediction_metrics(yy, predictions[right][index])
                values["delta_auroc"].append(c["auroc"] - b["auroc"])
                values["delta_average_precision"].append(
                    c["average_precision"] - b["average_precision"]
                )
                values["brier_improvement"].append(b["brier"] - c["brier"])
                values["log_loss_improvement"].append(
                    b["log_loss"] - c["log_loss"]
                )
            row = {
                "centre": centre,
                "comparison": label,
                "left": left,
                "right": right,
                "pairs": int(len(x)),
                "events": int(y.sum()),
            }
            for metric, estimate in point.items():
                low, high = np.quantile(values[metric], [0.025, 0.975])
                row[metric] = estimate
                row[f"{metric}_ci_low"] = float(low)
                row[f"{metric}_ci_high"] = float(high)
            row["bootstrap_reps"] = min(len(values[key]) for key in values)
            contrast_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(contrast_rows)


def make_figure(risks: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    ax = axes[0]
    x = np.arange(len(BANDS))
    for centre in ["INSPIRE", "MOVER"]:
        z = risks.loc[risks.centre.eq(centre)].set_index("prior_first_map_band").loc[BANDS]
        ax.errorbar(
            x,
            100 * z.risk,
            yerr=[100 * (z.risk - z.ci_low), 100 * (z.ci_high - z.risk)],
            marker="o",
            lw=2,
            capsize=2,
            color=COLOUR[centre],
            label=centre,
        )
    ax.set_xticks(x, BANDS)
    ax.set_xlabel("Prior first MAP (mmHg), among binary-alert-negative cases")
    ax.set_ylabel("Next-anaesthetic early low-MAP risk (%)")
    ax.set_title("A. Risk persists below the binary alert threshold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    z = contrasts.loc[contrasts.contrast.eq("first_MAP_65_84_vs_ge85")].reset_index(drop=True)
    for position, row in enumerate(z.itertuples(index=False)):
        ax.errorbar(
            100 * row.risk_difference,
            position,
            xerr=[
                [100 * (row.risk_difference - row.rd_ci_low)],
                [100 * (row.rd_ci_high - row.risk_difference)],
            ],
            fmt="o",
            markersize=8,
            capsize=3,
            color=COLOUR[row.centre],
        )
    ax.axvline(0, color="#555555", lw=1)
    ax.set_yticks(np.arange(len(z)), z.centre)
    ax.invert_yaxis()
    ax.set_xlabel("Risk difference: first MAP 65–84 vs ≥85 (percentage points)")
    ax.set_title("B. A simple sub-alert clinical interpretation")
    ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "A binary MAP<65 alert discards a continuous risk gradient",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Descriptive interpretation only: 85 mmHg is not a treatment or safety threshold, and the expanded rule is not a validated decision tool.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    fig.savefig(OUT / "fig3_subalert_gradient.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT / "fig3_subalert_gradient.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_data()
    risks = risk_table(d)
    contrasts = bootstrap_contrasts(d)
    triage = triage_table(d)
    oof_metrics, oof_contrasts = alert_negative_increment()
    risks.to_csv(OUT / "subalert_risk_gradient.csv", index=False)
    contrasts.to_csv(OUT / "subalert_contrasts.csv", index=False)
    triage.to_csv(OUT / "binary_vs_expanded_alert_performance.csv", index=False)
    oof_metrics.to_csv(OUT / "alert_negative_oof_metrics.csv", index=False)
    oof_contrasts.to_csv(OUT / "alert_negative_oof_contrasts.csv", index=False)
    make_figure(risks, contrasts)
    main_contrast = contrasts.loc[
        contrasts.contrast.eq("first_MAP_65_84_vs_ge85")
    ]
    change_contrast = contrasts.loc[
        contrasts.contrast.eq("second_minus_first_le_minus10_vs_other")
    ]
    checks = {
        "two_centres": set(risks.centre) == {"INSPIRE", "MOVER"},
        "four_nonempty_bands_per_centre": bool(
            risks.groupby("centre").prior_first_map_band.nunique().eq(4).all()
        ),
        "fixed_65_84_contrast_positive_both": bool(
            main_contrast.risk_difference.gt(0).all()
        ),
        "fixed_65_84_intervals_exclude_zero_both": bool(
            main_contrast.rd_ci_low.gt(0).all()
        ),
        "change_alone_is_not_promoted": bool(
            change_contrast.risk_difference.lt(main_contrast.risk_difference.min()).all()
        ),
        "alert_negative_level_delta_positive_both": bool(
            oof_contrasts.loc[
                oof_contrasts.comparison.eq("prior_level_beyond_context"),
                "delta_auroc",
            ].gt(0).all()
        ),
        "alert_negative_level_logloss_improves_both": bool(
            oof_contrasts.loc[
                oof_contrasts.comparison.eq("prior_level_beyond_context"),
                "log_loss_improvement_ci_low",
            ].gt(0).all()
        ),
        "figure_exists": (OUT / "fig3_subalert_gradient.png").exists(),
    }
    summary = {
        "status": "KEEP_AS_MECHANISTIC_CLINICAL_INTERPRETATION_NOT_NEW_ALERT",
        "checks": checks,
        "main_contrast": main_contrast.to_dict("records"),
        "alert_negative_prior_level_increment": oof_contrasts.loc[
            oof_contrasts.comparison.eq("prior_level_beyond_context")
        ].to_dict("records"),
        "interpretation": (
            "Among prior cases with no MAP<65 in the first two NIBP records, a first MAP of "
            "65-84 retained a higher next-case risk than >=85 in both centres. The continuous "
            "level, not an isolated >=10-mmHg first-to-second fall, explains most of the simple "
            "sub-alert gradient. Do not promote 85 mmHg as a treatment threshold."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
