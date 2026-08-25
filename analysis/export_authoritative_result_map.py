#!/usr/bin/env python3
"""Build the manuscript result map from aggregate analysis tables only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_IDS = (
    "ASSOC_BINARY_ALERT_INSPIRE",
    "ASSOC_BINARY_ALERT_MOVER",
    "ASSOC_PRIOR_LEVEL_INSPIRE",
    "ASSOC_PRIOR_LEVEL_MOVER",
    "ASSOC_EARLY_FALL_INSPIRE",
    "ASSOC_EARLY_FALL_MOVER",
    "MOVER_DRUG_TIMED_LEVEL",
    "MOVER_DRUG_TIMED_FALL",
    "PAIR_INSPIRE",
    "PAIR_MOVER",
    "SUBALERT_INSPIRE",
    "SUBALERT_MOVER",
    "MOVER_DRUG_TIMED",
    "MOVER_CURRENT_BASELINE",
    "MOVER_SUSTAINED_ABSOLUTE_BURDEN_OR",
    "MOVER_SUSTAINED_ABSOLUTE_BURDEN_RD",
    "MOVER_SUSTAINED_RELATIVE_BURDEN",
    "MOVER_RECOVERY_AFTER_EARLY_LOW",
    "MOVER_DELAYED_ONSET",
    "TRIPLET_MOVER",
    "TRIPLET_INSPIRE",
    "POST_CURRENT_MAP",
)


def read(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing aggregate source: {path}")
    return pd.read_csv(path)


def one(frame: pd.DataFrame, **selector: object) -> pd.Series:
    selected = frame
    for column, value in selector.items():
        if column not in selected:
            raise RuntimeError(f"missing selector column: {column}")
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise RuntimeError(f"expected one source row for {selector}; found {len(selected)}")
    return selected.iloc[0]


def inverse_odds_ratio(row: pd.Series) -> tuple[float, float, float]:
    return 1.0 / float(row.odds_ratio), 1.0 / float(row.ci_high), 1.0 / float(row.ci_low)


def selector_text(**selector: object) -> str:
    return "; ".join(f"{key}={value}" for key, value in selector.items())


def result_row(
    analysis_id: str,
    role: str,
    dataset: str,
    estimand: str,
    estimate: float,
    ci_low: float,
    ci_high: float,
    metric: str,
    source_table: str,
    source_selector: str,
    allowed_claim: str,
    *,
    n: int,
    events: int,
    patients: int | None = None,
    sample_label: str | None = None,
) -> dict[str, object]:
    return {
        "analysis_id": analysis_id,
        "manuscript_role": role,
        "dataset": dataset,
        "n": int(n),
        "events": int(events),
        "patients": pd.NA if patients is None else int(patients),
        "sample_label": sample_label or f"{int(n)} observations; {int(events)} events",
        "estimand": estimand,
        "estimate": float(estimate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "metric": metric,
        "source_table": source_table,
        "source_selector": source_selector,
        "allowed_claim": allowed_claim,
    }


def build_result_map(root: Path) -> pd.DataFrame:
    root = root.expanduser().resolve()
    rows: list[dict[str, object]] = []

    association_path = "deepening_v1/two_centre_harmonized_associations.csv"
    associations = read(root, association_path)
    association_specs = (
        (
            "prior_binary_alert",
            "ASSOC_BINARY_ALERT_{centre}",
            "binary prior alert adjusted with continuous prior level, change and case context",
            "binary alert is an association, not a causal effect; the adjusted result does not replicate in MOVER",
            False,
        ),
        (
            "prior_first_MAP_per_10mmHg",
            "ASSOC_PRIOR_LEVEL_{centre}",
            "prior first MAP per 10 mmHg lower adjusted with early change, binary alert and case context",
            "lower prior MAP is associated with higher next-anaesthetic early low-MAP odds; not a causal or treatment effect",
            True,
        ),
        (
            "prior_MAP_change_per_10mmHg",
            "ASSOC_EARLY_FALL_{centre}",
            "prior early MAP change per 10 mmHg greater fall adjusted with prior level, binary alert and case context",
            "greater prior early MAP fall is associated with higher next-anaesthetic early low-MAP odds; not a causal or treatment effect",
            True,
        ),
    )
    for term, id_template, estimand, claim, invert in association_specs:
        for centre in ("INSPIRE", "MOVER"):
            selector = {
                "centre": centre,
                "specification": "absolute_change_model",
                "term": term,
            }
            source = one(associations, **selector)
            if invert:
                estimate, low, high = inverse_odds_ratio(source)
            else:
                estimate, low, high = source.odds_ratio, source.ci_low, source.ci_high
            rows.append(
                result_row(
                    id_template.format(centre=centre),
                    "PRIMARY CLINICAL ASSOCIATION",
                    centre,
                    estimand,
                    estimate,
                    low,
                    high,
                    "adjusted odds ratio",
                    association_path,
                    selector_text(**selector),
                    claim,
                    n=source.n,
                    events=source.events,
                    patients=source.patients,
                    sample_label=f"{int(source.n)} pairs; {int(source.events)} events",
                )
            )

    strict_association_path = "hypnotic_anchored/strict_postminute/drug_adjusted_association.csv"
    strict_associations = read(root, strict_association_path)
    for term, analysis_id, estimand in (
        (
            "prior_post_MAP_per_10",
            "MOVER_DRUG_TIMED_LEVEL",
            "prior post-hypnotic first MAP per 10 mmHg lower adjusted with early change, binary alert, case and recorded drug context",
        ),
        (
            "prior_post_change_per_10",
            "MOVER_DRUG_TIMED_FALL",
            "prior post-hypnotic early MAP change per 10 mmHg greater fall adjusted with prior level, binary alert, case and recorded drug context",
        ),
    ):
        source = one(strict_associations, term=term)
        estimate, low, high = inverse_odds_ratio(source)
        rows.append(
            result_row(
                analysis_id,
                "KEY CONSTRUCT ASSOCIATION",
                "MOVER",
                estimand,
                estimate,
                low,
                high,
                "adjusted odds ratio",
                strict_association_path,
                selector_text(term=term),
                "association persists after actual hypnotic-time anchoring and recorded drug adjustment; not a treatment effect",
                n=source.n,
                events=source.events,
                patients=source.clusters,
                sample_label=f"{int(source.n)} pairs; {int(source.events)} events",
            )
        )

    inspire_metrics_path = "cross_database_minimal_bridge/metrics.csv"
    inspire_metrics = read(root, inspire_metrics_path)
    inspire = one(inspire_metrics, model="M2_continuous_prior_response")
    inspire_ci_path = "deepening_v1/signal_decomposition_bootstrap_ci.csv"
    inspire_ci = one(
        read(root, inspire_ci_path),
        centre="INSPIRE",
        model="M2_level_plus_change",
    )
    rows.append(
        result_row(
            "PAIR_INSPIRE",
            "PRIMARY",
            "INSPIRE",
            "continuous prior MAP response beyond binary prior alert and case context",
            inspire.delta_auroc_vs_M1,
            inspire_ci.delta_auroc_vs_M1_ci_low,
            inspire_ci.delta_auroc_vs_M1_ci_high,
            "delta AUROC",
            inspire_metrics_path,
            selector_text(model="M2_continuous_prior_response"),
            "same-specification feature-level replication; not validation of one transported probability model",
            n=inspire.n,
            events=inspire.events,
            sample_label=f"{int(inspire.n)} pairs; {int(inspire.events)} events",
        )
    )

    mover_increment_path = "mover_external_validation/m2_vs_m1_increment_cluster_bootstrap.csv"
    mover = one(
        read(root, mover_increment_path),
        analysis="MOVER_grouped_OOF_common_M2_vs_M1",
        metric="delta_auroc",
    )
    mover_metric = one(
        read(root, "mover_external_validation/mover_grouped_oof_metrics.csv"),
        analysis="MOVER_grouped_OOF_common",
        model="M2_continuous_prior_response",
    )
    rows.append(
        result_row(
            "PAIR_MOVER",
            "PRIMARY",
            "MOVER",
            "continuous prior MAP response beyond binary prior alert and case context",
            mover.point,
            mover.ci_low,
            mover.ci_high,
            "delta AUROC",
            mover_increment_path,
            selector_text(analysis=mover.analysis, metric=mover.metric),
            "same-specification feature-level replication; not validation of one transported probability model",
            n=mover_metric.n,
            events=mover_metric.events,
            sample_label=f"{int(mover_metric.n)} pairs; {int(mover_metric.events)} events",
        )
    )

    subalert_path = "subalert_gradient/subalert_contrasts.csv"
    subalert = read(root, subalert_path)
    for centre in ("INSPIRE", "MOVER"):
        selector = {"centre": centre, "contrast": "first_MAP_65_84_vs_ge85"}
        source = one(subalert, **selector)
        events = int(source.exposed_events + source.reference_events)
        rows.append(
            result_row(
                f"SUBALERT_{centre}",
                "MECHANISTIC CLINICAL INTERPRETATION",
                centre,
                "prior first MAP 65-84 versus >=85 among binary-alert-negative prior cases",
                source.risk_difference,
                source.rd_ci_low,
                source.rd_ci_high,
                "risk difference",
                subalert_path,
                selector_text(**selector),
                "sub-alert risk gradient; 85 mmHg is not a treatment or safety threshold",
                n=source.alert_negative_pairs,
                events=events,
                sample_label=f"{int(source.alert_negative_pairs)} binary-alert-negative pairs; {events} events",
            )
        )

    strict_increment_path = "hypnotic_anchored/strict_postminute/increment_ci.csv"
    strict_increment = one(
        read(root, strict_increment_path),
        comparison="continuous_beyond_case_binary_and_drug",
        metric="delta_auroc",
    )
    strict_count = one(strict_associations, term="prior_post_MAP_per_10")
    rows.append(
        result_row(
            "MOVER_DRUG_TIMED",
            "KEY CONSTRUCT VALIDATION",
            "MOVER",
            "continuous prior response after actual IV hypnotic, beyond binary alert and hypnotic context",
            strict_increment.point,
            strict_increment.ci_low,
            strict_increment.ci_high,
            "delta AUROC",
            strict_increment_path,
            selector_text(
                comparison="continuous_beyond_case_binary_and_drug",
                metric="delta_auroc",
            ),
            "signal persists after pharmacologically coherent re-anchoring",
            n=strict_count.n,
            events=strict_count.events,
            patients=strict_count.clusters,
            sample_label=f"{int(strict_count.n)} pairs; {int(strict_count.events)} events",
        )
    )

    current_path = "hypnotic_anchored/current_baseline_increment/increment_ci.csv"
    current = one(
        read(root, current_path),
        outcome="target_post_any_low",
        comparison="total_absolute_relative_history_beyond_current_baseline",
        metric="delta_auroc",
    )
    current_count = one(
        read(root, "hypnotic_anchored/current_baseline_increment/model_metrics.csv"),
        outcome="target_post_any_low",
        model="C3_plus_prior_absolute_and_relative",
    )
    rows.append(
        result_row(
            "MOVER_CURRENT_BASELINE",
            "CURRENT-BASELINE SENSITIVITY",
            "MOVER",
            "prior absolute and relative response beyond current pre-hypnotic MAP and actual current drug context",
            current.point,
            current.ci_low,
            current.ci_high,
            "delta AUROC",
            current_path,
            selector_text(
                outcome=current.outcome,
                comparison=current.comparison,
                metric=current.metric,
            ),
            "small residual historical information after current baseline; incomplete proper-score support for absolute low MAP",
            n=current_count.n,
            events=current_count.events,
            sample_label=f"{int(current_count.n)} pairs; {int(current_count.events)} events",
        )
    )

    burden_path = "post_hypnotic_burden/association_results.csv"
    burden = read(root, burden_path)
    burden_specs = (
        (
            "MOVER_SUSTAINED_ABSOLUTE_BURDEN_OR",
            {"measurement_rule": "ART_priority", "scale": "odds ratio"},
            "at least 10 fixed-bin minutes MAP<65 per 10-percentage-point larger prior relative decline",
            "adjusted odds ratio",
            1.0,
            "prior response marks a fixed-opportunity sustained low-MAP proxy in MOVER; not waveform duration or clinical benefit",
        ),
        (
            "MOVER_SUSTAINED_ABSOLUTE_BURDEN_RD",
            {"measurement_rule": "ART_priority", "scale": "risk difference, percentage points"},
            "standardized absolute risk change in at least 10 low minutes per 10-percentage-point prior decline",
            "risk difference",
            0.01,
            "approximately two additional sustained-low events per 100 operations on the observed MOVER case mix",
        ),
        (
            "MOVER_SUSTAINED_RELATIVE_BURDEN",
            {"measurement_rule": "NIBP_relative_80pct_baseline", "scale": "odds ratio"},
            "at least 10 NIBP-bin minutes below 80% of pre-hypnotic MAP per 10-percentage-point larger prior decline",
            "adjusted odds ratio",
            1.0,
            "directionally coherent individualized sustained-response sensitivity in MOVER only",
        ),
    )
    for analysis_id, selector, estimand, metric, scale, claim in burden_specs:
        source = one(burden, **selector)
        rows.append(
            result_row(
                analysis_id,
                "SECONDARY CLINICAL ENDPOINT",
                "MOVER",
                estimand,
                source.estimate * scale,
                source.ci_low * scale,
                source.ci_high * scale,
                metric,
                burden_path,
                selector_text(**selector),
                claim,
                n=source.n,
                events=source.events,
                sample_label=f"{int(source.n)} pairs; {int(source.events)} events",
            )
        )

    recovery_path = "post_hypnotic_recovery/trajectory_associations.csv"
    recovery = read(root, recovery_path)
    for analysis_id, estimand_key, estimand, event_label, claim in (
        (
            "MOVER_RECOVERY_AFTER_EARLY_LOW",
            "persistence_after_early_low",
            "persistent/recurrent low MAP at 10-30 minutes after low MAP at 0-10 minutes per 10-percentage-point larger prior decline, adjusted for current early nadir",
            "persistent/recurrent events",
            "recovery hypothesis unsupported; subgroup missed the prespecified 300-operation gate",
        ),
        (
            "MOVER_DELAYED_ONSET",
            "delayed_onset_after_early_normotension",
            "low MAP at 10-30 minutes after no low MAP at 0-10 minutes per 10-percentage-point larger prior decline, adjusted for current early nadir",
            "delayed-onset events",
            "exploratory localization toward delayed onset; not a treatment effect or independent confirmation",
        ),
    ):
        selector = {
            "measurement_rule": "NIBP_only",
            "estimand": estimand_key,
            "adjustment": "case context plus current early nadir",
        }
        source = one(recovery, **selector)
        rows.append(
            result_row(
                analysis_id,
                "EXPLORATORY TEMPORAL LOCALIZATION",
                "MOVER",
                estimand,
                source.odds_ratio,
                source.odds_ratio_ci_low,
                source.odds_ratio_ci_high,
                "adjusted odds ratio",
                recovery_path,
                selector_text(**selector),
                claim,
                n=source.n,
                events=source.events,
                patients=source.patients,
                sample_label=f"{int(source.n)} NIBP-only pairs; {int(source.events)} {event_label}",
            )
        )

    history_path = "clinical_workflow_deepening/history_state_effects.csv"
    history = read(root, history_path)
    for centre in ("MOVER", "INSPIRE"):
        source = one(history, centre=centre, estimand="both_vs_neither_rd")
        rows.append(
            result_row(
                f"TRIPLET_{centre}",
                "KEY DEPTH ANALYSIS",
                centre,
                "two prior positive responses versus neither for a distinct third anaesthetic",
                source.estimate,
                source.ci_low,
                source.ci_high,
                "risk difference",
                history_path,
                selector_text(centre=centre, estimand="both_vs_neither_rd"),
                "risk enrichment; additive synergy not established",
                n=source.triplets,
                events=source.events,
                sample_label=f"{int(source.triplets)} triplets; {int(source.events)} events",
            )
        )

    timing_path = "clinical_workflow_deepening/post_signal_management_oof_contrasts.csv"
    timing = one(
        read(root, timing_path),
        comparison="P2_plus_composite_history_vs_P0_current_MAP_and_context",
    )
    timing_count = one(
        read(root, "clinical_workflow_deepening/post_signal_management_oof_metrics.csv"),
        model="P2_plus_composite_history",
    )
    rows.append(
        result_row(
            "POST_CURRENT_MAP",
            "TIMING BOUNDARY",
            "MOVER",
            "prior composite history after current first two MAP observations",
            timing.delta_auroc,
            timing.delta_auroc_ci_low,
            timing.delta_auroc_ci_high,
            "delta AUROC",
            timing_path,
            selector_text(comparison=timing.comparison),
            "history is pre-induction information; it does not supersede real-time MAP",
            n=timing_count.n,
            events=timing_count.events,
            sample_label=f"{int(timing_count.n)} triplets; {int(timing_count.events)} post-MAP bolus events",
        )
    )

    result = pd.DataFrame(rows)
    verify_result_map(result)
    order = {analysis_id: index for index, analysis_id in enumerate(EXPECTED_IDS)}
    return result.sort_values("analysis_id", key=lambda values: values.map(order)).reset_index(drop=True)


def verify_result_map(result: pd.DataFrame) -> None:
    ids = tuple(result["analysis_id"])
    if len(ids) != len(set(ids)):
        raise RuntimeError("authoritative result map contains duplicate analysis IDs")
    if set(ids) != set(EXPECTED_IDS):
        missing = sorted(set(EXPECTED_IDS) - set(ids))
        extra = sorted(set(ids) - set(EXPECTED_IDS))
        raise RuntimeError(f"authoritative ID drift; missing={missing}; extra={extra}")
    numeric = result[["estimate", "ci_low", "ci_high", "n", "events"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError("authoritative result map contains non-finite required values")
    if (numeric["ci_low"] > numeric["estimate"]).any() or (
        numeric["estimate"] > numeric["ci_high"]
    ).any():
        raise RuntimeError("authoritative result map contains an estimate outside its CI")
    if (numeric["events"] < 0).any() or (numeric["events"] > numeric["n"]).any():
        raise RuntimeError("authoritative result map contains impossible event counts")
    strict = result.loc[result.analysis_id.str.startswith("MOVER_DRUG_TIMED")]
    if not strict.n.eq(4871).all() or not strict.events.eq(567).all():
        raise RuntimeError("drug-timed rows are not bound to the strict 4,871/567 cohort")
    if "patients" in strict and not strict.patients.eq(3645).all():
        raise RuntimeError("drug-timed rows are not bound to 3,645 strict-cohort patients")
    strict_level = result.loc[result.analysis_id.eq("MOVER_DRUG_TIMED_LEVEL")].iloc[0]
    strict_fall = result.loc[result.analysis_id.eq("MOVER_DRUG_TIMED_FALL")].iloc[0]
    if not (1.26 < float(strict_level.estimate) < 1.28):
        raise RuntimeError("strict postminute MAP-level OR is outside its audited range")
    if not (1.13 < float(strict_fall.estimate) < 1.16):
        raise RuntimeError("strict postminute MAP-change OR is outside its audited range")
    inspire = result.loc[result.analysis_id.eq("PAIR_INSPIRE")].iloc[0]
    if not (0.0346 < float(inspire.estimate) < 0.0347):
        raise RuntimeError("corrected INSPIRE primary increment is outside its audited range")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_result_map(args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"AUTHORITATIVE_RESULT_MAP_PASS rows={len(result)} output={args.output}")


if __name__ == "__main__":
    main()
