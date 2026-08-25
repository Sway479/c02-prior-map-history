#!/usr/bin/env python3
"""Prepare the frozen C02 v2 development cohort from raw INSPIRE files only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from c02_runtime import protect_file, require_private_path, secure_directory


MAP_ITEMS = ("art_mbp", "nibp_mbp")
WARD_ITEMS = ("nibp_sbp", "nibp_dbp", "hr")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def qdict(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {"n": 0}
    q = x.quantile([0, .1, .25, .5, .75, .9, 1])
    return {
        "n": int(len(x)), "min": float(q.loc[0]), "p10": float(q.loc[.1]),
        "p25": float(q.loc[.25]), "median": float(q.loc[.5]),
        "p75": float(q.loc[.75]), "p90": float(q.loc[.9]), "max": float(q.loc[1]),
    }


def immediate_adult_general_pairs(operations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pair before filtering so an unusable intermediate operation is never skipped."""
    x = operations.sort_values(["subject_id", "anstart_time", "op_id"]).copy()
    x["operation_number_all_records"] = x.groupby("subject_id", observed=True).cumcount() + 1
    x["procedure3"] = x["icd10_pcs"].astype("string").str[:3]
    x["bmi"] = x["weight"] / (x["height"] / 100.0) ** 2
    x["bmi"] = x["bmi"].where(x["bmi"].between(10, 80))
    x["an_duration_min"] = x["anend_time"] - x["anstart_time"]
    prior_columns = [
        "op_id", "hadm_id", "age", "asa", "emop", "department", "antype",
        "procedure3", "anstart_time", "anend_time", "an_duration_min",
    ]
    prior = x.groupby("subject_id", observed=True)[prior_columns].shift(1).add_prefix("prior_")
    pairs = pd.concat([x, prior], axis=1)
    adjacent = pairs.loc[pairs["operation_number_all_records"].gt(1)].copy()
    adjacent["interval_min"] = adjacent["anstart_time"] - adjacent["prior_anend_time"]
    adjacent["interval_days"] = adjacent["interval_min"] / 1440.0
    adjacent["log_interval_days"] = np.log1p(adjacent["interval_days"].clip(lower=0))
    criteria = {
        "strict_time": adjacent["anstart_time"].notna()
        & adjacent["prior_anend_time"].notna()
        & adjacent["interval_min"].gt(0),
        "adult_both": adjacent["age"].ge(18) & adjacent["prior_age"].ge(18),
        "general_both": adjacent["antype"].eq("General") & adjacent["prior_antype"].eq("General"),
    }
    eligible = adjacent.loc[criteria["strict_time"] & criteria["adult_both"] & criteria["general_both"]].copy()
    eligible["same_admission"] = eligible["hadm_id"].eq(eligible["prior_hadm_id"])
    audit = {
        "all_operations": int(len(operations)),
        "all_subjects": int(operations["subject_id"].nunique()),
        "all_immediate_adjacent_rows": int(len(adjacent)),
        "strict_time_rows": int(criteria["strict_time"].sum()),
        "adult_both_rows": int(criteria["adult_both"].sum()),
        "general_both_rows": int(criteria["general_both"].sum()),
        "strict_adult_general_immediate_pairs": int(len(eligible)),
        "strict_adult_general_subjects": int(eligible["subject_id"].nunique()),
        "intermediate_operations_skipped": 0,
    }
    return eligible, audit


def _choose_formula_map(
    start: float,
    sbp_times: np.ndarray,
    sbp_values: np.ndarray,
    dbp_times: np.ndarray,
    dbp_values: np.ndarray,
) -> tuple[float, float, float, float, float]:
    s_left = np.searchsorted(sbp_times, start - 1440, side="left")
    s_right = np.searchsorted(sbp_times, start, side="left")
    d_left = np.searchsorted(dbp_times, start - 1440, side="left")
    d_right = np.searchsorted(dbp_times, start, side="left")
    st = sbp_times[s_left:s_right]
    sv = sbp_values[s_left:s_right]
    dt = dbp_times[d_left:d_right]
    dv = dbp_values[d_left:d_right]
    if not len(st) or not len(dt):
        return (math.nan,) * 5
    candidates = []
    for s_time, s_value in zip(st, sv):
        lo = np.searchsorted(dt, s_time - 5, side="left")
        hi = np.searchsorted(dt, s_time + 5, side="right")
        for d_time, d_value in zip(dt[lo:hi], dv[lo:hi]):
            if not (40 <= s_value <= 300 and 20 <= d_value <= 200 and s_value > d_value):
                continue
            formula = (float(s_value) + 2 * float(d_value)) / 3.0
            if not 20 <= formula <= 250:
                continue
            later = max(float(s_time), float(d_time))
            candidates.append((
                -later,
                abs(float(s_time) - float(d_time)),
                float(s_time),
                float(d_time),
                formula,
                float(s_value),
                float(d_value),
            ))
    if not candidates:
        return (math.nan,) * 5
    chosen = min(candidates)
    s_time, d_time = chosen[2], chosen[3]
    freshness = float(start - max(s_time, d_time))
    return chosen[4], freshness, s_time, d_time, chosen[1]


def ward_preanaesthetic_features(
    ward_path: Path, operation_targets: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    subjects = set(operation_targets["subject_id"].astype(int))
    frames: list[pd.DataFrame] = []
    selected = 0
    for chunk in pd.read_csv(
        ward_path,
        usecols=["subject_id", "chart_time", "item_name", "value"],
        chunksize=1_000_000,
    ):
        x = chunk.loc[
            chunk["subject_id"].isin(subjects) & chunk["item_name"].isin(WARD_ITEMS)
        ].copy()
        selected += len(x)
        if not x.empty:
            frames.append(x)
    raw = pd.concat(frames, ignore_index=True)
    key = ["subject_id", "chart_time", "item_name"]
    grouped = raw.groupby(key, observed=True, as_index=False).agg(
        value=("value", "median"), distinct_values=("value", "nunique"), raw_records=("value", "size")
    )
    target_groups = {
        int(subject): group.sort_values(["anstart_time", "op_id"])
        for subject, group in operation_targets.groupby("subject_id", observed=True)
    }
    item_maps: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for item in WARD_ITEMS:
        item_maps[item] = {}
        frame = grouped.loc[grouped["item_name"].eq(item)].sort_values(["subject_id", "chart_time"])
        for subject, group in frame.groupby("subject_id", observed=True):
            item_maps[item][int(subject)] = (
                group["chart_time"].to_numpy(dtype=float), group["value"].to_numpy(dtype=float)
            )
    rows = []
    for subject, targets in target_groups.items():
        sbp_times, sbp_values = item_maps["nibp_sbp"].get(
            subject, (np.empty(0, dtype=float), np.empty(0, dtype=float))
        )
        dbp_times, dbp_values = item_maps["nibp_dbp"].get(
            subject, (np.empty(0, dtype=float), np.empty(0, dtype=float))
        )
        hr_times, hr_values = item_maps["hr"].get(
            subject, (np.empty(0, dtype=float), np.empty(0, dtype=float))
        )
        for target in targets.itertuples(index=False):
            formula, freshness, sbp_time, dbp_time, pair_gap = _choose_formula_map(
                float(target.anstart_time), sbp_times, sbp_values, dbp_times, dbp_values
            )
            left = np.searchsorted(hr_times, float(target.anstart_time) - 1440, side="left")
            right = np.searchsorted(hr_times, float(target.anstart_time), side="left")
            valid_hr_indices = [
                index for index in range(left, right) if 20 <= hr_values[index] <= 250
            ]
            if valid_hr_indices:
                hr_index = valid_hr_indices[-1]
                hr_value = float(hr_values[hr_index])
                hr_freshness = float(target.anstart_time - hr_times[hr_index])
            else:
                hr_value = math.nan
                hr_freshness = math.nan
            rows.append({
                "op_id": int(target.op_id), "subject_id": subject,
                "preop_formula_map": formula,
                "preop_formula_map_freshness_min": freshness,
                "preop_sbp_time": sbp_time,
                "preop_dbp_time": dbp_time,
                "preop_sbp_dbp_pair_gap_min": pair_gap,
                "preop_hr": hr_value,
                "preop_hr_freshness_min": hr_freshness,
            })
    output = pd.DataFrame(rows)
    audit = {
        "target_operations": int(len(operation_targets)),
        "target_subjects": int(operation_targets["subject_id"].nunique()),
        "selected_raw_rows": selected,
        "grouped_subject_time_item_keys": int(len(grouped)),
        "same_key_multiple_raw_records": int(grouped["raw_records"].gt(1).sum()),
        "same_key_conflicting_values_collapsed_by_median": int(grouped["distinct_values"].gt(1).sum()),
        "formula_map_24h_available": int(output["preop_formula_map"].notna().sum()),
        "formula_map_6h_available": int(
            output["preop_formula_map"].notna().sum()
            and output["preop_formula_map_freshness_min"].le(360).sum()
        ),
        "hr_24h_available": int(output["preop_hr"].notna().sum()),
    }
    return output, audit


def landmark_features(
    vitals_path: Path, operation_targets: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    timing = operation_targets.set_index("op_id")[["subject_id", "anstart_time"]]
    target_ids = set(operation_targets["op_id"].astype(int))
    frames: list[pd.DataFrame] = []
    selected = 0
    for chunk in pd.read_csv(
        vitals_path,
        usecols=["op_id", "subject_id", "chart_time", "item_name", "value"],
        chunksize=1_000_000,
    ):
        x = chunk.loc[
            chunk["op_id"].isin(target_ids) & chunk["item_name"].isin(MAP_ITEMS)
        ].copy()
        selected += len(x)
        if x.empty:
            continue
        x = x.merge(timing.reset_index(), on=["op_id", "subject_id"], how="inner", validate="many_to_one")
        x["relative_min"] = x["chart_time"] - x["anstart_time"]
        x = x.loc[x["relative_min"].between(10, 20, inclusive="both")]
        if not x.empty:
            frames.append(x)
    raw = pd.concat(frames, ignore_index=True)
    grouped = raw.groupby(
        ["op_id", "subject_id", "chart_time", "relative_min", "item_name"],
        observed=True, as_index=False,
    ).agg(
        value=("value", "median"), distinct_values=("value", "nunique"), raw_records=("value", "size")
    )
    grouped = grouped.loc[grouped["value"].between(20, 200)].copy()
    grouped["distance_to_15"] = (grouped["relative_min"] - 15).abs()
    selected_rows = (
        grouped.sort_values(["op_id", "item_name", "distance_to_15", "relative_min"])
        .groupby(["op_id", "item_name"], observed=True)
        .head(1)
    )
    value = selected_rows.pivot(index="op_id", columns="item_name", values="value")
    relative = selected_rows.pivot(index="op_id", columns="item_name", values="relative_min")
    value = value.rename(columns={item: f"{item}_landmark_value" for item in MAP_ITEMS})
    relative = relative.rename(columns={item: f"{item}_landmark_relative_min" for item in MAP_ITEMS})
    output = timing.reset_index().merge(value.reset_index(), on="op_id", how="left")
    output = output.merge(relative.reset_index(), on="op_id", how="left")
    for item in MAP_ITEMS:
        for suffix in ("landmark_value", "landmark_relative_min"):
            column = f"{item}_{suffix}"
            if column not in output:
                output[column] = np.nan
    art = output["art_mbp_landmark_value"]
    nibp = output["nibp_mbp_landmark_value"]
    output["best_art_priority_value"] = art.combine_first(nibp)
    output["best_art_priority_modality"] = np.select(
        [art.notna(), nibp.notna()], ["ART", "NIBP"], default=None
    )
    output["best_nibp_priority_value"] = nibp.combine_first(art)
    output["best_nibp_priority_modality"] = np.select(
        [nibp.notna(), art.notna()], ["NIBP", "ART"], default=None
    )
    output["both_modalities_landmark"] = art.notna() & nibp.notna()
    audit = {
        "target_operations": int(len(operation_targets)),
        "selected_raw_map_rows_before_landmark_filter": selected,
        "raw_rows_in_10_to_20_window": int(len(raw)),
        "valid_grouped_keys": int(len(grouped)),
        "same_key_multiple_raw_records": int(grouped["raw_records"].gt(1).sum()),
        "same_key_conflicting_values_collapsed_by_median": int(grouped["distinct_values"].gt(1).sum()),
        "operations_art_landmark": int(output["art_mbp_landmark_value"].notna().sum()),
        "operations_nibp_landmark": int(output["nibp_mbp_landmark_value"].notna().sum()),
        "operations_any_landmark": int(output["best_art_priority_value"].notna().sum()),
        "operations_both_modalities": int(output["both_modalities_landmark"].sum()),
    }
    return output, audit


def attach_operation_features(
    pairs: pd.DataFrame, ward: pd.DataFrame, landmark: pd.DataFrame
) -> pd.DataFrame:
    ward_current = ward.drop(columns=["subject_id"])
    landmark_current = landmark.drop(columns=["subject_id", "anstart_time"])
    result = pairs.merge(ward_current, on="op_id", how="left", validate="many_to_one")
    result = result.merge(landmark_current, on="op_id", how="left", validate="many_to_one")
    ward_prior = ward_current.add_prefix("prior_").rename(columns={"prior_op_id": "prior_op_id"})
    landmark_prior = landmark_current.add_prefix("prior_").rename(columns={"prior_op_id": "prior_op_id"})
    result = result.merge(ward_prior, on="prior_op_id", how="left", validate="many_to_one")
    result = result.merge(landmark_prior, on="prior_op_id", how="left", validate="many_to_one")
    return result


def derive_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["historical_eligible"] = (
        x["prior_preop_formula_map"].notna()
        & x["prior_best_art_priority_value"].notna()
    )
    x = x.loc[x["historical_eligible"]].copy()
    x["outcome_observed"] = x["best_art_priority_value"].notna().astype(int)
    x["outcome_low65"] = x["best_art_priority_value"].lt(65).where(x["outcome_observed"].eq(1))
    x["current_landmark_modality"] = x["best_art_priority_modality"]
    x["prior_landmark_modality"] = x["prior_best_art_priority_modality"]
    x["prior_delta15"] = x["prior_best_art_priority_value"] - x["prior_preop_formula_map"]
    x["nibp_outcome_observed"] = x["nibp_mbp_landmark_value"].notna().astype(int)
    x["nibp_outcome_low65"] = x["nibp_mbp_landmark_value"].lt(65).where(
        x["nibp_outcome_observed"].eq(1)
    )
    x["art_outcome_observed"] = x["art_mbp_landmark_value"].notna().astype(int)
    x["art_outcome_low65"] = x["art_mbp_landmark_value"].lt(65).where(
        x["art_outcome_observed"].eq(1)
    )
    x["prior_delta15_nibp"] = x["prior_nibp_mbp_landmark_value"] - x["prior_preop_formula_map"]
    x["prior_delta15_art"] = x["prior_art_mbp_landmark_value"] - x["prior_preop_formula_map"]
    x["reverse_outcome_observed"] = x["best_nibp_priority_value"].notna().astype(int)
    x["reverse_outcome_low65"] = x["best_nibp_priority_value"].lt(65).where(
        x["reverse_outcome_observed"].eq(1)
    )
    x["reverse_prior_landmark_modality"] = x["prior_best_nibp_priority_modality"]
    x["reverse_current_landmark_modality"] = x["best_nibp_priority_modality"]
    x["reverse_prior_delta15"] = x["prior_best_nibp_priority_value"] - x["prior_preop_formula_map"]
    x["preop_map_6h_both"] = (
        x["preop_formula_map"].notna()
        & x["preop_formula_map_freshness_min"].le(360)
        & x["prior_preop_formula_map_freshness_min"].le(360)
    )
    x["modality_transition"] = (
        x["prior_landmark_modality"].astype("string")
        + "->"
        + x["current_landmark_modality"].astype("string")
    ).where(x["outcome_observed"].eq(1))
    return x.sort_values(["subject_id", "anstart_time", "op_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True,
    )
    args = parser.parse_args()
    args.root = require_private_path(args.root)
    args.output = secure_directory(args.output)
    protocol = args.output / "protocol.md"
    amendment = args.output / "protocol_amendment_001.md"
    protocol_receipt = json.loads((args.output / "protocol_freeze_receipt.json").read_text(encoding="utf-8"))
    amendment_receipt = json.loads(
        (args.output / "protocol_amendment_001_freeze_receipt.json").read_text(encoding="utf-8")
    )
    if sha256(protocol) != protocol_receipt["protocol_sha256"]:
        raise AssertionError("parent protocol hash changed")
    if sha256(amendment) != amendment_receipt["amendment_sha256"]:
        raise AssertionError("protocol amendment hash changed")

    operations_path = args.root / "operations.csv.gz"
    ward_path = args.root / "ward_vitals.csv.gz"
    vitals_path = args.root / "vitals.csv.gz"
    operations = pd.read_csv(operations_path)
    pairs, pair_audit = immediate_adult_general_pairs(operations)
    operation_ids = set(pairs["op_id"].astype(int)) | set(pairs["prior_op_id"].astype(int))
    targets = operations.loc[operations["op_id"].isin(operation_ids), ["op_id", "subject_id", "anstart_time"]]
    ward, ward_audit = ward_preanaesthetic_features(ward_path, targets)
    landmark, landmark_audit = landmark_features(vitals_path, targets)
    all_pairs = derive_analysis_columns(attach_operation_features(pairs, ward, landmark))
    primary = (
        all_pairs.sort_values(["subject_id", "anstart_time", "op_id"])
        .groupby("subject_id", observed=True)
        .head(1)
        .reset_index(drop=True)
    )
    if not primary["subject_id"].is_unique:
        raise AssertionError("primary analysis must contain one row per subject")

    all_path = args.output / "all_consecutive_index_eligible_pairs.csv.gz"
    primary_path = args.output / "primary_first_pair_per_subject.csv.gz"
    ward_output = args.output / "operation_preanaesthetic_ward_features.csv.gz"
    landmark_output = args.output / "operation_landmark_features.csv.gz"
    all_pairs.to_csv(all_path, index=False, compression="gzip")
    primary.to_csv(primary_path, index=False, compression="gzip")
    ward.to_csv(ward_output, index=False, compression="gzip")
    landmark.to_csv(landmark_output, index=False, compression="gzip")
    for protected in (all_path, primary_path, ward_output, landmark_output):
        protect_file(protected)

    summary = {
        "created_utc": utc_now(),
        "protocol_sha256": sha256(protocol),
        "amendment_sha256": sha256(amendment),
        "pair_audit": pair_audit,
        "ward_audit": ward_audit,
        "landmark_audit": landmark_audit,
        "historically_eligible_all_consecutive": {
            "pairs": int(len(all_pairs)),
            "subjects": int(all_pairs["subject_id"].nunique()),
            "current_outcome_observed": int(all_pairs["outcome_observed"].sum()),
            "current_outcome_unobserved": int(all_pairs["outcome_observed"].eq(0).sum()),
        },
        "primary_first_pair": {
            "pairs": int(len(primary)),
            "subjects": int(primary["subject_id"].nunique()),
            "one_row_per_subject": bool(primary["subject_id"].is_unique),
            "current_outcome_observed": int(primary["outcome_observed"].sum()),
            "current_outcome_unobserved": int(primary["outcome_observed"].eq(0).sum()),
            "current_preop_formula_map_missing": int(primary["preop_formula_map"].isna().sum()),
            "current_preop_hr_missing": int(primary["preop_hr"].isna().sum()),
            "six_hour_both_preop_map_rows": int(primary["preop_map_6h_both"].sum()),
        },
        "rules": {
            "current_future_map_used_for_base_cohort": False,
            "current_hr_used_for_base_cohort": False,
            "prior_history_required": ["prior_preop_formula_map", "prior_best_art_priority_value"],
            "primary_selection": "earliest historically eligible immediate-adjacent current operation per subject",
        },
    }
    summary_path = args.output / "data_preparation_summary.json"
    write_json(summary_path, summary)
    artifacts = [all_path, primary_path, ward_output, landmark_output, summary_path, Path(__file__)]
    receipt = {
        "created_utc": utc_now(),
        "protocol_sha256": sha256(protocol),
        "amendment_sha256": sha256(amendment),
        "inputs": {
            "operations": {"path": str(operations_path), "sha256": sha256(operations_path)},
            "ward_vitals": {"path": str(ward_path), "sha256": sha256(ward_path)},
            "vitals": {"path": str(vitals_path), "sha256": sha256(vitals_path)},
        },
        "v1_artifact_read": False,
        "local_holdout_created": False,
        "artifacts": {
            path.name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
    }
    write_json(args.output / "data_preparation_receipt.json", receipt)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
