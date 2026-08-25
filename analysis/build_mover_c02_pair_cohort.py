#!/usr/bin/env python3
"""Build the strict MOVER C02 adjacent-operation cohort from extracted MAPs.

Patient-level rows remain under data/restricted. Only aggregate flow and data-
quality tables are written to the report directory. This script constructs the
cohort but does not fit or evaluate a prediction model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from c02_runtime import require_private_path, secure_directory


NIBP_MEAS_NAME = "UC ANE R BLOOD PRESSURE - MAP"
NIBP_DISPLAY_NAME = "NIBP - MAP"
ART_MEAS_NAME = "UC ANE R ARTERIAL LINE MAP - ART"
ART_DISPLAY_NAME = "MAP-ART A-line"
HEIGHT_RE = re.compile(r"^\s*(\d+)\s*'\s*(\d+(?:\.\d+)?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--map-subset", required=True, type=Path)
    parser.add_argument("--restricted-cohort", required=True, type=Path)
    parser.add_argument("--aggregate-out", required=True, type=Path)
    return parser.parse_args()


def parse_height_m(value: object) -> float:
    match = HEIGHT_RE.match("" if value is None else str(value))
    if not match:
        return np.nan
    inches = int(match.group(1)) * 12.0 + float(match.group(2))
    return inches * 0.0254


def clean_operations(info_path: Path, mapping_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(info_path, dtype=str, low_memory=False)
    exact_duplicates = int(raw.duplicated().sum())
    info = raw.drop_duplicates().copy()
    nonkey = [column for column in info.columns if column != "LOG_ID"]
    info_conflict = info.groupby("LOG_ID", dropna=False)[nonkey].nunique(dropna=False).gt(1).any(axis=1)
    conflict_info = set(info_conflict[info_conflict].index.dropna())

    mapping = pd.read_csv(mapping_path, dtype=str).drop_duplicates()
    mapping_conflict = mapping.groupby("LOG_ID")["MRN"].nunique(dropna=False).gt(1)
    conflict_mapping = set(mapping_conflict[mapping_conflict].index)
    excluded_log_ids = conflict_info | conflict_mapping
    info = info.loc[~info["LOG_ID"].isin(excluded_log_ids)].drop_duplicates("LOG_ID")
    mapping = mapping.loc[~mapping["LOG_ID"].isin(conflict_mapping)].drop_duplicates("LOG_ID")
    operations = info.merge(
        mapping[["LOG_ID", "MRN"]],
        on="LOG_ID",
        how="left",
        suffixes=("_info", "_map"),
        validate="one_to_one",
    )
    operations["patient_id"] = operations["MRN_map"].fillna(operations["MRN_info"])
    operations["anstart"] = pd.to_datetime(
        operations["AN_START_DATETIME"], format="%m/%d/%y %H:%M", errors="coerce"
    )
    operations["anstop"] = pd.to_datetime(
        operations["AN_STOP_DATETIME"], format="%m/%d/%y %H:%M", errors="coerce"
    )
    operations["positive_interval"] = operations["anstart"].lt(operations["anstop"])
    operations["is_general"] = (
        operations["PRIMARY_ANES_TYPE_NM"].fillna("").str.strip().eq("General")
    )

    operations["age_years"] = pd.to_numeric(operations["BIRTH_DATE"], errors="coerce")
    operations["height_m"] = operations["HEIGHT"].map(parse_height_m)
    operations["weight_kg"] = pd.to_numeric(operations["WEIGHT"], errors="coerce") * 0.028349523125
    operations["bmi_kg_m2"] = operations["weight_kg"] / operations["height_m"].pow(2)
    plausible_body = (
        operations["height_m"].between(1.2, 2.3)
        & operations["weight_kg"].between(20, 350)
        & operations["bmi_kg_m2"].between(10, 100)
    )
    operations.loc[~plausible_body, "bmi_kg_m2"] = np.nan
    operations["asa_numeric"] = pd.to_numeric(operations["ASA_RATING_C"], errors="coerce")
    operations.loc[~operations["asa_numeric"].between(1, 6), "asa_numeric"] = np.nan
    operations["sex_common"] = operations["SEX"].astype("string").str.strip().str.upper().replace(
        {"MALE": "M", "FEMALE": "F"}
    )
    operations.loc[~operations["sex_common"].isin(["M", "F"]), "sex_common"] = pd.NA
    operations["patient_class_common"] = (
        operations["PATIENT_CLASS_GROUP"].fillna("<MISSING>").str.strip()
    )
    operations["procedure_common"] = (
        operations["PRIMARY_PROCEDURE_NM"].fillna("<MISSING>").str.strip()
    )

    # Main adjacency never silently jumps over an operation with unavailable or
    # nonpositive anaesthesia timing.
    subject_orderable = operations.groupby("patient_id")["positive_interval"].all()
    strict = operations.loc[
        operations["patient_id"].isin(subject_orderable[subject_orderable].index)
    ].sort_values(["patient_id", "anstart", "anstop", "LOG_ID"])
    for column in [
        "LOG_ID",
        "anstart",
        "anstop",
        "is_general",
        "age_years",
        "bmi_kg_m2",
        "asa_numeric",
        "sex_common",
        "patient_class_common",
        "procedure_common",
    ]:
        strict["prior_" + column] = strict.groupby("patient_id", observed=True)[column].shift(1)
    strict["adjacent_order_valid"] = strict["anstart"].gt(strict["prior_anstop"])
    strict["general_to_general"] = strict["is_general"] & strict["prior_is_general"].fillna(False)
    pairs = strict.loc[strict["adjacent_order_valid"] & strict["general_to_general"]].copy()
    pairs["interval_days"] = (
        (pairs["anstart"] - pairs["prior_anstop"]).dt.total_seconds() / 86400.0
    )

    audit = {
        "raw_info_rows": int(len(raw)),
        "exact_duplicate_rows_removed": int(exact_duplicates),
        "conflicting_info_log_id_excluded": int(len(conflict_info)),
        "conflicting_mapping_log_id_excluded": int(len(conflict_mapping)),
        "clean_operation_rows": int(len(operations)),
        "patients_all_operations_orderable": int(subject_orderable.sum()),
        "strict_general_to_general_pairs_before_map_coverage": int(len(pairs)),
        "strict_pair_patients_before_map_coverage": int(pairs["patient_id"].nunique()),
        "age_nonmissing_rate_pairs": float(pairs["age_years"].notna().mean()),
        "bmi_nonmissing_rate_pairs": float(pairs["bmi_kg_m2"].notna().mean()),
        "asa_nonmissing_rate_pairs": float(pairs["asa_numeric"].notna().mean()),
        "sex_nonmissing_rate_pairs": float(pairs["sex_common"].notna().mean()),
        "weight_storage_interpretation": "ounces converted to kg; aggregate mean cross-checked to MOVER paper",
        "birth_date_storage_interpretation": "released two-digit age field; aggregate mean cross-checked to MOVER paper",
    }
    return pairs, audit


def operation_first_two(map_subset_path: Path, modality: str, intraop_only: bool) -> tuple[pd.DataFrame, dict]:
    usecols = [
        "LOG_ID",
        "RECORDED_TIME",
        "relative_min",
        "value",
        "UNITS",
        "RECORD_TYPE",
        "FLO_MEAS_NAME",
        "FLO_DISPLAY_NAME",
        "modality_hint",
    ]
    maps = pd.read_csv(map_subset_path, usecols=usecols, dtype={"LOG_ID": str}, low_memory=False)
    if modality == "NIBP":
        exact = maps["FLO_MEAS_NAME"].eq(NIBP_MEAS_NAME) & maps["FLO_DISPLAY_NAME"].eq(NIBP_DISPLAY_NAME)
    elif modality == "ART":
        exact = maps["FLO_MEAS_NAME"].eq(ART_MEAS_NAME) & maps["FLO_DISPLAY_NAME"].eq(ART_DISPLAY_NAME)
    else:
        raise ValueError("modality must be NIBP or ART")
    selected = maps.loc[
        exact
        & maps["modality_hint"].eq(modality)
        & pd.to_numeric(maps["relative_min"], errors="coerce").between(0, 30, inclusive="both")
        & pd.to_numeric(maps["value"], errors="coerce").between(20, 200, inclusive="both")
    ].copy()
    if intraop_only:
        selected = selected.loc[selected["RECORD_TYPE"].eq("INTRA-OP")].copy()
    selected["recorded"] = pd.to_datetime(selected["RECORDED_TIME"], errors="coerce")
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    selected["relative_min"] = pd.to_numeric(selected["relative_min"], errors="coerce")
    selected = selected.dropna(subset=["LOG_ID", "recorded", "value", "relative_min"])

    grouped = (
        selected.groupby(["LOG_ID", "recorded"], as_index=False, observed=True)
        .agg(
            relative_min=("relative_min", "min"),
            raw_records=("value", "size"),
            distinct_values=("value", "nunique"),
            sole_value=("value", "first"),
        )
    )
    conflict = grouped["distinct_values"].gt(1)
    valid = grouped.loc[~conflict].sort_values(["LOG_ID", "recorded"]).copy()
    first_two = valid.groupby("LOG_ID", observed=True, sort=False).head(2)
    rows = []
    for log_id, frame in first_two.groupby("LOG_ID", observed=True, sort=False):
        frame = frame.sort_values("recorded")
        if len(frame) < 2:
            continue
        values = frame["sole_value"].to_numpy(float)
        relative = frame["relative_min"].to_numpy(float)
        rows.append(
            {
                "LOG_ID": str(log_id),
                f"{modality.lower()}_map_0": float(values[0]),
                f"{modality.lower()}_map_1": float(values[1]),
                f"{modality.lower()}_rel_0": float(relative[0]),
                f"{modality.lower()}_rel_1": float(relative[1]),
                f"{modality.lower()}_change": float(values[1] - values[0]),
                f"{modality.lower()}_any_low": int(np.min(values) < 65),
                f"{modality.lower()}_both_low": int(np.all(values < 65)),
            }
        )
    features = pd.DataFrame(rows)
    audit = {
        "modality": modality,
        "intraop_only": bool(intraop_only),
        "exact_rows_0_30": int(len(selected)),
        "same_time_keys": int(len(grouped)),
        "conflicting_same_time_keys_excluded": int(conflict.sum()),
        "operations_with_1plus_valid_measurement": int(valid["LOG_ID"].nunique()),
        "operations_with_2plus_valid_measurements": int(len(features)),
    }
    return features, audit


def main() -> None:
    args = parse_args()
    args.info = require_private_path(args.info)
    args.mapping = require_private_path(args.mapping)
    args.map_subset = require_private_path(args.map_subset)
    args.restricted_cohort = require_private_path(
        args.restricted_cohort, must_exist=False
    )
    secure_directory(args.restricted_cohort.parent)
    args.aggregate_out = secure_directory(args.aggregate_out)
    pairs, operation_audit = clean_operations(args.info, args.mapping)
    nibp, nibp_audit = operation_first_two(args.map_subset, "NIBP", intraop_only=True)
    art, art_audit = operation_first_two(args.map_subset, "ART", intraop_only=True)
    nibp_any_type, nibp_any_type_audit = operation_first_two(args.map_subset, "NIBP", intraop_only=False)

    current_nibp = nibp.rename(columns={column: "current_" + column for column in nibp.columns if column != "LOG_ID"})
    prior_nibp = nibp.rename(columns={column: "prior_" + column for column in nibp.columns if column != "LOG_ID"})
    cohort = pairs.merge(current_nibp, on="LOG_ID", how="left", validate="many_to_one")
    cohort = cohort.merge(
        prior_nibp,
        left_on="prior_LOG_ID",
        right_on="LOG_ID",
        how="left",
        validate="many_to_one",
        suffixes=("", "_prior_lookup"),
    )
    if "LOG_ID_prior_lookup" in cohort:
        cohort = cohort.drop(columns=["LOG_ID_prior_lookup"])
    cohort["current_first2_nibp_observed"] = cohort["current_nibp_map_0"].notna()
    cohort["prior_first2_nibp_observed"] = cohort["prior_nibp_map_0"].notna()
    cohort["both_first2_nibp_observed"] = (
        cohort["current_first2_nibp_observed"] & cohort["prior_first2_nibp_observed"]
    )
    final = cohort.loc[cohort["both_first2_nibp_observed"]].copy()
    final["target_any_low_first2"] = final["current_nibp_any_low"].astype(int)
    final["target_both_low_first2"] = final["current_nibp_both_low"].astype(int)
    final["prior_first2_any_low"] = final["prior_nibp_any_low"].astype(int)
    final["prior_first_map"] = final["prior_nibp_map_0"].astype(float)
    final["prior_first2_change"] = final["prior_nibp_change"].astype(float)
    final["interval_log1p"] = np.log1p(final["interval_days"].clip(lower=0))

    final.to_csv(args.restricted_cohort, index=False, compression="gzip")
    args.restricted_cohort.chmod(0o600)
    flow = pd.DataFrame(
        [
            {"stage": "strict General-to-General adjacent pairs", "pairs": int(len(pairs)), "patients": int(pairs["patient_id"].nunique())},
            {"stage": "current first-two intra-op NIBP observed", "pairs": int(cohort["current_first2_nibp_observed"].sum()), "patients": int(cohort.loc[cohort["current_first2_nibp_observed"], "patient_id"].nunique())},
            {"stage": "prior first-two intra-op NIBP observed", "pairs": int(cohort["prior_first2_nibp_observed"].sum()), "patients": int(cohort.loc[cohort["prior_first2_nibp_observed"], "patient_id"].nunique())},
            {"stage": "both current and prior observed", "pairs": int(len(final)), "patients": int(final["patient_id"].nunique())},
        ]
    )
    flow.to_csv(args.aggregate_out / "cohort_flow.csv", index=False)
    summary = {
        "status": "MOVER_C02_COHORT_CONSTRUCTED_NO_MODEL",
        "operation_audit": operation_audit,
        "nibp_audit": nibp_audit,
        "art_audit": art_audit,
        "nibp_any_record_type_sensitivity_audit": nibp_any_type_audit,
        "final_pairs": int(len(final)),
        "final_patients": int(final["patient_id"].nunique()),
        "target_events_not_reported_before_model_stage": True,
        "restricted_cohort_name": args.restricted_cohort.name,
        "raw_identifiers_in_aggregate_outputs": False,
        "main_measurement_rule": (
            "first two conflict-free UC ANE R BLOOD PRESSURE - MAP / NIBP - MAP values, "
            "RECORD_TYPE=INTRA-OP, 0-30 min after Patient Information anesthesia start"
        ),
    }
    (args.aggregate_out / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
