#!/usr/bin/env python3
"""Rebuild C02 adjacent pairs without skipping operations with missing times.

This is a conservative exploratory rebuild: subjects with any missing
anaesthesia start are excluded; pairs are formed only between truly adjacent
operation rows with non-missing starts/ends and strict current-start >
prior-end. No later pair is substituted when a candidate adjacent pair fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from c02_runtime import (
    private_workspace_root,
    protect_file,
    require_private_path,
    secure_directory,
)


ROOT = private_workspace_root()
DATA = ROOT / "INSPIRE_v1.4.2"
SRC = ROOT / "outputs/inspire_curiosity/candidate_gates/c02_redesign_v1"
OUT = ROOT / "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild"


def minutes_to_days(value):
    """Convert INSPIRE relative-time differences from minutes to days."""
    return value / 1440.0


def minute_duration(start, end):
    """Return duration in minutes for INSPIRE minute-valued time fields."""
    return end - start


def load() -> pd.DataFrame:
    ops = pd.read_csv(DATA / "operations.csv.gz")
    keep = [
        "op_id", "subject_id", "age", "sex", "weight", "height", "asa", "emop",
        "department", "antype", "icd10_pcs", "admission_time", "discharge_time",
        "anstart_time", "anend_time", "opstart_time", "allcause_death_time",
    ]
    ops = ops[keep].copy()
    for c in ["op_id", "subject_id", "age", "weight", "height", "asa", "emop", "admission_time", "discharge_time", "anstart_time", "anend_time", "opstart_time", "allcause_death_time"]:
        ops[c] = pd.to_numeric(ops[c], errors="coerce")
    ops["procedure3"] = ops.icd10_pcs.fillna("").astype(str).str[:3].replace("", np.nan)
    ops["bmi"] = ops.weight / (ops.height / 100.0) ** 2
    ops.loc[~ops.bmi.between(10, 80), "bmi"] = np.nan
    # INSPIRE relative-time fields are already expressed in minutes.
    ops["an_duration_min"] = minute_duration(ops.anstart_time, ops.anend_time)
    fixed = pd.read_csv(SRC / "fixed_slot_operation_features.csv.gz")
    # Keep the authoritative timing fields from operations; the fixed-feature
    # table repeats anstart/opstart and would otherwise create suffixed columns.
    fixed = fixed.drop(columns=[c for c in ["anstart_time", "opstart_time"] if c in fixed.columns])
    ward = pd.read_csv(SRC / "current_preop_ward.csv.gz")
    labs = pd.read_csv(SRC / "current_preop_covariates.csv.gz", usecols=["op_id", "subject_id", "preop_lab_creatinine", "preop_lab_hb", "preop_lab_albumin"])
    x = ops.merge(fixed, on=["op_id", "subject_id"], validate="one_to_one")
    x = x.merge(ward, on=["op_id", "subject_id"], how="left", validate="one_to_one")
    x = x.merge(labs, on=["op_id", "subject_id"], how="left", validate="one_to_one")
    return x


def main() -> None:
    global DATA, SRC, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--fixed-feature-dir", required=True, type=Path)
    parser.add_argument("--restricted-output-dir", required=True, type=Path)
    args = parser.parse_args()
    DATA = require_private_path(args.dataset_dir)
    SRC = require_private_path(args.fixed_feature_dir)
    OUT = secure_directory(args.restricted_output_dir)
    x = load()
    missing_start_subjects = set(x.loc[x.anstart_time.isna(), "subject_id"].dropna().astype(int))
    x = x.loc[~x.subject_id.isin(missing_start_subjects)].copy()
    x = x.sort_values(["subject_id", "anstart_time", "anend_time", "op_id"], na_position="last")
    pairs = []
    all_pairs = []
    skipped_overlap = 0
    skipped_missing_time = 0
    for sid, g in x.groupby("subject_id", sort=False):
        rows = g.reset_index(drop=True)
        for i in range(1, len(rows)):
            prior, cur = rows.iloc[i - 1], rows.iloc[i]
            if pd.isna(prior.anstart_time) or pd.isna(prior.anend_time) or pd.isna(cur.anstart_time) or pd.isna(cur.anend_time):
                skipped_missing_time += 1
                continue
            if not (cur.anstart_time > prior.anend_time):
                skipped_overlap += 1
                continue
            cur_observed = bool(cur.fixed_window_eligible)
            row = {
                "op_id": int(cur.op_id), "subject_id": int(sid), "target_any_low": (int(bool(cur.target_any_low)) if cur_observed else np.nan),
                "target_two_low": (int(bool(cur.target_two_low)) if cur_observed else np.nan),
                "current_map_t0": cur.map_t0, "current_map_t5": cur.map_t5, "current_map_t10": cur.map_t10,
                "age": cur.age, "bmi": cur.bmi, "asa": cur.asa,
                "emop": cur.emop, "sex": cur.sex, "department": cur.department, "antype": cur.antype,
                "procedure3": cur.procedure3, "preop_ward_map_within_24h": cur.preop_ward_map_within_24h,
                "preop_lab_creatinine": cur.preop_lab_creatinine, "preop_lab_hb": cur.preop_lab_hb,
                "preop_lab_albumin": cur.preop_lab_albumin,
                "prior_asa": prior.asa, "prior_emop": prior.emop, "prior_department": prior.department,
                "prior_antype": prior.antype, "prior_procedure3": prior.procedure3,
                # INSPIRE relative-time fields are minute-valued.  Convert the
                # adjacent anaesthetic gap with 1,440 minutes per day.
                "interval_days": minutes_to_days(cur.anstart_time - prior.anend_time),
                "prior_an_duration_min": prior.an_duration_min,
                "prior_map_t0": prior.map_t0, "prior_map_t5": prior.map_t5, "prior_map_t10": prior.map_t10,
                "prior_map_delta_10_vs_0": prior.map_delta_10_vs_0, "prior_map_nadir_delta": prior.map_nadir_delta,
                "current_anstart_time": cur.anstart_time, "current_anend_time": cur.anend_time,
                "current_discharge_time": cur.discharge_time,
                "prior_anstart_time": prior.anstart_time, "prior_anend_time": prior.anend_time,
                "current_slot_observed": int(cur_observed),
                "prior_slot_observed": int(bool(prior.fixed_window_eligible)),
                "both_slot_observed": int(bool(prior.fixed_window_eligible) and bool(cur.fixed_window_eligible)),
            }
            all_pairs.append(row)
            if not bool(row["both_slot_observed"]):
                continue
            pairs.append(row)
    out = pd.DataFrame(pairs)
    all_out = pd.DataFrame(all_pairs)
    out.to_csv(OUT / "pair_cohort.csv.gz", index=False)
    all_out.to_csv(OUT / "all_adjacent_pairs.csv.gz", index=False)
    protect_file(OUT / "pair_cohort.csv.gz")
    protect_file(OUT / "all_adjacent_pairs.csv.gz")
    summary = {
        "status": "exploratory_clean_rebuild",
        "all_operations": int(len(x)),
        "subjects_excluded_for_any_missing_anstart": int(len(missing_start_subjects)),
        "retained_operation_subjects": int(x.subject_id.nunique()),
        "clean_pairs": int(len(out)),
        "clean_pair_subjects": int(out.subject_id.nunique()) if not out.empty else 0,
        "events": int(out.target_any_low.sum()) if not out.empty else 0,
        "two_low_events": int(out.target_two_low.sum()) if not out.empty else 0,
        "all_strict_adjacent_pairs": int(len(all_out)),
        "all_pair_subjects": int(all_out.subject_id.nunique()) if not all_out.empty else 0,
        "both_slot_observed_pairs": int(all_out.both_slot_observed.sum()) if not all_out.empty else 0,
        "current_slot_observed_pairs": int(all_out.current_slot_observed.sum()) if not all_out.empty else 0,
        "prior_slot_observed_pairs": int(all_out.prior_slot_observed.sum()) if not all_out.empty else 0,
        "adjacent_overlap_or_reverse_pairs_skipped": int(skipped_overlap),
        "pairs_skipped_for_missing_time": int(skipped_missing_time),
        "pair_rule": "all-operation adjacency; strict current anstart > prior anend; no replacement after a failed adjacent pair",
        "caveat": "This rebuild uses fixed-slot features prepared previously; it is not a fully independent raw-vital reconstruction.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
