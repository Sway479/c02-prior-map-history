#!/usr/bin/env python3
"""Stream MOVER EPIC measurements and retain only auditable MAP candidates.

The source tar.gz expands to roughly 171 GB, so this script never extracts the
full CSV. It scans the gzip/tar stream once, writes a restricted LOG_ID-level
MAP subset, and emits aggregate label/coverage counts without patient IDs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import tarfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from c02_runtime import require_private_path, secure_directory


MAP_PATTERN = re.compile(r"(?<![a-z])map(?![a-z])|mean arterial|mean blood pressure", re.I)
REQUIRED_MEASUREMENT_COLUMNS = {
    "LOG_ID",
    "AN_START_DATETIME",
    "AN_STOP_DATETIME",
    "FLO_NAME",
    "FLO_MEAS_NAME",
    "FLO_DISPLAY_NAME",
    "RECORD_TYPE",
    "RECORDED_TIME",
    "MEAS_VALUE",
    "UNITS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--info", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--restricted-out", required=True, type=Path)
    parser.add_argument("--aggregate-out", required=True, type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--allow-incomplete-prefix", action="store_true")
    return parser.parse_args()


def parse_info_time(value: object) -> datetime | None:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return datetime.strptime(text, "%m/%d/%y %H:%M")
    except ValueError:
        return None


def parse_iso_time(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def build_timing_lookup(info_path: Path, mapping_path: Path) -> tuple[dict[str, tuple[datetime, datetime | None]], dict]:
    raw = pd.read_csv(info_path, dtype=str, low_memory=False)
    exact_duplicates = int(raw.duplicated().sum())
    info = raw.drop_duplicates().copy()
    nonkey = [column for column in info.columns if column != "LOG_ID"]
    conflict = info.groupby("LOG_ID", dropna=False)[nonkey].nunique(dropna=False).gt(1).any(axis=1)
    conflict_info = set(conflict[conflict].index.dropna())

    mapping = pd.read_csv(mapping_path, dtype=str).drop_duplicates()
    mapping_conflict = mapping.groupby("LOG_ID")["MRN"].nunique(dropna=False).gt(1)
    conflict_mapping = set(mapping_conflict[mapping_conflict].index)
    exclude = conflict_info | conflict_mapping
    info = info.loc[~info["LOG_ID"].isin(exclude)].drop_duplicates("LOG_ID")

    lookup: dict[str, tuple[datetime, datetime | None]] = {}
    start_parse_fail = 0
    stop_parse_fail = 0
    nonpositive_interval = 0
    for row in info[["LOG_ID", "AN_START_DATETIME", "AN_STOP_DATETIME"]].itertuples(index=False):
        start = parse_info_time(row.AN_START_DATETIME)
        stop = parse_info_time(row.AN_STOP_DATETIME)
        if start is None:
            start_parse_fail += 1
            continue
        if stop is None:
            stop_parse_fail += 1
            continue
        if stop <= start:
            nonpositive_interval += 1
            continue
        lookup[str(row.LOG_ID)] = (start, stop)
    audit = {
        "info_raw_rows": int(len(raw)),
        "info_exact_duplicates": exact_duplicates,
        "excluded_conflicting_info_log_id": int(len(conflict_info)),
        "excluded_conflicting_mapping_log_id": int(len(conflict_mapping)),
        "timing_lookup_log_id": int(len(lookup)),
        "anstart_parse_fail_or_missing": int(start_parse_fail),
        "anstop_parse_fail_or_missing_among_start_available": int(stop_parse_fail),
        "nonpositive_anesthesia_interval": int(nonpositive_interval),
    }
    return lookup, audit


def is_map_candidate(flo_name: str, flo_meas_name: str, flo_display_name: str) -> bool:
    return bool(MAP_PATTERN.search(" | ".join((flo_name, flo_meas_name, flo_display_name))))


def modality_hint(flo_meas_name: str, flo_display_name: str) -> str:
    display = flo_display_name.strip().lower()
    measure = flo_meas_name.strip().lower()
    if display == "nibp - map" or "map cuff" in measure:
        return "NIBP"
    if display == "map-art a-line" or "map a-line" in measure or "arterial line map" in display:
        return "ART"
    return "OTHER_MAP"


def main() -> None:
    args = parse_args()
    args.measurements = require_private_path(args.measurements)
    args.info = require_private_path(args.info)
    args.mapping = require_private_path(args.mapping)
    args.restricted_out = require_private_path(args.restricted_out, must_exist=False)
    secure_directory(args.restricted_out.parent)
    args.aggregate_out = secure_directory(args.aggregate_out)
    os.chmod(args.restricted_out.parent, 0o700)

    timing, timing_audit = build_timing_lookup(args.info, args.mapping)
    label_counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    member_rows: Counter[str] = Counter()
    candidate_rows = 0
    retained_rows = 0
    numeric_invalid = 0
    value_outside_plausible_map = 0
    recorded_time_invalid = 0
    anstart_source = Counter()
    row_vs_info_start_disagreement = 0
    rows_scanned = 0
    csv_members = 0
    started = time.monotonic()
    incomplete_prefix_error = None

    output_columns = [
        "LOG_ID",
        "RECORDED_TIME",
        "relative_min",
        "value",
        "UNITS",
        "RECORD_TYPE",
        "FLO_NAME",
        "FLO_MEAS_NAME",
        "FLO_DISPLAY_NAME",
        "modality_hint",
    ]
    with gzip.open(args.restricted_out, "wt", encoding="utf-8", newline="") as out_handle:
        os.chmod(args.restricted_out, 0o600)
        writer = csv.DictWriter(out_handle, fieldnames=output_columns)
        writer.writeheader()
        try:
            with args.measurements.open("rb") as raw, gzip.GzipFile(fileobj=raw) as gzip_stream:
                with tarfile.open(fileobj=gzip_stream, mode="r|") as archive:
                    for member in archive:
                        if not member.isfile() or not member.name.lower().endswith(".csv"):
                            continue
                        csv_members += 1
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            continue
                        lines = (line.decode("utf-8-sig", errors="replace") for line in extracted)
                        reader = csv.reader(lines)
                        header = next(reader)
                        missing = sorted(REQUIRED_MEASUREMENT_COLUMNS - set(header))
                        if missing:
                            raise ValueError(f"{member.name} missing required columns: {missing}")
                        index = {column: header.index(column) for column in REQUIRED_MEASUREMENT_COLUMNS}
                        for row in reader:
                            rows_scanned += 1
                            member_rows[member.name] += 1
                            if len(row) < len(header):
                                continue
                            flo_name = row[index["FLO_NAME"]].strip()
                            flo_meas = row[index["FLO_MEAS_NAME"]].strip()
                            flo_display = row[index["FLO_DISPLAY_NAME"]].strip()
                            if not is_map_candidate(flo_name, flo_meas, flo_display):
                                if args.max_rows and rows_scanned >= args.max_rows:
                                    break
                                continue
                            candidate_rows += 1
                            record_type = row[index["RECORD_TYPE"]].strip()
                            units = row[index["UNITS"]].strip()
                            hint = modality_hint(flo_meas, flo_display)
                            label_counts[(flo_name, flo_meas, flo_display, record_type, units, hint)] += 1

                            try:
                                value = float(row[index["MEAS_VALUE"]].strip())
                            except ValueError:
                                numeric_invalid += 1
                                if args.max_rows and rows_scanned >= args.max_rows:
                                    break
                                continue
                            if not 20 <= value <= 200:
                                value_outside_plausible_map += 1
                                if args.max_rows and rows_scanned >= args.max_rows:
                                    break
                                continue

                            log_id = row[index["LOG_ID"]].strip()
                            recorded = parse_iso_time(row[index["RECORDED_TIME"]])
                            if recorded is None:
                                recorded_time_invalid += 1
                                if args.max_rows and rows_scanned >= args.max_rows:
                                    break
                                continue
                            info_timing = timing.get(log_id)
                            if info_timing is None:
                                anstart_source["excluded_or_unavailable_patient_information"] += 1
                                if args.max_rows and rows_scanned >= args.max_rows:
                                    break
                                continue
                            info_start, info_stop = info_timing
                            row_start = parse_iso_time(row[index["AN_START_DATETIME"]])
                            if row_start is None:
                                anstart_source["patient_information_only"] += 1
                            else:
                                anstart_source["measurement_and_patient_information"] += 1
                                if row_start != info_start:
                                    row_vs_info_start_disagreement += 1
                            start = info_start
                            relative_min = (recorded - start).total_seconds() / 60.0
                            duration_min = (info_stop - info_start).total_seconds() / 60.0
                            # Retain pre-anaesthetic baseline candidates and all
                            # candidate MAPs during the recorded anaesthetic.
                            if -1440.0 <= relative_min <= max(30.0, duration_min):
                                writer.writerow(
                                    {
                                        "LOG_ID": log_id,
                                        "RECORDED_TIME": recorded.isoformat(sep=" "),
                                        "relative_min": f"{relative_min:.6f}",
                                        "value": f"{value:.12g}",
                                        "UNITS": units,
                                        "RECORD_TYPE": record_type,
                                        "FLO_NAME": flo_name,
                                        "FLO_MEAS_NAME": flo_meas,
                                        "FLO_DISPLAY_NAME": flo_display,
                                        "modality_hint": hint,
                                    }
                                )
                                retained_rows += 1
                            if args.max_rows and rows_scanned >= args.max_rows:
                                break
                        if args.max_rows and rows_scanned >= args.max_rows:
                            break
        except (EOFError, OSError, tarfile.ReadError) as exc:
            if not args.allow_incomplete_prefix:
                raise
            incomplete_prefix_error = f"{type(exc).__name__}: {exc}"

    label_rows = [
        {
            "FLO_NAME": key[0],
            "FLO_MEAS_NAME": key[1],
            "FLO_DISPLAY_NAME": key[2],
            "RECORD_TYPE": key[3],
            "UNITS": key[4],
            "modality_hint": key[5],
            "rows": count,
        }
        for key, count in label_counts.most_common()
    ]
    pd.DataFrame(label_rows).to_csv(args.aggregate_out / "map_candidate_label_counts.csv", index=False)
    summary = {
        "status": "MOVER_MAP_STREAM_EXTRACTION_PREFIX" if args.max_rows else "MOVER_MAP_STREAM_EXTRACTION_FULL",
        "source_name": args.measurements.name,
        "csv_members_scanned": int(csv_members),
        "rows_scanned": int(rows_scanned),
        "member_rows": dict(member_rows),
        "broad_map_candidate_rows": int(candidate_rows),
        "retained_restricted_rows": int(retained_rows),
        "numeric_invalid_candidate_rows": int(numeric_invalid),
        "candidate_values_outside_20_200": int(value_outside_plausible_map),
        "recorded_time_invalid": int(recorded_time_invalid),
        "anstart_source": dict(anstart_source),
        "row_vs_info_start_disagreement": int(row_vs_info_start_disagreement),
        "timing_audit": timing_audit,
        "max_rows": args.max_rows,
        "incomplete_prefix_error": incomplete_prefix_error,
        "elapsed_minutes": float((time.monotonic() - started) / 60.0),
        "restricted_output_name": args.restricted_out.name,
        "raw_identifiers_in_aggregate_outputs": False,
    }
    (args.aggregate_out / "map_stream_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
