#!/usr/bin/env python3
"""Stream MOVER EPIC medication MAR and retain early vasopressor actions.

The 7.7-GB CSV is kept inside EPIC_EMR.tar.gz. This script never extracts it.
It binary-prefilters vasoactive drug names, aligns MED_ACTION_TIME to Patient
Information anaesthesia start, and writes a restricted LOG_ID-level subset plus
aggregate action/label counts. It does not label every row as rescue.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tarfile
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from extract_mover_map_stream import build_timing_lookup
from c02_runtime import require_private_path, secure_directory


TARGET_MEMBER = "EPIC_EMR/EMR/patient_medications.csv"
MARKERS = (
    b"norepinephrine", b"NOREPINEPHRINE", b"Levophed", b"LEVOPHED",
    b"phenylephrine", b"PHENYLEPHRINE", b"Neo-Synephrine", b"NEO-SYNEPHRINE",
    b"ephedrine", b"EPHEDRINE", b"vasopressin", b"VASOPRESSIN",
    b"epinephrine", b"EPINEPHRINE", b"adrenaline", b"ADRENALINE",
    b"metaraminol", b"METARAMINOL",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--restricted-out", type=Path, required=True)
    parser.add_argument("--aggregate-out", type=Path, required=True)
    parser.add_argument(
        "--full-intraop",
        action="store_true",
        help="Retain anesthesia start through stop instead of -10 to +30 minutes.",
    )
    return parser.parse_args()


def drug_class(display: str, medication: str) -> str | None:
    text = (display + " | " + medication).lower()
    # Exclude local-anaesthetic solutions containing epinephrine and topical
    # mixtures; only explicit primary vasoactive product names are retained.
    if "norepinephrine" in text or "levophed" in text:
        return "norepinephrine"
    if "phenylephrine" in text or "neo-synephrine" in text:
        return "phenylephrine"
    if "ephedrine" in text:
        return "ephedrine"
    if "vasopressin" in text:
        return "vasopressin"
    if "epinephrine" in text or "adrenaline" in text:
        if any(token in text for token in ["lidocaine", "bupivacaine", "ropivacaine", "articaine", "tetracaine"]):
            return None
        return "epinephrine"
    if "metaraminol" in text:
        return "metaraminol"
    return None


def main() -> None:
    args = parse_args()
    args.archive = require_private_path(args.archive)
    args.info = require_private_path(args.info)
    args.mapping = require_private_path(args.mapping)
    args.restricted_out = require_private_path(args.restricted_out, must_exist=False)
    secure_directory(args.restricted_out.parent)
    args.aggregate_out = secure_directory(args.aggregate_out)
    os.chmod(args.restricted_out.parent, 0o700)
    timing, timing_audit = build_timing_lookup(args.info, args.mapping)
    started = time.monotonic()
    total_rows = 0
    candidate_lines = 0
    parsed_candidates = 0
    retained = 0
    malformed = 0
    invalid_time = 0
    outside_window = 0
    timing_missing = 0
    class_counts: Counter[str] = Counter()
    action_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    display_counts: Counter[tuple[str, str, str, str]] = Counter()
    output_columns = [
        "LOG_ID", "drug_class", "MED_ACTION_TIME", "relative_min", "RECORD_TYPE",
        "MAR_ACTION_NM", "ADMIN_SIG", "DOSE_UNIT_NM", "MED_ROUTE_NM",
        "DISPLAY_NAME", "MEDICATION_NM", "ORDER_CLASS_NM", "ORDER_STATUS_NM",
    ]
    with gzip.open(args.restricted_out, "wt", encoding="utf-8", newline="") as output:
        os.chmod(args.restricted_out, 0o600)
        writer = csv.DictWriter(output, fieldnames=output_columns)
        writer.writeheader()
        with tarfile.open(args.archive, "r:gz") as archive:
            member = archive.getmember(TARGET_MEMBER)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("medication member could not be opened")
            header_raw = source.readline()
            header = next(csv.reader([header_raw.decode("utf-8-sig", errors="strict")]))
            required = set(output_columns) - {"drug_class", "relative_min"}
            missing = sorted(required - set(header))
            if missing:
                raise RuntimeError(f"missing medication columns: {missing}")
            index = {column: header.index(column) for column in required}
            for raw in source:
                total_rows += 1
                if not any(marker in raw for marker in MARKERS):
                    continue
                candidate_lines += 1
                try:
                    row = next(csv.reader([raw.decode("utf-8", errors="strict")]))
                except (UnicodeDecodeError, csv.Error):
                    malformed += 1
                    continue
                if len(row) != len(header):
                    malformed += 1
                    continue
                parsed_candidates += 1
                display = row[index["DISPLAY_NAME"]].strip()
                medication = row[index["MEDICATION_NM"]].strip()
                klass = drug_class(display, medication)
                if klass is None:
                    continue
                log_id = row[index["LOG_ID"]].strip()
                info_timing = timing.get(log_id)
                if info_timing is None:
                    timing_missing += 1
                    continue
                action_time = pd.to_datetime(
                    row[index["MED_ACTION_TIME"]].strip(), errors="coerce"
                )
                if pd.isna(action_time):
                    invalid_time += 1
                    continue
                relative = (action_time.to_pydatetime() - info_timing[0]).total_seconds() / 60
                window_start = 0.0 if args.full_intraop else -10.0
                window_end = (
                    (info_timing[1] - info_timing[0]).total_seconds() / 60.0
                    if args.full_intraop
                    else 30.0
                )
                if not window_start <= relative <= window_end:
                    outside_window += 1
                    continue
                record_type = row[index["RECORD_TYPE"]].strip()
                action = row[index["MAR_ACTION_NM"]].strip()
                sig = row[index["ADMIN_SIG"]].strip()
                unit = row[index["DOSE_UNIT_NM"]].strip()
                route = row[index["MED_ROUTE_NM"]].strip()
                class_counts[klass] += 1
                action_counts[(klass, record_type, action, unit, route)] += 1
                display_counts[(klass, display, medication, record_type)] += 1
                writer.writerow(
                    {
                        "LOG_ID": log_id,
                        "drug_class": klass,
                        "MED_ACTION_TIME": action_time.isoformat(sep=" "),
                        "relative_min": f"{relative:.6f}",
                        "RECORD_TYPE": record_type,
                        "MAR_ACTION_NM": action,
                        "ADMIN_SIG": sig,
                        "DOSE_UNIT_NM": unit,
                        "MED_ROUTE_NM": route,
                        "DISPLAY_NAME": display,
                        "MEDICATION_NM": medication,
                        "ORDER_CLASS_NM": row[index["ORDER_CLASS_NM"]].strip(),
                        "ORDER_STATUS_NM": row[index["ORDER_STATUS_NM"]].strip(),
                    }
                )
                retained += 1
    pd.DataFrame(
        [
            {
                "drug_class": key[0], "record_type": key[1], "mar_action": key[2],
                "dose_unit": key[3], "route": key[4], "rows": count,
            }
            for key, count in action_counts.most_common()
        ]
    ).to_csv(args.aggregate_out / "action_unit_route_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "drug_class": key[0], "display_name": key[1],
                "medication_name": key[2], "record_type": key[3], "rows": count,
            }
            for key, count in display_counts.most_common()
        ]
    ).to_csv(args.aggregate_out / "drug_label_counts.csv", index=False)
    summary = {
        "status": "MOVER_EARLY_VASOPRESSOR_MAR_EXTRACTED_NO_RESCUE_LABEL",
        "source": args.archive.name,
        "source_member": TARGET_MEMBER,
        "total_medication_rows_scanned": int(total_rows),
        "binary_candidate_lines": int(candidate_lines),
        "parsed_candidate_lines": int(parsed_candidates),
        "malformed_candidate_lines": int(malformed),
        "timing_unavailable": int(timing_missing),
        "invalid_action_time": int(invalid_time),
        "outside_selected_window": int(outside_window),
        "full_intraop": bool(args.full_intraop),
        "retained_rows": int(retained),
        "retained_by_drug_class": dict(class_counts),
        "timing_audit": timing_audit,
        "elapsed_minutes": float((time.monotonic() - started) / 60),
        "restricted_output": args.restricted_out.name,
        "credentials_or_identifiers_in_aggregate_outputs": False,
        "construct_warning": (
            "Retained rows are candidate MAR actions. Rescue requires downstream action, dose, route, "
            "record-type and temporal linkage checks; orders/holds/stops are not rescue."
        ),
    }
    (args.aggregate_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
