#!/usr/bin/env python3
"""Stream MOVER MAR for induction and early anaesthetic drug candidates."""

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
    b"propofol", b"PROPOFOL", b"Diprivan", b"DIPRIVAN",
    b"etomidate", b"ETOMIDATE", b"Amidate", b"AMIDATE",
    b"ketamine", b"KETAMINE", b"Ketalar", b"KETALAR",
    b"midazolam", b"MIDAZOLAM", b"Versed", b"VERSED",
    b"fentanyl", b"FENTANYL", b"Sublimaze", b"SUBLIMAZE",
    b"remifentanil", b"REMIFENTANIL", b"Ultiva", b"ULTIVA",
    b"sufentanil", b"SUFENTANIL", b"Sufenta", b"SUFENTA",
    b"alfentanil", b"ALFENTANIL", b"Alfenta", b"ALFENTA",
    b"dexmedetomidine", b"DEXMEDETOMIDINE", b"Precedex", b"PRECEDEX",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--restricted-out", type=Path, required=True)
    parser.add_argument("--aggregate-out", type=Path, required=True)
    return parser.parse_args()


def drug_class(display: str, medication: str) -> str | None:
    text = (display + " | " + medication).lower()
    # Avoid drug-name mentions in premixed or combination products only where
    # another active agent clearly defines the product. Labels remain in the
    # audit table so downstream strict rules are inspectable.
    if "propofol" in text or "diprivan" in text:
        return "propofol"
    if "etomidate" in text or "amidate" in text:
        return "etomidate"
    if "ketamine" in text or "ketalar" in text:
        return "ketamine"
    if "midazolam" in text or "versed" in text:
        return "midazolam"
    if "remifentanil" in text or "ultiva" in text:
        return "remifentanil"
    if "sufentanil" in text or "sufenta" in text:
        return "sufentanil"
    if "alfentanil" in text or "alfenta" in text:
        return "alfentanil"
    if "fentanyl" in text or "sublimaze" in text:
        return "fentanyl"
    if "dexmedetomidine" in text or "precedex" in text:
        return "dexmedetomidine"
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
    counts: Counter[str] = Counter()
    action_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    label_counts: Counter[tuple[str, str, str, str]] = Counter()
    output_columns = [
        "LOG_ID", "drug_class", "MED_ACTION_TIME", "relative_min", "RECORD_TYPE",
        "MAR_ACTION_NM", "ADMIN_SIG", "DOSE_UNIT_NM", "MED_ROUTE_NM",
        "DISPLAY_NAME", "MEDICATION_NM", "ORDER_CLASS_NM", "ORDER_STATUS_NM",
    ]
    required = set(output_columns) - {"drug_class", "relative_min"}
    with gzip.open(args.restricted_out, "wt", encoding="utf-8", newline="") as output:
        os.chmod(args.restricted_out, 0o600)
        writer = csv.DictWriter(output, fieldnames=output_columns)
        writer.writeheader()
        with tarfile.open(args.archive, "r:gz") as archive:
            member = archive.getmember(TARGET_MEMBER)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("medication member could not be opened")
            header = next(csv.reader([source.readline().decode("utf-8-sig")]))
            missing = sorted(required - set(header))
            if missing:
                raise RuntimeError(f"missing medication columns: {missing}")
            index = {column: header.index(column) for column in required}
            for raw in source:
                counts["rows_scanned"] += 1
                if not any(marker in raw for marker in MARKERS):
                    continue
                counts["binary_candidates"] += 1
                try:
                    row = next(csv.reader([raw.decode("utf-8")]))
                except (UnicodeDecodeError, csv.Error):
                    counts["malformed"] += 1
                    continue
                if len(row) != len(header):
                    counts["malformed"] += 1
                    continue
                counts["parsed_candidates"] += 1
                display = row[index["DISPLAY_NAME"]].strip()
                medication = row[index["MEDICATION_NM"]].strip()
                klass = drug_class(display, medication)
                if klass is None:
                    continue
                log_id = row[index["LOG_ID"]].strip()
                if log_id not in timing:
                    counts["timing_unavailable"] += 1
                    continue
                action_time = pd.to_datetime(
                    row[index["MED_ACTION_TIME"]].strip(), errors="coerce"
                )
                if pd.isna(action_time):
                    counts["invalid_action_time"] += 1
                    continue
                relative = (
                    action_time.to_pydatetime() - timing[log_id][0]
                ).total_seconds() / 60
                if not -15 <= relative <= 15:
                    counts["outside_window"] += 1
                    continue
                record_type = row[index["RECORD_TYPE"]].strip()
                action = row[index["MAR_ACTION_NM"]].strip()
                sig = row[index["ADMIN_SIG"]].strip()
                unit = row[index["DOSE_UNIT_NM"]].strip()
                route = row[index["MED_ROUTE_NM"]].strip()
                action_counts[(klass, record_type, action, unit, route)] += 1
                label_counts[(klass, display, medication, record_type)] += 1
                counts["retained"] += 1
                counts[f"retained_{klass}"] += 1
                writer.writerow(
                    {
                        "LOG_ID": log_id, "drug_class": klass,
                        "MED_ACTION_TIME": action_time.isoformat(sep=" "),
                        "relative_min": f"{relative:.6f}",
                        "RECORD_TYPE": record_type, "MAR_ACTION_NM": action,
                        "ADMIN_SIG": sig, "DOSE_UNIT_NM": unit,
                        "MED_ROUTE_NM": route, "DISPLAY_NAME": display,
                        "MEDICATION_NM": medication,
                        "ORDER_CLASS_NM": row[index["ORDER_CLASS_NM"]].strip(),
                        "ORDER_STATUS_NM": row[index["ORDER_STATUS_NM"]].strip(),
                    }
                )
    pd.DataFrame(
        [
            {
                "drug_class": key[0], "record_type": key[1],
                "mar_action": key[2], "dose_unit": key[3], "route": key[4],
                "rows": value,
            }
            for key, value in action_counts.most_common()
        ]
    ).to_csv(args.aggregate_out / "action_unit_route_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "drug_class": key[0], "display_name": key[1],
                "medication_name": key[2], "record_type": key[3], "rows": value,
            }
            for key, value in label_counts.most_common()
        ]
    ).to_csv(args.aggregate_out / "drug_label_counts.csv", index=False)
    summary = {
        "status": "MOVER_EARLY_ANAESTHETIC_MAR_CANDIDATES_EXTRACTED",
        "source": args.archive.name,
        "source_member": TARGET_MEMBER,
        "window_relative_to_anaesthesia_start_min": [-15, 15],
        "counts": dict(counts),
        "timing_audit": timing_audit,
        "elapsed_minutes": (time.monotonic() - started) / 60,
        "restricted_output": args.restricted_out.name,
        "raw_identifiers_in_aggregate_outputs": False,
        "construct_warning": (
            "Candidate MAR rows are not yet comparable induction doses. Downstream rules must "
            "require actual administration, coherent route/unit, pre-first-MAP timing and weight normalization."
        ),
    }
    (args.aggregate_out / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
