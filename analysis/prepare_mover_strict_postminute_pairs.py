#!/usr/bin/env python3
"""Build the MOVER sensitivity cohort strictly after IV-hypnotic time.

The cohort excludes MAP observations with a zero timestamp difference from the
recorded hypnotic administration.  It writes one protected row per adjacent
anaesthetic pair and a separate aggregate audit receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from c02_runtime import (
    configure_private_workspace,
    protect_file,
    require_private_path,
    secure_directory,
)


STRICT_COLUMNS = {
    "current_post_map_0": "cur_map0",
    "current_post_map_1": "cur_map1",
    "current_post_map_change": "cur_change",
    "current_post_any_low": "cur_anylow",
    "current_post_map_rel_0": "cur_r0",
    "current_post_map_rel_1": "cur_r1",
    "current_anchor_agent": "cur_anchor_agent",
    "current_anchor_propofol_mg": "cur_anchor_propofol_mg",
    "current_anchor_etomidate_mg": "cur_anchor_etomidate_mg",
    "current_anchor_ketamine_mg": "cur_anchor_ketamine_mg",
    "prior_post_post_map_0": "pr_map0",
    "prior_post_post_map_1": "pr_map1",
    "prior_post_post_map_change": "pr_change",
    "prior_post_post_any_low": "pr_anylow",
    "prior_post_post_map_rel_0": "pr_r0",
    "prior_post_post_map_rel_1": "pr_r1",
    "prior_post_anchor_agent": "pr_anchor_agent",
    "prior_post_anchor_propofol_mg": "pr_anchor_propofol_mg",
    "prior_post_anchor_etomidate_mg": "pr_anchor_etomidate_mg",
    "prior_post_anchor_ketamine_mg": "pr_anchor_ketamine_mg",
}


def build_strict_pair(
    pair_path: Path,
    map_path: Path,
    mar_path: Path,
    restricted_output: Path,
    aggregate_output: Path,
) -> dict[str, object]:
    from analyze_c02_hypnotic_anchored_reproducibility import (
        operation_hypnotics,
        operation_post_anchor_map,
        prepare_pair,
    )

    pair_path = require_private_path(pair_path)
    map_path = require_private_path(map_path)
    mar_path = require_private_path(mar_path)
    restricted_output = require_private_path(restricted_output, must_exist=False)
    aggregate_output = secure_directory(aggregate_output)
    secure_directory(restricted_output.parent)

    pair = pd.read_csv(
        pair_path,
        dtype={"LOG_ID": str, "prior_LOG_ID": str},
        low_memory=False,
    )
    pair_columns = list(pair.columns)
    operation_ids = set(pair["LOG_ID"]) | set(pair["prior_LOG_ID"])

    mar = pd.read_csv(mar_path, dtype={"LOG_ID": str}, low_memory=False)
    anchors, drug_audit = operation_hypnotics(mar.loc[mar["LOG_ID"].isin(operation_ids)])
    maps = pd.read_csv(map_path, dtype={"LOG_ID": str}, low_memory=False)
    operation, map_audit = operation_post_anchor_map(
        maps.loc[maps["LOG_ID"].isin(operation_ids)],
        anchors,
        strictly_after_anchor=True,
    )
    prepared = prepare_pair(operation, pair)
    strict = prepared[pair_columns + list(STRICT_COLUMNS)].rename(columns=STRICT_COLUMNS)

    if not strict["cur_r0"].gt(0).all() or not strict["pr_r0"].gt(0).all():
        raise RuntimeError("strict postminute cohort contains a non-positive MAP time")
    if strict.duplicated(["patient_id", "LOG_ID"]).any():
        raise RuntimeError("strict postminute cohort has duplicate patient-operation rows")
    observed_counts = (
        int(len(strict)),
        int(strict["patient_id"].nunique()),
        int(strict["cur_anylow"].sum()),
    )
    expected_counts = (4871, 3645, 567)
    if observed_counts != expected_counts:
        raise RuntimeError(
            "strict postminute cohort count drift: "
            f"observed pairs/patients/events={observed_counts}; "
            f"expected={expected_counts}"
        )

    strict.to_csv(restricted_output, index=False, compression="gzip")
    protect_file(restricted_output)
    receipt = {
        "status": "STRICT_POSTMINUTE_COHORT_BUILT",
        "pairs": int(len(strict)),
        "patients": int(strict["patient_id"].nunique()),
        "events": int(strict["cur_anylow"].sum()),
        "rule": "first two NIBP MAP observations with timestamp strictly after recorded IV-hypnotic administration and no later than 15 minutes",
        "drug_audit": drug_audit,
        "map_audit": map_audit,
        "restricted_output_name": restricted_output.name,
        "identifiers_in_aggregate_output": False,
    }
    (aggregate_output / "strict_postminute_cohort_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-workspace", required=True, type=Path)
    parser.add_argument("--pair", required=True, type=Path)
    parser.add_argument("--map-stream", required=True, type=Path)
    parser.add_argument("--hypnotic-mar", required=True, type=Path)
    parser.add_argument("--restricted-output", required=True, type=Path)
    parser.add_argument("--aggregate-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_private_workspace(args.private_workspace)
    receipt = build_strict_pair(
        args.pair,
        args.map_stream,
        args.hypnotic_mar,
        args.restricted_output,
        args.aggregate_output,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
