#!/usr/bin/env python3
"""Compare the independent base-R coefficients with the result map."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


COMPARISONS = (
    ("ASSOC_BINARY_ALERT_INSPIRE", "INSPIRE", "primary_absolute_change", "prior_binary_alert", False),
    ("ASSOC_BINARY_ALERT_MOVER", "MOVER", "primary_absolute_change", "prior_binary_alert", False),
    ("ASSOC_PRIOR_LEVEL_INSPIRE", "INSPIRE", "primary_absolute_change", "prior_first_MAP_per_10mmHg", True),
    ("ASSOC_PRIOR_LEVEL_MOVER", "MOVER", "primary_absolute_change", "prior_first_MAP_per_10mmHg", True),
    ("ASSOC_EARLY_FALL_INSPIRE", "INSPIRE", "primary_absolute_change", "prior_MAP_change_per_10mmHg", True),
    ("ASSOC_EARLY_FALL_MOVER", "MOVER", "primary_absolute_change", "prior_MAP_change_per_10mmHg", True),
    ("MOVER_DRUG_TIMED_LEVEL", "MOVER", "strict_post_hypnotic", "prior_post_MAP_per_10", True),
    ("MOVER_DRUG_TIMED_FALL", "MOVER", "strict_post_hypnotic", "prior_post_change_per_10", True),
)


def one(frame: pd.DataFrame, **selector: object) -> pd.Series:
    selected = frame
    for column, value in selector.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise RuntimeError(f"expected one row for {selector}; found {len(selected)}")
    return selected.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-map", required=True, type=Path)
    parser.add_argument("--r-results", required=True, type=Path)
    args = parser.parse_args()

    result_map = pd.read_csv(args.result_map)
    r_results = pd.read_csv(args.r_results)
    for analysis_id, centre, model, term, invert in COMPARISONS:
        expected = one(result_map, analysis_id=analysis_id)
        observed = one(r_results, centre=centre, model=model, term=term)
        if invert:
            point = 1.0 / observed.odds_ratio
            low = 1.0 / observed.ci_high
            high = 1.0 / observed.ci_low
        else:
            point, low, high = observed.odds_ratio, observed.ci_low, observed.ci_high
        for label, left, right in (
            ("estimate", expected.estimate, point),
            ("ci_low", expected.ci_low, low),
            ("ci_high", expected.ci_high, high),
        ):
            if not math.isclose(float(left), float(right), rel_tol=2e-6, abs_tol=2e-6):
                raise RuntimeError(
                    f"independent replication mismatch: {analysis_id}/{label}: "
                    f"map={left}; R={right}"
                )
        if int(expected.n) != int(observed.n) or int(expected.events) != int(observed.events):
            raise RuntimeError(f"independent replication count mismatch: {analysis_id}")
    print(f"INDEPENDENT_REPLICATION_VERIFY_PASS rows={len(COMPARISONS)}")


if __name__ == "__main__":
    main()
