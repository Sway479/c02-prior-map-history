#!/usr/bin/env python3
"""Synthetic fail-closed checks for the two corrected provenance errors."""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS))


def main() -> None:
    private_root = tempfile.TemporaryDirectory(prefix="c02-unit-test-")
    os.environ["C02_PRIVATE_WORKSPACE"] = private_root.name
    from prepare_inspire_pairs import minute_duration, minutes_to_days
    from c02_cluster_logit import fit_clustered_logit
    from c02_preflight import validate_primary_cohorts
    from export_authoritative_result_map import EXPECTED_IDS, verify_result_map

    assert minutes_to_days(1440.0) == 1.0
    assert minutes_to_days(60.0) == 1.0 / 24.0
    assert minute_duration(10.0, 70.0) == 60.0

    inspire_analysis_n = 9306
    inspire = pd.DataFrame(
        {
            "subject_id": np.arange(inspire_analysis_n) % 7372,
            "target_any_low": np.r_[
                np.ones(1127), np.zeros(inspire_analysis_n - 1127)
            ],
            "antype": "General",
            "prior_antype": "General",
            "interval_days": 1.0,
            "prior_an_duration_min": 60.0,
            "current_anstart_time": 1500.0,
            "prior_anstart_time": 0.0,
            "prior_anend_time": 60.0,
        }
    )
    ignored_n = 13231 - inspire_analysis_n
    ignored = pd.DataFrame(
        {
            "subject_id": np.arange(ignored_n),
            "target_any_low": 0,
            "antype": "Other",
            "prior_antype": "General",
            "interval_days": 1.0,
            "prior_an_duration_min": 60.0,
            "current_anstart_time": 1500.0,
            "prior_anstart_time": 0.0,
            "prior_anend_time": 60.0,
        }
    )
    inspire = pd.concat([inspire, ignored], ignore_index=True)
    mover_n = 7721
    mover = pd.DataFrame(
        {
            "patient_id": np.arange(mover_n) % 5297,
            "target_any_low_first2": np.r_[np.ones(240), np.zeros(mover_n - 240)],
            "adjacent_order_valid": True,
            "general_to_general": True,
            "interval_days": 1.0,
            "anstart": "2020-01-02 00:00:00",
            "prior_anstop": "2020-01-01 00:00:00",
        }
    )
    inspire_path = Path(private_root.name) / "inspire.csv.gz"
    mover_path = Path(private_root.name) / "mover.csv.gz"
    inspire.to_csv(inspire_path, index=False)
    mover.to_csv(mover_path, index=False)
    assert validate_primary_cohorts(inspire_path, mover_path) == {
        "INSPIRE": {"pairs": 9306, "patients": 7372, "events": 1127},
        "MOVER": {"pairs": 7721, "patients": 5297, "events": 240},
    }
    extra_ignored = pd.concat([inspire, ignored.iloc[[0]]], ignore_index=True)
    extra_ignored.to_csv(inspire_path, index=False)
    try:
        validate_primary_cohorts(inspire_path, mover_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unexpected ignored INSPIRE source row was not rejected")
    bad_inspire = inspire.copy()
    bad_inspire.loc[0, "interval_days"] = 24.0
    bad_inspire.to_csv(inspire_path, index=False)
    try:
        validate_primary_cohorts(inspire_path, mover_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("INSPIRE interval-unit drift was not rejected")

    frame = pd.DataFrame({"x": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] * 4})
    outcome = np.array(
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
         0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
    )
    groups = np.repeat(np.arange(12), 2)
    fitted = fit_clustered_logit(frame, outcome, groups, ["x"])
    assert bool(fitted.fit_converged.all())
    assert np.isfinite(fitted[["odds_ratio", "ci_low", "ci_high"]].to_numpy()).all()

    strict_source = (ANALYSIS / "analyze_c02_strict_postminute_sensitivity.py").read_text(
        encoding="utf-8"
    )
    assert "association = clustered_logit(association_input)" in strict_source
    assert "association_input = d.assign(" in strict_source
    assert '"association_source": "same strict-postminute cohort"' in strict_source
    builder_source = (
        ANALYSIS / "prepare_mover_strict_postminute_pairs.py"
    ).read_text(encoding="utf-8")
    assert "strictly_after_anchor=True" in builder_source
    assert 'strict["cur_r0"].gt(0).all()' in builder_source
    assert "prepare_pair(operation, pair)" in builder_source
    assert "expected_counts = (4871, 3645, 567)" in builder_source

    synthetic_map = pd.DataFrame(
        {
            "analysis_id": EXPECTED_IDS,
            "estimate": 0.5,
            "ci_low": 0.4,
            "ci_high": 0.6,
            "n": 100,
            "events": 20,
        }
    )
    strict_rows = synthetic_map.analysis_id.str.startswith("MOVER_DRUG_TIMED")
    synthetic_map.loc[strict_rows, ["n", "events"]] = [4871, 567]
    synthetic_map["patients"] = pd.NA
    synthetic_map.loc[strict_rows, "patients"] = 3645
    synthetic_map.loc[
        synthetic_map.analysis_id.eq("MOVER_DRUG_TIMED_LEVEL"),
        ["estimate", "ci_low", "ci_high"],
    ] = [1.267, 1.18, 1.36]
    synthetic_map.loc[
        synthetic_map.analysis_id.eq("MOVER_DRUG_TIMED_FALL"),
        ["estimate", "ci_low", "ci_high"],
    ] = [1.144, 1.08, 1.22]
    synthetic_map.loc[synthetic_map.analysis_id.eq("PAIR_INSPIRE"), "estimate"] = 0.03465
    synthetic_map.loc[synthetic_map.analysis_id.eq("PAIR_INSPIRE"), "ci_low"] = 0.02
    synthetic_map.loc[synthetic_map.analysis_id.eq("PAIR_INSPIRE"), "ci_high"] = 0.05
    verify_result_map(synthetic_map)
    bad_map = synthetic_map.copy()
    bad_map.loc[bad_map.analysis_id.eq("MOVER_DRUG_TIMED_LEVEL"), "n"] = 4892
    try:
        verify_result_map(bad_map)
    except RuntimeError:
        pass
    else:
        raise AssertionError("mixed strict-postminute provenance was not rejected")
    private_root.cleanup()
    print("UNIT_AND_PROVENANCE_TEST_PASS")


if __name__ == "__main__":
    main()
