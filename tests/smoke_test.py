#!/usr/bin/env python3
"""Small code-only smoke test using synthetic records."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS))


def require_import(filename: str) -> None:
    path = ANALYSIS / filename
    if not path.exists():
        raise AssertionError(f"missing analysis script: {filename}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot create import specification for {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def synthetic_metric_check() -> None:
    from run_mover_c02_external_validation import canonicalize

    frame = pd.DataFrame(
        {
            "age_years": [40, 55, 70, 62],
            "bmi_kg_m2": [22.0, 27.5, np.nan, 31.0],
            "asa_numeric": [1, 2, 3, 2],
            "prior_asa_numeric": [1, 2, 2, 2],
            "interval_log1p": np.log1p([10, 40, 90, 15]),
            "prior_first2_any_low": [0, 0, 1, 1],
            "prior_first_map": [90, 78, 60, 66],
            "prior_first2_change": [-2, -10, -15, 4],
            "target_any_low_first2": [0, 0, 1, 1],
            "sex_common": ["F", "M", "F", "M"],
            "patient_class_common": ["outpatient"] * 4,
            "procedure_common": ["other"] * 4,
            "prior_patient_class_common": ["outpatient"] * 4,
            "prior_procedure_common": ["other"] * 4,
        }
    )
    out = canonicalize(frame)
    assert len(out) == 4
    assert out["target_any_low_first2"].tolist() == [0, 0, 1, 1]


def main() -> None:
    private_root = tempfile.TemporaryDirectory(prefix="c02-code-smoke-")
    os.environ["C02_PRIVATE_WORKSPACE"] = private_root.name
    required = [
        "c02_preflight.py",
        "run_c02_cross_database_minimal_bridge.py",
        "run_mover_c02_external_validation.py",
        "analyze_c02_deepening_v1.py",
        "analyze_c02_subalert_gradient.py",
        "analyze_c02_hypnotic_anchored_reproducibility.py",
        "analyze_c02_post_hypnotic_burden.py",
        "analyze_c02_post_hypnotic_recovery.py",
        "analyze_c02_clinical_workflow_deepening.py",
        "prepare_mover_strict_postminute_pairs.py",
        "extract_mover_early_vasopressor_mar.py",
        "export_authoritative_result_map.py",
        "verify_authoritative_result_map.py",
    ]
    for filename in required:
        require_import(filename)
    synthetic_metric_check()
    private_root.cleanup()
    print("SMOKE_TEST_PASS")


if __name__ == "__main__":
    main()
