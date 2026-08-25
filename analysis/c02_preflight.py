#!/usr/bin/env python3
"""Fail-closed provenance checks for the two frozen primary C02 cohorts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COUNTS = {
    "INSPIRE": {"pairs": 9306, "patients": 7372, "events": 1127},
    "MOVER": {"pairs": 7721, "patients": 5297, "events": 240},
}
EXPECTED_SOURCE_ROWS = {"INSPIRE": 13231, "MOVER": 7721}


def _read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read only the protected columns needed for provenance checks."""
    try:
        return pd.read_csv(path, usecols=columns)
    except ValueError as exc:
        raise RuntimeError(
            f"required cohort columns are missing from {path.name}: {exc}"
        ) from exc


def _numeric(series: pd.Series, *, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise RuntimeError(f"{label} contains missing or non-numeric values")
    return values


def _binary_events(series: pd.Series, *, label: str) -> int:
    values = _numeric(series, label=label)
    if not values.isin([0, 1]).all():
        raise RuntimeError(f"{label} is not binary")
    return int(values.sum())


def _check_counts(
    centre: str,
    *,
    pairs: int,
    patients: int,
    events: int,
) -> dict[str, int]:
    observed = {"pairs": pairs, "patients": patients, "events": events}
    expected = EXPECTED_COUNTS[centre]
    if observed != expected:
        raise RuntimeError(
            f"{centre} frozen-cohort count gate failed: "
            f"observed={observed}; expected={expected}"
        )
    return observed


def validate_inspire_cohort(path: Path) -> dict[str, int]:
    """Verify the corrected INSPIRE general-to-general analytic cohort."""
    columns = [
        "subject_id",
        "target_any_low",
        "antype",
        "prior_antype",
        "interval_days",
        "prior_an_duration_min",
        "current_anstart_time",
        "prior_anstart_time",
        "prior_anend_time",
    ]
    frame = _read_columns(path, columns)
    if len(frame) != EXPECTED_SOURCE_ROWS["INSPIRE"]:
        raise RuntimeError(
            "INSPIRE frozen source-row gate failed: "
            f"observed={len(frame)}; expected={EXPECTED_SOURCE_ROWS['INSPIRE']}"
        )
    general = (
        frame["antype"].astype("string").str.strip().str.casefold().eq("general")
        & frame["prior_antype"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("general")
    )
    cohort = frame.loc[general].copy()
    if cohort["subject_id"].isna().any():
        raise RuntimeError("INSPIRE subject_id contains missing values")

    current_start = _numeric(
        cohort["current_anstart_time"], label="INSPIRE current_anstart_time"
    )
    prior_start = _numeric(
        cohort["prior_anstart_time"], label="INSPIRE prior_anstart_time"
    )
    prior_end = _numeric(
        cohort["prior_anend_time"], label="INSPIRE prior_anend_time"
    )
    interval = _numeric(cohort["interval_days"], label="INSPIRE interval_days")
    duration = _numeric(
        cohort["prior_an_duration_min"], label="INSPIRE prior_an_duration_min"
    )
    expected_interval = (current_start - prior_end) / 1440.0
    expected_duration = prior_end - prior_start
    if not np.allclose(interval, expected_interval, rtol=0.0, atol=1e-9):
        raise RuntimeError("INSPIRE interval_days is not derived from minutes / 1440")
    if not np.allclose(duration, expected_duration, rtol=0.0, atol=1e-9):
        raise RuntimeError("INSPIRE prior_an_duration_min is not a minute duration")
    if not interval.gt(0).all() or not duration.gt(0).all():
        raise RuntimeError("INSPIRE interval or prior anaesthetic duration is non-positive")

    return _check_counts(
        "INSPIRE",
        pairs=len(cohort),
        patients=int(cohort["subject_id"].nunique()),
        events=_binary_events(cohort["target_any_low"], label="INSPIRE target_any_low"),
    )


def _all_true(series: pd.Series, *, label: str) -> None:
    normalized = series.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().casefold() in {"1", "true"}
    )
    if series.isna().any() or not normalized.all():
        raise RuntimeError(f"{label} is not true for every MOVER pair")


def validate_mover_cohort(path: Path) -> dict[str, int]:
    """Verify the frozen adjacent general-to-general MOVER analytic cohort."""
    columns = [
        "patient_id",
        "target_any_low_first2",
        "adjacent_order_valid",
        "general_to_general",
        "interval_days",
        "anstart",
        "prior_anstop",
    ]
    cohort = _read_columns(path, columns)
    if len(cohort) != EXPECTED_SOURCE_ROWS["MOVER"]:
        raise RuntimeError(
            "MOVER frozen source-row gate failed: "
            f"observed={len(cohort)}; expected={EXPECTED_SOURCE_ROWS['MOVER']}"
        )
    if cohort["patient_id"].isna().any():
        raise RuntimeError("MOVER patient_id contains missing values")
    _all_true(cohort["adjacent_order_valid"], label="adjacent_order_valid")
    _all_true(cohort["general_to_general"], label="general_to_general")

    current_start = pd.to_datetime(cohort["anstart"], errors="coerce")
    prior_end = pd.to_datetime(cohort["prior_anstop"], errors="coerce")
    if current_start.isna().any() or prior_end.isna().any():
        raise RuntimeError("MOVER anaesthetic timestamps contain missing or invalid values")
    interval = _numeric(cohort["interval_days"], label="MOVER interval_days")
    expected_interval = (current_start - prior_end).dt.total_seconds() / 86400.0
    if not np.allclose(interval, expected_interval, rtol=0.0, atol=1e-9):
        raise RuntimeError("MOVER interval_days does not match the source timestamps")
    if not interval.gt(0).all():
        raise RuntimeError("MOVER interval_days contains non-positive values")

    return _check_counts(
        "MOVER",
        pairs=len(cohort),
        patients=int(cohort["patient_id"].nunique()),
        events=_binary_events(
            cohort["target_any_low_first2"], label="MOVER target_any_low_first2"
        ),
    )


def validate_primary_cohorts(
    inspire_path: Path, mover_path: Path
) -> dict[str, dict[str, int]]:
    """Run both frozen-cohort gates without exposing row-level values."""
    return {
        "INSPIRE": validate_inspire_cohort(inspire_path),
        "MOVER": validate_mover_cohort(mover_path),
    }
