#!/usr/bin/env python3
"""Build the conventional two-centre clinical association table for C02.

The script uses the same paired cohorts, variable definitions, median
imputation, and patient-clustered sandwich standard errors as the existing
harmonized association analysis.  It adds one-predictor models for the
univariable columns and retains every coefficient from the full eight-variable
model.  Only aggregate coefficient results are written.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from analyze_c02_deepening_v1 import add_response_terms, canonicalize_inspire
from c02_cluster_logit import fit_clustered_logit
from run_mover_c02_external_validation import canonicalize


from c02_runtime import private_workspace_root


ROOT = private_workspace_root()
BASE = ROOT / "outputs/inspire_curiosity/statistical_strictness_review/c02_clean_rebuild"
INSPIRE = BASE / "first_two_sampling_bridge/first_two_pair_cohort.csv.gz"
MOVER = ROOT / "data/restricted/mover/extracted/mover_c02_cleaned_pair_cohort.csv.gz"
REFERENCE = BASE / "deepening_v1/two_centre_harmonized_associations.csv"
OUT = BASE / "final_submission_core/reporting_v2/table2_univariable_multivariable_associations.csv"

TERMS = [
    "age_per_10y",
    "bmi_per_5",
    "asa_class",
    "male",
    "log1p_interval_days",
    "prior_binary_alert",
    "prior_first_MAP_per_10mmHg",
    "prior_MAP_change_per_10mmHg",
]

# Multipliers convert fitted log-odds coefficients to the direction and unit
# shown in the manuscript.  The interval term is reported per doubling of
# (interval days + 1), which is equivalent to multiplying its coefficient by
# ln(2) and does not alter the fitted model.
DISPLAY_MULTIPLIER = {
    "age_per_10y": 1.0,
    "bmi_per_5": 1.0,
    "asa_class": 1.0,
    "male": 1.0,
    "log1p_interval_days": math.log(2.0),
    "prior_binary_alert": 1.0,
    "prior_first_MAP_per_10mmHg": -1.0,
    "prior_MAP_change_per_10mmHg": -1.0,
}

LABELS = {
    "age_per_10y": "年龄（每增加10岁）",
    "bmi_per_5": "BMI（每增加5 kg/m²）",
    "asa_class": "ASA分级（每升高1级）",
    "male": "男性（vs 女性）",
    "log1p_interval_days": "两次手术间隔（天数+1每增加一倍）",
    "prior_binary_alert": "既往二分类提醒（有 vs 无）",
    "prior_first_MAP_per_10mmHg": "既往首次MAP（每降低10 mmHg）",
    "prior_MAP_change_per_10mmHg": "既往早期MAP（每进一步下降10 mmHg）",
}


def common_frame(d: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reproduce the exact common-covariate frame from the primary analysis."""
    common = pd.DataFrame(
        {
            "age_per_10y": pd.to_numeric(d["age_years"], errors="coerce") / 10,
            "bmi_per_5": pd.to_numeric(d["bmi_kg_m2"], errors="coerce") / 5,
            "asa_class": pd.to_numeric(d["asa_numeric"], errors="coerce"),
            "male": d["sex_common"].astype(str).str.upper().eq("M").astype(float),
            "log1p_interval_days": pd.to_numeric(d["interval_log1p"], errors="coerce"),
            "prior_binary_alert": pd.to_numeric(
                d["prior_first2_any_low"], errors="coerce"
            ),
            "prior_first_MAP_per_10mmHg": pd.to_numeric(
                d["prior_first_map"], errors="coerce"
            )
            / 10,
            "prior_MAP_change_per_10mmHg": pd.to_numeric(
                d["prior_first2_change"], errors="coerce"
            )
            / 10,
        }
    )
    missing = {column: int(common[column].isna().sum()) for column in TERMS}
    for column in TERMS:
        common[column] = common[column].fillna(common[column].median())
    if not np.isfinite(common[TERMS].to_numpy(float)).all():
        raise RuntimeError("non-finite value after harmonized median imputation")
    return common, missing


def fit_cluster_logistic(
    common: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    columns: list[str],
) -> dict[str, tuple[float, float, float]]:
    """Fit an unpenalized logistic model with patient-clustered SEs."""
    fitted = fit_clustered_logit(common, y, groups, columns)
    return {
        row.term: (
            float(row.log_odds),
            float(row.cluster_se),
            float(row.p_value),
        )
        for row in fitted.itertuples(index=False)
    }


def transformed_result(
    estimate: float, standard_error: float, p_value: float, multiplier: float
) -> dict[str, float]:
    displayed_estimate = multiplier * estimate
    displayed_se = abs(multiplier) * standard_error
    return {
        "log_odds": displayed_estimate,
        "standard_error": displayed_se,
        "odds_ratio": math.exp(displayed_estimate),
        "ci_low": math.exp(displayed_estimate - 1.959963984540054 * displayed_se),
        "ci_high": math.exp(displayed_estimate + 1.959963984540054 * displayed_se),
        "p_value_cluster_robust": p_value,
    }


def analyse_centre(
    centre: str,
    d: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    common, missing = common_frame(d)
    adjusted = fit_cluster_logistic(common, y, groups, TERMS)
    rows: list[dict[str, object]] = []
    for order, term in enumerate(TERMS, start=1):
        univariable = fit_cluster_logistic(common, y, groups, [term])[term]
        uni = transformed_result(*univariable, DISPLAY_MULTIPLIER[term])
        adj = transformed_result(*adjusted[term], DISPLAY_MULTIPLIER[term])
        rows.append(
            {
                "centre": centre,
                "variable_order": order,
                "term": term,
                "variable_label_zh": LABELS[term],
                "univariable_odds_ratio": uni["odds_ratio"],
                "univariable_ci_low": uni["ci_low"],
                "univariable_ci_high": uni["ci_high"],
                "univariable_p_value": uni["p_value_cluster_robust"],
                "adjusted_odds_ratio": adj["odds_ratio"],
                "adjusted_ci_low": adj["ci_low"],
                "adjusted_ci_high": adj["ci_high"],
                "adjusted_p_value": adj["p_value_cluster_robust"],
                "n": int(len(y)),
                "events": int(y.sum()),
                "patients": int(pd.Series(groups).nunique()),
                "missing_before_median_imputation": missing[term],
            }
        )
    return pd.DataFrame(rows)


def verify_primary_coefficients(result: pd.DataFrame) -> None:
    """Guard against accidental drift from the already adjudicated analysis."""
    reference = pd.read_csv(REFERENCE)
    reference = reference.loc[
        reference["specification"].eq("absolute_change_model")
    ].copy()
    primary = [
        "prior_binary_alert",
        "prior_first_MAP_per_10mmHg",
        "prior_MAP_change_per_10mmHg",
    ]
    for centre in ["INSPIRE", "MOVER"]:
        for term in primary:
            expected = reference.loc[
                reference["centre"].eq(centre) & reference["term"].eq(term)
            ].iloc[0]
            observed = result.loc[
                result["centre"].eq(centre) & result["term"].eq(term)
            ].iloc[0]
            if term == "prior_binary_alert":
                expected_or = float(expected.odds_ratio)
                expected_low = float(expected.ci_low)
                expected_high = float(expected.ci_high)
            else:
                expected_or = 1 / float(expected.odds_ratio)
                expected_low = 1 / float(expected.ci_high)
                expected_high = 1 / float(expected.ci_low)
            comparisons = {
                "OR": (observed.adjusted_odds_ratio, expected_or),
                "CI low": (observed.adjusted_ci_low, expected_low),
                "CI high": (observed.adjusted_ci_high, expected_high),
                "P": (observed.adjusted_p_value, float(expected.p_value_cluster_robust)),
            }
            for label, (actual, target) in comparisons.items():
                if not math.isclose(float(actual), float(target), rel_tol=1e-8, abs_tol=1e-10):
                    raise RuntimeError(
                        f"primary coefficient drift: {centre}/{term}/{label}: "
                        f"{actual} != {target}"
                    )


def main() -> None:
    inspire = canonicalize_inspire(pd.read_csv(INSPIRE, low_memory=False))
    mover = add_response_terms(canonicalize(pd.read_csv(MOVER, low_memory=False)))
    result = pd.concat(
        [
            analyse_centre(
                "INSPIRE",
                inspire,
                inspire["target"].to_numpy(int),
                inspire["subject_id"].to_numpy(),
            ),
            analyse_centre(
                "MOVER",
                mover,
                mover["target_any_low_first2"].to_numpy(int),
                mover["patient_id"].to_numpy(),
            ),
        ],
        ignore_index=True,
    )
    verify_primary_coefficients(result)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
