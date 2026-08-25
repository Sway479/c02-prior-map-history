#!/usr/bin/env python3
"""Run the paper-facing C02 analysis stages in a protected workspace."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency diagnostic
        raise SystemExit(
            "Python below 3.11 requires tomli. Install requirements.txt first."
        ) from exc


CODE_ROOT = Path(__file__).resolve().parent
ANALYSIS = CODE_ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS))

from c02_runtime import (  # noqa: E402
    configure_private_workspace,
    protect_file,
    require_private_path,
    secure_directory,
)


STAGES = (
    "primary",
    "drug_timed",
    "burden_recovery",
    "history",
    "ml",
    "export",
    "validate",
    "all",
)

STAGE_INPUTS = {
    "primary": ("inspire_pair_cohort", "mover_pair_cohort"),
    "drug_timed": (
        "mover_pair_cohort",
        "mover_map_stream",
        "mover_hypnotic_mar",
        "mover_patient_information",
    ),
    "burden_recovery": ("mover_full_map_stream",),
    "history": (
        "mover_map_stream",
        "mover_hypnotic_mar",
        "mover_vasopressor_mar",
        "mover_epic_archive",
        "inspire_history_pairs",
        "inspire_exclusion_flags",
        "inspire_triplet_conflict_audit",
    ),
    "ml": ("inspire_pair_cohort", "mover_pair_cohort"),
    "export": (),
    "validate": ("inspire_pair_cohort", "mover_pair_cohort"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=STAGES, default="primary")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    paths = config.get("paths", {})
    required = ("private_workspace", "private_output_root", "private_derived_root")
    missing = [key for key in required if not paths.get(key)]
    if missing:
        raise RuntimeError(f"missing [paths] entries: {', '.join(missing)}")
    configure_private_workspace(
        Path(paths["private_workspace"]),
        paths.get("allowed_private_real_roots", []),
    )
    return config


def configured_path(config: dict, key: str, *, must_exist: bool) -> Path:
    value = config.get("paths", {}).get(key)
    if not value:
        raise RuntimeError(f"missing [paths] entry: {key}")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(config["paths"]["private_workspace"]).expanduser() / candidate
    return require_private_path(candidate, must_exist=must_exist)


def output_root(config: dict) -> Path:
    return secure_directory(configured_path(config, "private_output_root", must_exist=False))


def derived_root(config: dict) -> Path:
    return secure_directory(configured_path(config, "private_derived_root", must_exist=False))


def release_root(config: dict) -> Path:
    if config.get("paths", {}).get("aggregate_release_root"):
        return secure_directory(
            configured_path(config, "aggregate_release_root", must_exist=False)
        )
    return secure_directory(output_root(config) / "aggregate_release_candidate")


def stages_for(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return (
            "primary",
            "drug_timed",
            "burden_recovery",
            "history",
            "ml",
            "export",
            "validate",
        )
    return (stage,)


def describe(config: dict, stage: str) -> dict:
    selected = stages_for(stage)
    required_keys = sorted({key for item in selected for key in STAGE_INPUTS[item]})
    inputs = {
        key: str(configured_path(config, key, must_exist=False)) for key in required_keys
    }
    return {
        "stages": list(selected),
        "private_workspace": str(
            configured_path(config, "private_workspace", must_exist=False)
        ),
        "private_output_root": str(
            configured_path(config, "private_output_root", must_exist=False)
        ),
        "private_derived_root": str(
            configured_path(config, "private_derived_root", must_exist=False)
        ),
        "required_inputs": inputs,
        "privacy_boundary": (
            "The private output root contains row-level OOF predictions, fitted "
            "models and bootstrap artifacts. Publish only reviewed aggregate exports."
        ),
    }


def require_stage_inputs(config: dict, stage: str) -> None:
    for key in STAGE_INPUTS[stage]:
        configured_path(config, key, must_exist=True)


def preflight_primary_cohorts(config: dict) -> dict[str, dict[str, int]]:
    """Reject cohort or unit drift before any paper-facing primary model."""
    checker = importlib.import_module("c02_preflight")
    return checker.validate_primary_cohorts(
        configured_path(config, "inspire_pair_cohort", must_exist=True),
        configured_path(config, "mover_pair_cohort", must_exist=True),
    )


def run_primary(config: dict) -> None:
    require_stage_inputs(config, "primary")
    preflight_primary_cohorts(config)
    inspire = configured_path(config, "inspire_pair_cohort", must_exist=True)
    mover = configured_path(config, "mover_pair_cohort", must_exist=True)
    output = output_root(config)

    bridge = importlib.import_module("run_c02_cross_database_minimal_bridge")
    bridge.INPUT = inspire
    bridge.OUT = output / "cross_database_minimal_bridge"
    bridge.main()

    validation = importlib.import_module("run_mover_c02_external_validation")
    validation.COHORT = mover
    validation.INSPIRE_MODELS = bridge.OUT
    validation.OUT = output / "mover_external_validation"
    validation.main()

    deepening = importlib.import_module("analyze_c02_deepening_v1")
    deepening.INSPIRE = inspire
    deepening.MOVER = mover
    deepening.FIXED_MODELS = bridge.OUT
    deepening.OUT = output / "deepening_v1"
    deepening.main()

    gradient = importlib.import_module("analyze_c02_subalert_gradient")
    gradient.INSPIRE = inspire
    gradient.MOVER = mover
    gradient.OUT = output / "subalert_gradient"
    gradient.main()

    associations = importlib.import_module(
        "build_c02_univariable_multivariable_associations"
    )
    associations.INSPIRE = inspire
    associations.MOVER = mover
    associations.REFERENCE = deepening.OUT / "two_centre_harmonized_associations.csv"
    associations.OUT = (
        output / "reporting/table2_univariable_multivariable_associations.csv"
    )
    associations.main()


def run_drug_timed(config: dict) -> None:
    require_stage_inputs(config, "drug_timed")
    output = output_root(config)
    derived = derived_root(config)
    mover_pair = configured_path(config, "mover_pair_cohort", must_exist=True)
    map_stream = configured_path(config, "mover_map_stream", must_exist=True)
    hypnotic_mar = configured_path(config, "mover_hypnotic_mar", must_exist=True)
    patient_info = configured_path(config, "mover_patient_information", must_exist=True)

    anchored_pair = derived / "mover_c02_hypnotic_anchored_pair.csv.gz"
    strict_pair = derived / "mover_c02_hypnotic_strict_post_pair.csv.gz"
    relative_pair = derived / "mover_c02_relative_hypnotic_pair.csv.gz"

    anchored = importlib.import_module("analyze_c02_hypnotic_anchored_reproducibility")
    anchored.PAIR = mover_pair
    anchored.MAP = map_stream
    anchored.MAR = hypnotic_mar
    anchored.RESTRICTED = anchored_pair
    anchored.OUT = output / "hypnotic_anchored"
    anchored.main()

    strict_builder = importlib.import_module("prepare_mover_strict_postminute_pairs")
    strict_builder.build_strict_pair(
        mover_pair,
        map_stream,
        hypnotic_mar,
        strict_pair,
        output / "hypnotic_anchored/strict_postminute",
    )

    strict = importlib.import_module("analyze_c02_strict_postminute_sensitivity")
    strict.PAIR = strict_pair
    strict.PATIENT_INFO = patient_info
    strict.OUT = output / "hypnotic_anchored/strict_postminute"
    strict.main()

    relative = importlib.import_module("analyze_c02_relative_hypnotic_response")
    relative.PAIR = mover_pair
    relative.MAP = map_stream
    relative.MAR = hypnotic_mar
    relative.RESTRICTED = relative_pair
    relative.OUT = output / "hypnotic_anchored/relative_response"
    relative.main()

    current = importlib.import_module("analyze_c02_current_baseline_increment")
    current.PAIR = relative_pair
    current.OUT = output / "hypnotic_anchored/current_baseline_increment"
    current.main()


def run_burden_recovery(config: dict) -> None:
    require_stage_inputs(config, "burden_recovery")
    output = output_root(config)
    derived = derived_root(config)
    full_map = configured_path(config, "mover_full_map_stream", must_exist=True)
    anchored_pair = require_private_path(derived / "mover_c02_hypnotic_anchored_pair.csv.gz")
    relative_pair = require_private_path(derived / "mover_c02_relative_hypnotic_pair.csv.gz")

    burden = importlib.import_module("analyze_c02_post_hypnotic_burden")
    burden.MAP_PATH = full_map
    burden.ANCHOR_PAIR = anchored_pair
    burden.RELATIVE_PAIR = relative_pair
    burden.RESTRICTED_METRICS = derived / "mover_c02_posthypnotic_burden.csv.gz"
    burden.RESTRICTED_RELATIVE_METRICS = (
        derived / "mover_c02_posthypnotic_relative_burden.csv.gz"
    )
    burden.OUT = output / "post_hypnotic_burden"
    burden.main()

    recovery = importlib.import_module("analyze_c02_post_hypnotic_recovery")
    recovery.RESTRICTED = derived / "mover_c02_posthypnotic_recovery.csv.gz"
    recovery.EXISTING_METRICS = burden.RESTRICTED_METRICS
    recovery.BURDEN = burden.OUT
    recovery.OUT = output / "post_hypnotic_recovery"
    recovery.main()


def run_history(config: dict) -> None:
    require_stage_inputs(config, "history")
    output = output_root(config)
    derived = derived_root(config)

    memory = importlib.import_module("analyze_c02_multiepisode_memory")
    memory.MOVER_PAIR = require_private_path(
        derived / "mover_c02_relative_hypnotic_pair.csv.gz"
    )
    memory.INSPIRE_PAIR = configured_path(config, "inspire_history_pairs", must_exist=True)
    memory.INSPIRE_EXCLUSION = configured_path(
        config, "inspire_exclusion_flags", must_exist=True
    )
    memory.INSPIRE_TRIPLET_CONFLICT_AUDIT = configured_path(
        config, "inspire_triplet_conflict_audit", must_exist=True
    )

    management = importlib.import_module(
        "analyze_c02_multiepisode_management_instability"
    )
    management.PAIR = memory.MOVER_PAIR
    management.MAP = configured_path(config, "mover_map_stream", must_exist=True)
    management.HYPNOTIC_MAR = configured_path(
        config, "mover_hypnotic_mar", must_exist=True
    )
    management.VASOPRESSOR_MAR = configured_path(
        config, "mover_vasopressor_mar", must_exist=True
    )

    workflow = importlib.import_module("analyze_c02_clinical_workflow_deepening")
    workflow.MOVER_ARCHIVE = configured_path(config, "mover_epic_archive", must_exist=True)
    workflow.OUT = output / "clinical_workflow_deepening"
    workflow.main()


def run_ml(config: dict) -> None:
    require_stage_inputs(config, "ml")
    preflight_primary_cohorts(config)
    module = importlib.import_module("analyze_c02_ml_modeling")
    module.INSPIRE = configured_path(config, "inspire_pair_cohort", must_exist=True)
    module.MOVER = configured_path(config, "mover_pair_cohort", must_exist=True)
    module.OUT = output_root(config) / "machine_learning_attempt"
    module.main()


def run_export(config: dict) -> Path:
    exporter = importlib.import_module("export_authoritative_result_map")
    result = exporter.build_result_map(output_root(config))
    destination = release_root(config) / "authoritative_result_map.csv"
    result.to_csv(destination, index=False)
    protect_file(destination)
    return destination


def run_validate(config: dict) -> None:
    require_stage_inputs(config, "validate")
    preflight_primary_cohorts(config)
    result_map = release_root(config) / "authoritative_result_map.csv"
    if not result_map.is_file():
        result_map = run_export(config)
    verifier = importlib.import_module("verify_authoritative_result_map")
    verifier.verify_result_map(pd.read_csv(result_map))

    strict_pair = require_private_path(
        derived_root(config) / "mover_c02_hypnotic_strict_post_pair.csv.gz"
    )
    r_output = secure_directory(output_root(config) / "independent_r_validation")
    command = [
        "Rscript",
        str(CODE_ROOT / "validation/audit_core_associations_independent.R"),
        str(configured_path(config, "inspire_pair_cohort", must_exist=True)),
        str(configured_path(config, "mover_pair_cohort", must_exist=True)),
        str(strict_pair),
        str(r_output),
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "validation/verify_independent_replication.py"),
            "--result-map",
            str(result_map),
            "--r-results",
            str(r_output / "independent_r_cluster_logit.csv"),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    plan = describe(config, args.stage)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    stage_runners = {
        "primary": run_primary,
        "drug_timed": run_drug_timed,
        "burden_recovery": run_burden_recovery,
        "history": run_history,
        "ml": run_ml,
        "export": run_export,
        "validate": run_validate,
    }
    for stage in stages_for(args.stage):
        stage_runners[stage](config)
    print(json.dumps({**plan, "status": "COMPLETE"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
