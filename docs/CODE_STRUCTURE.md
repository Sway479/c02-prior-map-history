# Code structure

- `run_pipeline.py`: the only full manuscript orchestrator; configures the
  private workspace before importing analysis modules.
- `analysis/c02_runtime.py`: fail-closed path and permission boundary.
- `analysis/c02_cluster_logit.py`: clustered logistic inference.
- `analysis/c02_preflight.py`: frozen cohort, event-count, timestamp, and unit
  gates for the two primary databases.
- `analysis/prepare_*.py`, `analysis/extract_*.py`, and
  `analysis/build_mover_c02_pair_cohort.py`: protected input construction.
- `analysis/run_*.py` and `analysis/analyze_*.py`: paper-facing and support
  analyses, grouped by the stages in `METHODS_MAP.md`.
- `analysis/export_authoritative_result_map.py`: source-row selection and the
  fixed 22-result contract.
- `analysis/verify_authoritative_result_map.py`: independent file-level result
  map gate.
- `validation/`: base-R coefficient replication.
- `tests/`: synthetic import, unit, correction, and provenance gates.
- `tools/check_release.py`: code-only release scanner.

Output and manuscript assembly code is deliberately absent. The public package
generates scientific evidence; it does not rewrite Word files or reproduce
submission layout.
