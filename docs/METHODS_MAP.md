# Methods-to-code map

This index distinguishes protected cohort construction, paper-facing analysis,
shared support code, and release validation.

| Component | Stage or script | Output boundary |
|---|---|---|
| INSPIRE adjacent-pair audit | `analysis/prepare_inspire_pairs.py` | protected pairs; minute-valued timestamps use `/1440` for days and no second duration conversion |
| MOVER MAP extraction | `analysis/extract_mover_map_stream.py` | protected MAP stream plus aggregate extraction receipt |
| MOVER hypnotic and vasopressor extraction | `analysis/extract_mover_early_anesthetic_mar.py`, `analysis/extract_mover_early_vasopressor_mar.py` | protected medication streams plus aggregate label counts |
| MOVER adjacent-pair construction | `analysis/build_mover_c02_pair_cohort.py` | protected pair cohort plus aggregate flow |
| Frozen primary-cohort preflight | `analysis/c02_preflight.py`, called before `primary`, `ml`, and `validate` | count, adjacency, event, timestamp, and corrected-unit gates; no row-level output |
| Primary grouped models and fixed transport | `run_pipeline.py --stage primary` | private OOF/models plus aggregate performance and association tables |
| Binary-alert-negative gradient | `analysis/analyze_c02_subalert_gradient.py`, called by `primary` | aggregate risks and patient-cluster contrasts |
| Actual hypnotic-time anchoring | `analysis/analyze_c02_hypnotic_anchored_reproducibility.py`, called by `drug_timed` | protected anchored cohort plus aggregate results |
| Strict post-administration-minute cohort | `analysis/prepare_mover_strict_postminute_pairs.py`, called by `drug_timed` | protected 4,871-pair cohort and aggregate receipt |
| Strict drug-timed models and adjusted ORs | `analysis/analyze_c02_strict_postminute_sensitivity.py`, called by `drug_timed` | aggregate results; ORs and n/events share the exact strict cohort |
| Relative response and current baseline | `analysis/analyze_c02_relative_hypnotic_response.py`, `analysis/analyze_c02_current_baseline_increment.py`, called by `drug_timed` | protected relative-response cohort plus aggregate contrasts |
| Fixed-opportunity 30-minute burden | `run_pipeline.py --stage burden_recovery` | protected operation metrics plus aggregate associations |
| Recovery and delayed onset | `analysis/analyze_c02_post_hypnotic_recovery.py`, called by `burden_recovery` | protected trajectories plus aggregate associations |
| Three-anaesthetic history and post-current-MAP boundary | `run_pipeline.py --stage history` | aggregate history-state and OOF contrasts |
| Flexible-model robustness | `run_pipeline.py --stage ml` | private nested-CV artifacts and aggregate summaries |
| Authoritative result map | `analysis/export_authoritative_result_map.py`, `analysis/verify_authoritative_result_map.py`, called by `export` | exactly 22 aggregate, source-selected rows |
| Independent coefficient replication | `validation/audit_core_associations_independent.R`, called by `validate` | aggregate base-R coefficient table |

## Shared support modules

Several filenames retain the original analysis name but are imported for shared
functions. They are not additional manuscript claims:

- `analyze_c02_fixed_four_endpoint_adjudication.py`: grouped OOF and
  patient-cluster bootstrap helpers;
- `analyze_c02_repeated_alert_comparator.py` and
  `analyze_c02_two_centre_repeated_low.py`: harmonized feature and endpoint
  helpers;
- `analyze_mover_c02_early_vasopressor_action.py`: medication-action filters;
- `analyze_c02_multiepisode_memory.py` and
  `analyze_c02_multiepisode_management_instability.py`: history-triplet and
  time-ordered management-cohort construction;
- `mover_c02_procedure_family.py`: frozen outcome-blind procedure grouping;
- `c02_cluster_logit.py`, `c02_preflight.py`, and `c02_runtime.py`: statistical,
  frozen-cohort provenance, and protected-path infrastructure.

These modules cannot be deleted independently because paper-facing modules
import their functions. Future refactoring may extract those helpers, but the
current release keeps the audited implementations intact to avoid numerical
drift.

Manuscript/DOCX assembly, figure-layout variants, rejected exploratory
endpoints, credentials, download helpers, and local operational logs are
excluded.
