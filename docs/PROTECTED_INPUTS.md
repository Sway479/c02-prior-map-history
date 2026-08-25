# Protected input contract

None of these files belongs in the public repository. Paths are configured
under `private_workspace`.

| Config key | Role | Minimum provenance check |
|---|---|---|
| `inspire_pair_cohort` | corrected INSPIRE adjacent-pair source, filtered by analysis modules to General-to-General | INSPIRE 1.4.2; 13,231 source rows; 9,306 analysed pairs; 7,372 patients; 1,127 events; interval in days from minutes; duration in minutes |
| `mover_pair_cohort` | MOVER adjacent General-to-General pairs | 7,721 pairs; 5,297 patients; 240 events; patient-grouped ordering; interval agrees with timestamps |
| `mover_map_stream` | early cleaned MOVER MAP records | source archive/member receipt; conflict and modality rules |
| `mover_full_map_stream` | full intraoperative MOVER MAP records for 30-minute analyses | same source version and cleaning rules as early stream |
| `mover_hypnotic_mar` | early IV hypnotic MAR candidates | explicit route/action/dose filters and extraction receipt |
| `mover_vasopressor_mar` | time-ordered vasopressor MAR candidates | explicit medication/action/route filters; not a treatment label |
| `mover_patient_information` | weight and operation context | MOVER source archive version |
| `mover_epic_archive` | protected MOVER EPIC archive | accepted DUA and verified archive identity |
| `inspire_history_pairs` | frozen consecutive history-depth pairs | protocol/amendment hashes retained privately |
| `inspire_exclusion_flags` | pre-analysis operation/subject correction flags | flag-generation receipt and row-key uniqueness |
| `inspire_triplet_conflict_audit` | selected-source conflict flags | raw-key conflict audit completed before history analysis |

## Fail-closed gates

- Every configured path must be below the private workspace and outside the
  public code directory.
- Both normalized logical paths and resolved symlink targets are checked.
  Offloaded targets require an explicit `allowed_private_real_roots` entry.
- Primary, ML, and validation stages hard-gate both frozen cohort counts and
  event counts. Corrected INSPIRE interval/duration assertions and MOVER
  timestamp-derived intervals must also pass.
- The strict postminute builder must reproduce 4,871 pairs, 3,645 patients, and
  567 events before drug-timed results are exported.
- The authoritative exporter reads aggregate source tables, never an earlier
  manuscript result map.
- Source versions, checksums, and protocol receipts remain private audit
  artifacts because some may reveal protected file structure or identifiers.

The included INSPIRE pair builder consumes protected fixed-slot feature files;
it is an auditable unit-correction and adjacency implementation, not a complete
raw-vital reconstruction. The history stage similarly starts from frozen
protected inputs. Those limits are explicit and should not be described as
one-command raw-data reproducibility.

Random seeds, fold counts, bootstrap repetitions, and clinical thresholds are
versioned protocol constants inside the audited analysis modules. They are not
user-configurable because changing them would create a different analysis.
