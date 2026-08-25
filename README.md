# Prior anaesthetic MAP history: analysis code

This code accompanies a two-centre retrospective study of whether MAP
information from the immediately preceding general anaesthetic adds information
about early low MAP during the next general anaesthetic. The protected source
datasets are INSPIRE 1.4.2 and MOVER.

## Data boundary

This repository is code-only. It contains no clinical records, identifier
values, derived cohorts, fitted models, bootstrap replicates, credentials,
manuscript files, or result CSVs.

All execution must occur in an access-controlled workspace outside this code
directory. `C02_PRIVATE_WORKSPACE` and the TOML config define that boundary;
the code fails closed if a protected path points inside the public repository.
New protected files are written with owner-only permissions where the platform
supports POSIX modes.

The configured `private_output_root` is intentionally private. It contains
row-level out-of-fold predictions, serialized models, and bootstrap artifacts
as well as aggregate tables. Do not publish that directory. Only the reviewed
contents of `aggregate_release_root` are candidates for release.

## Reproducibility scope

The main pipeline starts from the two protected analytic pair cohorts used in
the manuscript. Dataset-specific extraction and cohort-audit scripts are
included, but the repository does not claim a one-command reconstruction from
the credentialed raw archives: some INSPIRE fixed-slot and history-depth
intermediates depend on frozen, access-controlled protocol artifacts that
cannot be redistributed. Every such input is declared in
`config.example.toml` and documented in `docs/PROTECTED_INPUTS.md`.

The public package does provide a single staged route for all paper-facing
models, the strict post-hypnotic cohort, the corrected authoritative result
map, and independent R coefficient replication.

Before the primary, ML, or validation stage, a fail-closed preflight verifies
the frozen source size and cohort counts (INSPIRE: 13,231 source rows filtered
to 9,306 pairs / 7,372 patients / 1,127 events;
MOVER: 7,721 / 5,297 / 240), general-to-general adjacency, and both databases'
interval derivations. It also rechecks that INSPIRE intervals use minutes /
1,440 and that anaesthetic duration remains in minutes.

## Setup

Python 3.9 or newer is supported. The final audited analysis used Python 3.9.6;
see `environment-lock.txt`. On Python below 3.11, `tomli` is installed by
`requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt  # only for the ML stage
```

Copy `config.example.toml` to the private workspace and replace its
placeholders. Do not keep a populated config in this repository.

## Run

Inspect the complete plan without accessing data:

```bash
python run_pipeline.py --config /secure/path/config.local.toml --stage all --dry-run
```

Run individual stages:

```bash
python run_pipeline.py --config /secure/path/config.local.toml --stage primary
python run_pipeline.py --config /secure/path/config.local.toml --stage drug_timed
python run_pipeline.py --config /secure/path/config.local.toml --stage burden_recovery
python run_pipeline.py --config /secure/path/config.local.toml --stage history
python run_pipeline.py --config /secure/path/config.local.toml --stage ml
python run_pipeline.py --config /secure/path/config.local.toml --stage export
python run_pipeline.py --config /secure/path/config.local.toml --stage validate
```

`--stage all` runs those stages in order. The `export` stage creates exactly
22 source-linked manuscript result rows. The `validate` stage rejects result-ID
drift, mixed sample provenance, and the two known correction regressions before
running the independent base-R clustered-logit replication.

When stages are run separately, keep the same private output and derived roots.
`burden_recovery` and `history` require the protected anchored/relative cohorts
created by `drug_timed`; `validate` requires its strict cohort plus the
aggregate tables from the preceding analysis stages. Use `--stage all` for a
fresh end-to-end run. Missing prerequisites fail rather than being inferred.

Standalone cohort/extraction scripts also require the workspace boundary:

```bash
export C02_PRIVATE_WORKSPACE=/secure/path/c02-private-workspace
python analysis/extract_mover_map_stream.py --help
```

## Release checks

Run from this code directory:

```bash
python tools/check_release.py .
python -m compileall -q analysis validation tools tests run_pipeline.py
python tests/smoke_test.py
python tests/test_units_and_provenance.py
python tests/test_release_contract.py
python run_pipeline.py --config /secure/path/config.local.toml --stage all --dry-run
```

Then independently review the complete file list and repository history. A
passing scanner does not make a generated table safe to publish.

## Interpretation limits

- The two centres are analysed separately; this code does not justify pooling
  them or interpreting associations as treatment effects.
- Transport analyses separate ranking from calibration and do not establish a
  deployable clinical probability model.
- Exact numerical reproduction requires the same dataset versions, protected
  cohort definitions, and frozen protocol inputs.
- Verify the current INSPIRE, PhysioNet, MOVER, and institutional data-use terms
  before any redistribution.
