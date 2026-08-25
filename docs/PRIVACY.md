# Privacy and release rules

## Public repository

Allowed contents are source code, documentation, configuration templates, and
tests using synthetic records. Do not include:

- INSPIRE or MOVER source files;
- row-level cohorts, predictions, trajectories, or bootstrap replicates;
- identifier values or clinical free text;
- fitted models or serialized preprocessors;
- credentials, signed agreements, tokens, cookies, or private download links;
- populated configs, local logs, manuscript files, figures, or result tables;
- personal absolute paths or operating-system metadata.

Identifier column names such as `subject_id`, `patient_id`, and `LOG_ID`
are required for patient-grouped analysis. They are not identifier values.

## Runtime boundary

`C02_PRIVATE_WORKSPACE` is mandatory for standalone scripts.
`run_pipeline.py` sets it from the config before importing analysis modules.
The runtime rejects protected paths inside the public code directory and uses a
restrictive process umask. Restricted row-level artifacts are explicitly
written with owner-only permissions where supported.

Both normalized logical paths and resolved symlink targets are checked.
Lexical `..` escapes and arbitrary symlink targets are rejected. A protected
path may leave the real workspace root only when its verified target is listed
explicitly in `allowed_private_real_roots`.

`private_output_root` is not an aggregate-only directory. It contains:

- row-level OOF prediction tables with operation and patient identifiers;
- fitted model objects;
- bootstrap replicate tables;
- aggregate metrics and figures.

Never publish that directory. `aggregate_release_root` is only a release
candidate: every file still requires disclosure review, small-cell review, and
claim/provenance review.

## Required release gate

1. Run `python tools/check_release.py .`.
2. Run the compile, smoke, unit/provenance, dry-run, result-map, and independent
   replication checks in the README.
3. Review the complete candidate repository file list and `git status`.
4. Confirm no generated output path is tracked or nested below this code root.
5. Inspect every non-code file manually.
6. Confirm repository history never contained restricted material. Deleting a
   current file is insufficient if sensitive data appeared in history.
7. Confirm current dataset and institutional terms before public dissemination.

The current development package is nested under a larger private project
workspace. Do not publish or reuse that parent repository's history. For a
formal release, copy only this reviewed package into a new empty directory,
rerun every release gate there, inspect the copied file list, and initialize a
new repository whose root is exactly the copied package.

## Collaboration

Do not send restricted files to public issue trackers, code hosts, generic file
sharing, or services not approved for the applicable data agreement. A
collaborator who needs row-level access must independently satisfy the relevant
access and governance requirements.
