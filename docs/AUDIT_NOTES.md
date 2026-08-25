# Final correction and provenance notes

The final re-audit identified and corrected three release-blocking defects:

1. INSPIRE anaesthesia timestamps are minute-valued. Adjacent-operation
   intervals are divided by 1,440 to obtain days.
2. INSPIRE anaesthesia duration is already an end-minus-start value in minutes
   and is not divided by 60 again.
3. Drug-timed adjusted ORs are fitted on the same strict post-administration
   cohort used for the reported sample: 4,871 pairs, 567 events, and 3,645
   patients.

The corrected INSPIRE primary AUROC increment is approximately 0.034653. The
strict drug-timed risk-direction ORs are reconstructed from the same source
rows as their sample counts. The final exporter rejects the former mixed
4,892/483 versus 4,871/567 provenance and rejects a return to the pre-correction
INSPIRE interval result.

The old workspace-level `authoritative_result_map.csv` is not an input to this
package and must not be published. `export_authoritative_result_map.py` rebuilds
all 22 rows from aggregate source tables, records a source selector for every
row, and verifies the exact result-ID set.

Independent base-R clustered-logit replication is parameterized with explicit
protected inputs; it no longer infers data paths from the public repository.
The corrections changed numerical values but did not change the main clinical
conclusion. The corrected INSPIRE spline AUROC interval required one local
wording update because that interval no longer crossed zero, while the pattern
did not replicate in MOVER.

The shared clustered-logit helper now standardizes predictors only during
optimization and transforms estimates back to their original clinical units.
On the 4,892-pair descriptive hypnotic-anchored sensitivity, this stable solver
changes coefficients by at most about 0.000002 and cluster standard errors by
at most about 0.0017 relative to the earlier unstandardized output. The model
formula, cohort, effect direction, and interpretation are unchanged. The
paper-facing strict 4,871-pair association is rebuilt and independently
replicated from the corrected cohort rather than copied from either earlier
table.
