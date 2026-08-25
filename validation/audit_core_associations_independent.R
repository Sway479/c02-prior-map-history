#!/usr/bin/env Rscript

# Independent base-R validation of the manuscript-facing C02 logistic models.
#
# This script deliberately does not import the Python analysis code. It rebuilds
# the three principal design matrices from the corrected private cohorts, fits
# unpenalized binomial GLMs, and computes patient-clustered sandwich standard
# errors directly from cluster-summed score contributions. Only aggregate
# coefficients and fit diagnostics are written.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop(paste(
    "usage: audit_core_associations_independent.R",
    "<inspire_pair.csv.gz> <mover_pair.csv.gz>",
    "<strict_postminute_pair.csv.gz> <private_output_dir>"
  ))
}

inspire_path <- normalizePath(args[[1]], mustWork = TRUE)
mover_path <- normalizePath(args[[2]], mustWork = TRUE)
strict_path <- normalizePath(args[[3]], mustWork = TRUE)
out_dir <- args[[4]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_private_csv <- function(path) {
  if (!file.exists(path)) stop(sprintf("missing private input: %s", path))
  read.csv(gzfile(path), stringsAsFactors = FALSE, check.names = FALSE)
}

median_impute <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  value <- median(x[is.finite(x)], na.rm = TRUE)
  if (!is.finite(value)) stop("cannot median-impute an all-missing predictor")
  x[!is.finite(x)] <- value
  x
}

cluster_glm <- function(frame, outcome, cluster, terms, centre, model) {
  needed <- c(outcome, cluster, terms)
  if (!all(needed %in% names(frame))) {
    stop(sprintf("missing columns for %s/%s: %s", centre, model,
                 paste(setdiff(needed, names(frame)), collapse = ", ")))
  }
  work <- frame[, needed, drop = FALSE]
  work[[outcome]] <- as.integer(work[[outcome]])
  if (!all(work[[outcome]] %in% c(0L, 1L))) stop("non-binary outcome")
  for (term in terms) work[[term]] <- median_impute(work[[term]])
  work[[cluster]] <- as.character(work[[cluster]])
  if (any(is.na(work[[cluster]]) | work[[cluster]] == "")) stop("missing cluster id")

  formula <- as.formula(sprintf("%s ~ %s", outcome, paste(terms, collapse = " + ")))
  fit <- glm(formula, data = work, family = binomial(link = "logit"),
             control = glm.control(maxit = 100, epsilon = 1e-10))
  if (!isTRUE(fit$converged)) stop(sprintf("GLM failed: %s/%s", centre, model))

  X <- model.matrix(fit)
  y <- work[[outcome]]
  p <- as.numeric(fitted(fit))
  if (any(!is.finite(p)) || any(p <= 0) || any(p >= 1)) {
    stop(sprintf("invalid fitted probabilities: %s/%s", centre, model))
  }
  w <- p * (1 - p)
  bread <- crossprod(X, X * w)
  bread_inv <- solve(bread)
  score_rows <- X * (y - p)
  cluster_scores <- rowsum(score_rows, group = work[[cluster]], reorder = FALSE)
  meat <- crossprod(cluster_scores)
  n <- nrow(X)
  k <- ncol(X)
  g <- nrow(cluster_scores)
  correction <- (g / (g - 1)) * ((n - 1) / (n - k))
  covariance <- correction * bread_inv %*% meat %*% bread_inv
  robust_se <- sqrt(diag(covariance))
  beta <- coef(fit)

  rows <- lapply(terms, function(term) {
    b <- unname(beta[[term]])
    s <- unname(robust_se[[term]])
    data.frame(
      centre = centre,
      model = model,
      term = term,
      log_odds = b,
      cluster_se = s,
      odds_ratio = exp(b),
      ci_low = exp(b - qnorm(0.975) * s),
      ci_high = exp(b + qnorm(0.975) * s),
      p_value = 2 * pnorm(-abs(b / s)),
      n = n,
      events = sum(y),
      patients = g,
      converged = fit$converged,
      iterations = fit$iter,
      deviance = fit$deviance,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

inspire <- read_private_csv(inspire_path)
inspire <- inspire[
  trimws(inspire$antype) == "General" & trimws(inspire$prior_antype) == "General",
]
inspire$target <- as.integer(inspire$target_any_low)
inspire$age_per_10y <- as.numeric(inspire$age) / 10
inspire$bmi_per_5 <- as.numeric(inspire$bmi) / 5
inspire$asa_class <- as.numeric(inspire$asa)
inspire$male <- as.numeric(toupper(trimws(inspire$sex)) == "M")
inspire$log1p_interval_days <- log1p(as.numeric(inspire$interval_days))
inspire$prior_binary_alert <- as.numeric(
  pmin(inspire$prior_first2_map_0, inspire$prior_first2_map_1) < 65
)
inspire$prior_first_MAP_per_10mmHg <- as.numeric(inspire$prior_first2_map_0) / 10
inspire$prior_MAP_change_per_10mmHg <- (
  as.numeric(inspire$prior_first2_map_1) - as.numeric(inspire$prior_first2_map_0)
) / 10

primary_terms <- c(
  "age_per_10y", "bmi_per_5", "asa_class", "male",
  "log1p_interval_days", "prior_binary_alert",
  "prior_first_MAP_per_10mmHg", "prior_MAP_change_per_10mmHg"
)

mover <- read_private_csv(mover_path)
mover$target <- as.integer(mover$target_any_low_first2)
mover$age_per_10y <- as.numeric(mover$age_years) / 10
mover$bmi_per_5 <- as.numeric(mover$bmi_kg_m2) / 5
mover$asa_class <- as.numeric(mover$asa_numeric)
mover$male <- as.numeric(toupper(trimws(mover$sex_common)) == "M")
mover$log1p_interval_days <- as.numeric(mover$interval_log1p)
mover$prior_binary_alert <- as.numeric(mover$prior_first2_any_low)
mover$prior_first_MAP_per_10mmHg <- as.numeric(mover$prior_first_map) / 10
mover$prior_MAP_change_per_10mmHg <- as.numeric(mover$prior_first2_change) / 10

strict <- read_private_csv(strict_path)
strict$target <- as.integer(strict$cur_anylow)
strict$prior_post_MAP_per_10 <- as.numeric(strict$pr_map0) / 10
strict$prior_post_change_per_10 <- as.numeric(strict$pr_change) / 10
strict$prior_binary_alert <- as.numeric(strict$pr_anylow)
strict$current_age_per_10 <- as.numeric(strict$age_years) / 10
strict$current_BMI_per_5 <- as.numeric(strict$bmi_kg_m2) / 5
strict$current_ASA <- as.numeric(strict$asa_numeric)
strict$interval_log1p <- as.numeric(strict$interval_log1p)
strict$prior_propofol_dose_per_100mg <- as.numeric(strict$pr_anchor_propofol_mg) / 100
strict$prior_etomidate_any <- as.numeric(as.numeric(strict$pr_anchor_etomidate_mg) > 0)
strict$prior_ketamine_any <- as.numeric(as.numeric(strict$pr_anchor_ketamine_mg) > 0)
strict$current_propofol_dose_per_100mg <- as.numeric(strict$cur_anchor_propofol_mg) / 100
strict$current_etomidate_any <- as.numeric(as.numeric(strict$cur_anchor_etomidate_mg) > 0)
strict$current_ketamine_any <- as.numeric(as.numeric(strict$cur_anchor_ketamine_mg) > 0)
strict_terms <- c(
  "prior_post_MAP_per_10", "prior_post_change_per_10", "prior_binary_alert",
  "current_age_per_10", "current_BMI_per_5", "current_ASA",
  "interval_log1p", "prior_propofol_dose_per_100mg", "prior_etomidate_any",
  "prior_ketamine_any", "current_propofol_dose_per_100mg",
  "current_etomidate_any", "current_ketamine_any"
)

result <- rbind(
  cluster_glm(inspire, "target", "subject_id", primary_terms,
              "INSPIRE", "primary_absolute_change"),
  cluster_glm(mover, "target", "patient_id", primary_terms,
              "MOVER", "primary_absolute_change"),
  cluster_glm(strict, "target", "patient_id", strict_terms,
              "MOVER", "strict_post_hypnotic")
)

write.csv(
  result,
  file.path(out_dir, "independent_r_cluster_logit.csv"),
  row.names = FALSE,
  quote = TRUE
)

diagnostics <- data.frame(
  dataset = c("INSPIRE primary", "MOVER primary", "MOVER strict post-hypnotic"),
  pairs = c(nrow(inspire), nrow(mover), nrow(strict)),
  patients = c(length(unique(inspire$subject_id)), length(unique(mover$patient_id)),
               length(unique(strict$patient_id))),
  events = c(sum(inspire$target), sum(mover$target), sum(strict$target)),
  stringsAsFactors = FALSE
)
write.csv(
  diagnostics,
  file.path(out_dir, "independent_r_cohort_diagnostics.csv"),
  row.names = FALSE,
  quote = TRUE
)

cat(sprintf("R_INDEPENDENT_ASSOCIATION_AUDIT_PASS rows=%d\n", nrow(result)))
