# Bayesian ETWFE with partial pooling of cohort-time ATTs (brms)
# Same data as 01-etwfe-ols-small-sample.R: hh_id, village, wave, first_treat (0/2/3/4)
# Outcome here: temp (indoor temperature, degrees C). For temperature, wave 3 is missing,
# so treated cells are g2_t2, g2_t4, g3_t4, g4_t4.
#
# Key points (see background.md, section 3):
#   - (1 | village:wave) is the model-based analogue of clustering by village; without it,
#     village-by-wave shocks are treated as independent household noise and intervals are
#     too narrow.
#   - (0 + D | cell) partially pools ATT(g,t) around the mean effect b_D with SD tau.
#     With only 4 treated cells, tau's posterior is mostly prior: show sensitivity.

library(dplyr)
library(brms)
library(marginaleffects)

options(mc.cores = 4, brms.backend = "cmdstanr")

dfb <- df |>
  mutate(
    cohort = factor(first_treat),
    w      = factor(wave),
    D      = as.integer(first_treat > 0 & wave >= first_treat),
    cell   = if_else(D == 1, paste0("g", first_treat, "_t", wave), "none")
  )

# ------------------------------------------------------------------
# 1. Main model: village + village-by-wave + household random effects
#    Cohort dummies + wave dummies + treated cells = ETWFE (Wooldridge 2021).
#    Priors on the degrees C scale; adjust to substantive knowledge.
# ------------------------------------------------------------------
pri <- c(
  prior(normal(0, 3), class = b, coef = D),       # mean ATT
  prior(normal(0, 5), class = b),                 # cohort and wave terms
  prior(normal(0, 1), class = sd, group = cell),  # tau: between-cell SD of ATTs
  prior(normal(0, 2), class = sd)                 # village, village:wave, household SDs
)

fit_vw <- brm(
  temp ~ cohort + w + D +
    (0 + D | cell) +
    (1 | village) + (1 | village:w) +
    (1 | hh_id),
  data = dfb, prior = pri,
  family = gaussian(),
  chains = 4, iter = 2000, seed = 20261021,
  file = "fits/brms-etwfe-vw"
)

summary(fit_vw)   # look at sd(village:w Intercept) and sd(cell D)

# ------------------------------------------------------------------
# 2. Same model without village-by-wave REs, for comparison
#    If sd(village:w) above is non-trivial, expect narrower (overconfident) ATT intervals here.
# ------------------------------------------------------------------
fit_v <- update(fit_vw,
                formula. = ~ . - (1 | village:w),
                seed = 20261021, file = "fits/brms-etwfe-v")

# ------------------------------------------------------------------
# 3. ATTs from posterior draws
#    newdata restricted to treated observations = ATT; by = "cohort" for cohort ATTs.
# ------------------------------------------------------------------
treated <- subset(dfb, D == 1)

att_vw <- avg_comparisons(fit_vw, variables = "D", newdata = treated)
att_v  <- avg_comparisons(fit_v,  variables = "D", newdata = treated)
att_vw; att_v

avg_comparisons(fit_vw, variables = "D", newdata = treated, by = "cohort")

# Heterogeneity contrast, e.g. cohort 2 vs late cohorts (3 and 4 pooled)
# avg_comparisons(fit_vw, variables = "D", newdata = treated,
#                 by = "cohort", hypothesis = "b1 - (b2 + b3)/2 = 0")

# ------------------------------------------------------------------
# 4. Prior sensitivity for tau
#    Compare the posterior for sd(cell D) with its prior, and refit with 2-3 alternatives.
# ------------------------------------------------------------------
tau_priors <- list(
  tight  = prior(normal(0, 0.5), class = sd, group = cell),
  medium = prior(normal(0, 1),   class = sd, group = cell),
  wide   = prior(normal(0, 3),   class = sd, group = cell)
)
# fits_tau <- lapply(names(tau_priors), \(nm)
#   update(fit_vw, prior = c(pri[pri$group != "cell", ], tau_priors[[nm]]),
#          seed = 20261021, file = paste0("fits/brms-etwfe-tau-", nm)))
# lapply(fits_tau, \(f) avg_comparisons(f, variables = "D", newdata = treated, by = "cohort"))

# ------------------------------------------------------------------
# 5. Nonlinear outcomes
#    Swap family = bernoulli() or poisson() (priors then on the link scale);
#    avg_comparisons() returns response-scale ATTs directly.
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 6. Calibrated-Bayes check (slow): refit under permuted adoption timing
#    (as in the RI loops in scripts 01/02) and record how often the 95% interval
#    for the pooled ATT excludes zero. Run on a cluster or with a small number of permutations.
# ------------------------------------------------------------------
