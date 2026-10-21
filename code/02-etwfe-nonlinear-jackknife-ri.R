# Nonlinear ETWFE (logit / Poisson) + marginaleffects: small-sample inference
# Same data as 01-etwfe-ols-small-sample.R: hh_id, village, wave, first_treat (0/2/3/4), y
# Approach: keep etwfe + emfx (marginaleffects) for estimation; change only the
# variance (village-level jackknife) and the reference distribution (t, G-1 df),
# and add studentized randomization inference.

library(dplyr)
library(etwfe)
library(marginaleffects)

G <- n_distinct(df$village)   # 50

fit_etwfe <- function(d, family = "logit") {
  # Wooldridge (2023) nonlinear ETWFE: pooled QMLE with cohort dummies,
  # not unit FE (avoids incidental parameters in logit)
  etwfe(fml = y ~ 0, tvar = wave, gvar = first_treat,
        data = d, family = family)
}

# ------------------------------------------------------------------
# 0. Point estimates + default delta-method SEs (CV1 by village, z-based)
# ------------------------------------------------------------------
m <- fit_etwfe(df, family = "logit")   # or "poisson"
att_simple <- emfx(m, type = "simple", vcov = ~village)
att_group  <- emfx(m, type = "group",  vcov = ~village)
att_simple; att_group

# Check for separation / extreme fitted values in the 3-village cohort cell.
# With a binary outcome, ~60 households and a saturated cell, logit can
# give huge coefficients, which may explain odd late-cohort estimates.
# For logit:  table(df$first_treat, df$wave, df$y)
# For Poisson: look for all-zero cells.

# ------------------------------------------------------------------
# 1. Delete-one-village jackknife of the whole estimate -> emfx pipeline
#    (nonlinear analogue of CV3 / the Karim-Nielsen-MacKinnon-Webb approach)
# ------------------------------------------------------------------
est_fun <- function(d, type) {
  emfx(fit_etwfe(d, family = "logit"), type = type, vcov = FALSE)$estimate
}

villages <- unique(df$village)
theta_simple <- est_fun(df, "simple")
theta_group  <- est_fun(df, "group")

jk_simple <- sapply(villages, \(v) est_fun(filter(df, village != v), "simple"))
jk_group  <- sapply(villages, \(v) est_fun(filter(df, village != v), "group"))
# jk_group: one row per cohort, one column per dropped village

jk_se <- function(jk, theta) {
  jk <- as.matrix(jk)
  if (ncol(jk) == 1) jk <- t(jk)
  sqrt((G - 1) / G * rowSums((jk - theta)^2))   # centred on full-sample estimate
}

se_simple <- jk_se(t(jk_simple), theta_simple)
se_group  <- jk_se(jk_group, theta_group)

tcrit <- qt(0.975, df = G - 1)
data.frame(
  term = c("ATT", paste0("cohort_", sort(unique(df$first_treat[df$first_treat > 0])))),
  est  = c(theta_simple, theta_group),
  se   = c(se_simple, se_group)
) |>
  mutate(lo = est - tcrit * se, hi = est + tcrit * se)
# The pooled ATT interval is the one to trust. The cohort-4 interval (3 villages)
# is still likely too narrow; report it descriptively.

# Heterogeneity contrast (e.g. cohort 2 vs late cohorts), same jackknife:
# contrast <- jk_group[1, ] - jk_group[3, ]
# se_c <- sqrt((G - 1) / G * sum((contrast - (theta_group[1] - theta_group[3]))^2))

# ------------------------------------------------------------------
# 2. Studentized randomization inference for the pooled ATT
#    Permute adoption timing across villages (cohort sizes fixed 30/10/7/3);
#    studentize with the cheap delta-method SE. The SE need not be "correct",
#    studentizing just makes the RI test more robust (MacKinnon & Webb 2020).
# ------------------------------------------------------------------
t_stat <- function(d) {
  e <- emfx(fit_etwfe(d, family = "logit"), type = "simple", vcov = ~village)
  e$estimate / e$std.error
}

t_obs <- t_stat(df)
vill  <- distinct(df, village, first_treat)

set.seed(20261020)
B <- 999   # each draw refits the model; increase if time allows
t_perm <- replicate(B, {
  perm <- mutate(vill, first_treat = sample(first_treat))
  t_stat(df |> select(-first_treat) |> left_join(perm, by = "village"))
})
p_ri <- mean(abs(c(t_perm, t_obs)) >= abs(t_obs))
p_ri

# ------------------------------------------------------------------
# 3. OLS ETWFE through marginaleffects: pass a better vcov and a t reference
#    The ATT is a linear combination, so the delta method is exact; only the
#    vcov and the reference distribution (normal by default) need changing.
# ------------------------------------------------------------------
m_ols <- etwfe(fml = y ~ 0, tvar = wave, gvar = first_treat, data = df)
emfx(m_ols, type = "simple", vcov = ~village, df = G - 1)
# For CR2 + Satterthwaite df, use the lm() refit in 01-etwfe-ols-small-sample.R:
#   linear_contrast() gives the CR2 SE and Satterthwaite df directly.
# The jackknife in section 1 also works unchanged with family = "gaussian".
