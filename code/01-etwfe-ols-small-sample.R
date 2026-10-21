# ETWFE with few treated clusters: CR2/CR3 inference, leverage, and randomization inference
# Setup: ~1000 households, 50 villages, waves 1-4; village-level treatment
#   first_treat: 0 = never (30 villages), 2 (10), 3 (7), 4 (3)
# Expects df with columns: hh_id, village, wave, first_treat, y
# NOTE: if the outcome is missing in wave 3, the treated cells are g2_t2, g2_t4, g3_t4, g4_t4.

library(dplyr)
library(etwfe)        # McDermott's ETWFE (fixest backend)
library(clubSandwich) # CR2 + Satterthwaite / HTZ tests
library(sandwich)     # CV3 via vcovCL(type = "HC3")

# ------------------------------------------------------------------
# 0. Reference fit with etwfe (default CV1 clustered by village)
# ------------------------------------------------------------------
m_etwfe <- etwfe(fml = y ~ 0, tvar = wave, gvar = first_treat,
                 data = df, vcov = ~village)
emfx(m_etwfe, type = "simple")  # pooled ATT
emfx(m_etwfe, type = "group")   # by cohort

# ------------------------------------------------------------------
# 1. Same model as plain lm(), so small-sample tools apply.
#    Wooldridge (2021): with no covariates and a balanced panel, pooled OLS
#    with cohort + time dummies + cohort-by-post-period treated cells gives the
#    same ATT(g,t) as ETWFE with unit FE. If the panel is unbalanced, check
#    against m_etwfe, or swap cohort for factor(hh_id).
# ------------------------------------------------------------------
build_cells <- function(d) {
  d |>
    mutate(
      cohort = factor(first_treat),
      w      = factor(wave),
      cell   = if_else(first_treat > 0 & wave >= first_treat,
                       paste0("g", first_treat, "_t", wave), "none"),
      cell   = relevel(factor(cell), ref = "none")
    )
}

df  <- build_cells(df)
fit <- lm(y ~ cohort + w + cell, data = df)

cells <- grep("^cell", names(coef(fit)), value = TRUE)
cells                                # should match emfx cohort-time cells
# compare coef(fit)[cells] with emfx(m_etwfe, type = "event") / by-cell output

# ------------------------------------------------------------------
# 2. CR2 with Satterthwaite df (Pustejovsky & Tipton 2018)
#    Look at the df column: for g4_t4 (3 villages) expect df around 2.
# ------------------------------------------------------------------
V_cr2 <- vcovCR(fit, cluster = df$village, type = "CR2")
coef_test(fit, vcov = V_cr2, test = "Satterthwaite", coefs = cells)

# Pooled ATT = weighted average of cells (weights = treated obs per cell,
# matching emfx "simple")
n_cell <- table(df$cell[df$cell != "none"])[sub("^cell", "", cells)]
wts    <- as.numeric(n_cell / sum(n_cell))
C      <- matrix(0, nrow = 1, ncol = length(coef(fit)),
                 dimnames = list("ATT", names(coef(fit))))
C[1, cells] <- wts
linear_contrast(fit, vcov = V_cr2, contrasts = C, test = "Satterthwaite")

# Cohort heterogeneity test (all treated cells equal), HTZ small-sample Wald
Wald_test(fit, constraints = constrain_equal(cells), vcov = V_cr2, test = "HTZ")

# ------------------------------------------------------------------
# 3. CV3 (cluster jackknife-type) as a cross-check
# ------------------------------------------------------------------
V_cv3 <- vcovCL(fit, cluster = ~village, type = "HC3")
se_cv3 <- sqrt(diag(V_cv3))[cells]
att_se_cv3 <- sqrt(drop(C %*% V_cv3 %*% t(C)))
se_cv3; att_se_cv3   # use t with G - 1 = 49 df for the pooled ATT;
                     # for single-cohort cells, CV3 is still unreliable

# ------------------------------------------------------------------
# 4. Leverage / influence diagnostics (summclust; MacKinnon, Nielsen & Webb)
#    install.packages("summclust") or from GitHub (s3alfisc/summclust)
# ------------------------------------------------------------------
# library(summclust)
# sc <- summclust(fit, cluster = ~village, params = cells)
# summary(sc); plot(sc)   # high partial leverage for the 3 late villages expected

# ------------------------------------------------------------------
# 5. Randomization inference: permute adoption timing across villages,
#    holding cohort sizes fixed (30/10/7/3). Uses the CR2 t statistic
#    (studentized RI is more robust than RI on the coefficient;
#    MacKinnon & Webb 2020).
# ------------------------------------------------------------------
att_t <- function(d) {
  d   <- build_cells(d)
  f   <- lm(y ~ cohort + w + cell, data = d)
  cl  <- grep("^cell", names(coef(f)), value = TRUE)
  n   <- table(d$cell[d$cell != "none"])[sub("^cell", "", cl)]
  Cm  <- matrix(0, 1, length(coef(f)), dimnames = list(NULL, names(coef(f))))
  Cm[1, cl] <- as.numeric(n / sum(n))
  V   <- vcovCR(f, cluster = d$village, type = "CR2")
  drop(Cm %*% coef(f)) / sqrt(drop(Cm %*% V %*% t(Cm)))
}

t_obs <- att_t(df)
vill  <- distinct(df, village, first_treat)

set.seed(20261020)
B <- 1999
t_perm <- replicate(B, {
  perm <- mutate(vill, first_treat = sample(first_treat))
  d    <- df |> select(-first_treat, -cohort, -w, -cell) |>
          left_join(perm, by = "village")
  att_t(d)
})
p_ri <- mean(abs(c(t_perm, t_obs)) >= abs(t_obs))
p_ri

# Same idea for a single cell (e.g. g4_t4): swap the contrast for that
# coefficient. With 3 treated villages, read the result alongside the CR2
# Satterthwaite df rather than as a stand-alone answer.
