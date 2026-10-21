# CLAUDE.md — dd-het

Guidance for Claude (and humans) working in this repo.

## What this repo is for

Methods work on **inference and heterogeneity in staggered difference-in-differences** when only a
few clusters are treated, motivated by the coal-ban study (village-level ban, indoor temperature and
related household outcomes). The core estimator is Wooldridge's extended TWFE (ETWFE), linear and
nonlinear, with aggregation and heterogeneity contrasts done via `marginaleffects`.

The question driving the work: how do we get honest uncertainty for (a) the pooled ATT, (b)
cohort-specific ATTs, and (c) tests of heterogeneity across cohorts, when some cohorts contain only
a handful of treated villages?

See `background.md` for the literature, the reasoning behind each method, and open questions.

## Study design (keep these numbers in mind)

- ~1000 households in 50 villages, 4 survey waves; treatment assigned at the **village** level.
- Adoption cohorts: 30 never treated, 10 first treated in wave 2, 7 in wave 3, 3 in wave 4.
- Indoor temperature was measured in waves 1, 2 and 4 only (no wave 3 because of COVID-19). For that
  outcome the wave-3 and wave-4 cohorts are both first observed as treated in wave 4.
- Inference is driven by the number of **treated villages per estimand**, not households.
  Pooled ATT ≈ 20 treated villages (fine); cohort 2 = 10 (borderline); cohort 3 = 7 (needs
  small-sample corrections); cohort 4 = 3 (no variance estimator rescues this; report descriptively).

## Repo layout

```
CLAUDE.md          this file
background.md      literature, rationale, decisions, open questions
code/              R scripts (see below)
data/              local only, git-ignored; never commit household data
```

### Scripts in `code/`

| Script | What it does |
|---|---|
| `01-etwfe-ols-small-sample.R` | Linear ETWFE refit as `lm()`; CR2 + Satterthwaite df (clubSandwich), HTZ Wald test of cohort heterogeneity, CV3 cross-check, summclust leverage, studentized randomization inference permuting adoption timing across villages. |
| `02-etwfe-nonlinear-jackknife-ri.R` | Logit/Poisson ETWFE via `etwfe` + `emfx`; delete-one-village jackknife of the full estimate → aggregate pipeline; studentized RI; t(G − 1) reference for the OLS `emfx` case. |
| `03-etwfe-bayes-brms.R` | Bayesian ETWFE in `brms` with partial pooling of cohort-time ATTs and village and **village × wave** random effects; response-scale ATTs via `avg_comparisons()`; with/without village × wave comparison. |

All scripts expect a data frame `df` with columns `hh_id`, `village`, `wave`, `first_treat`
(0 = never, else first treated wave), and an outcome `y` (or `temp` in the Bayesian script).
They are templates: check variable names and the balanced-panel assumptions before running.

## Conventions

- Language: R. Main packages: `etwfe`, `fixest`, `marginaleffects`, `clubSandwich`, `sandwich`,
  `summclust`, `brms` (cmdstanr backend), `dplyr`.
- Cluster at the village level everywhere. Use t critical values with G − 1 = 49 df (or
  Satterthwaite df) rather than the normal default from `marginaleffects`.
- Set seeds explicitly for permutation and MCMC runs; record B (permutation draws) in output.
- When adding a method, check it on simulated data first (e.g. a DGP with village × wave shocks)
  before running on the study data.

## Things to be careful about

- **Never commit data** or anything that could identify households or villages. `data/` is
  git-ignored; keep it that way. The repo may be public.
- Do not present cohort-4 estimates or cohort-4 heterogeneity contrasts as confirmatory.
- For nonlinear models, state the scale (link vs response) whenever discussing heterogeneity.
- Check for separation (logit) or all-zero cells (Poisson) in the late cohort before interpreting
  odd estimates.
- Bayesian models must include `(1 | village:wave)`; village-only random effects understate
  uncertainty when there are village-by-wave shocks (weather on survey days, enforcement, etc.).
- Permutation inference assumes adoption timing is exchangeable across villages; the rollout was
  administrative, so treat RI as a check and document how villages were scheduled.

## Working with Sam

Sam is an epidemiologist with expertise in causal inference and biostatistics; analyses are often
run by PhD students. Explanations can be technical, but code should be readable and well commented
for students.
