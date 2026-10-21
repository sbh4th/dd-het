# Background: inference with few treated clusters in staggered DiD

*Started 21 October 2026. A working document: add decisions, results and new references as we go.*

## 1. The problem

We estimate the effect of a village-level coal ban with a staggered rollout using Wooldridge's
extended TWFE (ETWFE), in linear and nonlinear (logit, Poisson) forms, and use `marginaleffects`
to build the pooled ATT, cohort ATTs and heterogeneity contrasts.

Cluster-robust inference relies on the number of **treated clusters** being large, not the total
number of clusters or households. In a staggered design that requirement applies separately to
each estimand:

| Estimand | Treated villages | Assessment |
|---|---|---|
| Pooled ATT | ~20 | Not really a few-treated problem; CR2/CV3 as insurance |
| Cohort 2 ATT | 10 | Borderline; CR2 or CV3 workable |
| Cohort 3 ATT | 7 | Default CV1 likely too narrow; use CR2 + Satterthwaite df |
| Cohort 4 ATT | 3 | No variance estimator is reliable; report descriptively |
| Heterogeneity contrasts involving cohort 4 | 3 on one side | Low power and unreliable size |

The odd estimates for the 3-village cohort are what this literature predicts.

**Design wrinkle.** Indoor temperature was not measured in wave 3 (COVID-19). Cohorts 3 and 4 are
therefore both first observed as treated in wave 4, with waves 1–2 as their pre-period. A contrast
between them confounds cohort differences with one extra wave of exposure. Pooling them into a
single "late" cohort of 10 villages is a defensible option for heterogeneity work.

## 2. What the literature says

**Standard CRVE and the wild cluster bootstrap fail with few treated clusters.** t tests based on
CV1 over-reject badly; wild cluster bootstrap variants can over- or under-reject depending on the
setup (MacKinnon & Webb 2017/2018). Their subcluster wild bootstrap needs similar cluster sizes
across treated and control groups, which rarely holds in DiD.

**Modern staggered estimators don't escape this.** Karim, Nielsen, MacKinnon & Webb (2026) find
over-rejection with few (treated) clusters is at least as severe for CSDID as for TWFE, and propose
a cluster jackknife (Stata `csdidjack`, R `didjack`). The jackknife helps a lot, but the problem
remains when very few clusters are treated. The lesson carries over to ETWFE.

**Wooldridge's own answer.** Lee & Wooldridge (2025; Stata `lwdid`) collapse the panel to
cross-sectional regressions (with or without unit-specific detrending) and use exact t inference
under normality, or randomization inference. Works best with many pre/post periods; with our three
or four waves it leans on normality.

**Methods designed for few treated units.**
- Conley & Taber (2011): use control-cluster residuals to approximate the treated clusters' error
  distribution.
- Ferman & Pinto (2019): corrects for heteroskedasticity from varying cluster sizes, which
  otherwise makes few-treated methods over-reject when treated groups are small.
- Alvarez & Ferman: extensions to staggered timing and uniform bands; a separate paper shows that
  with more than one treated unit, spatially correlated errors among treated units can make these
  methods invalid, and proposes conservative alternatives.
- MacKinnon & Webb (2020): randomization inference on t statistics; fails mainly when treated
  clusters are few *and* atypical.

**Best entry point:** Alvarez, Ferman & Wüthrich (2025), a survey of inference with few treated
units, including the argument that design-based methods share the limitations of model-based ones.

## 3. Toolkit and rationale

### Linear ETWFE (`code/01-etwfe-ols-small-sample.R`)

ETWFE is an OLS regression, so the regression small-sample toolkit applies directly.

- **CR2 + Bell–McCaffrey/Satterthwaite df** (Pustejovsky & Tipton 2018; `clubSandwich`). With few
  treated clusters the df collapse toward the number of treated clusters, which signals the problem
  honestly (Imbens & Kolesár 2016). Expect df ≈ 2 for the cohort-4 cell.
- **CV3 / cluster jackknife** as a cross-check.
- **Leverage diagnostics** (`summclust`): partial leverage and effective number of clusters.
- **Wild cluster bootstrap** as a diagnostic only: large gaps between restricted and unrestricted
  versions signal that neither should be trusted.
- **Carry the correction through aggregation.** `emfx()` uses CV1 + delta method + normal reference
  by default. Either aggregate from the corrected vcov (`linear_contrast()`), or pass a better vcov
  and `df = G - 1`.

### Nonlinear ETWFE (`code/02-etwfe-nonlinear-jackknife-ri.R`)

CR2 + Satterthwaite has no settled analogue for logit/Poisson. The most defensible option is a
**delete-one-village jackknife of the whole fit → `emfx` pipeline** (50 refits), with t(49)
critical values. It works for any quantity `marginaleffects` produces and avoids the delta method.

Additional checks for nonlinear models:
1. Separation (logit) or all-zero cells (Poisson) in the sparse late-cohort cell.
2. **Scale of heterogeneity.** Response-scale cohort ATTs can differ because baselines differ, even
   with identical link-scale effects. State the scale; check whether patterns survive on both scales
   or in the OLS version.

### Jackknife vs permutation: why both

They answer different questions and fail in opposite situations.

| | Jackknife | Permutation (RI) |
|---|---|---|
| Source of uncertainty | Sampling of villages | Assignment of adoption timing |
| Noise learned from | The treated villages themselves | All villages, mostly controls |
| Breaks down when | Few treated clusters | Treated villages atypical (size, variance, selection into timing) |
| Output | SE / CI, plus leave-one-out influence | p-value |

- Both agree → result doesn't hinge on either set of assumptions.
- Jackknife significant, RI not → estimate sits within placebo range; jackknife likely too narrow
  (the classic few-treated pattern; watch the late cohort).
- RI significant, jackknife not → look for an influential village or noisier treated villages.

**Caveat for this study:** rollout order was administrative, not random. RI requires arguing timing
wasn't related to anything that makes treated villages' outcomes more or less variable. Document
how villages were scheduled; treat RI as a check rather than primary inference.

### Bayesian ETWFE (`code/03-etwfe-bayes-brms.R`)

What Bayes buys:
1. Integrates over uncertainty in variance components rather than plugging in noisy estimates.
2. **Partial pooling of cohort-time ATTs** around a common mean with SD τ: a principled answer to
   "what do we believe about cohort 4 given cohorts 2 and 3?"
3. Easier nonlinear inference: response-scale ATTs computed per draw; weakly informative priors
   tame separation (Gelman et al. 2008).

Where to be careful:
- **Clustering has to be in the model.** What drives DiD uncertainty is village-by-wave shocks
  (e.g. weather on survey days, local fuel supply, enforcement, fieldworker effects). These neither
  difference out nor average away within villages. Omitting `(1 | village:wave)` treats 1000
  households as independent; with ~20 households per village and an intra-village share ρ of the
  time-varying variance, the design effect is ≈ 1 + 19ρ (ρ = 0.2 → SEs ~2.2× too narrow).
  Village + village × wave REs also impose exchangeable serial correlation, unlike clustered SEs.
- In simulations without village × time shocks (e.g. the earlier `did-exp` exercise), village-only
  REs and clustered SEs agree because there is nothing for either to get wrong.
- **With 3–4 treated cells the posterior for τ is mostly prior** (an eight-schools problem with
  even fewer schools; Gelman 2006). Pooling assumes exchangeable cohort effects, which biases toward
  homogeneity. Report τ's posterior next to its prior and show sensitivity to 2–3 priors on the °C
  scale.
- Priors don't touch identification; parallel trends still does the causal work. A prior on
  post-period deviations from parallel trends would be a Bayesian analogue of Rambachan & Roth.

**Calibrated Bayes check:** refit under placebo timing assignments and record how often the 95%
interval excludes zero. This ties jackknife, RI and Bayes to the same placebo assignments.

## 4. Decisions so far

- Pooled ATT is the headline estimand; cohort-specific estimates and heterogeneity contrasts
  involving cohort 4 are descriptive.
- Report CR2/CV3 (linear) or village jackknife (nonlinear) intervals with t(G − 1) or Satterthwaite
  df, plus studentized RI as a robustness check.

## 5. Open questions / to do

- [ ] Confirm how adoption timing was determined (supports or undermines RI).
- [ ] Decide whether to pool cohorts 3 and 4 for temperature outcomes given the missing wave 3.
- [ ] Run `summclust` leverage diagnostics on the main specifications.
- [ ] Estimate the village × wave variance share from the data (Bayesian SD or a variance-components
      model); compare Bayesian fits with and without `(1 | village:wave)`.
- [ ] Simulation study: DGP mimicking 30/10/7/3 cohorts with village × wave shocks; compare
      coverage of CV1, CR2, CV3, jackknife, RI and Bayesian intervals by estimand.
- [ ] Prior sensitivity for τ (between-cohort-time SD of ATTs).
- [ ] Check whether an R implementation of `lwdid` exists, or port the `small` option.

## References

- Alvarez, L., & Ferman, B. Extensions for inference in difference-in-differences with few treated
  clusters. arXiv:2302.03131.
- Alvarez, L., & Ferman, B. Inference in difference-in-differences with few treated units and
  spatial correlation. arXiv:2006.16997.
- Alvarez, L., Ferman, B., & Wüthrich, K. (2025). Inference with few treated units.
  arXiv:2504.19841.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for
  inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414–427.
- Conley, T. G., & Taber, C. R. (2011). Inference with "difference in differences" with a small
  number of policy changes. *Review of Economics and Statistics*, 93(1), 113–125.
- Ferman, B., & Pinto, C. (2019). Inference in differences-in-differences with few treated groups
  and heteroskedasticity. *Review of Economics and Statistics*, 101(3), 452–467.
- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models.
  *Bayesian Analysis*, 1(3), 515–534.
- Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y.-S. (2008). A weakly informative default prior
  distribution for logistic and other regression models. *Annals of Applied Statistics*, 2(4),
  1360–1383.
- Imbens, G. W., & Kolesár, M. (2016). Robust standard errors in small samples: Some practical
  advice. *Review of Economics and Statistics*, 98(4), 701–712.
- Karim, S. R., Nielsen, M. Ø., MacKinnon, J. G., & Webb, M. D. (2026). Improved inference for
  CSDID using the cluster jackknife. arXiv:2602.12043.
- Lee, S. J., & Wooldridge, J. M. (2025). Simple approaches to inference with difference-in-
  differences estimators with small cross-sectional sample sizes. Working paper; Stata `lwdid`.
- MacKinnon, J. G., Nielsen, M. Ø., & Webb, M. D. (2023). Cluster-robust inference: A guide to
  empirical practice. *Journal of Econometrics*, 232(2), 272–299.
- MacKinnon, J. G., & Webb, M. D. (2018). The wild bootstrap for few (treated) clusters.
  *Econometrics Journal*, 21(2), 114–135.
- MacKinnon, J. G., & Webb, M. D. (2020). Randomization inference for difference-in-differences
  with few treated clusters. *Journal of Econometrics*, 218(2), 435–450.
- Pustejovsky, J. E., & Tipton, E. (2018). Small-sample methods for cluster-robust variance
  estimation and hypothesis testing in fixed effects models. *Journal of Business & Economic
  Statistics*, 36(4), 672–683.
- Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends. *Review of
  Economic Studies*, 90(5), 2555–2591.
- Wooldridge, J. M. (2021). Two-way fixed effects, the two-way Mundlak regression, and
  difference-in-differences estimators. SSRN 3906345.
- Wooldridge, J. M. (2023). Simple approaches to nonlinear difference-in-differences with panel
  data. *Econometrics Journal*, 26(3), C31–C66.
