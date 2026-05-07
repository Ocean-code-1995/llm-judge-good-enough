# Methodological Audit — `core/llm_good_enough/evaluator.py`

> **Date:** 2026-02-19  
> **Scope:** Full review of `evaluator.py` against the methodology described in `README.md`.  
> **Focus:** Statistical and methodological correctness only (code quality out of scope).

---

## Verdict Summary

| Area | Status | Details |
|------|--------|---------|
| H-H disagreement computation | ✅ Correct | All pairwise \|H_i − H_j\|, NaN-safe |
| LLM-H disagreement computation | ✅ Correct | All \|LLM − H_j\|, NaN-safe |
| MWU test direction | ✅ Correct | `alternative='greater'` consistently applied |
| MWU argument order | ✅ Correct | Always `x=model_dis, y=human_dis` |
| Random judge generation | ✅ Correct | Uniform integer on [min_score, max_score] |
| Monte Carlo simulation | ✅ Correct | Fresh random judge per iteration; items fixed |
| Human stability analysis | ✅ Correct | Bootstrap + fresh random judge per iteration |
| LLM stability analysis | ✅ Correct | Bootstrap + real LLM ratings per iteration |
| Seed sensitivity analysis | ✅ Correct | Multiple seeds, t-based CI |
| Convergence (adaptive sampling) | ✅ Correct | Split-half on decisions, relative/absolute threshold |
| Delta-mean computation | ✅ Correct | mean(model-H) − mean(H-H) |
| Histogram normalization | ✅ Correct | `density=True` with unit-width bins → heights = probabilities |
| Missing-value handling | ✅ Correct | Consistent masking throughout |
| Within-item dependency (MWU assumption) | ⚠️ Caveat | Standard in the literature; not a bug, but worth noting |
| README ↔ Code API match | 🔧 Minor gap | `reseed()` listed in README but not implemented |

**No methodological errors found.** Two caveats and one documentation gap are discussed below.

---

## Detailed Analysis

### 1. Core Disagreement Computations

#### `compute_human_disagreements()` (lines 257–297)

**README says:** "For each item, all pairwise absolute differences between human scores are computed."

**Code does:**
- Uses `np.triu_indices(n_humans, k=1)` to enumerate all unique (i < j) rater pairs.
- For each pair, masks out rows where either rating is NaN.
- Computes `|H_i − H_j|` on valid rows and concatenates results.

**Assessment: ✅ Correct.** Matches the README exactly. All C(n,2) pairs per item, NaN-safe.

---

#### `compute_llm_human_disagreements()` (lines 302–342)

**README says:** "Compute the absolute difference between each human score and the LLM's score."

**Code does:**
- Broadcasts `|human_matrix − llm_vec[:, None]|`.
- Masks for NaN in both human matrix and LLM vector.
- Returns flattened array of valid differences.

**Assessment: ✅ Correct.** All |LLM − H_j| for non-missing pairs.

---

### 2. Mann-Whitney U Test

#### `run_mannwhitneyu_test()` (lines 369–395)

**README says:**
- H₀: LLM–Human disagreement is *not greater* than Human–Human disagreement  
- H₁: LLM–Human disagreement is *greater* than Human–Human disagreement

**Code does:**
```python
mannwhitneyu(x=llm_human_disagreements, y=human_human_disagreements, alternative='greater')
```

SciPy's `mannwhitneyu` with `alternative='greater'` tests whether the distribution of `x` is stochastically greater than `y` — i.e., whether `P(X > Y) > 0.5`.

**Assessment: ✅ Correct.** The argument order and alternative are exactly right.

#### Consistency check across all MWU call sites

| Call site | x (first arg) | y (second arg) | alternative | Correct? |
|-----------|---------------|----------------|-------------|----------|
| `run_mannwhitneyu_test()` | LLM-Human | Human-Human | `'greater'` | ✅ |
| `_monte_carlo_random_judges()` | Random-Human | Human-Human | `'greater'` | ✅ |
| `_compute_stability_for_percentage()` | Random-Human | Human-Human | `'greater'` | ✅ |
| `_compute_llm_stability_for_percentage()` | LLM-Human | Human-Human | `'greater'` | ✅ |
| `_run_stability_single_seed()` | Random-Human | Human-Human | `'greater'` | ✅ |
| `_render_robustness_panels()` (Panel C) | LLM-Human | Human-Human | `'greater'` | ✅ |
| `plot_monte_carlo_robustness_multi()` | LLM-Human | Human-Human | `'greater'` | ✅ |

**All 7 call sites are consistent.** The model/random distribution is always `x`, human-human is always `y`, and the one-sided alternative is always `'greater'`.

---

### 3. Random Judge Generation

#### `init_random_judge()` (lines 142–165) and inline usages

**Code:**
```python
rng.integers(min_score, max_score + 1, size=...)
```

`np.random.Generator.integers(low, high)` draws from `[low, high)`. With `max_score + 1`, this yields uniform integers on `[min_score, max_score]` inclusive.

**Assessment: ✅ Correct.** Matches the README description of "uniformly random integer scores in the same rating range."

All inline random-judge generations (`_monte_carlo_random_judges`, `_compute_stability_for_percentage`, `_run_stability_single_seed`) use the same `rng.integers(self.min_score, self.max_score + 1, ...)` pattern. **Consistent.**

---

### 4. Monte Carlo Simulation

#### `_monte_carlo_random_judges()` (lines 748–827)

**README says:**
> Each Monte Carlo iteration corresponds to a *new item-wise random scoring function*. For each draw: Δ mean = mean(Random–Human) − mean(Human–Human), and p-value from a one-sided MWU test.

**Code does (per iteration):**
1. Draws one random integer rating per item via `rng_i.integers(...)` — ✅ fresh random judge
2. Computes Random–Human disagreements: `|R − H_j|` for all non-missing H — ✅
3. Computes delta: `mean(random_human_dis) − mean_hh` — ✅
4. Computes p-value: `mannwhitneyu(random_human_dis, human_human_dis, alternative="greater")` — ✅

**Key design choice:** Human–Human disagreements (`human_human_dis`) are computed *once* outside the loop and reused. This is correct because human ratings don't change; only the random judge varies per iteration.

**RNG isolation:** Each iteration gets its own child seed via `SeedSequence.spawn()`, guaranteeing parallel-safe, reproducible results regardless of execution order.

**Assessment: ✅ Methodologically correct.** Matches the README description exactly.

---

### 5. Human Stability Analysis

#### `_compute_stability_for_percentage()` (lines 1429–1561)

**README says:**
> For each percentage of sampled data: (1) Bootstrap-sample that percentage of rows, (2) Compute human–human and random-judge–human disagreements, (3) Run a one-sided MWU test, (4) Record a boolean decision, (5) Repeat until convergence.

**Code does (per iteration at percentage p):**
1. Bootstrap-samples `n_rows = max(1, int(len(df) * (p / 100)))` rows with replacement — ✅
2. Computes H-H disagreements on the *bootstrap sample* via `self.compute_human_disagreements(df=sample)` — ✅
3. Generates a fresh random judge on the bootstrap sample — ✅
4. Computes Random-Human disagreements on the sample — ✅
5. Runs MWU: `mannwhitneyu(pseudo_dis, human_dis, alternative="greater")` — ✅
6. Records `p_val > 0.05` as the boolean decision — ✅

**Critical correctness point:** Both H-H and Random-H disagreements are computed on the *same* bootstrap sample. This is correct — they must reflect the same subset of items.

**Assessment: ✅ Correct.** Each iteration draws *both* a new bootstrap sample and a new random judge. The convergence mechanism (split-half on decisions) is sound.

---

### 6. LLM Stability Analysis

#### `_compute_llm_stability_for_percentage()` (lines 1631–1725)

**Structurally mirrors** `_compute_stability_for_percentage()`, but replaces the random judge with the real LLM column.

**Per iteration:**
1. Bootstrap-sample rows — ✅
2. Compute H-H disagreements on the sample — ✅
3. Compute LLM-H disagreements on the sample via `self.compute_llm_human_disagreements(llm_col, df=sample)` — ✅
4. MWU test + record decision and delta-mean — ✅
5. Convergence on acceptance rate — ✅

**Assessment: ✅ Correct.** The LLM ratings come from the bootstrap sample (not the full dataset), ensuring the comparison is internally consistent.

---

### 7. Seed Sensitivity Analysis

#### `_run_stability_single_seed()` (lines 2199–2283)

This method duplicates the logic of `_compute_stability_for_percentage()` inline (rather than calling it). I verified line-by-line that the algorithms are **functionally identical**:

- Same RNG derivation: `np.random.default_rng(seed + p)`
- Same bootstrap sampling
- Same H-H and Random-H computation
- Same MWU test and decision recording
- Same convergence check via `_has_converged()`

#### `plot_human_stability_seed_robustness()` (lines 2286–2456)

- Generates `n_seeds` independent seeds via `rng.integers(0, 2**31, size=n_seeds)` — ✅
- Runs stability for each seed — ✅
- Computes mean + t-based CI: `stats.t.interval(confidence_level, df=n-1, loc=mean, scale=sem)` — ✅ appropriate for small n

**Assessment: ✅ Correct.** No methodological issues.

---

### 8. Convergence Mechanism

#### `_has_converged()` (lines 1370–1425)

**Two modes:**
- **Absolute:** `|mean_A − mean_B| < threshold` — straightforward
- **Relative:** `|mean_A − mean_B| < max(threshold × mean, threshold/10)` — proportional stability with a floor to prevent premature convergence when acceptance rate ≈ 0

**Assessment: ✅ Sound.** The floor `threshold/10` is a sensible guard. Without it, an acceptance rate near 0 would yield `effective_threshold ≈ 0`, which is nearly impossible to satisfy, and the simulation would always hit `max_iterations`.

---

### 9. Histogram Visualization

#### `_plot_disagreement_grid()` and `visualize_good_enough()`

- **Bins:** `np.arange(0, max_possible_disagreement + 2)` — integer bin edges [0, 1, 2, ..., max_dis+1], bin width = 1.0.
- **`density=True`:** With bin width 1.0, density values equal probabilities. The y-axis label "Probability" is therefore correct.
- **Side-by-side bars:** The model histogram uses `bins - 0.35` (shifted bin edges) with `width=0.35`. Since disagreements are integer-valued, each integer still falls into the corresponding shifted bin (e.g., value 0 falls in [-0.35, 0.65)). Normalization is unaffected since bin widths remain 1.0.

**Assessment: ✅ Correct.** The `width` kwarg is passed through `hist()` → `bar()` internally by matplotlib. Both distributions are properly normalized.

---

### 10. Missing Value Handling

| Method | NaN handling | Correct? |
|--------|-------------|----------|
| `_filter_minimum_raters()` | Drops rows with < 2 non-NaN human ratings | ✅ |
| `compute_human_disagreements()` | Masks pairs where either rating is NaN | ✅ |
| `compute_llm_human_disagreements()` | Masks pairs where human or LLM is NaN | ✅ |
| `_monte_carlo_random_judges()` | Masks by `~np.isnan(human_matrix)` (random judge never NaN) | ✅ |
| `_compute_stability_for_percentage()` | Same as above | ✅ |

---

## Caveats (Not Bugs)

### Caveat 1: Within-item dependency violates MWU independence assumption

The MWU test assumes independent observations. However, both distributions contain **dependent** values from the same item:

- **H-H:** For an item with 3 raters, the differences |H₁−H₂|, |H₁−H₃|, |H₂−H₃| are correlated (they share raters).
- **LLM-H:** For the same item, |LLM−H₁|, |LLM−H₂|, |LLM−H₃| are correlated (they share the LLM score).

**Impact:** P-values may be *anti-conservative* (somewhat too small) due to inflated effective sample size. This means the test could reject H₀ slightly more often than the nominal 5% level.

**Does the multi-layered validation fix this?**

No — the dependency violation exists in every single MWU call and cannot be removed without changing the approach. However, the framework's conclusions **do not rest on any single p-value**. What the additional analyses provide is resilience against being misled by a single (slightly biased) test:

1. **Monte Carlo robustness:** Thousands of random judges pass through the *same* dependency-affected test as the LLM. The relative positioning of the LLM vs. the random-judge cloud is therefore a **like-for-like comparison under the same bias**. If the dependency makes p-values slightly too small for *everyone*, the LLM's position relative to the random cloud is still meaningful.

2. **Stability analysis:** The trend of acceptance rate vs. sample size is informative even if the individual p-values at each sample size are slightly anti-conservative. The dependency affects every bootstrap iteration equally, so the *shape of the curve* (downhill, flat, erratic) is a valid diagnostic.

3. **Seed sensitivity:** The dependency is constant across seeds, so if the conclusion is stable across seeds, the dependency is not driving the result.

**In short:** The dependency makes absolute p-values slightly unreliable, but the framework draws conclusions from **patterns across many tests** (trends, distributions, relative positioning), not from any single p-value. That is the real defense.

**Severity: Low.** This is a known, accepted trade-off in the literature, not an implementation error. It is standard practice in NLP/evaluation research (e.g., inter-annotator agreement studies), and the README correctly frames the result as a "practical rule" rather than a strict mathematical guarantee.

---

### Caveat 2: Unequal distribution sizes between H-H and H-Judge

The two distributions fed to MWU can have different sizes depending on the number of human raters:

| Human raters (n) | H-H pairs per item | LLM-H pairs per item | Ratio |
|:-:|:-:|:-:|:-:|
| 2 | C(2,2) = 1 | 2 | 1:2 |
| 3 | C(3,2) = 3 | 3 | 1:1 |
| 5 | C(5,2) = 10 | 5 | 2:1 |
| 7 | C(7,2) = 21 | 7 | 3:1 |

With missing values the imbalance can be more uneven (e.g., a row with only 2 non-NaN human ratings contributes 1 H-H pair but 2 LLM-H pairs if the LLM rated that item).

**Is this a problem?** The MWU test handles unequal sample sizes natively — this is a well-known property of the test and is not an error. The reason this caveat is noted is that it *interacts* with Caveat 1: the dependency structures within each distribution differ (H-H pairs share two raters; LLM-H pairs share one LLM + one rater), and the distributions have different effective sample sizes. This is inherent to the pairwise-difference formulation and cannot be avoided without fundamentally changing the approach (e.g., item-level aggregation, which would discard distributional information).

**Severity: Low.** Inherent to the pairwise-difference approach. MWU itself is fine with unequal sizes.

---

## Documentation-Code Discrepancy

### `reseed(new_seed)` — listed in README but not implemented

The README's API reference (under "Utility") lists:

> `reseed(new_seed)` — Reseed the random number generators for reproducibility or variation.

This method does **not exist** in `evaluator.py`. The closest equivalent is `_set_global_seed()` (private) and `_init_seed()` (used during `__init__` only).

**Recommendation:** Either implement `reseed()` as a public method that updates `self.seed` and calls `_set_global_seed()`, or remove it from the README.

---

## Conclusion

The implementation in `evaluator.py` is **methodologically sound**. Every statistical computation — pairwise disagreements, MWU tests, Monte Carlo simulations, bootstrap stability analyses, and convergence diagnostics — faithfully implements the methodology described in the README. The MWU test direction and argument order are consistent across all 7 call sites. No errors, no silent bugs, no incorrect statistical reasoning.

The two caveats (within-item dependency and unequal distribution sizes) are inherent properties of the pairwise-difference evaluation paradigm, not implementation errors. They are widely accepted in the literature and are mitigated by the multi-layered validation approach (Monte Carlo robustness, stability analyses, seed sensitivity) that the framework provides.
