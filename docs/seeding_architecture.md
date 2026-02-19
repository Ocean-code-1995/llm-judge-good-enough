# Seeding Architecture — `core/llm_good_enough/evaluator.py`

> **Date:** 2026-02-19  
> **Scope:** How randomness is managed across the framework, why, and one noted inconsistency.

---

## Design Goal

The framework runs thousands of Monte Carlo iterations and bootstrap simulations, optionally in parallel across CPU cores. The seeding architecture must guarantee three things:

1. **Reproducibility** — given a fixed seed, every run produces identical results.
2. **Independence** — each iteration draws a genuinely fresh random judge; no iteration reuses another's randomness.
3. **Parallel determinism** — serial and parallel execution produce the same results, regardless of worker scheduling order.

---

## Why Independent RNG Streams Matter

Without independent streams, three things can go wrong:

**1. Correlated random judges invalidate the Monte Carlo.**
The Monte Carlo robustness analysis asks: "across many *independent* random judges, how does their disagreement with humans compare to the LLM's?" If iterations share or reuse the same RNG state, the random judges are correlated — the Monte Carlo cloud contracts artificially, and conclusions about where the LLM sits relative to the cloud become unreliable.

**2. A shared global RNG breaks under parallelism.**
If all workers draw from a single `np.random` global state, the sequence of random numbers depends on which worker runs first — i.e., on OS thread scheduling. The same code with the same seed produces different results on every run, destroying reproducibility. Worse, concurrent reads from a shared RNG can produce duplicate values or corrupt internal state.

**3. Reusing a fixed random judge column conflates randomness with structure.**
If every stability iteration tested the *same* random judge (e.g., the static `RANDOM_as_a_judge` column), you'd be measuring the bootstrap variability of one specific random judge's performance — not the variability *across* random judges. The acceptance rate would reflect that one judge's luck, not the general behavior of random scoring. Each iteration must draw a fresh random judge to properly estimate how often *any* random judge passes the test.

---

## Seed Lifecycle

### Initialization

```
__init__(seed=42)
    │
    ├─ self.seed = 42                          # stored as instance attribute
    ├─ np.random.seed(42)                      # legacy global RNG (for df.sample fallback etc.)
    ├─ random.seed(42)                         # Python stdlib RNG
    └─ RANDOM_as_a_judge column                # via default_rng(self.seed) — one-time static baseline
```

The baseline `RANDOM_as_a_judge` column is created once during initialization and is **only used for static bar plots** (`visualize_good_enough`, `plot_judges_grid`). Every simulation (Monte Carlo, stability, seed sensitivity) generates its own fresh random judges per iteration and never touches this column.

### Where `self.seed` flows

| Consumer | Derivation | RNG pattern |
|----------|-----------|-------------|
| Baseline column (`init_random_judge`) | `default_rng(self.seed)` | Single draw, stored in `self.df` |
| Monte Carlo (`_monte_carlo_random_judges`) | `SeedSequence(self.seed).spawn(iterations)` | One child stream per iteration |
| Human stability (`_compute_stability_for_percentage`) | `default_rng(self.seed + p)` | One stream per percentage |
| LLM stability (`_compute_llm_stability_for_percentage`) | `default_rng(self.seed + p)` | One stream per percentage |
| Seed sensitivity (`_run_stability_single_seed`) | `default_rng(seed_k + p)` | One stream per (seed, percentage) pair |

---

## Two RNG Patterns Used

### Pattern A — `SeedSequence.spawn()` (Monte Carlo)

```python
ss = np.random.SeedSequence(self.seed)
child_seeds = ss.spawn(iterations)          # e.g. 25,000 children
for i, child_ss in enumerate(child_seeds):
    rng_i = np.random.default_rng(child_ss) # independent stream for iteration i
    r = rng_i.integers(min_score, max_score + 1, size=n_items)
```

`SeedSequence.spawn()` is NumPy's recommended mechanism for creating independent RNG streams. It uses a counter-based derivation with cryptographic-quality mixing, formally guaranteeing that:

- Parent and children produce non-overlapping sequences.
- Children are mutually independent regardless of how many are spawned.
- Results are identical no matter which order the children are consumed.

This is the gold standard for parallel-safe randomness.

### Pattern B — `seed + p` offset (Stability analyses)

```python
rng = np.random.default_rng(seed + p)       # e.g. seed=42, p=10 → default_rng(52)
# All randomness for this percentage drawn from this one stream:
random_state = int(rng.integers(0, 2**32 - 1))   # bootstrap sample seed
pseudo_vals  = rng.integers(min_score, max_score + 1, size=n_items)  # random judge
```

Each percentage `p` (10, 20, 30, ...) produces a distinct integer `seed + p`, which is passed to `default_rng()`. Internally, `default_rng(int)` constructs `SeedSequence(int)` and derives the PCG64 initial state from it. So `default_rng(52)` and `default_rng(62)` go through the same cryptographic mixing — they just start from different input integers.

This is practically sound: SeedSequence's mixing function (based on ThreeFry/hashing) ensures that `SeedSequence(52)` and `SeedSequence(62)` produce unrelated output, even though the inputs differ by only 10. There is no formal guarantee of non-overlap (unlike `spawn()`), but the probability of correlation between streams is astronomically small — well below any practical concern.

---

## The Inconsistency

The Monte Carlo simulation uses Pattern A (`SeedSequence.spawn`), while all stability analyses use Pattern B (`seed + p`). Both produce independent RNG streams in practice, and both are correct. The inconsistency is stylistic, not functional.

### Why it doesn't matter

| Property | Pattern A (`spawn`) | Pattern B (`seed + p`) |
|----------|:---:|:---:|
| Reproducibility | Yes | Yes |
| Independence (practical) | Yes | Yes |
| Independence (formal guarantee) | Yes — by construction | No formal guarantee, but cryptographic mixing makes collision/correlation negligible |
| Parallel-safe | Yes | Yes |
| Execution-order invariant | Yes | Yes |

The only difference is in the **strength of the independence guarantee**:

- `spawn()` provides a **formal, mathematical** guarantee of non-overlapping sequences, backed by NumPy's SeedSequence counter mechanism.
- `seed + p` provides a **practical** guarantee via cryptographic mixing. For the input ranges used here (seed ∈ [0, 2³²), p ∈ [5, 200]), the probability of correlated streams is effectively zero.

### Why both exist

The Monte Carlo simulation is a single method that runs one loop over `iterations` — spawning all children upfront is natural and efficient. The stability analyses dispatch independent per-percentage computations (potentially to parallel workers), where each worker only needs its own stream — passing `seed + p` as a single integer is simpler and avoids pickling SeedSequence objects across process boundaries.

### Does it need to be fixed?

No. Both patterns are correct and produce the same quality of results. A future refactor could unify them under Pattern A for formal consistency, for example:

```python
ss = np.random.SeedSequence(self.seed)
children = ss.spawn(len(percentages))
for child, p in zip(children, percentages):
    rng = np.random.default_rng(child)
    ...
```

This would be a pure style change with no effect on statistical behavior or results.

---

## Fresh Random Judges — Separation of Concerns

A critical design invariant: **every simulation iteration generates its own random judge from scratch.** The static `RANDOM_as_a_judge` column (created once in `__init__`) is never read by any simulation method.

| Method | Random judge source | Uses `RANDOM_as_a_judge` column? |
|--------|---|:---:|
| `visualize_good_enough()` | Reads static column | Yes (static plot only) |
| `plot_judges_grid()` | Reads static column | Yes (static plot only) |
| `_monte_carlo_random_judges()` | `rng_i.integers(...)` per iteration | No |
| `_compute_stability_for_percentage()` | `rng.integers(...)` per iteration | No |
| `_run_stability_single_seed()` | `rng.integers(...)` per iteration | No |
| `_compute_llm_stability_for_percentage()` | No random judge (uses real LLM) | No |

The `RANDOM_as_a_judge` column does ride along inside `self.df` when passed to stability workers, but it is never referenced — the workers only access `self.human_cols` from the sample and generate their own `pseudo_vals`.

---

## Parallel Execution Detail

When `parallel=True`, joblib's `loky` backend spawns separate processes:

1. Each worker receives its own `p` value and a pickled copy of `self` + `df`.
2. Inside the worker: `rng = default_rng(seed + p)` creates a **local** RNG. No shared mutable state.
3. `df.sample(random_state=int(rng.integers(...)))` uses a per-call random state derived from the local RNG — not the global `np.random`.
4. Since each `p` is unique, each worker's entire random sequence is fully determined by `seed + p`, independent of when or on which core the worker runs.

Result: **serial == parallel**, guaranteed.
