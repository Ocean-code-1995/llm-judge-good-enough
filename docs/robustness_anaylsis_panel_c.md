# 🧩 How Panel C (ΔMean vs p-value) is Computed and Interpreted

---

## 1️⃣ What Panel C Represents

Panel C in the robustness analysis plot shows the relationship between **how much more a model disagrees than humans (ΔMean)** and **how statistically significant that difference is (p-value)**, across multiple randomized evaluations.

It helps you see:
- How **consistent** the LLM’s performance is under randomness.
- How **different** it is from a random baseline judge.

---

## 2️⃣ Step-by-Step Computation

The computation happens inside the `visualize_robustness()` method.

---

### Step 1 — Precompute fixed disagreement distributions

Before the randomized loop begins, two disagreement distributions are computed once and remain constant:

1. **Human–Human disagreements**
   - Take all pairs of human judges.
   - Compute absolute differences in their scores per item.
   - Collect all these differences into one array.
   - Represents the natural variability among humans.

   ***Human–Human disagreements = |score_human_i − score_human_j| for all pairs (i < j)***

2. **LLM–Human disagreements**
   - Compute the absolute difference between the LLM’s score and each human judge’s score.
   - Collect all those differences into one array.
   - Fixed across all iterations since both the LLM and human scores are constant.

   ***LLM–Human disagreements = |score_LLM − score_human_i| for all human judges***

---

### Step 2 — Run randomized iterations for both models

A Monte Carlo-style loop is run several times (e.g., 25 iterations).  
In each iteration, two models are evaluated:
1. The **LLM Judge** (fixed ratings)
2. The **Random Judge** (new random ratings each loop)

For each iteration:

1. **Generate a new Random Judge**
   - Assign random integer scores between the min and max rating.
   - Compute its disagreements with each human judge.

   ***Random–Human disagreements = |score_Random − score_human_i| for all human judges***

2. **Compute ΔMean and p-value for both models**
   - For the **LLM Judge**, reuse the precomputed disagreements (values stay constant).
   - For the **Random Judge**, use the freshly generated disagreements.
   - For each model, calculate:

     - **ΔMean** = Average(Model–Human disagreements) − Average(Human–Human disagreements)
     - **p-value** = Mann–Whitney U test(Model–Human vs Human–Human, alternative = "greater")

   The ΔMean and p-value for the **LLM Judge** remain constant across iterations,  
   while the **Random Judge** values change each loop due to new random scores.

---

### Step 3 — Collect and visualize results

After all iterations:
- You have **one fixed pair (ΔMean, p-value)** for the LLM Judge.  
- You have **many (ΔMean, p-value) pairs** for the Random Judge, one per iteration.

Panel C plots these points:
- **X-axis:** ΔMean (how much more the model disagrees than humans)  
- **Y-axis:** p-value (how significant that difference is)  
- **Dashed line at p = 0.05:** significance threshold

This visualization shows how stable the LLM’s evaluation is compared to the range of outcomes expected from random behavior.


---

### Step 4 — Repeat for many iterations

This process repeats (e.g., 25 times):

- The **Random Judge** changes every iteration → produces a *cloud* of (ΔMean, p-value) points.
- The **LLM** uses fixed ratings → produces *one point* (since its disagreements don’t change).

All results are collected in a DataFrame:

| model | mean_diff | p_value |
|--------|------------|---------|
| LLM | ΔMeanₗₗₘ | pₗₗₘ |
| Random Judge | ΔMeanᵣₐₙdₒₘ | pᵣₐₙdₒₘ |
| ... | ... | ... |

---

## 3️⃣ Panel C Plot Construction

- **X-axis:** ΔMean (Average(Model–Human) − Average(Human–Human))  
  → “How much more the model disagrees than humans.”

- **Y-axis:** Mann–Whitney p-value  
  → “How statistically significant that difference is.”

- **Each point:** One random-seed trial (for the Random Judge) or one LLM evaluation.

- **Dashed line at y = 0.05:** Standard significance threshold.

---

## 4️⃣ How to Interpret Panel C

| Visual Feature | Interpretation |
|-----------------|----------------|
| **LLM point near 0 on X, above 0.05 on Y** | The LLM disagrees about as much as humans; no significant difference → “good enough.” |
| **LLM point below 0.05 line** | The LLM disagrees significantly more than humans → not good enough. |
| **Random Judge cloud (usually far right, low p)** | Baseline: random model disagrees far more and fails significance every time. |
| **Tight LLM point vs wide Random Judge cloud** | LLM performance is consistent; Random baseline is noisy. |

---

## 5️⃣ Summary

- **ΔMean** measures **effect size** (how much worse or better the model disagrees than humans).  
- **p-value** measures **statistical significance** (is that difference real or just noise).  
- **Panel C** combines both in a scatter plot:
  - The **LLM** should appear **left and high** (low ΔMean, high p).  
  - The **Random Judge** should appear **right and low** (high ΔMean, low p).  
  - The visual gap between them shows how robustly “human-like” the LLM is.

---
