# ***`When is an LLM-as-a-judge good enough?`***
---
---
## ***`Abstract`***

*Generative AI models allow us to generate human-like content, such as large language models (LLMs) generate texts. However, such created contend can only be used to automate processes if certain requirements regarding trustworthiness and correctness are fulfilled. In order to relieve people from controlling the created content, a concept known as LLM-as-a-judge is available. In this scenario, another instance of an LLM is prompted to act as a judge, and check a created text for certain quality requirements. However, such a judgement might not always align with a human judgement, which requires benchmarking the LLM-judge as well. Since a full alignment might not be achieved, the question is answered in this work, when an LLM-as-a-judge is good enough to judge a specific task fulfillment to break evaluating evaluations. For this purpose, it is made use of the fact that for evaluating generative content, judgements from different human evaluators can also differ as there might be no universal or unequivocally assessment of the quality, called a diversity of opinion. As long as the deviations of the LLM-judge from human estimations remain within the human diversity of opinion, it is suggested to call an LLM-judge as good enough.*

---

> ***`Paper accessible at the following link:`*** xxxxxxxx.xx

## ***`General Approach — Is the LLM "Good Enough"?`***

The goal is to test whether a language model's judgments align with human-level variability — that is, whether it behaves like *another human judge* rather than a random or systematically biased rater.

![General Approach](diagrams/svg/general_approach.svg)

#### 1. **Measure Inter-Human Disagreement**
For each task or item, multiple human raters provide scores.  
All **pairwise absolute differences** between human scores are computed, forming a distribution that captures the *natural variability* in human judgment — the “diversity of human opinion.”

#### 2. **Measure Human-LLM Disagreement**
For the same items, the **absolute difference** between each human score and the LLM’s score is calculated.  
This yields a second distribution describing how much the LLM diverges from humans.

#### 3. **Compare Distributions Statistically**
A **Mann–Whitney U test** compares the two distributions of disagreement magnitudes:  
- **Null hypothesis (H₀):** the LLM’s disagreement with humans is *not greater* than the typical disagreement among humans — the LLM behaves within normal human variability.  
- **Alternative hypothesis (H₁):** the LLM’s disagreement with humans is *greater* than that among humans — meaning the LLM and humans do **not agree** to the same extent as humans agree with each other.  

If the resulting *p*-value is **high (≥ 0.05)**, there is no evidence that the LLM’s disagreement differs from human-level variability — suggesting it is *“good enough.”*  
If it is **low (< 0.05)**, the LLM’s disagreement is significantly larger, indicating it diverges meaningfully from human judgment and is *not yet human-like.*


#### 4. **Establish a Random Baseline**
To calibrate expectations, a **random judge** is simulated by assigning scores uniformly across the rating range.  
This random baseline provides a clear contrast — it typically shows high disagreement and significant differences from humans, marking what “not good enough” looks like.

> **In essence:**  
> The method tests whether an LLM’s variability in judgment falls within the *natural human range* rather than resembling random noise.


## ***Getting Started***

### ***`0. Repository Structure`***
```text
LLM-AS-A-JUDGE-GOOD-ENOUGH
│
├── README.md
├── __init__.py
├── llm_as_a_judge
│   ├── inference.py
│   └── llm_as_a_judge.py
│
├── llm_good_enough
│   ├── __init__.py
│   ├── evaluator.py
│   └── example.ipynb
│
├── movielens
│   ├── data.zip
│   ├── prompts
│   │   ├── movielen.txt
│   └── src
│       └── analysis.ipynb
│
├── newsroom
│   ├── data.zip
│   ├── prompts
│   │   ├── Coherence_prompt.txt
│   │   ├── Fluency_prompt.txt
│   │   ├── Informativeness_prompt.txt
│   │   └── Relevance_prompt.txt
│   └── src
│       └── analysis.ipynb
│
├── politifact
│   ├── data.zip
│   ├── prompts
│   │   ├── politifact.txt
│   └── src
│       └── analysis.ipynb
│
├─── wmt-human
│    ├── data.zip
│    ├── prompts
│    │   └── prompt.txt
│    └── src
│        └── analysis.ipynb
│
└── requirements.txt
```

### ***`1. Clone Repository`***
```bash
git clone https://github.com/Ocean-code-1995/LLM_as_a_Judge_good_enough.git
```

```bash
cd path/to/.../LLM_as_a_Judge_good_enough
```

### ***`2. Install dependencies`***

##### Create conda envirnoment:
```bash
conda create --name llm-good-enough python=3.11.9
```

##### Activate virtual environment:
```bash
conda activate llm-good-enough
```
##### Install dependencies:
```bash
pip install -r requirements.txt
```

### ***`3. Usage Example`***

#### ***3.1 Run LLM-as-a-judge inference***

```bash
cd llm_as_a_judge
```

```bash
python llm_as_a_judge.py
```

#### ***3.2 Is the selected LLM-as-a-judge good enough?***

```bash
touch notebook_name.ipynb
```

```python
import pandas as pd
from llm_good_enough import LLMGoodEnough

# Load your DataFrame (replace with your own data)
df = pd.read_csv('your_data.csv')

# Initialize evaluator
LLM_Evaluator = LLMGoodEnough(
    df=df,
    human_cols=['human_#1', 'human_#2', 'human_#3']  # replace with your human judge columns
    min_score=0,
    max_score=6
)

# visualize LLM-as-a-judge good enough
LLM_Evaluator.visulize_good_enough(
    llm_col="GPT_as_a_judge",          # LLM-as-a-judge column name
    y_lim=0.6,                         # y-axis limit -> probability (0-1)
)
```

`See the following notebooks for more detailed examples:`

    - LLM-as-a-Judge/LLM-as-a-Judge-good-enough/llm_good_enough/example.ipynb
    - LLM-as-a-Judge/LLM-as-a-Judge-good-enough/wmt-human/src/analysis.ipynb
    - LLM-as-a-Judge/LLM-as-a-Judge-good-enough/newsroom/src/analysis.ipynb


---

## ***`API Reference — Public Methods`***

The `LLMGoodEnough` class provides the following public methods:

### **Core Computation**

| Method | Description |
|--------|-------------|
| `compute_human_disagreements()` | Compute all pairwise absolute differences between human raters. Returns the "diversity of opinion" distribution. |
| `compute_llm_human_disagreements(llm_col)` | Compute absolute differences between a specific LLM and all human raters. |
| `compute_all_llm_disagreements()` | Compute LLM-human disagreements for all LLM columns at once. |
| `run_mannwhitneyu_test(llm_dis, human_dis)` | Run Mann-Whitney U test comparing LLM-human vs human-human disagreements. Returns p-value. |
| `summarize(llm_col)` | Print summary statistics (means, p-value) for a specific LLM. |

### **Visualization**

| Method | Description |
|--------|-------------|
| `visualize_good_enough(llm_col, y_lim)` | Main visualization comparing LLM vs random judge against human baseline. |
| `plot_judges_grid(y_lim)` | Grid comparing multiple LLM judges + random baseline in one figure. |
| `plot_monte_carlo_robustness(llm_col, iterations)` | Monte Carlo robustness analysis for a single LLM (3-panel figure). |
| `plot_monte_carlo_robustness_multi(llm_cols, iterations)` | Monte Carlo robustness analysis comparing multiple LLMs simultaneously. |

### **Human Stability Analysis**

| Method | Description |
|--------|-------------|
| `plot_human_stability_analysis()` | Test if humans are stable/coherent annotators using adaptive sampling until convergence. |
| `plot_human_stability_seed_robustness(n_seeds)` | Seed sensitivity analysis — test robustness of stability conclusions across multiple random seeds. |

### **Utility**

| Method | Description |
|--------|-------------|
| `reseed(new_seed)` | Reseed the random number generators for reproducibility or variation. |
| `init_random_judge(min_score, max_score)` | Initialize the random baseline judge column. |

---

## ***`Analysis Methodologies`***


### **Disagreement Distribution Comparison**
Implemented in `visualize_good_enough()`, this plot directly compares the **distribution of human–human**, **LLM–human**, and **random–human** disagreements.  
It visually illustrates whether the LLM's disagreement pattern overlaps with natural human variability or drifts toward random behavior.

### **Multi-Model Comparison Grid**
Implemented in `plot_judges_grid()`, this method extends the same logic to multiple candidate models.  
It allows quick visual comparison across several LLMs to identify which behave most like human judges.


### ***Robustness Visualization***
The robustness visualization assesses how *consistent* and *human-like* the LLM’s judgments are across multiple randomized evaluations.  
Each iteration reseeds the random judge, recomputes disagreement distributions, and runs a Mann–Whitney U test comparing the LLM and random judge against human disagreement.  
For every run, two values are recorded per model:
- The **Δ mean disagreement** = mean(Model–Human) − mean(Human–Human)  
- The **p-value** from the Mann–Whitney test.  

These values are aggregated across runs to generate the following three panels:

#### **Panel (A) — P-value Distribution**
**How it’s computed:**  
All *p*-values from repeated runs are collected and plotted as distributions (via kernel density estimation) for the LLM and the random judge.  

**Interpretation:**  
This panel shows how often each judge’s disagreement with humans is *statistically different* from human–human disagreement.  
If most values lie **above the 0.05 dashed line**, it means differences are not significant — the judge behaves within human variability.  
If they lie **below**, disagreements are consistently significant — the judge diverges from humans.

**Why it’s useful:**  
It reveals the *statistical reliability* of the LLM’s alignment with human judgment across random seeds.  
A strong model’s curve should sit high and right (mostly non-significant), while the random baseline clusters left and below 0.05.

---

#### **Panel (B) — Δ Mean Disagreement (Boxplot)**
**How it’s computed:**  
For each run, compute the Δ mean disagreement — the difference in average disagreement between each model and the human baseline.  
The resulting values across runs are summarized as boxplots for the LLM and random judge.

**Interpretation:**  
This panel shows how much and how consistently the model’s disagreement deviates from human–human variability.  
Δ mean ≈ 0 → behaves like humans.  
Δ mean > 0 → disagrees more than humans.  
Δ mean < 0 → disagrees less (possibly over-consistent).

**Why it’s useful:**  
It visualizes *effect size stability* — whether the LLM’s deviation from humans is small and consistent (good) or large and erratic (bad).  
A narrow, centered box for the LLM and a wide, higher one for the random judge indicate robustness.

---

#### **Panel (C) — Δ Mean vs P-value (Scatterplot)**
**How it’s computed:**  
Each run provides a paired (Δ mean, *p*-value). These pairs are plotted — x-axis = Δ mean (effect size), y-axis = *p*-value (significance).

**Interpretation:**  
This panel links the *size* of disagreement with the *strength* of statistical evidence for it.  
Points high on the plot (large *p*) indicate human-like behavior; points low and right (large Δ, small *p*) show strong divergence.

**Why it’s useful:**  
It bridges *practical difference* and *statistical certainty*, showing whether large deviations consistently translate into significant differences.  
Ideally, the LLM clusters near the top (non-significant, human-like) while the random judge clusters lower and farther right (significantly worse).

---

> **In essence:**  
> Panels (A), (B), and (C) together evaluate whether the LLM's disagreement pattern is *consistently within human variability*,  
> and whether that conclusion remains *robust* across randomized baselines.

---

### ***Human Stability Analysis***

Before evaluating an LLM judge, it's critical to verify that the human annotations themselves form a reliable baseline. The **Human Stability Analysis** answers: *"Are humans judging coherently, or is the dataset too small/noisy to evaluate an LLM reliably?"*

Implemented in `plot_human_stability_analysis()`, this method uses **adaptive sampling**: for each sample percentage, bootstrap iterations continue until the acceptance rate converges (split-half difference < threshold) rather than using fixed iteration counts.

#### **How It Works**
For each percentage of sampled data (5%, 10%, …, 100%):
1. Bootstrap-sample that percentage of rows
2. Compute human–human and random-judge–human disagreements
3. Run MWU test (accepted if p > 0.05)
4. Repeat until split-half acceptance rates converge or max_iterations reached

#### **What the Plot Shows**
- **X-axis:** Percentage of the dataset sampled
- **Y-axis:** Acceptance rate (how often the random judge is "accepted" as human-like)
- **Green dot:** Converged — the acceptance rate stabilized
- **Red dot:** Max iterations reached — did not converge
- **Annotations:** Number of iterations needed (reveals "difficulty" per sample size)

#### **Interpretation — Trend Patterns**

| Trend | Meaning |
|-------|---------|
| **Downhill** (high→low) | Expected for random judges. MWU test gains power with more data — humans are distinguishable from random. ✅ Methodology working. |
| **Flat high** (~1.0) | Random judge always accepted — humans are too noisy to distinguish from random. ⚠️ Dataset may not be suitable. |
| **Flat low** (~0.0) | Random judge always rejected — humans are very consistent. ✅ Strong baseline. |
| **Uphill** | Suspicious — investigate data quality. |
| **Erratic** (many red) | High variance — may need more data or iterations. |

> **Key insight:** A **converging downtrend** confirms three things:
> 1. ✅ **Enough data** — the statistical test gains power with larger samples
> 2. ✅ **Reliable humans** — annotators share a consistent judgment pattern (not random noise)
> 3. ✅ **Valid baseline** — the dataset is suitable for benchmarking LLM judges
>
> If humans were just noise (like the random judge), acceptance would stay flat and high (~1.0) regardless of sample size — because you can never distinguish noise from noise.

---

### ***Seed Sensitivity Analysis***

Results can vary across different random initializations. The **Seed Sensitivity Analysis** tests how robust conclusions are across multiple seeds.

Implemented in `plot_human_stability_seed_robustness(n_seeds)`, this method:
1. Runs the stability analysis with N different random seeds
2. Collects acceptance rates for each percentage across all seeds
3. Plots mean ± confidence interval

#### **What the Plot Shows**
- **Blue band:** 95% CI across seeds
- **Blue line:** Mean acceptance rate
- **Annotations:** % of seeds that converged at each sample size

#### **Interpretation**

| Pattern | Meaning |
|---------|---------|
| Tight CI band | Robust conclusion — doesn't depend on random seed ✅ |
| Wide CI band | Sensitive to initialization — results may be unstable ⚠️ |
| Consistent downhill trend | Methodology is reliable across seeds ✅ |
| High convergence % (green annotation) | Estimates are stable at that sample size |
| Low convergence % (orange annotation) | That sample size is noisy/difficult |

> **In essence:** This analysis confirms whether your stability conclusions hold regardless of which random seed you happened to use.
