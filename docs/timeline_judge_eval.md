# Case Study: Interpreting a Borderline LLM Judge Evaluation

> Real-world example of the framework producing a coherent "not good enough" verdict across all analysis layers.

---

## Results

**Human stability (curve goes down, converges at 100%):** The human baseline is confirmed usable. The test reliably rejects random judges at full sample size, which means the humans share a consistent signal. This is the prerequisite check passing — everything downstream is interpretable.

**LLM stability (curve goes down, settles at 38% acceptance at 100%):** This is the key result. At full data, the MWU test rejects the LLM 62% of the time and fails to reject it 38% of the time. This is a **borderline / not-good-enough** result. A genuinely human-like LLM would maintain high acceptance (80–100%) even as sample size grows. The downward trend tells you that small-sample acceptance earlier in the curve was partly due to low power, not genuine human-likeness. At full power, the test detects a real gap more often than not.

**Monte Carlo Panel C (LLM lands in the random judge band):** This is consistent with the 38% acceptance. The LLM's (Δ, p-value) on the full dataset falls where random judges typically land, meaning the test can't clearly separate it from random. A "good enough" LLM would sit well above the random cloud (high p-value, low Δ).

**Accuracy metrics (LLM–Human A: 66%, LLM–Human B: 59%, Human–Human: 74%):** This confirms the same story from a different angle. Humans agree with each other 74% of the time, but the LLM agrees with humans only 59–66% of the time. The LLM is better than pure random (which would be much lower depending on the scale), but measurably worse than human-level agreement.

---

## Synthesis

All four signals point the same direction:

| Signal | Says... |
|--------|---------|
| Human stability converges down | Humans are reliable, baseline is usable |
| LLM stability at 38% | LLM is detectably worse than humans at full power |
| Panel C: LLM in random cloud | LLM not clearly separable from random judges |
| Accuracy gap (59–66% vs 74%) | LLM agrees with humans less than humans agree with each other |

---

## Verdict

**The LLM is not good enough for this task.** It's not catastrophically bad (it's not at 0% acceptance), but the framework is detecting a real, consistent gap between LLM–human disagreement and human–human disagreement. The 38% acceptance means "sometimes the test can't catch it, but most of the time it can" — which is not the same as "human-like."

This is exactly the kind of result the framework is designed to surface. A naive accuracy-only evaluation might see 66% and think "not bad," but the disagreement-based analysis reveals the LLM systematically deviates from human judgment patterns more than humans deviate from each other.
