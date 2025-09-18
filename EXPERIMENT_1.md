# Experiment 1 — Eggs Don’t Un-Break

## Hypothesis (H₁)
For current LLMs, **Contradiction Rate** CR(L) **increases** with generation length L on narratives containing irreversible events.

Null (H₀): CR does not increase with L.

## Setup
- Model (default): `Qwen/Qwen2.5-1.5B-Instruct` (CPU/MPS friendly).  
  Optional: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
- Decoding: temperature=0.7, top_p=0.95, max_new_tokens ∈ {64,128,256,512}, samples=3 per seed.
- Seeds: 50 short scenarios with explicit irreversible events in `data/seeds.jsonl`.

## Metrics
- **CR(L)**: fraction of seeds with ≥1 contradiction by length L (rule-based, optional NLI confirmation).
- Optional: **Repair Overhead** (future work).

## Procedure
1. Generate continuations for each seed at lengths L.
2. Run rule-based contradiction checks (and optional NLI).
3. Compute CR(L), bootstrap 95% CIs, and plot.

## Threats to Validity
- Rule bias / figurative language; mitigate via NLI.
- Prompt domain bias; use diverse seeds.
- Sampling variance; average across k samples.

## Results (to fill in after run)
- Table: CR(L) with CIs.
- Plot: `results/cr_curve.png`.
- 1–2 illustrative reversal examples.

## Next
- Add “forward vs reversed” perplexity gap.
- Try a simple constraint (“Remember the mug remains shattered.”) and compare CR slope.