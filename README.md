# The Thermodynamic Manifesto (v0)

**Thesis.** LLMs are masters of the local, novices of the global. As generation length grows, **informational entropy** (loss of global coherence / causal consistency) rises unless actively countered. Intelligence at scale requires respecting the **arrow of time** (irreversibility).

**Operational definition (v0).** We quantify entropy as **Contradiction Rate (CR)** over horizon length on narratives that contain **irreversible events** (e.g., “the mug shattered”, “the egg was cooked”). A contradiction is any later statement that *undoes* the irreversible state (e.g., mug becomes “as good as new”).

**Experiment 1 (Eggs Don’t Un-Break).**  
Seed short irreversible scenarios → generate continuations to growing lengths → detect reversals via rules (and optionally NLI) → plot **CR vs length**. Expect CR to increase with length.

See `EXPERIMENT_1.md` for the exact protocol and reporting template.