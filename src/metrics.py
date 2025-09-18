from typing import List, Dict
import pandas as pd
import numpy as np

def compute_cr(rows: List[Dict]) -> pd.DataFrame:
    """
    rows: dicts with keys [seed_id, length, is_contradiction]
    Returns CR per length with bootstrap 95% CI.
    """
    df = pd.DataFrame(rows)
    out = []
    for L, g in df.groupby("length"):
        # Aggregate to per-seed: any contradiction across samples for that seed at <= L
        # Expect rows already at each checkpoint; ensure per-seed-any
        seed_any = g.groupby("seed_id")["is_contradiction"].max().values.astype(int)
        p = seed_any.mean()
        # bootstrap
        B = 1000
        bs = np.random.choice(seed_any, (B, len(seed_any)), replace=True).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append({"length": L, "CR": p, "CR_lo": lo, "CR_hi": hi, "n_seeds": len(seed_any)})
    return pd.DataFrame(out).sort_values("length")
