import matplotlib.pyplot as plt
import pandas as pd

def plot_cr(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(6,4))
    L = df["length"].astype(int)
    y = df["CR"]
    lo = df["CR_lo"]
    hi = df["CR_hi"]
    plt.plot(L, y, marker="o")
    plt.fill_between(L, lo, hi, alpha=0.2, linewidth=0)
    plt.xlabel("Generation length (tokens)")
    plt.ylabel("Contradiction Rate (CR)")
    plt.title("Entropy proxy: Contradictions rise with horizon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
