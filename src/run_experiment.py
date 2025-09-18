import argparse, os
from typing import List, Dict
from tqdm import tqdm
from utils import ensure_dir, load_jsonl, save_jsonl, set_seed
from rollouts import load_lm, generate_samples
from rules import find_contradiction
from metrics import compute_cr
from plotting import plot_cr
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seeds", default="data/seeds.jsonl")
    ap.add_argument("--lengths", default="64,128,256,512")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_nli", action="store_true", help="Optional NLI confirmation (slower)")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    set_seed(args.seed)
    lengths = [int(x) for x in args.lengths.split(",")]

    seeds = load_jsonl(args.seeds)
    tok, model, device = load_lm(args.model)

    # 1) Generate samples at checkpoints
    gens = generate_samples(tok, model, device, seeds, lengths, args.samples, args.temperature, args.top_p)
    save_jsonl(os.path.join(args.outdir, "generations.jsonl"), gens)

    # 2) Rule-based contradiction scan
    eval_rows: List[Dict] = []
    for g in tqdm(gens, desc="Rule-check"):
        is_contra, reason = find_contradiction(g["text"])
        eval_rows.append({
            "seed_id": g["seed_id"],
            "length": g["length"],
            "sample_idx": g["sample_idx"],
            "is_contradiction": int(is_contra),
            "reason": reason
        })

    # Optional NLI pass (premise = earliest irreversible sentence; hypothesis = each later sentence)
    if args.use_nli:
        from nli_eval import NLI
        nli = NLI()
        # Very light heuristic: if rule flagged false, try to see if any sentence pair contradicts
        # (You can expand this later. We keep it minimal here.)
        pass  # placeholder for v1—rules are primary, NLI comes in v1.1

    # 3) Save per-checkpoint eval
    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(os.path.join(args.outdir, "eval_rows.csv"), index=False)

    # 4) Aggregate CR(L) with bootstrap CI
    df_cr = compute_cr(eval_rows)
    df_cr.to_csv(os.path.join(args.outdir, "cr_by_length.csv"), index=False)

    # 5) Plot
    plot_path = os.path.join(args.outdir, "cr_curve.png")
    plot_cr(df_cr, plot_path)

    # 6) Print summary
    print("\n=== Contradiction Rate by Length ===")
    print(df_cr.to_string(index=False))
    print(f"\nSaved: {plot_path}")
    print(f"Generations: results/generations.jsonl")
    print(f"Eval rows:   results/eval_rows.csv")

if __name__ == "__main__":
    main()
