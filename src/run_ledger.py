import re, os, json, random, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from rollouts import load_lm
from utils import ensure_dir, set_seed

# ---------- Scenario + reference ----------

@dataclass
class LedgerState:
    Cash: int
    Inventory: int
    Payables: int

def gen_random_scenarios(n: int, seed: int = 123) -> List[Dict]:
    random.seed(seed)
    scenarios = []
    for i in range(n):
        s0 = LedgerState(
            Cash=random.randint(80, 200),
            Inventory=random.randint(10, 60),
            Payables=random.randint(20, 100),
        )
        T = random.randint(20, 40)
        txs = []
        for t in range(T):
            kind = random.choice(["buy", "return", "sell", "pay", "borrow", "waste"])
            if kind == "buy":
                qty = random.randint(2, 8)
                price = random.randint(1, 4) * qty
                txs.append(f"Buy {qty} supplies using Cash for {price}.")
            elif kind == "return":
                qty = random.randint(1, 5)
                dec = random.randint(1, 3) * qty
                txs.append(f"Return {qty} supplies to vendor; Payables decreases by {dec}.")
            elif kind == "sell":
                qty = random.randint(1, 6)
                rev = random.randint(3, 6) * qty
                txs.append(f"Sell {qty} supplies for {rev}; Cash increases, Inventory decreases.")
            elif kind == "pay":
                amt = random.randint(5, 20)
                txs.append(f"Pay vendor {amt} from Cash; Payables decreases by {amt}.")
            elif kind == "borrow":
                amt = random.randint(10, 30)
                txs.append(f"Borrow {amt}; Cash increases by {amt}; Payables increases by {amt}.")
            elif kind == "waste":
                qty = random.randint(1, 4)
                txs.append(f"Waste {qty} supplies due to damage; Inventory decreases.")
        scenarios.append({
            "id": f"ledger_{i:03d}",
            "init": {"Cash": s0.Cash, "Inventory": s0.Inventory, "Payables": s0.Payables},
            "transactions": txs
        })
    return scenarios

def apply_tx(state: LedgerState, tx: str) -> LedgerState:
    s = LedgerState(**vars(state))
    m = re.search(r"Buy (\d+) supplies .* for (\d+)", tx, re.I)
    if m:
        qty, price = int(m.group(1)), int(m.group(2))
        s.Cash -= price; s.Inventory += qty; return s
    m = re.search(r"Return (\d+) supplies .* Payables decreases by (\d+)", tx, re.I)
    if m:
        qty, dec = int(m.group(1)), int(m.group(2))
        s.Inventory -= qty; s.Payables -= dec; return s
    m = re.search(r"Sell (\d+) supplies .* for (\d+)", tx, re.I)
    if m:
        qty, revenue = int(m.group(1)), int(m.group(2))
        s.Inventory -= qty; s.Cash += revenue; return s
    m = re.search(r"Pay vendor (\d+) from Cash; Payables decreases by \1", tx, re.I)
    if m:
        amt = int(m.group(1))
        s.Cash -= amt; s.Payables -= amt; return s
    m = re.search(r"Borrow (\d+); Cash increases by \1; Payables increases by \1", tx, re.I)
    if m:
        amt = int(m.group(1))
        s.Cash += amt; s.Payables += amt; return s
    m = re.search(r"Waste (\d+) supplies", tx, re.I)
    if m:
        qty = int(m.group(1))
        s.Inventory -= qty; return s
    return s

def rollout_reference(init: Dict[str,int], txs: List[str]) -> List[LedgerState]:
    st = LedgerState(**init)
    hist = [LedgerState(**vars(st))]
    for tx in txs:
        st = apply_tx(st, tx); hist.append(LedgerState(**vars(st)))
    return hist

# ---------- MC options ----------

def fmt_state(s: LedgerState) -> str:
    return f"Cash={s.Cash}, Inventory={s.Inventory}, Payables={s.Payables}"

def make_distractors(correct: LedgerState) -> List[LedgerState]:
    # simple but effective: perturb one field each, small deltas
    deltas = [-7, -5, -3, +3, +5, +7]
    def vary(field: str) -> LedgerState:
        s = LedgerState(**vars(correct))
        d = random.choice(deltas)
        setattr(s, field, getattr(s, field) + d)
        return s
    ds = [vary("Cash"), vary("Inventory"), vary("Payables")]
    # ensure uniqueness vs correct
    uniq = []
    for s in ds:
        if (s.Cash, s.Inventory, s.Payables) != (correct.Cash, correct.Inventory, correct.Payables):
            uniq.append(s)
    # If any collapsed onto correct, tweak again
    while len(uniq) < 3:
        uniq.append(vary(random.choice(["Cash","Inventory","Payables"])))
    return uniq[:3]

def build_mc_chat(tok, step_k: int, prev: LedgerState, tx: str, options: List[str]):
    system = (
        "You are a precise bookkeeper. Choose the correct next state after applying the transaction.\n"
        "Answer with a single letter: A, B, C, or D. No explanation."
    )
    user = (
        f"Step {step_k}\n"
        f"Current state: {fmt_state(prev)}\n"
        f"Transaction: {tx}\n\n"
        "Which option is the correct next state?\n"
        f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n\n"
        "Answer (A/B/C/D) only:"
    )
    return tok.apply_chat_template(
        [{"role":"system","content":system},{"role":"user","content":user}],
        add_generation_prompt=True, return_tensors="pt"
    )

LETTER_RE = re.compile(r"\b([ABCD])\b", re.I)

def ask_mc(tok, model, device, enc, temperature: float, max_new_tokens: int):
    out = model.generate(
        enc.to(device),
        do_sample = temperature > 0.0,
        temperature = max(temperature, 1e-5),
        top_p = 0.9,
        max_new_tokens = max_new_tokens,
        pad_token_id = tok.eos_token_id,
        eos_token_id = tok.eos_token_id,
    )
    full = tok.decode(out[0], skip_special_tokens=True)
    prompt_txt = tok.decode(enc[0], skip_special_tokens=True)
    new = full[len(prompt_txt):].strip()
    m = LETTER_RE.search(new)
    if not m:
        return None, new
    return m.group(1).upper(), new

# ---------- Metrics / runner ----------

def l1(a: LedgerState, b: LedgerState) -> int:
    return abs(a.Cash-b.Cash)+abs(a.Inventory-b.Inventory)+abs(a.Payables-b.Payables)

def run_episode(tok, model, device, scen: Dict, temperature: float, max_new_tokens: int):
    init = scen["init"]; txs = scen["transactions"]
    ref_hist = rollout_reference(init, txs)
    prev = LedgerState(**init)
    rows = []
    first_err = None

    for k, tx in enumerate(txs, start=1):
        # correct next
        corr = ref_hist[k]
        # candidates
        distractors = make_distractors(corr)
        cands = [fmt_state(corr)] + [fmt_state(d) for d in distractors]
        # shuffle options
        idxs = list(range(4)); random.shuffle(idxs)
        options = [cands[i] for i in idxs]
        correct_letter = "ABCD"[idxs.index(0)]  # position of true state

        enc = build_mc_chat(tok, k, prev, tx, options)
        letter, raw = ask_mc(tok, model, device, enc, temperature, max_new_tokens)
        if letter is None:
            # treat as wrong with large deviation
            chosen = LedgerState(10**9,10**9,10**9)
            ok = 0
        else:
            pos = "ABCD".index(letter)
            ok = int(letter == correct_letter)
            # parse chosen state to continue episode from model's belief
            m = re.search(r"Cash=(-?\d+),\s*Inventory=(-?\d+),\s*Payables=(-?\d+)", options[pos])
            if m:
                chosen = LedgerState(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            else:
                chosen = LedgerState(10**9,10**9,10**9); ok = 0

        dev = l1(chosen, corr)
        if ok == 0 and first_err is None: first_err = k
        prev = chosen  # continue from model's chosen state (lets errors compound)

        rows.append({
            "scenario_id": scen["id"], "step": k, "ok": ok, "dev_L1": dev,
            "correct": fmt_state(corr), "chosen": fmt_state(prev),
            "letter": letter if letter else "", "correct_letter": correct_letter
        })
    return rows, first_err

def aggregate(rows: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows)
    acc = df.groupby("step")["ok"].mean().reset_index().rename(columns={"ok":"accuracy"})
    # survival: fraction with no mistake up to k
    first_err = df[df["ok"]==0].groupby("scenario_id")["step"].min().reset_index().rename(columns={"step":"first_error"})
    ids = df["scenario_id"].unique()
    surv = []
    max_k = df["step"].max()
    for k in range(1,max_k+1):
        alive = 0
        for sid in ids:
            r = first_err[first_err["scenario_id"]==sid]
            if r.empty or r["first_error"].iloc[0] > k: alive += 1
        surv.append({"step":k, "survival": alive/len(ids)})
    return acc, pd.DataFrame(surv)

def plot_curve(df, x, y, title, out):
    plt.figure(figsize=(6,4))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel(x.capitalize()); plt.ylabel(y.capitalize())
    plt.title(title); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=180)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--n_scen", type=int, default=20)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--outdir", default="results/ledger_mc")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_new_tokens", type=int, default=4)
    args = ap.parse_args()

    ensure_dir(args.outdir); set_seed(args.seed)
    tok, model, device = load_lm(args.model)

    all_rows = []
    scenarios = gen_random_scenarios(args.n_scen, args.seed)
    for s in tqdm(scenarios, desc="Scenarios"):
        for ep in range(args.samples):
            rows, _ = run_episode(tok, model, device, s, args.temperature, args.max_new_tokens)
            for r in rows: r["episode"] = ep
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.outdir, "eval_rows.csv"), index=False)

    acc, surv = aggregate(all_rows)
    acc.to_csv(os.path.join(args.outdir, "accuracy_by_step.csv"), index=False)
    surv.to_csv(os.path.join(args.outdir, "survival_by_step.csv"), index=False)

    plot_curve(acc, "step", "accuracy", "MC-Ledger accuracy vs step", os.path.join(args.outdir, "accuracy_curve.png"))
    plot_curve(surv, "step", "survival", "MC-Ledger survival vs step", os.path.join(args.outdir, "survival_curve.png"))

    print("\n=== Accuracy (first/last) ===")
    print(acc.head(3).to_string(index=False)); print(acc.tail(3).to_string(index=False))
    print("\n=== Survival (first/last) ===")
    print(surv.head(3).to_string(index=False)); print(surv.tail(3).to_string(index=False))
    print(f"\nSaved: {args.outdir}")
if __name__ == "__main__":
    main()