import os, json, random
from typing import List, Dict

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def set_seed(seed: int = 42):
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
