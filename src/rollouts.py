from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_lm(model_name: str, device=None):
    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device.type!="cpu" else torch.float32)
    model.to(device)
    model.eval()
    return tok, model, device

def promptify(seed_text: str) -> str:
    # Simple system/user format; small instruct models behave well with plain prompts too
    return (seed_text + "\nContinue the story realistically, maintaining consistency with irreversible events. "
            "Do not contradict earlier facts.\n")

@torch.inference_mode()
def generate_samples(
    tok, model, device, seeds: List[Dict],
    lengths: List[int], samples_per_seed: int = 3,
    temperature: float = 0.7, top_p: float = 0.95, max_input_tokens: int = 512
) -> List[Dict]:
    outputs = []
    for seed in tqdm(seeds, desc="Generating"):
        base = promptify(seed["text"])
        enc = tok(base, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(device)
        for s in range(samples_per_seed):
            gen = model.generate(
                **enc,
                do_sample=True, temperature=temperature, top_p=top_p,
                max_new_tokens=max(lengths), pad_token_id=tok.eos_token_id
            )
            full = tok.decode(gen[0], skip_special_tokens=True)
            # slice checkpoints by token count (approx using tokenizer on the continuation)
            cont = full[len(base):]
            cont_tokens = tok(cont, add_special_tokens=False).input_ids
            for L in lengths:
                cut_ids = cont_tokens[:L]
                cut_text = tok.decode(cut_ids, skip_special_tokens=True)
                outputs.append({
                    "seed_id": seed["id"],
                    "seed_text": seed["text"],
                    "sample_idx": s,
                    "length": L,
                    "text": seed["text"] + "\n" + cut_text.strip()
                })
    return outputs
