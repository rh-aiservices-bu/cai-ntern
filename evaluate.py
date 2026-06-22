"""
Holdout evaluation script.

Loads holdout.csv, generates responses from both the base model and the
fine-tuned LoRA model, then prints a side-by-side comparison and saves
results to eval_results.csv.

Usage:
    python evaluate.py
"""

from kfp.dsl import component


@component
def evaluate_component():
    """KFP component: side-by-side evaluation of base vs fine-tuned model on holdout set."""
    import evaluate
    evaluate.main()

import re
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Configuration ──────────────────────────────────────────────────────────────
HOLDOUT_DATA  = "./holdout.csv"
BASE_MODEL    = "Qwen/Qwen2-0.5B-Instruct"
ADAPTER_DIR   = "./lora_adapter"
OUTPUT_FILE   = "./eval_results.csv"
MAX_NEW_TOKENS = 512
# ──────────────────────────────────────────────────────────────────────────────


def _parse_input_to_messages(input_str: str) -> list[dict]:
    messages = []
    chunks = re.split(r'\n(?=User: |Assistant: )', input_str.strip())
    for chunk in chunks:
        if chunk.startswith("User: "):
            messages.append({"role": "user", "content": chunk[6:].strip()})
        elif chunk.startswith("Assistant: "):
            messages.append({"role": "assistant", "content": chunk[11:].strip()})
    return messages


def build_prompt(input_str: str, tokenizer) -> str:
    """Build a generation prompt from the conversation history (no final assistant turn)."""
    messages = _parse_input_to_messages(input_str)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(prompt: str, model, tokenizer) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_base_model(tokenizer):
    has_gpu = torch.cuda.is_available()
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto" if has_gpu else None,
        torch_dtype=torch.bfloat16 if has_gpu else None,
        trust_remote_code=True,
    )


def main():
    if not Path(HOLDOUT_DATA).exists():
        print(f"ERROR: {HOLDOUT_DATA} not found. Run fine_tune.py first.")
        return
    if not Path(ADAPTER_DIR).exists():
        print(f"ERROR: {ADAPTER_DIR} not found. Run fine_tune.py first.")
        return

    df = pd.read_csv(HOLDOUT_DATA)
    print(f"Evaluating {len(df)} holdout examples\n")

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model ...")
    base_model = load_base_model(tokenizer)
    base_model.eval()

    print("Loading fine-tuned model ...")
    finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    finetuned_model.eval()

    results = []
    for i, row in df.iterrows():
        print(f"\n{'='*60}")
        print(f"Example {i+1}/{len(df)}  (source: {row['source']})")
        print(f"{'='*60}")

        prompt = build_prompt(row["input"], tokenizer)

        # Show the last user turn as context
        messages = _parse_input_to_messages(row["input"])
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        print(f"\nUser: {last_user[:200]}{'...' if len(last_user) > 200 else ''}")

        print("\n--- Reference answer ---")
        print(row["output"][:400] + ("..." if len(str(row["output"])) > 400 else ""))

        print("\n--- Base model ---")
        base_response = generate(prompt, base_model, tokenizer)
        print(base_response[:400] + ("..." if len(base_response) > 400 else ""))

        print("\n--- Fine-tuned model ---")
        ft_response = generate(prompt, finetuned_model, tokenizer)
        print(ft_response[:400] + ("..." if len(ft_response) > 400 else ""))

        results.append({
            "source": row["source"],
            "user_input": last_user,
            "reference": row["output"],
            "base_response": base_response,
            "finetuned_response": ft_response,
        })

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")
