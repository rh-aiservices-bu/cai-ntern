"""
Holdout evaluation script.

Loads holdout.csv, generates responses from both the base model and the
fine-tuned LoRA model, then prints a side-by-side comparison and saves
results to eval_results.csv.
"""

from kfp.dsl import component


@component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-pytorch-cuda-py312-rhel9@sha256:74e130efd4386125d852a69080b61a591899e068b0296814e3c99cc5fe2e44a2",
    packages_to_install=["transformers", "peft", "bitsandbytes", "pandas"],
)
def evaluate_component(
    base_model: str = "Qwen/Qwen2-0.5B-Instruct",
    max_new_tokens: int = 512,
):
    """KFP component: side-by-side evaluation of base vs fine-tuned model on holdout set."""
    import re
    import sys
    from pathlib import Path

    import pandas as pd
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HOLDOUT_DATA = "/workspace/holdout.csv"
    ADAPTER_DIR  = "/workspace/lora_adapter"
    OUTPUT_FILE  = "/workspace/eval_results.csv"

    if not Path(HOLDOUT_DATA).exists():
        print(f"ERROR: {HOLDOUT_DATA} not found")
        sys.exit(1)
    if not Path(ADAPTER_DIR).exists():
        print(f"ERROR: {ADAPTER_DIR} not found")
        sys.exit(1)

    def _parse_input_to_messages(input_str):
        messages = []
        for chunk in re.split(r'\n(?=User: |Assistant: )', input_str.strip()):
            if chunk.startswith("User: "):
                messages.append({"role": "user", "content": chunk[6:].strip()})
            elif chunk.startswith("Assistant: "):
                messages.append({"role": "assistant", "content": chunk[11:].strip()})
        return messages

    has_gpu = torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_prompt(input_str):
        messages = _parse_input_to_messages(input_str)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(prompt, model):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("Loading base model ...")
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if has_gpu else None,
        torch_dtype=torch.bfloat16 if has_gpu else None,
        trust_remote_code=True,
    )
    base_model_obj.eval()

    df = pd.read_csv(HOLDOUT_DATA)
    print(f"Evaluating {len(df)} holdout examples")

    # Generate base model responses BEFORE wrapping with PeftModel.
    # PeftModel.from_pretrained modifies base_model_obj in place, so any
    # generate() call after that point will include LoRA weights regardless
    # of which object you call it on.
    print("Generating base model responses ...")
    prompts = [build_prompt(row["input"]) for _, row in df.iterrows()]
    base_responses = [generate(p, base_model_obj) for p in prompts]

    print("Loading fine-tuned model ...")
    finetuned_model = PeftModel.from_pretrained(base_model_obj, ADAPTER_DIR)
    finetuned_model.eval()

    results = []
    for i, row in df.iterrows():
        messages  = _parse_input_to_messages(row["input"])
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        base_resp = base_responses[i]
        ft_resp   = generate(prompts[i], finetuned_model)

        print(f"\n[{i+1}/{len(df)}] source={row['source']}")
        print(f"  User      : {last_user[:120]}")
        print(f"  Reference : {str(row['output'])[:120]}")
        print(f"  Base      : {base_resp[:120]}")
        print(f"  Finetuned : {ft_resp[:120]}")

        results.append({
            "source":             row["source"],
            "user_input":         last_user,
            "reference":          row["output"],
            "base_response":      base_resp,
            "finetuned_response": ft_resp,
        })

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"\nEval results saved to {OUTPUT_FILE}")
