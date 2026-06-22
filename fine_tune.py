"""
LoRA fine-tuning script.

Loads training_data.csv produced by sdg_pipeline/pipeline.py and fine-tunes
a causal LM using LoRA via the PEFT + transformers Trainer stack.

Requirements:
    pip install transformers peft datasets accelerate bitsandbytes
"""

from kfp.dsl import component


@component
def fine_tune_component():
    """KFP component: LoRA fine-tune on training_data.csv produced by sdg_component."""
    import fine_tune
    fine_tune.main()

import re
import sys
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# ── Configuration ──────────────────────────────────────────────────────────────
TRAINING_DATA = "training_data.csv"
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"   # swap to any HF causal LM
OUTPUT_DIR = "./lora_adapter"
MAX_SEQ_LENGTH = 2048

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 2                 # effective batch = BATCH_SIZE * GRAD_ACCUM
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1
LR_SCHEDULER = "cosine"
EVAL_SPLIT = 0.1               # fraction used for validation during training
HOLDOUT_SPLIT = 0.1            # fraction held out entirely for post-training evaluation
HOLDOUT_OUTPUT = "./holdout.csv"
LOAD_IN_4BIT = True            # set False if you have >= 24 GB VRAM
# ──────────────────────────────────────────────────────────────────────────────


def _parse_input_to_messages(input_str: str) -> list[dict]:
    """
    Parse the "input" column from training_data.csv back into a list of
    {role, content} dicts.

    The SDG pipeline formats multi-turn history as:
        User: <content>
        Assistant: <content>
        User: <content>
    """
    messages = []
    chunks = re.split(r'\n(?=User: |Assistant: )', input_str.strip())
    for chunk in chunks:
        if chunk.startswith("User: "):
            messages.append({"role": "user", "content": chunk[6:].strip()})
        elif chunk.startswith("Assistant: "):
            messages.append({"role": "assistant", "content": chunk[11:].strip()})
    return messages


def format_example(row: dict, tokenizer) -> str:
    """
    Build the full chat-formatted string for one training example using the
    model's own chat template (keeps formatting consistent with pre-training).
    """
    messages = _parse_input_to_messages(row["input"])
    messages.append({"role": "assistant", "content": row["output"]})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_dataset(tokenizer):
    """
    Load and format training_data.csv, then produce three splits:
      - holdout : saved to HOLDOUT_OUTPUT, never seen during training
      - train   : used for gradient updates
      - val     : used for loss tracking / early stopping during training

    The holdout set is the unbiased set you evaluate the final model against.
    The val set influences model selection (load_best_model_at_end), so it
    cannot serve as an impartial post-training benchmark.
    """
    if not Path(TRAINING_DATA).exists():
        print(f"ERROR: {TRAINING_DATA} not found. Run sdg_pipeline/pipeline.py first.")
        sys.exit(1)

    df = pd.read_csv(TRAINING_DATA)
    print(f"Loaded {len(df)} examples from {TRAINING_DATA}")
    print(df["source"].value_counts().to_string())

    # Carve out the holdout set first, before any model sees the data
    holdout_df = df.sample(frac=HOLDOUT_SPLIT, random_state=42)
    trainval_df = df.drop(holdout_df.index).reset_index(drop=True)
    holdout_df.to_csv(HOLDOUT_OUTPUT, index=False)
    print(f"Holdout set : {len(holdout_df)} examples saved to {HOLDOUT_OUTPUT}")

    trainval_df["text"] = trainval_df.apply(lambda row: format_example(row, tokenizer), axis=1)

    # Print a sample so you can verify formatting before committing to a full run
    print("\n--- Sample formatted example ---")
    print(trainval_df["text"].iloc[0][:600])
    print("--------------------------------\n")

    dataset = Dataset.from_pandas(trainval_df[["text"]], preserve_index=False)
    split = dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)

    train = split["train"].map(tokenize, batched=True, remove_columns=["text"])
    val = split["test"].map(tokenize, batched=True, remove_columns=["text"])
    print(f"Train: {len(train)}  Val: {len(val)}  Holdout: {len(holdout_df)}")
    return train, val


def build_model_and_tokenizer():
    """Load base model with optional 4-bit quantization and configure tokenizer."""
    has_gpu = torch.cuda.is_available()
    use_4bit = LOAD_IN_4BIT and has_gpu
    print(f"Device     : {'GPU (' + torch.cuda.get_device_name(0) + ')' if has_gpu else 'CPU'}")
    print(f"4-bit quant: {use_4bit}")

    print(f"Loading tokenizer from {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if use_4bit
        else None
    )

    print(f"Loading model from {BASE_MODEL} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto" if has_gpu else None,
        trust_remote_code=True,
        dtype=torch.bfloat16 if (has_gpu and not use_4bit) else None,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    return model, tokenizer, has_gpu


def main():
    print("=== LoRA Fine-Tuning ===")
    print(f"Base model : {BASE_MODEL}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"LoRA rank  : {LORA_R}  alpha={LORA_ALPHA}")
    print(f"Epochs     : {EPOCHS}  lr={LEARNING_RATE}  eff-batch={BATCH_SIZE * GRAD_ACCUM}")

    model, tokenizer, has_gpu = build_model_and_tokenizer()
    train_dataset, eval_dataset = load_dataset(tokenizer)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=has_gpu,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=max(1, int(WARMUP_RATIO * (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS)),
        bf16=has_gpu,
        use_cpu=not has_gpu,
        optim="adamw_torch",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",     # swap to "wandb" or "mlflow" for experiment tracking
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\n=== Starting training ===")
    trainer.train()

    print(f"\n=== Saving LoRA adapter to {OUTPUT_DIR} ===")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done. To use the adapter, load the base model and call:")
    print(f"  model = PeftModel.from_pretrained(base_model, '{OUTPUT_DIR}')")
