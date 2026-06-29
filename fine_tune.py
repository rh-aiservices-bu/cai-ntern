"""
LoRA fine-tuning script.

Loads training_data.csv produced by sdg_pipeline/pipeline.py and fine-tunes
a causal LM using LoRA via the PEFT + transformers Trainer stack.

Requirements:
    pip install transformers peft datasets accelerate bitsandbytes
"""

from kfp.dsl import component


@component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-pytorch-cuda-py312-rhel9@sha256:74e130efd4386125d852a69080b61a591899e068b0296814e3c99cc5fe2e44a2",
    packages_to_install=["transformers", "peft", "datasets", "accelerate", "bitsandbytes", "pandas"],
)
def fine_tune_component(
    base_model: str = "Qwen/Qwen2-0.5B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
):
    """KFP component: LoRA fine-tune base_model on /workspace/training_data.csv."""
    import re
    import sys
    from pathlib import Path

    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    TRAINING_DATA  = "/workspace/training_data.csv"
    OUTPUT_DIR     = "/workspace/lora_adapter"
    HOLDOUT_OUTPUT = "/workspace/holdout.csv"
    MAX_SEQ_LENGTH = 2048
    GRAD_ACCUM     = 2
    EVAL_SPLIT     = 0.1
    HOLDOUT_SPLIT  = 0.1
    LORA_DROPOUT   = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    LOAD_IN_4BIT   = False

    if not Path(TRAINING_DATA).exists():
        print(f"ERROR: {TRAINING_DATA} not found")
        sys.exit(1)

    def _parse_input_to_messages(input_str):
        messages = []
        for chunk in re.split(r'\n(?=User: |Assistant: )', input_str.strip()):
            if chunk.startswith("User: "):
                messages.append({"role": "user", "content": chunk[6:].strip()})
            elif chunk.startswith("Assistant: "):
                messages.append({"role": "assistant", "content": chunk[11:].strip()})
        return messages

    def format_example(row, tokenizer):
        messages = _parse_input_to_messages(row["input"])
        messages.append({"role": "assistant", "content": row["output"]})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    has_gpu = torch.cuda.is_available()
    use_4bit = LOAD_IN_4BIT and has_gpu
    print(f"Device     : {'GPU (' + torch.cuda.get_device_name(0) + ')' if has_gpu else 'CPU'}")
    print(f"4-bit quant: {use_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    df = pd.read_csv(TRAINING_DATA)
    print(f"Loaded {len(df)} examples  ({df['source'].value_counts().to_dict()})")

    holdout_df  = df.sample(frac=HOLDOUT_SPLIT, random_state=42)
    trainval_df = df.drop(holdout_df.index).reset_index(drop=True)
    holdout_df.to_csv(HOLDOUT_OUTPUT, index=False)
    print(f"Holdout: {len(holdout_df)} examples -> {HOLDOUT_OUTPUT}")

    trainval_df["text"] = trainval_df.apply(lambda row: format_example(row, tokenizer), axis=1)
    dataset = Dataset.from_pandas(trainval_df[["text"]], preserve_index=False)
    split    = dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)

    train_ds = split["train"].map(tokenize, batched=True, remove_columns=["text"])
    val_ds   = split["test"].map(tokenize,  batched=True, remove_columns=["text"])
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

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

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto" if has_gpu else None,
        trust_remote_code=True,
        dtype=torch.bfloat16 if (has_gpu and not use_4bit) else None,
    )
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    warmup_steps = max(1, int(0.1 * (len(train_ds) // (batch_size * GRAD_ACCUM)) * epochs))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=GRAD_ACCUM,
            gradient_checkpointing=has_gpu,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            bf16=has_gpu,
            use_cpu=not has_gpu,
            optim="adamw_torch",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="none",
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("=== Starting training ===")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")
