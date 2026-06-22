"""
SDG Hub pipeline: fetch sessions from MLflow, build conversation training examples,
judge quality with an LLM, and generate synthetic fine-tuning data.

Steps:
  1. Fetch     — pull traces from MLflow and group them into sessions
  2. Unroll    — for each session, create sliding-window conversation examples
  3. Judge     — LLM scores each example 1-10; keep those >= 7
  4. Generate  — produce NUM_GENERATIONS new examples per high-quality seed
  5. Save      — write curated examples to CSV and synthetic data to JSONL

Requirements:
    pip install sdg-hub datasets pandas mlflow
"""

from kfp.dsl import component


@component(
    base_image="python:3.11",
    packages_to_install=["mlflow", "sdg-hub", "pandas", "datasets", "urllib3"],
)
def sdg_component(
    model_url: str,
    model_name: str = "openai/Qwen3.6-35B-A3B",
    mlflow_tracking_uri: str = "https://mlflow.redhat-ods-applications.svc.cluster.local:8443",
    experiment_name: str = "it-helpdesk-sdg-finetune",
    api_key: str = "no-key-required",
    num_generations: int = 3,
    num_of_traces: int = 20,
):
    """KFP component: fetch MLflow traces → judge → generate synthetic data → save."""
    import ast
    import json
    import os
    import re
    import shutil
    from pathlib import Path

    import mlflow
    import pandas as pd
    import urllib3
    from datasets import Dataset
    from sdg_hub import Flow

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # ── MLflow auth ──────────────────────────────────────────────────────────
    os.environ["MLFLOW_TRACKING_AUTH"]         = "kubernetes"
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

    ns_path = "/run/secrets/kubernetes.io/serviceaccount/namespace"
    if os.path.exists(ns_path):
        with open(ns_path) as f:
            os.environ["MLFLOW_WORKSPACE"] = f.read().strip()

    token_path = "/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.exists(token_path):
        with open(token_path) as f:
            os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # ── Output paths ─────────────────────────────────────────────────────────
    WORKSPACE       = Path("/workspace")
    FLOWS_DIR       = WORKSPACE / "flows"
    PROMPTS_DIR     = WORKSPACE / "prompts"
    CURATED_OUTPUT  = str(WORKSPACE / "curated_traces.csv")
    SCORES_OUTPUT   = str(WORKSPACE / "scores_report.csv")
    TRAINING_OUTPUT = str(WORKSPACE / "training_data.csv")

    FLOWS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Write flow + prompt YAMLs ─────────────────────────────────────────────
    (FLOWS_DIR / "judge_flow.yaml").write_text("""\
metadata:
  id: judge-flow
  name: Judge Flow
  version: "1.0.0"
  dataset_requirements:
    required_columns: [conversation_str, target_response]
blocks:
  - block_type: PromptBuilderBlock
    block_config:
      block_name: build_judge_prompt
      input_cols: [conversation_str, target_response]
      output_cols: judge_messages
      prompt_config_path: /workspace/prompts/judge_prompt.yaml
  - block_type: LLMChatBlock
    block_config:
      block_name: judge_llm
      input_cols: judge_messages
      output_cols: judge_raw
      async_mode: true
      temperature: 0.0
      max_tokens: 1024
""")

    (FLOWS_DIR / "generate_flow.yaml").write_text("""\
metadata:
  id: generate-flow
  name: Generate Flow
  version: "1.0.0"
  dataset_requirements:
    required_columns: [conversation_str, target_response]
blocks:
  - block_type: PromptBuilderBlock
    block_config:
      block_name: build_gen_prompt
      input_cols: [conversation_str, target_response]
      output_cols: gen_messages
      prompt_config_path: /workspace/prompts/generate_prompt.yaml
  - block_type: LLMChatBlock
    block_config:
      block_name: generate_example
      input_cols: gen_messages
      output_cols: generated_output
      async_mode: true
      temperature: 0.8
      max_tokens: 8192
""")

    (PROMPTS_DIR / "judge_prompt.yaml").write_text("""\
- role: system
  content: |
    You are an expert data quality judge for AI training datasets. Your job is to evaluate
    a conversation and the assistant's final response, and decide if this is a high-quality
    training example suitable for fine-tuning a chatbot.

    A HIGH QUALITY example (score 7-10) has:
    - A clear, specific, and realistic user request
    - A helpful, accurate, and substantive assistant response
    - A response that directly addresses what was asked
    - Enough depth and detail to be a useful training example

    A LOW QUALITY example (score 1-6) has:
    - Vague, unclear, or trivially simple requests
    - Empty, very short, or unhelpful responses
    - Off-topic or irrelevant responses
    - Responses that appear cut off or incomplete

    Respond ONLY in this exact format — nothing before or after:
    Score: <integer from 1 to 10>
    Reason: <one sentence>

- role: user
  content: |
    Conversation:
    {{ conversation_str }}

    Assistant response to judge:
    {{ target_response }}
""")

    (PROMPTS_DIR / "generate_prompt.yaml").write_text("""\
- role: system
  content: |
    You are a synthetic training data generator. You will be given a real conversation example.
    Your job is to generate ONE new, distinct conversation that:
    - Covers a similar topic or domain as the example
    - Has a realistic, specific user request a real person might ask
    - Has a detailed, helpful assistant response that fully addresses the request
    - Is clearly different from the original — not a paraphrase or minor rewording

    Respond ONLY in this exact format, with no extra text before or after:
    <user_request>
    [the new user request here]
    </user_request>
    <assistant_response>
    [the new assistant response here]
    </assistant_response>

- role: user
  content: |
    Original conversation:
    {{ conversation_str }}

    Assistant response:
    {{ target_response }}

    Generate a new conversation in the same domain.
""")

    SYSTEM_MESSAGE_PATTERNS = [
        "review the conversation above",
        "update the skill library",
        "a pass that does nothing",
    ]

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _parse_metadata(raw):
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return ast.literal_eval(raw)
        return {}

    def _extract_session_id(meta):
        return meta.get("mlflow.trace.session", "").strip('"').strip("'")

    def _extract_judge_content(raw):
        if isinstance(raw, str):
            return raw
        try:
            for item in raw:
                if isinstance(item, dict) and item.get("content"):
                    return item["content"]
        except TypeError:
            pass
        return str(raw)

    def _is_placeholder(text):
        clean = re.sub(r'[\s\\n\\r]+', '', text)
        if len(clean) < 15:
            return True
        if re.fullmatch(r'\.{2,}', clean):
            return True
        if re.fullmatch(r'\[.*?\]', clean):
            return True
        return False

    def _format_input(messages):
        lines = []
        for msg in messages:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    # ── Fetch sessions ────────────────────────────────────────────────────────
    df = mlflow.search_traces(max_results=num_of_traces)
    status_col = "status" if "status" in df.columns else "state"
    sessions = {}

    for _, row in df.iterrows():
        if row.get(status_col) != "OK":
            continue
        request  = str(row.get("request",  "") or "").strip()
        response = str(row.get("response", "") or "").strip()
        if not request or not response:
            continue
        if any(p in request.lower() for p in SYSTEM_MESSAGE_PATTERNS):
            continue
        meta = _parse_metadata(row.get("trace_metadata") or row.get("request_metadata") or {})
        session_id = _extract_session_id(meta)
        if not session_id:
            continue
        sessions.setdefault(session_id, []).append({
            "request_time": row.get("request_time") or row.get("timestamp_ms", 0),
            "request": request,
            "response": response,
        })

    for turns in sessions.values():
        turns.sort(key=lambda t: t["request_time"])
    print(f"Fetched {len(sessions)} sessions")

    # ── Build examples ────────────────────────────────────────────────────────
    rows = []
    for session_id, turns in sessions.items():
        history = []
        for turn in turns:
            input_messages = history + [{"role": "user", "content": turn["request"]}]
            lines = [
                f"{'User:' if m['role'] == 'user' else 'Assistant:'} {m['content']}"
                for m in input_messages
            ]
            rows.append({
                "session_id":      session_id,
                "messages":        json.dumps(input_messages),
                "target_response": turn["response"],
                "conversation_str": "\n".join(lines),
            })
            history = input_messages + [{"role": "assistant", "content": turn["response"]}]

    examples_df = pd.DataFrame(rows)
    print(f"Built {len(examples_df)} examples")

    # ── Judge real examples ───────────────────────────────────────────────────
    for _dir in ["/workspace/checkpoints/judge", "/workspace/checkpoints/generate", "/workspace/checkpoints/judge_synthetic"]:
        shutil.rmtree(_dir, ignore_errors=True)

    judge_flow = Flow.from_yaml(str(FLOWS_DIR / "judge_flow.yaml"))
    judge_flow.set_model_config(model=model_name, api_base=model_url, api_key=api_key)

    result_df = judge_flow.generate(
        Dataset.from_pandas(examples_df),
        checkpoint_dir="/workspace/checkpoints/judge",
        max_concurrency=8,
    ).to_pandas()

    result_df["judge_content"] = result_df["judge_raw"].apply(_extract_judge_content)
    result_df["quality_score"] = result_df["judge_content"].apply(
        lambda t: int(m.group(1)) if (m := re.search(r"Score:\s*(\d+)", t)) else None
    )
    result_df["judge_reason"] = result_df["judge_content"].apply(
        lambda t: m.group(1).strip() if (m := re.search(r"Reason:\s*(.+)", t)) else ""
    )

    scores_df = result_df[["session_id", "conversation_str", "target_response", "quality_score", "judge_reason"]].copy()
    scores_df["included"] = result_df["quality_score"].ge(7)
    scores_df.to_csv(SCORES_OUTPUT, index=False)

    before = len(result_df)
    curated_df = result_df[result_df["quality_score"].notna() & (result_df["quality_score"] >= 7)].reset_index(drop=True)
    print(f"Judge: {before} → {len(curated_df)} kept (score ≥ 7)")
    curated_df.to_csv(CURATED_OUTPUT, index=False)

    if curated_df.empty:
        print("No examples passed quality filter.")
        return

    # ── Generate synthetic examples ───────────────────────────────────────────
    copies = [curated_df.assign(generation_idx=i) for i in range(num_generations)]
    seed_df = pd.concat(copies, ignore_index=True)

    gen_flow = Flow.from_yaml(str(FLOWS_DIR / "generate_flow.yaml"))
    gen_flow.set_model_config(model=model_name, api_base=model_url, api_key=api_key)

    gen_result = gen_flow.generate(
        Dataset.from_pandas(seed_df),
        checkpoint_dir="/workspace/checkpoints/generate",
        max_concurrency=8,
    ).to_pandas()

    def extract_tag(text, tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else None

    gen_result["generated_content"]  = gen_result["generated_output"].apply(_extract_judge_content)
    gen_result["generated_request"]  = gen_result["generated_content"].apply(lambda t: extract_tag(t, "user_request"))
    gen_result["generated_response"] = gen_result["generated_content"].apply(lambda t: extract_tag(t, "assistant_response"))

    # ── Judge synthetic examples ──────────────────────────────────────────────
    synth_judge_df = gen_result.copy()
    synth_judge_df["conversation_str"] = synth_judge_df["generated_request"].apply(lambda r: f"User: {r}")
    synth_judge_df["target_response"]  = synth_judge_df["generated_response"]

    synth_flow = Flow.from_yaml(str(FLOWS_DIR / "judge_flow.yaml"))
    synth_flow.set_model_config(model=model_name, api_base=model_url, api_key=api_key)

    synth_result = synth_flow.generate(
        Dataset.from_pandas(synth_judge_df),
        checkpoint_dir="/workspace/checkpoints/judge_synthetic",
        max_concurrency=8,
    ).to_pandas()

    synth_result["judge_content"] = synth_result["judge_raw"].apply(_extract_judge_content)
    synth_result["quality_score"] = synth_result["judge_content"].apply(
        lambda t: int(m.group(1)) if (m := re.search(r"Score:\s*(\d+)", t)) else None
    )

    before_synth = len(synth_result)
    synthetic_df = synth_result[synth_result["quality_score"].notna() & (synth_result["quality_score"] >= 7)].reset_index(drop=True)
    print(f"Synthetic judge: {before_synth} → {len(synthetic_df)} kept (score ≥ 7)")

    # ── Save training_data.csv ────────────────────────────────────────────────
    training_rows = []

    for _, row in curated_df.iterrows():
        try:
            msgs   = json.loads(row["messages"])
            output = row["target_response"].strip()
            if not _is_placeholder(output):
                training_rows.append({"source": "curated", "input": _format_input(msgs), "output": output})
        except Exception:
            pass

    skipped = 0
    for _, row in synthetic_df.iterrows():
        req = re.sub(r"^user:\s*", "", str(row.get("generated_request") or "").strip(), flags=re.IGNORECASE)
        res = str(row.get("generated_response") or "").strip()
        if req and res and not _is_placeholder(req) and not _is_placeholder(res):
            training_rows.append({
                "source": "synthetic",
                "input":  _format_input([{"role": "user", "content": req}]),
                "output": res,
            })
        else:
            skipped += 1
            print(f"  [skipped] req={req!r:.80} | res={res!r:.80}")

    pd.DataFrame(training_rows).to_csv(TRAINING_OUTPUT, index=False)
    print(f"Saved {len(training_rows)} training examples to {TRAINING_OUTPUT}")
    if skipped:
        print(f"  ({skipped} synthetic rows skipped — empty or placeholder output)")
