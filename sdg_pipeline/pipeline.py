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

import ast
import json
import re
import shutil
import sys
from pathlib import Path
import os

import mlflow
import pandas as pd
from datasets import Dataset
from sdg_hub import Flow

# ── Configuration ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "<TRACKING_URI>"

MLFLOW_WORKSPACE = "<WORKSPACE>"
# os.environ["MLFLOW_TRACKING_AUTH"] = "kubernetes"
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
os.environ["MLFLOW_WORKSPACE"] = MLFLOW_WORKSPACE
os.environ["MLFLOW_TRACKING_TOKEN"] = "<TRACKING_TOKEN>"
EXPERIMENT_NAME = "<EXPERIMENT_NAME>"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
# client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
print(f"✓ Connected to MLflow — workspace: {MLFLOW_WORKSPACE}, experiment: {EXPERIMENT_NAME}")
MODEL = "openai/Qwen3.6-35B-A3B"
MODEL_URL = "<MODEL_URL>"
API_KEY = "<API_KEY>"
NUM_GENERATIONS = 3          # synthetic examples to produce per high-quality seed
NUM_OF_TRACES = 20
CURATED_OUTPUT = "curated_traces.csv"
SCORES_OUTPUT = "scores_report.csv"
TRAINING_OUTPUT = "training_data.csv"
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_MESSAGE_PATTERNS = [
    "review the conversation above",
    "update the skill library",
    "a pass that does nothing",
]

FLOWS_DIR = Path(__file__).parent / "flows"


def _parse_metadata(raw) -> dict:
    """Parse trace_metadata whether it arrives as a dict, JSON string, or Python repr."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return ast.literal_eval(raw)
    return {}


def _extract_session_id(meta: dict) -> str:
    """Pull the session ID out of trace metadata and strip surrounding quotes."""
    return meta.get("mlflow.trace.session", "").strip('"').strip("'")


def fetch_sessions() -> dict[str, list[dict]]:
    """
    Fetch all OK traces from MLflow, filter noise, and group into sessions.
    Each session is a list of turn dicts sorted by request_time (ascending).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    df = mlflow.search_traces(max_results=NUM_OF_TRACES)

    print(df)

    # MLflow Python API uses 'status'; CSV export uses 'state' — handle both
    status_col = "status" if "status" in df.columns else "state"

    sessions: dict[str, list[dict]] = {}

    for _, row in df.iterrows():
        if row.get(status_col) != "OK":
            continue

        request = str(row.get("request", "") or "").strip()
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

    print(f"Fetched {len(sessions)} sessions from MLflow")
    for session_id, turns in sessions.items():
        print(f"\n  Session: {session_id} ({len(turns)} turn(s))")
        for i, turn in enumerate(turns):
            print(f"    Turn {i+1}:")
            print(f"      request:  {turn['request'][:120]!r}")
            print(f"      response: {turn['response'][:120]!r}")
    return sessions


def build_conversation_examples(sessions: dict[str, list[dict]]) -> pd.DataFrame:
    """
    Unroll each session into sliding-window training examples.

    For a session with turns [t1, t2, t3]:
      row 1 — messages: [user: t1.request]                         target: t1.response
      row 2 — messages: [user: t1, asst: t1, user: t2]            target: t2.response
      row 3 — messages: [user: t1, asst: t1, user: t2, asst: t2, user: t3]  target: t3.response

    Columns produced:
      session_id       — source session
      messages         — JSON list of {"role","content"} dicts (the input side)
      target_response  — the assistant turn the model should produce
      conversation_str — human-readable version of messages + target, used in prompts
    """
    rows = []
    for session_id, turns in sessions.items():
        history: list[dict] = []
        for turn in turns:
            input_messages = history + [{"role": "user", "content": turn["request"]}]

            # Readable string for judge/generate prompts
            lines = []
            for msg in input_messages:
                prefix = "User:" if msg["role"] == "user" else "Assistant:"
                lines.append(f"{prefix} {msg['content']}")
            conversation_str = "\n".join(lines)

            rows.append({
                "session_id": session_id,
                "messages": json.dumps(input_messages),
                "target_response": turn["response"],
                "conversation_str": conversation_str,
            })

            history = input_messages + [{"role": "assistant", "content": turn["response"]}]

    df = pd.DataFrame(rows)
    print(f"Built {len(df)} conversation examples from {len(sessions)} sessions")
    return df


def debug_judge_raw(df: pd.DataFrame):
    """Run just the prompt-builder + LLM blocks and print judge_raw so we can
    see exactly what the model returns before the regex tries to parse it."""
    flow = Flow.from_yaml(str(FLOWS_DIR / "judge_flow.yaml"))
    flow.set_model_config(model=MODEL, api_base=MODEL_URL, api_key=API_KEY)

    result = flow.generate(Dataset.from_pandas(df.head(1))).to_pandas()
    for i, raw in enumerate(result["judge_raw"]):
        print(f"\n  judge_raw[{i}]: {raw!r}")


def _extract_judge_content(raw) -> str:
    """Pull the text content out of judge_raw regardless of its format.

    LLMChatBlock can return either a plain string or a numpy array of response
    dicts (e.g. [{'role': 'assistant', 'content': '...', ...}]). Handle both.
    """
    if isinstance(raw, str):
        return raw
    try:
        for item in raw:
            if isinstance(item, dict) and item.get("content"):
                return item["content"]
    except TypeError:
        pass
    return str(raw)


def run_judge_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Score each example with an LLM judge; keep those scoring >= 7.

    Uses the 2-block LLM flow for the actual call, then extracts the score
    and filters in Python (LLMChatBlock returns raw response objects, not plain
    strings, so SDG Hub's RegexParserBlock can't match directly).
    """


    flow = Flow.from_yaml(str(FLOWS_DIR / "judge_flow.yaml"))
    flow.set_model_config(model=MODEL, api_base=MODEL_URL, api_key=API_KEY)

    result_df = flow.generate(
        Dataset.from_pandas(df), checkpoint_dir="./checkpoints/judge", max_concurrency=8
    ).to_pandas()

    result_df["judge_content"] = result_df["judge_raw"].apply(_extract_judge_content)
    result_df["quality_score"] = result_df["judge_content"].apply(
        lambda t: int(m.group(1)) if (m := re.search(r"Score:\s*(\d+)", t)) else None
    )
    result_df["judge_reason"] = result_df["judge_content"].apply(
        lambda t: m.group(1).strip() if (m := re.search(r"Reason:\s*(.+)", t)) else ""
    )

    # Save full scores report before filtering
    scores_cols = ["session_id", "conversation_str", "target_response", "quality_score", "judge_reason"]
    scores_df = result_df[scores_cols].copy()
    scores_df["included"] = result_df["quality_score"].ge(7)
    scores_df["request_preview"] = scores_df["conversation_str"].str[:120]
    scores_df["response_preview"] = scores_df["target_response"].str[:120]
    scores_df.drop(columns=["conversation_str", "target_response"]).to_csv(SCORES_OUTPUT, index=False)
    print(f"Scores report saved to: {SCORES_OUTPUT}")

    before = len(result_df)
    result_df = result_df[result_df["quality_score"].notna() & (result_df["quality_score"] >= 7)].reset_index(drop=True)
    print(f"Judge: {before} examples -> {len(result_df)} kept (score >= 7)")
    return result_df


def run_generate_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Generate NUM_GENERATIONS synthetic examples per high-quality seed.

    Uses a 2-block flow (prompt + LLM) and parses the XML tags in Python,
    since LLMChatBlock returns raw response objects that TagParserBlock can't handle.
    """
    copies = [df.assign(generation_idx=i) for i in range(NUM_GENERATIONS)]
    seed_df = pd.concat(copies, ignore_index=True)
    print(f"Generate flow: {len(df)} seeds x{NUM_GENERATIONS} -> {len(seed_df)} to generate")

    flow = Flow.from_yaml(str(FLOWS_DIR / "generate_flow.yaml"))
    flow.set_model_config(model=MODEL, api_base=MODEL_URL, api_key=API_KEY)

    result_df = flow.generate(
        Dataset.from_pandas(seed_df), checkpoint_dir="./checkpoints/generate", max_concurrency=8
    ).to_pandas()

    def extract_tag(text, tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else None

    result_df["generated_content"] = result_df["generated_output"].apply(_extract_judge_content)
    result_df["generated_request"] = result_df["generated_content"].apply(lambda t: extract_tag(t, "user_request"))
    result_df["generated_response"] = result_df["generated_content"].apply(lambda t: extract_tag(t, "assistant_response"))

    return result_df


def judge_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Run the same LLM judge on synthetic examples; keep those scoring >= 7."""
    judge_df = df.copy()
    judge_df["conversation_str"] = judge_df["generated_request"].apply(lambda r: f"User: {r}")
    judge_df["target_response"] = judge_df["generated_response"]

    flow = Flow.from_yaml(str(FLOWS_DIR / "judge_flow.yaml"))
    flow.set_model_config(model=MODEL, api_base=MODEL_URL, api_key=API_KEY)

    result_df = flow.generate(
        Dataset.from_pandas(judge_df), checkpoint_dir="./checkpoints/judge_synthetic", max_concurrency=8
    ).to_pandas()

    result_df["judge_content"] = result_df["judge_raw"].apply(_extract_judge_content)
    result_df["quality_score"] = result_df["judge_content"].apply(
        lambda t: int(m.group(1)) if (m := re.search(r"Score:\s*(\d+)", t)) else None
    )

    before = len(result_df)
    result_df = result_df[result_df["quality_score"].notna() & (result_df["quality_score"] >= 7)].reset_index(drop=True)
    print(f"Synthetic judge: {before} examples -> {len(result_df)} kept (score >= 7)")
    return result_df


def _is_placeholder(text: str) -> bool:
    """Return True if text is model-generated filler rather than real content.

    Catches cases like '...', '...\\n', '[request]', '[response]', etc.
    that appear when the model copies the prompt format instead of generating content.
    """

    # Collapse all whitespace and escape sequences so we compare the bare text
    clean = re.sub(r'[\s\\n\\r]+', '', text)
    if len(clean) < 15:
        return True
    if re.fullmatch(r'\.{2,}', clean):          # "...", "....", etc.
        return True
    if re.fullmatch(r'\[.*?\]', clean):          # "[request]", "[response]", etc.
        return True
    return False


def _format_input(messages: list[dict]) -> str:
    """Format a list of chat messages into a readable input string."""
    lines = []
    for msg in messages:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {msg['content']}")
    return "\n".join(lines)


def save_training_csv(curated_df: pd.DataFrame, synthetic_df: pd.DataFrame):  # noqa: C901
    """
    Write training_data.csv with columns: source, input, output.

    Curated rows use the full multi-turn conversation history as input.
    Synthetic rows are single-turn (generated_request / generated_response).
    Placeholder outputs ('...', '[response]', etc.) are filtered out.
    """
    rows = []

    for _, row in curated_df.iterrows():
        try:
            msgs = json.loads(row["messages"])
            output = row["target_response"].strip()
            if not _is_placeholder(output):
                rows.append({
                    "source": "curated",
                    "input": _format_input(msgs),
                    "output": output,
                })
        except Exception:
            pass

    skipped = 0
    for _, row in synthetic_df.iterrows():
        req = str(row.get("generated_request") or "").strip()
        res = str(row.get("generated_response") or "").strip()
        # Strip any "User:" prefix the model may have included in the request
        req = re.sub(r"^user:\s*", "", req, flags=re.IGNORECASE)
        if req and res and not _is_placeholder(req) and not _is_placeholder(res):
            rows.append({
                "source": "synthetic",
                "input": _format_input([{"role": "user", "content": req}]),
                "output": res,
            })
        else:
            skipped += 1
            content = str(row.get("generated_content") or "")
            print(f"  [skipped] req={req!r:.80} | res={res!r:.80}")

    pd.DataFrame(rows).to_csv(TRAINING_OUTPUT, index=False)
    print(f"Saved {len(rows)} training examples to {TRAINING_OUTPUT}")
    if skipped:
        print(f"  ({skipped} synthetic rows skipped — empty or placeholder output)")


if __name__ == "__main__":
    for _dir in ["./checkpoints/judge", "./checkpoints/generate", "./checkpoints/judge_synthetic"]:
        shutil.rmtree(_dir, ignore_errors=True)

    print("\n=== Step 1: Fetching sessions from MLflow ===")
    sessions = fetch_sessions()

    print("\n=== Step 2: Building conversation examples ===")
    examples_df = build_conversation_examples(sessions)

    print("\n=== Step 3: LLM judging for quality ===")
    curated_df = run_judge_flow(examples_df)
    curated_df.to_csv(CURATED_OUTPUT, index=False)
    print(f"Curated examples saved to: {CURATED_OUTPUT}")

    if curated_df.empty:
        print("No examples passed the quality filter. Exiting.")
        sys.exit(0)

    print("\n=== Step 4: Generating synthetic training examples ===")
    synthetic_df = run_generate_flow(curated_df)

    print("\n=== Step 5: Judging synthetic examples for quality ===")
    synthetic_df = judge_synthetic(synthetic_df)

    print("\n=== Step 6: Saving training data ===")
    save_training_csv(curated_df, synthetic_df)
