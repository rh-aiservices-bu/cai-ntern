# SDG Pipeline

Fetches real conversation traces from MLflow, curates high-quality examples using an LLM judge, generates synthetic training data from those examples, and judges the synthetic data before saving.

## Flow

```
MLflow
  |
  | search_traces()
  v
+------------------+
|  Fetch Sessions  |  Groups traces by session ID, sorts turns chronologically,
|                  |  filters noise (empty turns, system messages, failed calls)
+------------------+
  |
  | sessions: dict[session_id -> list[turn]]
  v
+------------------+
|  Build Examples  |  Sliding-window unroll: for a session with N turns,
|                  |  produces N training rows (row K has turns 1..K as context)
+------------------+
  |
  | examples_df: [session_id, messages, target_response, conversation_str]
  v
+------------------+
|   Judge (real)   |  LLM scores each example 1-10
|                  |  Saves scores_report.csv (all examples + score + reason + included flag)
|                  |  Keeps examples with score >= 7
+------------------+
  |
  | curated_df  ->  curated_traces.csv
  v
+------------------+
|    Generate      |  Produces NUM_GENERATIONS new synthetic examples per seed
|                  |  Uses temperature=0.8 for diversity
|                  |  Parses <user_request> / <assistant_response> XML tags
+------------------+
  |
  | synthetic_df: [generated_request, generated_response]
  v
+------------------+
| Judge (synth)    |  Same LLM judge re-applied to synthetic examples
|                  |  Keeps those scoring >= 7
+------------------+
  |
  v
+------------------+
|      Save        |  Merges curated + synthetic into training_data.csv
|                  |  Columns: source, input, output
+------------------+
  |
  v
training_data.csv
```

## Output Files

| File | Description |
|---|---|
| `training_data.csv` | Final training examples (`source`, `input`, `output`) |
| `curated_traces.csv` | High-quality real examples that passed the judge |
| `scores_report.csv` | All examples with scores, reasons, and `included` flag |

## Configuration

All settings are globals at the top of `sdg_pipeline/pipeline.py`:

| Variable | Description |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow server URL |
| `EXPERIMENT_NAME` | MLflow experiment to pull traces from |
| `MODEL` | LiteLLM model string (e.g. `openai/Qwen3-35B`) |
| `MODEL_URL` | API base URL for the model |
| `API_KEY` | API key |
| `NUM_GENERATIONS` | Synthetic examples to generate per curated seed |
| `NUM_OF_TRACES` | Max traces to fetch from MLflow per run |

## Structure

```
sdg_pipeline/
  pipeline.py          # Main orchestration script
  flows/
    judge_flow.yaml    # PromptBuilderBlock + LLMChatBlock for judging
    generate_flow.yaml # PromptBuilderBlock + LLMChatBlock for generation
  prompts/
    judge_prompt.yaml  # Jinja2 template: scores conversation 1-10
    generate_prompt.yaml # Jinja2 template: generates new (request, response) pair
```

## Running

```bash
cd sdg_pipeline
python pipeline.py
```

