# Deploy an IT Help Desk AI Fine-Tuning Pipeline on OpenShift AI

Automate the full MLOps lifecycle for an IT Help Desk AI assistant: generate synthetic training data from real conversation traces, fine-tune a small language model with LoRA, and evaluate it — all as a single Kubeflow Pipeline on OpenShift AI.

## Table of Contents

- [Description](#description)
- [Requirements](#requirements)
- [Deploy](#deploy)
- [Delete](#delete)
- [Reference](#reference)
- [Tags](#tags)

## Description

Enterprise IT Help Desk teams handle high volumes of repetitive requests — password resets, VPN issues, software installs, onboarding — that are well-suited for an AI assistant. But general-purpose models answer these questions by reasoning from scratch every time: multiple tool calls, long chains of thought, high token consumption. Fine-tuning bakes that institutional knowledge directly into the model weights, so it responds accurately with far fewer turns, no retrieval overhead, and significantly lower token usage.

This quickstart shows how to build that fine-tuned assistant from your own operational data. It captures real IT Help Desk conversations as MLflow traces, curates the highest-quality examples using a large teacher LLM as a judge, and generates 3x synthetic training data to expand coverage. A small student model (Qwen2-0.5B) is then fine-tuned with LoRA on that dataset — resulting in a model that has internalized your organization's IT policies, terminology, and resolution patterns rather than having to look them up.

All pipeline steps run as containers on OpenShift AI using Kubeflow Pipelines v2. Artifacts (training data, LoRA adapter, evaluation results) are passed between steps via a shared PVC. MLflow is used for trace storage and experiment tracking throughout.

### Architecture

![Architecture Diagram](docs/images/architecture.png)
<!-- TODO: add architecture diagram showing: generate_traces.py -> MLflow -> SDG component -> fine_tune component -> evaluate component, with shared PVC between pipeline steps and teacher/student model split -->

## Requirements

### Hardware Requirements

- At least 1 GPU node with `nvidia.com/gpu` available (required for fine-tuning and evaluation steps)
- Minimum 50 GB PVC storage available in the cluster

### Software Requirements

- OpenShift AI 2.16 or later with Kubeflow Pipelines enabled
- MLflow operator enabled in OpenShift AI
- An MLflow instance created in your project namespace (see [Deploy](#deploy))
- A running LLM endpoint compatible with the OpenAI API (e.g. vLLM or LiteLLM) for the teacher model used in SDG
- Python 3.11+ with `kfp`, `kfp-kubernetes` installed locally to submit the pipeline

## Deploy

### 1. Create an MLflow Instance

Apply the following CR to create an MLflow instance in your namespace:

```bash
oc apply -f mlflow.yaml
```

`mlflow.yaml`:

```yaml
apiVersion: mlflow.opendatahub.io/v1
kind: MLflow
metadata:
  name: mlflow
spec:
  artifactsDestination: 'file:///mlflow/artifacts'
  backendStoreUri: 'sqlite:////mlflow/mlflow.db'
  replicas: 1
  serveArtifacts: true
  serviceAccountName: mlflow-sa
  storage:
    accessModes:
      - ReadWriteOnce
    resources:
      requests:
        storage: 100Gi
  workers: 1
```

Wait for the MLflow instance to be ready:

```bash
oc wait --for=condition=Ready mlflow/mlflow --timeout=120s
```

### 2. Apply RBAC

The pipeline runner service account needs access to the MLflow service. Apply the included RBAC manifest, replacing `<YOUR_NAMESPACE>` with your project namespace:

```bash
sed 's/<YOUR_NAMESPACE>/your-namespace/g' rbac.yaml | oc apply -f -
```

### 3. Generate Traces

Before running the pipeline, seed MLflow with real IT Help Desk conversation traces. Edit `generate_traces.py` and set your LLM endpoint and API key:

```python
LLM_ENDPOINT = "https://your-llm-endpoint/v1"
MODEL_NAME    = "your-model-name"
API_KEY       = "sk-..."
```

Then run:

```bash
pip install openai mlflow
python generate_traces.py
```

This sends 100 example IT Help Desk prompts to your model and logs the responses as MLflow traces under the `it-helpdesk-sdg-finetune` experiment.

### 4. Submit the Pipeline

Install the required Python packages:

```bash
pip install kfp kfp-kubernetes
```

Edit the `__main__` block in `kfp_pipeline.py` and set your pipeline arguments:

```python
arguments={
    "model_url":   "https://your-llm-endpoint/v1",   # teacher LLM for SDG
    "api_key":     "sk-...",                          # LiteLLM virtual key
    "storage_class": "gp3-csi",                       # PVC storage class for your cluster
}
```

Submit the pipeline from inside your OpenShift AI workbench (where the SA token is mounted):

```bash
python kfp_pipeline.py
```

The pipeline will run three sequential steps:

| Step | What it does |
|---|---|
| **SDG** | Fetches traces from MLflow, judges quality, generates 3x synthetic training data |
| **Fine-tune** | LoRA fine-tunes `Qwen/Qwen2-0.5B-Instruct` on the generated data (requires GPU) |
| **Evaluate** | Runs side-by-side comparison of base vs fine-tuned model on a holdout set |

Outputs are written to the shared PVC:

| File | Description |
|---|---|
| `training_data.csv` | Final training examples used for fine-tuning |
| `curated_traces.csv` | High-quality real examples that passed the judge |
| `scores_report.csv` | All examples with quality scores and reasons |
| `lora_adapter/` | Trained LoRA adapter weights |
| `eval_results.csv` | Side-by-side evaluation results |

### Delete

To clean up all resources:

```bash
# Delete the MLflow instance
oc delete mlflow/mlflow

# Delete the RBAC
oc delete -f rbac.yaml

# Delete pipeline runs and PVCs via the OpenShift AI dashboard
# or via CLI:
oc delete pvc -l app=ithelpdesk-pvc
```

## Reference

- [Kubeflow Pipelines v2 documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [OpenShift AI documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed)
- [SDG Hub](https://github.com/instructlab/sdg)
- [PEFT / LoRA documentation](https://huggingface.co/docs/peft)
- [MLflow on OpenShift AI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed)

## Tags

- **Industry:** Information Technology
- **Product:** Red Hat OpenShift AI
- **Use case:** Fine-tuning, Synthetic Data Generation, MLOps
- **Model:** Qwen2-0.5B-Instruct
