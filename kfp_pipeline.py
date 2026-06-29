"""
Kubeflow Pipeline: SDG -> Fine-Tune -> Evaluate

Components are imported directly from the existing scripts.
Each step runs in its own container on the cluster.

Submit to the pipeline server:
    python kfp_pipeline.py
"""

import kfp
import kfp.dsl as dsl
from kfp import kubernetes

from sdg_pipeline.pipeline import sdg_component
from fine_tune import fine_tune_component
from evaluate import evaluate_component


@dsl.pipeline(
    name="IT Help Desk - SDG + Fine-Tune + Evaluate",
    description="Fetch traces, judge quality, generate 3x synthetic data, LoRA fine-tune, evaluate",
)
def ithelpdesk_pipeline(
    model_url: str,
    storage_class: str = "gp3-csi",
    model_name: str = "openai/Qwen3.6-35B-A3B",
    mlflow_tracking_uri: str = "https://mlflow.redhat-ods-applications.svc.cluster.local:8443",
    experiment_name: str = "it-helpdesk-sdg-finetune",
    api_key: str = "no-key-required",
    num_generations: int = 5,
    num_of_traces: int = 100,
    base_model: str = "Qwen/Qwen2-0.5B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_new_tokens: int = 512,
):
    # Shared PVC - carries all artifacts between steps
    pvc = kubernetes.CreatePVC(
        pvc_name_suffix="-ithelpdesk-pvc",
        access_modes=["ReadWriteOnce"],
        size="50Gi",
        storage_class_name=storage_class,
    )

    # Step 1: SDG
    sdg_task = sdg_component(
        model_url=model_url,
        model_name=model_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        api_key=api_key,
        num_generations=num_generations,
        num_of_traces=num_of_traces,
    )
    kubernetes.mount_pvc(sdg_task, pvc_name=pvc.outputs["name"], mount_path="/workspace")

    # Step 2: Fine-tune (GPU)
    ft_task = fine_tune_component(
        base_model=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    ft_task.after(sdg_task)
    kubernetes.mount_pvc(ft_task, pvc_name=pvc.outputs["name"], mount_path="/workspace")
    kubernetes.add_node_selector(ft_task, label_key="nvidia.com/gpu.present", label_value="true")
    kubernetes.add_toleration(ft_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    ft_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)

    # Step 3: Evaluate (GPU)
    eval_task = evaluate_component(
        base_model=base_model,
        max_new_tokens=max_new_tokens,
    )
    eval_task.after(ft_task)
    kubernetes.mount_pvc(eval_task, pvc_name=pvc.outputs["name"], mount_path="/workspace")
    kubernetes.add_node_selector(eval_task, label_key="nvidia.com/gpu.present", label_value="true")
    kubernetes.add_toleration(eval_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    eval_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)


if __name__ == "__main__":
    namespace_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    with open(namespace_path) as f:
        namespace = f.read().strip()

    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    with open(token_path) as f:
        bearer_token = f.read().strip()

    kubeflow_endpoint = f"https://ds-pipeline-dspa.{namespace}.svc:8443"
    ssl_ca_cert       = "/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"

    print(f"Connecting to Data Science Pipelines: {kubeflow_endpoint}")
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert,
    )

    client.create_run_from_pipeline_func(
        ithelpdesk_pipeline,
        arguments={
            "model_url": "<MODEL_URL>",  # set your vLLM endpoint here
            "storage_class": "gp3-csi",
            "api_key": "sk-..."
        },
        experiment_name="it-helpdesk-sdg-finetune",
        enable_caching=False,
    )
