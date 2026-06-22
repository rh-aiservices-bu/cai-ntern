"""
Kubeflow Pipeline: SDG → Fine-Tune → Evaluate

Components are imported directly from the existing scripts.

Run locally:
    python kfp_pipeline.py
"""

import kfp.dsl as dsl
from kfp import local

from sdg_pipeline.pipeline import sdg_component
from fine_tune import fine_tune_component
from evaluate import evaluate_component

local.init(runner=local.SubprocessRunner())


@dsl.pipeline(
    name="IT Help Desk — SDG + Fine-Tune + Evaluate",
    description="Fetch traces → judge → generate 3x synthetic data → LoRA fine-tune → evaluate",
)
def ithelpdesk_pipeline():
    sdg_task = sdg_component()

    ft_task = fine_tune_component()
    ft_task.after(sdg_task)

    eval_task = evaluate_component()
    eval_task.after(ft_task)


if __name__ == "__main__":
    ithelpdesk_pipeline()
