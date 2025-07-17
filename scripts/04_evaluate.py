#!/usr/bin/env python
"""Evaluate predictions against ground truth labels."""
import argparse
from pathlib import Path
import pandas as pd
import yaml
from nnunetv2.evaluation.evaluate_predictions import (
    compute_metrics_on_folder_simple,
    load_summary_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task_name = cfg["dataset"]["task_name"].replace(" ", "")
    task_id = int(cfg["dataset"]["task_id"])
    dataset = f"Task{task_id:03d}_{task_name}"

    nnunet_raw = Path(cfg["paths"]["nnunet_raw"])
    labelsTs = nnunet_raw / dataset / "labelsTs"
    pred_folder = Path(cfg["paths"]["outputs_dir"]) / "predictions"
    summary_file = pred_folder / "summary.json"

    label_defs = cfg["dataset"]["labels"]
    label_numbers = [int(v) for k, v in label_defs.items() if int(v) != 0]

    compute_metrics_on_folder_simple(
        str(labelsTs),
        str(pred_folder),
        label_numbers,
        output_file=str(summary_file),
    )

    results = load_summary_json(str(summary_file))
    per_case = []
    for entry in results["metric_per_case"]:
        case = entry["reference"].split("/")[-1]
        metrics = {f"Dice_label_{l}": entry["metrics"][l]["Dice"] for l in label_numbers}
        per_case.append({"case": case, **metrics})
    df = pd.DataFrame(per_case)
    df.to_csv(Path(cfg["paths"]["outputs_dir"]) / "summary.csv", index=False)
    print("Evaluation complete. Results saved to outputs/summary.csv")


if __name__ == "__main__":
    main()
