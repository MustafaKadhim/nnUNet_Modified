#!/usr/bin/env python
"""Run inference on the test set using a trained nnU-Net model."""
import argparse
import subprocess
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run nnU-Net inference")
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
    imagesTs = nnunet_raw / dataset / "imagesTs"
    output_dir = Path(cfg["paths"]["outputs_dir"]) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    configuration = cfg["training"]["configuration"]
    trainer = cfg["training"].get("trainer", "nnUNetTrainer")
    device = cfg["training"].get("device", "cuda")

    cmd = [
        "nnUNetv2_predict",
        "-i", str(imagesTs),
        "-o", str(output_dir),
        "-d", dataset,
        "-c", configuration,
        "-tr", trainer,
        "-f", "all",
        "-device", device,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
