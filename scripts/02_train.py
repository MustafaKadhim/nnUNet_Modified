#!/usr/bin/env python
"""Wrapper for nnUNetv2 training."""
import argparse
import subprocess
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train nnU-Net model")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task_name = cfg["dataset"]["task_name"].replace(" ", "")
    task_id = int(cfg["dataset"]["task_id"])
    dataset = f"Task{task_id:03d}_{task_name}"

    configuration = cfg["training"]["configuration"]
    trainer = cfg["training"].get("trainer", "nnUNetTrainer")
    fold = str(cfg["training"].get("fold", 0))
    device = cfg["training"].get("device", "cuda")

    cmd = [
        "nnUNetv2_train",
        dataset,
        configuration,
        fold,
        "-tr",
        trainer,
        "-device",
        device,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
