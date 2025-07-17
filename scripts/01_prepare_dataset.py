#!/usr/bin/env python
"""Prepare dataset for nnU-Net training.

This script splits raw data into training and test subsets and
converts them to the nnU-Net folder structure. Metadata is written
into a dataset.json file using nnunetv2 utilities.
"""
import argparse
import shutil
from pathlib import Path
import random
import yaml
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_images = Path(cfg["paths"]["raw_images"])
    raw_labels = Path(cfg["paths"]["raw_labels"])
    nnunet_raw = Path(cfg["paths"]["nnunet_raw"])

    task_name = cfg["dataset"]["task_name"].replace(" ", "")
    task_id = int(cfg["dataset"]["task_id"])
    dataset_name = f"Task{task_id:03d}_{task_name}"
    out_dir = nnunet_raw / dataset_name
    imagesTr = out_dir / "imagesTr"
    labelsTr = out_dir / "labelsTr"
    imagesTs = out_dir / "imagesTs"
    labelsTs = out_dir / "labelsTs"
    for d in (imagesTr, labelsTr, imagesTs, labelsTs):
        d.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(raw_images.glob("*.nii.gz")))
    label_files = {f.name: raw_labels / f.name for f in raw_labels.glob("*.nii.gz")}

    random.seed(cfg["split"]["random_seed"])
    random.shuffle(img_files)
    split_idx = int(len(img_files) * (1 - cfg["split"]["test_split_fraction"]))
    train_files = img_files[:split_idx]
    test_files = img_files[split_idx:]

    def copy_pair(files, dest_img, dest_lbl):
        for img in files:
            lbl = label_files[img.name]
            shutil.copy(img, dest_img / f"{img.stem}_0000.nii.gz")
            shutil.copy(lbl, dest_lbl / img.name)

    copy_pair(train_files, imagesTr, labelsTr)
    copy_pair(test_files, imagesTs, labelsTs)

    labels = cfg["dataset"]["labels"]
    labels_int = {k: int(v) if isinstance(v, str) and v.isdigit() else int(v) for k, v in labels.items()}
    generate_dataset_json(
        str(out_dir),
        channel_names={0: "MRI"},
        labels=labels_int,
        num_training_cases=len(train_files),
        file_ending=".nii.gz",
        dataset_name=dataset_name,
    )
    print(f"Dataset prepared under {out_dir}")


if __name__ == "__main__":
    main()
