# nnU-Net Segmentation Framework

This project provides a configuration-driven workflow for training nnU-Net models using the ResEnc M preset. It wraps the [`nnunetv2`](https://github.com/MIC-DKFZ/nnUNet) library with simple scripts and a clear directory structure.

## Directory Structure

```
/data             # raw and nnU-Net formatted datasets
/scripts          # command line tools
/models           # trained model weights
/outputs          # predictions, metrics and logs
```

## Setup

Create the conda environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate nnunet_env
```

## Data Preparation

Place your original images and labels in `data/raw/images` and `data/raw/labels` respectively. Run the preparation script to split the data and convert it to nnU-Net format:

```bash
python scripts/01_prepare_dataset.py --config config.yaml
```

## Training

Start training using the specified configuration and trainer:

```bash
python scripts/02_train.py --config config.yaml
```

## Inference

After training, run inference on the hidden test set:

```bash
python scripts/03_inference.py --config config.yaml
```

## Evaluation

Evaluate predictions against the ground truth labels:

```bash
python scripts/04_evaluate.py --config config.yaml
```

The evaluation script saves a `summary.csv` file in the `outputs` directory with Dice scores and other metrics.

## Configuration File

The `config.yaml` centralizes all parameters:

- **paths**: location of raw data and nnU-Net folders
- **dataset**: task name, id and label definitions
- **split**: fraction of cases used for the hidden test set
- **training**: nnU-Net configuration, trainer, fold and device

Modify these fields to adapt the pipeline to new datasets or tasks.

## Adding New Tasks

Update `config.yaml` with a new `task_name`, `task_id` and label mapping. Place the raw data in the same folder structure and rerun the scripts in the order listed above.
