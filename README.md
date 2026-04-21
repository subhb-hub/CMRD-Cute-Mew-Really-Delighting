# CMRD: Confidence-Modulation Reference Deviation Representation for Target-Free Cross-Subject EEG Emotion Recognition

Official implementation of **CMRD** for target-free cross-subject EEG emotion recognition on **SEED** and **SEED-IV**.

## Overview

This repository contains:

- a unified training script for SEED and SEED-IV;
- a SEED feature-fusion utility for combining existing JSD and DE folds;
- a SEED-IV end-to-end preprocessing pipeline from raw `.mat` files to LOOCV folds.

The current public version focuses on reproducibility and release readiness:

- local absolute paths have been removed from defaults;
- training and preprocessing now use explicit command-line interfaces;
- outputs are written to predictable repo-relative directories;
- the best fold checkpoint is restored before final evaluation.

## Repository Structure

```text
.
├── Preprocess/
│   ├── Pre-SEED.py
│   └── Pre-SEED-IV.py
├── Train/
│   └── Train.py
├── env.yml
└── requirements.txt
```

## Environment

Conda:

```bash
conda env create -f env.yml
conda activate cmrd
```

Pip:

```bash
pip install -r requirements.txt
```

If you plan to use GPU training, install the PyTorch build that matches your CUDA setup.

## Data Preparation

The datasets are **not** redistributed in this repository. Please obtain SEED and SEED-IV from their official sources and place them under your own data directory.

### SEED

`Preprocess/Pre-SEED.py` expects that you already have matching LOOCV fold directories for:

- JSD features;
- DE features.

Both roots should contain the same fold structure, for example:

```text
/path/to/seed/
├── _fold_jsd/
│   └── fold_subj_01/
│       ├── train_source/
│       └── test_target/
└── _fold_de/
    └── fold_subj_01/
        ├── train_source/
        └── test_target/
```

Run:

```bash
python Preprocess/Pre-SEED.py \
  --jsd_root /path/to/seed/_fold_jsd \
  --de_root /path/to/seed/_fold_de \
  --save_root data/SEED/_fold_jsd_degate
```

### SEED-IV

`Preprocess/Pre-SEED-IV.py` can run the full pipeline from raw files:

```bash
python Preprocess/Pre-SEED-IV.py \
  --base-path /path/to/SEED-IV-RAW \
  --save-root data/SEED-IV \
  --steps 1 2 3 4
```

Useful options:

- `--steps 1 2` only runs the early preprocessing stages.
- `--alpha 0.5` changes the zDE gate strength.
- `--no-save-feat` skips the concatenated `feat` tensor.

The final training-ready folds will be written to:

```text
data/SEED-IV/_fold_jsd_degate/
```

## Training

### SEED

```bash
python Train/Train.py \
  --dataset seed \
  --data-root data/SEED/_fold_jsd_degate \
  --out-dir runs/seed
```

### SEED-IV

```bash
python Train/Train.py \
  --dataset seed_iv \
  --data-root data/SEED-IV/_fold_jsd_degate \
  --out-dir runs/seed_iv
```

Optional arguments:

- `--graph-bias-root /path/to/folder` points to a directory containing `A_spatial.npy`.
- `--device auto|cpu|cuda:0` controls device selection.
- `--feature-keys jsd_gated zde` concatenates multiple saved feature tensors.
- `--skip-existing` reuses completed fold checkpoints instead of retraining them.
- `--eval-only` evaluates existing `model_final.pth` checkpoints without training.

## Outputs

Each training run writes:

- `config.json` with the resolved runtime configuration;
- `fold_sidXX/train.log` or `eval.log`;
- `fold_sidXX/metrics.json`;
- `fold_sidXX/model_final.pth`;
- `overall_metrics.json`.

## Release Notes

Before making the repository public, you may still want to add:

- a project `LICENSE`;
- a paper citation entry (`CITATION.cff` or BibTeX);
- a short changelog or release tag for the exact paper version.
