# CMRD

Code for **Confidence-Modulation Reference Deviation Representation (CMRD)** on
SEED and SEED-IV for cross-subject EEG emotion recognition.

This repository will be fully released when we are required during the review or the paper is accepted.

## Environment

```bash
conda env create -f env.yml
conda activate cmrd
```

or

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if GPU training is
used.

## Files

```text
Preprocess/
├── Pre-RJSD-SEED.py   # SEED JSD folds
├── Pre-DE-SEED.py     # SEED DE folds
├── Pre-SEED.py        # SEED zDE-gated fusion
├── Pre-SEED-IV.py     # SEED-IV end-to-end preprocessing
└── Spatial.py         # spatial adjacency A_spatial.npy
Train/
└── Train.py
```

## Data Layout

Place the raw datasets under `data/`:

```text
data/
├── SEED-RAW/
│   ├── Preprocessed_EEG/
│   │   ├── *.mat
│   │   └── label.mat
│   ├── channel-order.xlsx
│   └── channel_62_pos.locs
└── SEED-IV-RAW/
    ├── eeg_raw_data/
    │   ├── 1/*.mat
    │   ├── 2/*.mat
    │   └── 3/*.mat
    ├── Zehn-Channel Order.xlsx
    └── channel_62_pos.locs
```

If your file names differ, pass the corresponding paths with the command-line
arguments shown below.

## Run SEED

```bash
python Preprocess/Pre-RJSD-SEED.py \
  --base-path data/SEED-RAW \
  --save-root data/SEED

python Preprocess/Pre-DE-SEED.py \
  --base-path data/SEED-RAW \
  --save-root data/SEED

python Preprocess/Pre-SEED.py \
  --jsd_root data/SEED/_fold_jsd \
  --de_root data/SEED/_fold_de \
  --save_root data/SEED/_fold_jsd_degate

python Preprocess/Spatial.py \
  --channel-order data/SEED-RAW/channel-order.xlsx \
  --locs data/SEED-RAW/channel_62_pos.locs \
  --save-root data/SEED/_fold_jsd_degate
```

Then edit `SeedConfig` in `Train/Train.py`:

```python
data_root = "data/SEED/_fold_jsd_degate"
out_dir = "runs/SEED"
graph_bias_root = "data/SEED/_fold_jsd_degate"
```

## Run SEED-IV

```bash
python Preprocess/Pre-SEED-IV.py \
  --base-path data/SEED-IV-RAW \
  --save-root data/SEED-IV

python Preprocess/Spatial.py \
  --channel-order "data/SEED-IV-RAW/Zehn-Channel Order.xlsx" \
  --locs data/SEED-IV-RAW/channel_62_pos.locs \
  --save-root data/SEED-IV/_fold_jsd_degate
```

Then edit `SeedIVConfig` in `Train/Train.py`:

```python
data_root = "data/SEED-IV/_fold_jsd_degate"
out_dir = "runs/SEED-IV"
jsd_root = "data/SEED-IV/_fold_jsd_degate"
```

## Training

In `Train/Train.py`, keep only the dataset block you want to run at the bottom,
then start training:

```bash
python Train/Train.py
```

Outputs are written to the configured `out_dir`.
