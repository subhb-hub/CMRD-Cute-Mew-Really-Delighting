# CMRD

Code for **Confidence-Modulation Reference Deviation Representation (CMRD)** for
target-free cross-subject EEG emotion recognition on SEED and SEED-IV.

They will be released if required during review or after the paper is
accepted.

## Files

```text
Preprocess/
├── Spatial.py
├── Pre-DE-SEED.py
├── Pre-RJSD-SEED.py
├── Pre-SEED.py
└── Pre-SEED-IV.py
Train/
└── Train.py
```

## Environment

```bash
conda env create -f env.yml
conda activate cmrd
```

or

```bash
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA version for GPU training.

## Spatial Graph

`A_spatial.npy` can be built from the channel order file and the `.locs` file:

```bash
python Preprocess/Spatial.py \
  --channel-order /path/to/Channel\ Order.xlsx \
  --locs /path/to/channel_62_pos.locs \
  --save-root data/SEED-IV
```

This writes `A_spatial.npy` and `A_meta.npz`. By default it also writes
`W_topo_csr.npz` and `topo_mapping.npz`; use `--no-save-topo` to skip them.

## Training Data

`Train/Train.py` expects prebuilt LOOCV folds:

```text
<data_root>/
├── A_spatial.npy
├── fold_subj_01/
│   ├── train_source/
│   │   └── *.npz
│   └── test_target/
│       └── *.npz
└── fold_subj_02/
```

Each `.npz` trial uses `jsd_gated` by default, with shape `(T, 62, B)`, plus
`label`. SEED labels are mapped from `-1, 0, 1` to `0, 1, 2`; SEED-IV labels are
expected to be `0, 1, 2, 3`.

## Preprocessing

The preprocessing scripts are in `Preprocess/`. 

```bash
python Preprocess/Pre-SEED-IV.py 
```

## Training

Edit `SeedConfig` or `SeedIVConfig` in `Train/Train.py`.

```bash
python Train/Train.py
```

Each fold writes `train.log` or `eval.log`, `metrics.json`,
`model_final.pth`, and `used_channel_graph.npy`. The full run summary is saved
as `overall_metrics.json`.


