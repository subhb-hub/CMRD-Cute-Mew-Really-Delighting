# CMRD reproducible DE/RD baselines

This repository contains the reproducible first-stage rebuild of the SEED and
SEED-IV cross-subject EEG experiments. The current release intentionally covers
only differential entropy (DE) and reference deviation (RD, called RJSD in the
legacy code) with one shared masked Transformer.

## Environment

The verified local environment is `bilstm` (Python 3.11 with CUDA-enabled
PyTorch):

```powershell
conda activate bilstm
pip install -e .
```

Installing the package is optional for the four scripts because they add
`src/` to their import path. To create the environment elsewhere:

```powershell
conda env create -f env.yml
conda activate bilstm
```

## Dataset layout

The default configuration expects the dataset as a sibling of this repository:

```text
../Dataset/
├── Ori/
│   ├── SEED/Preprocessed_EEG/
│   │   ├── 1_*.mat ... 15_*.mat
│   │   └── label.mat
│   └── SEED-IV/eeg_raw_data/
│       ├── 1/*.mat
│       ├── 2/*.mat
│       └── 3/*.mat
└── Processed/CMRD/                 # generated feature caches
```

Change `paths.data_root` or `paths.processed_root` in YAML when the data lives
elsewhere. Paths are always resolved from the repository root, not the current
shell directory.

## Four experiment entry points

SEED DE:

```powershell
python scripts/preprocess_de.py --config configs/seed/de.yaml --resume
python scripts/train_de.py --config configs/seed/de.yaml --mode tune
python scripts/train_de.py --config configs/seed/de.yaml --mode final
```

SEED RD:

```powershell
python scripts/preprocess_rd.py --config configs/seed/rd.yaml --resume
python scripts/train_rd.py --config configs/seed/rd.yaml --mode tune
python scripts/train_rd.py --config configs/seed/rd.yaml --mode final
```

Use `configs/seediv/de.yaml` and `configs/seediv/rd.yaml` for SEED-IV. A single
fold can be checked with `--fold 1`. All scripts accept dotted configuration
overrides, for example:

```powershell
python scripts/train_de.py --config configs/seed/de.yaml --mode tune --fold 1 `
  --set training.epochs=2 --set training.device=cpu
```

`--resume` reuses complete cache/job artifacts, while `--force` explicitly
recomputes them. Formal `final` training requires a completed source-only tuning
selection for every requested fold.

## Scientific protocol

- Both datasets use 62 channels, 200 Hz signals, non-overlapping 1 s windows,
  and delta/theta/alpha/beta/gamma bands.
- DE is `0.5 * log(2*pi*e*variance + eps)` after full-trial band filtering.
- RD is Jensen-Shannon divergence between each spectral histogram and a
  fold-specific reference fitted only from source-training subjects.
- Every outer LOSO fold holds out one target subject and deterministically holds
  out two additional complete source subjects for validation. All sessions and
  trials of a subject stay in one split.
- Normalization is fitted only on real source-training windows. Padding is
  excluded from attention and masked mean pooling.
- Tuning evaluates 12 declared candidates using source validation Macro-F1
  (accuracy tie-break). Target trials are not loaded until the candidate has
  been fixed. Final results use seeds 42, 3407, and 2026.

## Artifacts

Large feature caches are written to `../Dataset/Processed/CMRD`. Each cache has
a preprocessing signature, dataset inventory, per-trial metadata, and, for RD,
the exact reference provenance.

Training outputs are written to ignored, immutable directories under:

```text
runs/<dataset>/<feature>/<timestamp>_<mode>_<config-hash>/
├── resolved_config.json
├── environment.json
├── manifest.json
├── selected_by_fold.json            # tuning
├── tuning_summary.csv                # tuning
├── fold_results.csv                  # final
├── summary.json                      # final
└── folds/fold-XX/...                 # epoch CSV, logs, results, checkpoints
```

The final summary reports variance across subjects separately from variance
across random seeds.

## Verification

```powershell
$env:PYTHONPATH = "src"
conda run -n bilstm python -m unittest discover -s tests -v
```

The tests cover formulas and shapes, configuration/path handling, cache
signatures, subject isolation, source-only normalization, padding invariance,
target-free tuning, checkpoint/result writing, and resume behavior.

