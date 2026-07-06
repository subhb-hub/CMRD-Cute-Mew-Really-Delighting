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

### SEED-IV ICA-cleaned DE + RJSD preprocessing

The separate legacy-protocol diagnostic pipeline adds 50 Hz notch filtering,
1 Hz high-pass before ICA, `find_bads_eog`, `find_bads_muscle`, final 1-75 Hz
band-pass filtering, and 1 s windows with a 0.5 s hop. It saves DE globally and
fits fold-specific RJSD references from source-training subjects only:

```powershell
conda activate cmrd
python scripts/preprocess_seediv_de_rjsd_ica.py --stage all --strict-ica --resume
```

Use `--stage trials` and `--stage folds` to run the expensive ICA and fold
construction separately. Generated data are placed under
`../Dataset/Processed/CMRD/seediv/de_rjsd_ica_1s_hop05/<signature>/`.

The matching SEED pipeline starts from the official `Preprocessed_EEG` trials,
does not repeat downsampling or the official 0-75 Hz preprocessing, verifies
that the SEED/SEED-IV channel orders match before reusing the montage, and
interpolates extreme-variance bad channels before ICA:

```powershell
conda activate cmrd
python scripts/preprocess_seed_de_rjsd_ica.py --stage all --strict-ica --resume
```

Its generated data are placed under
`../Dataset/Processed/CMRD/seed/de_rjsd_ica_1s_hop05/<signature>/`.

For 4 s windows with a 1 s hop, use the matching entry point:

```powershell
python scripts/preprocess_seediv_de_rjsd_ica_4s_hop1.py --stage all --strict-ica --resume
# Or, for SEED:
python scripts/preprocess_seed_de_rjsd_ica_4s_hop1.py --stage all --strict-ica --resume
```

These scripts store ICA-cleaned continuous trials in the window-independent
`../Dataset/Processed/CMRD/<dataset>/ica_cleaned/<cleaning-signature>/` cache.
Later window configurations reuse that cache and only repeat feature windowing.
The older `de_rjsd_ica_1s_hop05` archives contain only windowed `de` and
`p_hist`, so they cannot seed this cache; the first run after this change must
perform ICA once. Use `--force` to rebuild windowed features while retaining
the ICA cache, or `--force-ica` to rebuild both.

Train the completed SEED-IV 4s/1s cache with the matching entry point:

```powershell
python scripts/train_seediv_de_rjsd_ica_4s_hop1.py --validate-only
python scripts/train_seediv_de_rjsd_ica_4s_hop1.py --feature CMRD --fold 1 --resume
```

For epoch-by-epoch diagnostic target monitoring, open
`notebooks/seediv_feature_tuning_4s_hop1_monitor.ipynb`. Its outputs and shared
source-statistics cache are isolated under the `seediv_feature_tuning_4s_hop1`
diagnostic directory.

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
