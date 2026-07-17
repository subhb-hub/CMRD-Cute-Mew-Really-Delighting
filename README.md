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

### DEAP original-BDF ICA-cleaned DE + RJSD preprocessing

DEAP uses the original 512 Hz BDF archive, never the official 4-45 Hz
preprocessed Python/Matlab release. The loader uses status code 4 as the video
start, cuts exactly 60 s (excluding the pre-trial baseline), reorders every EEG
channel by name to the canonical BioSemi32 order, resamples to 200 Hz, and keeps
EXG1-EXG4 as EOG references during ICA. The formal default is a non-overlapping
1 s window:

```powershell
conda activate cmrd
python scripts/preprocess_deap_de_rjsd_ica.py --stage all --strict-ica --resume
python scripts/validate_deap_de_rjsd_ica.py --deep --write
```

The default label is the valence/arousal quadrant at threshold 5.0:
`LVLA=0`, `LVHA=1`, `HVLA=2`, and `HVHA=3`. Every trial manifest also retains
the continuous valence, arousal, dominance, and liking ratings plus separately
derived binary labels. The completed cache is stored under
`../Dataset/Processed/CMRD/deap/de_rjsd_ica_1s_hop1/<signature>/`; the reusable
ICA time series are under `../Dataset/Processed/CMRD/deap/ica_cleaned/`.

DEAP subjects 23-32 use a different on-disk EEG order, subjects 24-28 leave the
status-channel name blank, and subjects 29-32 contain duplicated unnamed status
channels. The loader handles these documented source-file variations by channel
name and by the low byte of the BioSemi status word; the end marker is retained
for audit only because its timestamp is not reliable enough to define the fixed
60 s stimulus cut.

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

## Fixed-protocol RJSD matrix

The revision experiments use a separate, immutable first-stage protocol:
15-fold LOSO, 1 s non-overlapping windows, and seed 42. Each formal fold trains
on all 14 source subjects for one predeclared dataset-level epoch count, then
evaluates the held-out target subject exactly once. The cached 12/2 source split
is used only by later mechanism diagnostics; it is not used for formal model
selection. The full matrix contains 600 fold jobs across both datasets, four
representations, and five models.

The default fixed epoch is 80 for both datasets. If it should be changed, edit
`matrix.fixed_epoch` in the two fixed-protocol configs before `LockEpoch`. The
lock is immutable once formal tasks start and prevents accidental epoch changes.

Prepare and validate the matching 1 s / 1 s caches, lock the dataset-level
epochs, and run the short smoke checks first:

```powershell
.\scripts\run_fixed_protocol.ps1 -Stage Prepare
.\scripts\run_fixed_protocol.ps1 -Stage Validate
.\scripts\run_fixed_protocol.ps1 -Stage LockEpoch
.\scripts\run_fixed_protocol.ps1 -Stage Smoke
```

Start or resume the long matrix without interactive monitoring:

```powershell
.\scripts\run_fixed_protocol.ps1 -Stage Matrix -Resume -RetryFailed
```

The small MLP uses trial-level temporal mean and standard deviation features,
standardized using all 14 source subjects only. To run or resume just the MLP
conditions before continuing the rest of the matrix:

```powershell
.\scripts\run_fixed_protocol.ps1 -Stage MLP -Resume -RetryFailed
```

The fixed-protocol configs include an RTX 5080 Laptop 16GB runtime profile:
batch sizes 64 for the pooled MLP, 64 for the plain Transformer, and 16 for
hierarchical attention. The MLP remains float32 because BF16 did not improve
its measured throughput and reduced source-validation accuracy; both
Transformer models use BF16 autocast. Deterministic math attention remains
enabled. Windows data loaders stay single-process because samples are already
resident in memory and worker processes would duplicate large arrays.

Feature archives are decompressed once per dataset process and reused through
an in-memory LRU cache. The first classical task for each underlying feature
family warms that cache, after which folds run four at a time with eight
numerical threads per task. GPU models remain sequential to avoid device-memory
contention. Separate `Classic`, `MLP`, `Transformer`, and `Hierarchical` stages
allow long model families to be resumed independently.

Linear SVM uses `tol=1e-3` and `max_iter=5000`; this converged faster in the
source-only benchmark than the stricter default tolerance. Strict summaries
reject older SVM artifacts that reached the previous 1000-iteration ceiling.

Inspect progress without attaching to the training process:

```powershell
.\scripts\run_fixed_protocol.ps1 -Stage Status
```

After all 600 tasks complete, create the strict statistical summary and then
run the source-only mechanism analyses:

```powershell
.\scripts\run_fixed_protocol.ps1 -Stage Summarize
.\scripts\run_fixed_protocol.ps1 -Stage Mechanism
```

The handoff artifacts are `runs/fixed_protocol_seed42/matrix_manifest.json`,
`summary.json`, `failed_tasks.csv` (when applicable), and the per-dataset files
under `mechanism/`. Historical target-monitoring runs are never reused by this
pipeline.

## Native-grid compact feature pilot

The fold-1 pilot compares three one-scalar-per-channel-band representations on
the unpadded one-second FFT grid: square-root JSD, an unsupervised first
Fisher-Rao tangent principal coordinate, and support-diameter-normalized
Wasserstein-1. All source references, the Fisher-Rao PCA state, and z-score
statistics are fitted from the 14 non-target subjects only. Training uses the
Base architecture and v2 settings for 200 epochs; target metrics are recorded
every 10 epochs for diagnosis but never select a checkpoint.

Validate the reusable ICA-cleaned caches without starting training:

```powershell
.\scripts\run_native_compact.ps1 -Stage Validate -CondaEnv cmrd
```

Run both datasets and all three fold-1 conditions, with safe resume behavior:

```powershell
.\scripts\run_native_compact.ps1 -Stage All -CondaEnv cmrd -Resume -RetryFailed
```

Use `-Stage SqrtJsd`, `-Stage FisherRao`, or `-Stage Wasserstein` to run one
representation after `-Stage Lock`. `-Dataset Seed` and `-Dataset SeedIV`
restrict execution to one dataset. Artifacts are isolated under
`runs/native_compact_v1_seed42/`; `-Stage Status` reads progress without
starting work.
