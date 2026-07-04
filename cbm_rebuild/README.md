# CBM DE + Plain Transformer Rebuild

This directory is an independent, reproducible baseline for the *Computers in Biology and Medicine* submission. It does not import from or modify the legacy `Preprocess/` and `Train/` code. The only implemented model is a plain Transformer over five-band differential-entropy (DE) windows; CMRD, graph priors, GAT, multi-branch fusion, and ablations are intentionally out of scope.

## Protocol

Both datasets use the same processing after a time-domain trial is loaded:

1. Validate and orient the signal as `[62, time_points]`.
2. Resample to 200 Hz when the configured input rate differs from 200 Hz.
3. Apply a fourth-order zero-phase Butterworth 1–75 Hz broad-band filter.
4. Apply full-trial band-pass filters for delta (1–4), theta (4–8), alpha (8–14), beta (14–31), and gamma (31–50 Hz). Full-trial filtering is done before segmentation to limit short-window edge artifacts.
5. Split into non-overlapping 1 s windows (200 samples, 200-sample hop).
6. Compute Gaussian DE, `0.5 * log(2*pi*e*variance + eps)`, per channel and band. Each window is `[62, 5]`, flattened channel-major to 310 features.
7. Pad each dataset independently and save a Boolean real-window mask.

The supplied official files in this project are already represented at 200 Hz, so both configs set `original_sampling_rate` to 200 and correctly skip resampling. If another official release has a different input rate, change only that config value. SEED starts from `Preprocessed_EEG`; SEED-IV starts from `eeg_raw_data`. These sources are not claimed to have equivalent upstream preprocessing.

Official MAT variables include subject-dependent prefixes in the files available here (`ww_eeg1`, `tyc_eeg1`, and so on), although dataset descriptions often show `eeg_1` or `cz_eeg1`. The loaders accept only variables ending in `eegN`/`eeg_N`, validate the complete trial range, and ignore macOS `._*.mat` sidecar files.

SEED labels are mapped `negative -> 0`, `neutral -> 1`, `positive -> 2`. SEED-IV retains its official mapping `neutral -> 0`, `sad -> 1`, `fear -> 2`, `happy -> 3`.

## Environment and commands

Run from the repository root in the requested conda environment:

```powershell
conda activate bilstm

python cbm_rebuild/scripts/preprocess_seed.py --config cbm_rebuild/configs/seed_de_transformer.yaml
python cbm_rebuild/scripts/preprocess_seediv.py --config cbm_rebuild/configs/seediv_de_transformer.yaml

python cbm_rebuild/scripts/train_seed_de_transformer.py --config cbm_rebuild/configs/seed_de_transformer.yaml
python cbm_rebuild/scripts/train_seediv_de_transformer.py --config cbm_rebuild/configs/seediv_de_transformer.yaml


```

The `.yaml` files are deliberately JSON-compatible YAML. They work with PyYAML when present and with the built-in JSON fallback in the current `bilstm` environment, where PyYAML is absent.

Preprocessing is CPU- and disk-intensive because it reads every official time-domain trial and saves a padded dataset. SEED should yield 675 trials (45 sessions × 15 trials); SEED-IV should yield 1080 trials (45 sessions × 24 trials). The scripts fail with contextual errors on missing variables, invalid channel counts, short trials, or non-finite signals.

## Leakage-free LOSO

Each fold holds out all sessions and trials from exactly one target subject. A stratified validation subset is selected only from source-subject trials. Feature-wise mean and standard deviation are fitted only on real windows from the remaining source-training trials; padding, source validation, and the target subject do not contribute. The same source-training statistics transform training, source validation, and target test trials.

Early stopping and best-epoch selection use source validation macro-F1 (accuracy breaks ties). The target fold is evaluated only once after the best source-validation state is restored. The Transformer receives `src_key_padding_mask=~mask`, and classification uses masked mean pooling.

## Outputs

Generated artifacts are written below `cbm_rebuild/outputs/` (ignored by Git):

```text
processed/   padded NPZ arrays and per-trial metadata JSON
checkpoints/ one model plus source-only normalization statistics per fold
logs/        preprocessing, overall training, and per-fold logs
results/     fold/summary CSV, complete JSON, and resolved config JSON
```

The result JSON includes dataset name, seed, window/hop length, processed shape, dataset-specific maximum sequence length, trial count per subject, fold accuracy and macro-F1, best epoch, confusion matrices, timings, and mean/population-standard-deviation summaries. Expected result tables are:

```text
cbm_rebuild/outputs/results/seed_de_transformer_1s_loso.csv
cbm_rebuild/outputs/results/seediv_de_transformer_1s_loso.csv
```

## Verification

Run the lightweight synthetic tests without processing the full datasets:

```powershell
conda run -n bilstm python -m unittest discover -s cbm_rebuild/tests -v
```

The tests cover DE shape, padding/mask semantics, source-only normalization, LOSO isolation, and invariance of Transformer logits to masked padding values.
