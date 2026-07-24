from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "faced_hierarchical_frequency_band_channel_time.ipynb"
OUTPUT = ROOT / "notebooks" / "faced_hierarchical_de_band_channel_time.ipynb"


def replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label!r} occurrence, found {count}")
    return source.replace(old, new, 1)


notebook = nbformat.read(SOURCE, as_version=4)
if len(notebook.cells) != 24:
    raise RuntimeError(f"Expected 24 source cells, found {len(notebook.cells)}")

notebook.cells[0].source = """# FACED hierarchical DE Band–Channel–Time Transformer

## Purpose

This experiment keeps the source-fit/source-development split, target lock, hierarchical
spatial/temporal backbone, optimizer and checkpoint rule of the sqrt-JSD experiment, while
changing only the input representation to five-band differential entropy (DE).

```text
five DE values per channel (delta/theta/alpha/beta/gamma)
    → five independent scalar-to-token band encoders
    → within-channel band Transformer
    → cross-channel Transformer
    → temporal Transformer over 30 one-second windows
    → attentive mean + standard-deviation pooling
    → nine-class classifier
```

DE is loaded from the existing FACED native cache. It was computed from official processed
signals using fourth-order Butterworth subband filtering and Gaussian differential entropy.
The source-fit subjects alone determine channel×band z-score statistics. The outer target is
not read unless the final locked-target cell is explicitly enabled.

This is a representation comparison. Bias features, quality features and FCCA are not used.
"""

notebook.cells[1].source = """## Parameters

Change `RUN_NAME` for every material experiment. Keep target evaluation disabled while
choosing architecture or training settings.
"""

parameters = notebook.cells[2]
parameters.source = replace_once(
    parameters.source,
    'RUN_NAME = "faced_hierarchical_fbct_base_seed42"',
    'RUN_NAME = "faced_hierarchical_de_bct_base_seed42"',
    label="RUN_NAME",
)
parameters.source = replace_once(
    parameters.source,
    "# ------------------------------ Feature -------------------------------\n"
    "EPSILON = 1e-12\n"
    "FEATURE_STORAGE_DTYPE = \"float16\"       # PSD/JSD materialization cache\n"
    "STANDARDIZE_SOURCE_FEATURES = True\n"
    "STANDARDIZED_TENSOR_DTYPE = \"float32\"   # store standardized trials; AMP handles compute dtype\n"
    "STANDARDIZE_CHUNK_TRIALS = 32",
    "# ------------------------------ Feature -------------------------------\n"
    "STANDARDIZE_SOURCE_FEATURES = True\n"
    "STANDARDIZED_TENSOR_DTYPE = \"float32\"   # source-fit zDE storage; AMP handles compute dtype\n"
    "STANDARDIZE_CHUNK_TRIALS = 64",
    label="feature parameters",
)

imports = notebook.cells[4]
imports.source = replace_once(
    imports.source,
    "from cmrd.faced_psd_jsd_experiment import (\n"
    "    SpectraStore,\n"
    "    fit_reference,\n"
    "    materialize_split,\n"
    ")\n",
    "",
    label="sqrt-JSD imports",
)

notebook.cells[5].source = """## Data

### 2. Locate the complete DE cache and freeze source/dev/target subjects
"""

notebook.cells[6].source = """BASE_CACHE = REPO_ROOT / "runs" / "faced_native_compact_base_seed42" / "cache" / "native_spectra"
manifest_paths = sorted(BASE_CACHE.glob("*/manifest.json"))
complete = []
for manifest_path in manifest_paths:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("all_subjects_complete") and len(payload.get("subjects_complete", [])) == SUBJECTS:
        complete.append((manifest_path, payload))
if len(complete) != 1:
    raise RuntimeError(f"Expected exactly one complete FACED native/DE cache, found {len(complete)}")

SPECTRA_MANIFEST_PATH, spectra_manifest = complete[0]
SPECTRA_ROOT = SPECTRA_MANIFEST_PATH.parent
BAND_NAMES = tuple(spectra_manifest["band_names"])
BAND_SIZES = (1,) * len(BAND_NAMES)
DE_FEATURES = len(BAND_NAMES)

source_subjects, outer_target_subjects = official_fold_subjects(FOLD)
shuffled_source = np.random.default_rng(SEED).permutation(source_subjects)
dev_subjects = tuple(sorted(map(int, shuffled_source[:SOURCE_DEV_SUBJECTS])))
fit_subjects = tuple(sorted(map(int, shuffled_source[SOURCE_DEV_SUBJECTS:])))
outer_target_subjects = tuple(map(int, outer_target_subjects))

assert len(fit_subjects) == 74 and len(dev_subjects) == 37
assert set(fit_subjects).isdisjoint(dev_subjects)
assert (set(fit_subjects) | set(dev_subjects)).isdisjoint(outer_target_subjects)

display(pd.DataFrame({
    "role": ["source fit", "source validation", "outer target (locked)"],
    "subjects": [len(fit_subjects), len(dev_subjects), len(outer_target_subjects)],
    "trials": [len(fit_subjects) * VIDEOS, len(dev_subjects) * VIDEOS, len(outer_target_subjects) * VIDEOS],
}))
display(pd.DataFrame({"band": BAND_NAMES, "DE_values_per_channel": [1] * DE_FEATURES}))
print("DE model input per window:", (len(EEG_CHANNEL_NAMES), DE_FEATURES))
print("DE estimator:", spectra_manifest["de_estimator"])
"""

notebook.cells[7].source = """### 3. Load five-band DE and fit source-only standardization

The cached flattened order is `[channel, band]`; it is restored to
`[trial, time, channel, band]`. Only source-fit trials determine the mean and standard
deviation. No pseudo-baseline, bias, quality or FCCA term is applied.
"""

notebook.cells[8].source = """loaded_subjects: set[int] = set()


def load_de_subjects(subjects: tuple[int, ...]):
    feature_parts = []
    label_parts = []
    subject_parts = []
    for subject in subjects:
        path = SPECTRA_ROOT / "subjects" / f"sub{subject:03d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing FACED DE cache: {path}")
        with np.load(path, allow_pickle=False) as archive:
            de = np.asarray(archive["de"], dtype=np.float32)
        expected = (VIDEOS, 30, len(EEG_CHANNEL_NAMES) * DE_FEATURES)
        if de.shape != expected or not np.isfinite(de).all():
            raise ValueError(f"Invalid DE cache for subject {subject}: {de.shape}")
        de = de.reshape(VIDEOS, 30, len(EEG_CHANNEL_NAMES), DE_FEATURES)
        feature_parts.append(de)
        label_parts.append(np.asarray(VIDEO_LABELS, dtype=np.int64))
        subject_parts.append(np.full(VIDEOS, subject, dtype=np.int64))
        loaded_subjects.add(int(subject))
    return (
        np.concatenate(feature_parts, axis=0),
        np.concatenate(label_parts),
        np.concatenate(subject_parts),
    )


def fit_compact_standardizer(values: np.ndarray, chunk_trials: int = 64):
    total = np.zeros(values.shape[2:], dtype=np.float64)
    square = np.zeros_like(total)
    count = 0
    for start in range(0, len(values), chunk_trials):
        chunk = values[start:start + chunk_trials].astype(np.float32)
        total += chunk.sum(axis=(0, 1), dtype=np.float64)
        square += np.square(chunk).sum(axis=(0, 1), dtype=np.float64)
        count += chunk.shape[0] * chunk.shape[1]
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-7] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


train_features, train_labels, train_subject_ids = load_de_subjects(fit_subjects)
dev_features, dev_labels, dev_subject_ids = load_de_subjects(dev_subjects)

feature_mean, feature_std = fit_compact_standardizer(train_features)
if not STANDARDIZE_SOURCE_FEATURES:
    feature_mean = np.zeros_like(feature_mean)
    feature_std = np.ones_like(feature_std)

target_overlap = loaded_subjects & set(outer_target_subjects)
if target_overlap:
    raise RuntimeError(f"Outer target was loaded during source preparation: {sorted(target_overlap)}")

standardized_train_mean = np.mean(
    (train_features.astype(np.float64) - feature_mean) / feature_std,
    axis=(0, 1),
)
standardized_train_std = np.std(
    (train_features.astype(np.float64) - feature_mean) / feature_std,
    axis=(0, 1),
)
audit = {
    "feature": "source_fit_zscored_five_band_differential_entropy",
    "de_estimator": spectra_manifest["de_estimator"],
    "band_names": list(BAND_NAMES),
    "feature_shape_per_trial": [30, len(EEG_CHANNEL_NAMES), DE_FEATURES],
    "standardization_scope": "source_fit_subjects_only",
    "maximum_abs_standardized_train_mean": float(np.max(np.abs(standardized_train_mean))),
    "maximum_abs_standardized_train_std_error": float(np.max(np.abs(standardized_train_std - 1.0))),
    "fit_subjects": list(fit_subjects),
    "development_subjects": list(dev_subjects),
    "outer_target_subjects": list(outer_target_subjects),
    "loaded_subjects": sorted(loaded_subjects),
    "target_loaded": False,
}
(RUN_ROOT / "source_isolation_audit.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("Train shape:", train_features.shape)
print("Dev shape:", dev_features.shape)
print("Outer target loaded:", audit["target_loaded"])
print("Max |source zDE mean|:", audit["maximum_abs_standardized_train_mean"])
print("Max |source zDE std - 1|:", audit["maximum_abs_standardized_train_std_error"])
"""

notebook.cells[9].source = """### 4. Standardize once and cache tensor splits

The source-fit channel×band statistics are applied once before training. The complete DE
tensor is small, so GPU caching is enabled by default.
"""

notebook.cells[11].source = """## Model

### 5. Hierarchical DE Band–Channel–Time Transformer

Each of the five scalar DE values has an independent scalar-to-token MLP. Importantly, a
single scalar is **not** passed through `LayerNorm(1)`, which would erase its value. The
within-channel band, cross-channel and temporal stages otherwise match the sqrt-JSD model.
"""

model_cell = notebook.cells[12]
model_cell.source = model_cell.source.replace("FrequencyBandEncoder", "DifferentialEntropyBandEncoder")
model_cell.source = replace_once(
    model_cell.source,
    "                nn.LayerNorm(size),\n                nn.Linear(size, frequency_hidden),",
    "                nn.Linear(size, frequency_hidden),",
    label="scalar LayerNorm removal",
)
model_cell.source = model_cell.source.replace("# value: [B,T,C,46]", "# value: [B,T,C,5] standardized DE")
model_cell.source = model_cell.source.replace("HierarchicalFBCT", "HierarchicalDEBCT")
model_cell.source = model_cell.source.replace("self.frequency_band_encoder", "self.de_band_encoder")
model_cell.source = model_cell.source.replace("channel_tokens = self.frequency_band_encoder(value)", "channel_tokens = self.de_band_encoder(value)")

notebook.cells[13].source = """### 6. Strict 18-trial overfit gate

Before source training, the full DE model must memorize two trials per class in float32.
Dropout, label smoothing, channel-vote loss and subject adversarial loss are disabled for
this diagnostic gate.
"""

training_cell = notebook.cells[18]
training_cell.source = training_cell.source.replace('                    "reference": reference,\n', "")
training_cell.source = training_cell.source.replace('        "reference": reference,\n', "")
training_cell.source = training_cell.source.replace(
    '"model": "hierarchical Frequency-Band-Channel-Time Transformer"',
    '"model": "hierarchical DE Band-Channel-Time Transformer"',
)
training_cell.source = training_cell.source.replace(
    '"feature": "frequency_resolved_sqrt_jsd_46_native_bins"',
    '"feature": "source_fit_zscored_five_band_differential_entropy"',
)
training_cell.source = replace_once(
    training_cell.source,
    "    for epoch in range(1, EPOCHS + 1):\n        training_model.train()",
    "    for epoch in range(1, EPOCHS + 1):\n"
    "        epoch_learning_rate = optimizer.param_groups[0][\"lr\"]\n"
    "        training_model.train()",
    label="epoch learning-rate capture",
)
training_cell.source = replace_once(
    training_cell.source,
    '            "learning_rate": optimizer.param_groups[0]["lr"],',
    '            "learning_rate": epoch_learning_rate,',
    label="learning-rate history",
)

notebook.cells[21].source = """## Optional locked target evaluation

This remains disabled by default. Enable it once, only after the DE architecture, training
configuration and source-development checkpoint rule have been frozen.
"""

notebook.cells[22].source = """if not EVALUATE_TARGET_AFTER_LOCK:
    print("Outer target remains locked and unread.")
else:
    checkpoint_path = BEST_PATH if TARGET_CHECKPOINT == "best_source_dev" else FINAL_PATH
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Train and lock the source checkpoint first: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    locked_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    locked_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)

    target_features, target_labels, target_subject_ids = load_de_subjects(outer_target_subjects)
    target_split = make_tensor_split(
        target_features,
        target_labels,
        target_subject_ids,
        locked_mean,
        locked_std,
        None,
    )
    del target_features
    gc.collect()

    locked_model = build_model().to(device)
    locked_model.load_state_dict(checkpoint["model_state_dict"])
    target_metrics = evaluate_model(locked_model, target_split)

    target_result = {
        "status": "outer_target_evaluated_after_source_lock",
        "feature": "source_fit_zscored_five_band_differential_entropy",
        "checkpoint": TARGET_CHECKPOINT,
        "metrics": target_metrics,
        "target_subjects": list(outer_target_subjects),
        "target_used_for_selection": False,
        "post_target_tuning_permitted": False,
    }
    (RUN_ROOT / "locked_target_result.json").write_text(
        json.dumps(target_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(target_result, indent=2, ensure_ascii=False))
    plot_confusion(
        target_metrics["confusion_matrix"],
        "Locked outer-target confusion matrix",
        RUN_ROOT / "locked_target_confusion.png",
    )
"""

notebook.cells[23].source = """## Checks and interpretation boundary

- `source_isolation_audit.json` must report `target_loaded=false`.
- The sanity gate must pass before full-training metrics are interpreted.
- Compare this run with sqrt-JSD using the same fold, subject split, architecture preset,
  optimizer and checkpoint rule.
- Do not enable subject adversarial learning for the first DE comparison.
- A short validation run only verifies the execution path; it is not a performance estimate.
- The outer target stays unread until all source-development choices are frozen.
- This is one outer fold, not the final multi-fold result.
"""

for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"{OUTPUT.name}:cell-{index}", "exec")
        cell.execution_count = None
        cell.outputs = []

notebook.metadata["kernelspec"] = {
    "display_name": "Python (cmrd)",
    "language": "python",
    "name": "cmrd",
}
notebook.metadata["generated"] = {
    "architecture": "hierarchical DE Band-Channel-Time Transformer",
    "feature": "source-fit z-scored five-band differential entropy",
    "source_notebook": SOURCE.name,
    "target_locked_by_default": True,
}
notebook.metadata.pop("cmrd_bugfix", None)
nbformat.write(notebook, OUTPUT)
print(OUTPUT)
