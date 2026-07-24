from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "faced_de_simple_structured_cnn_10fold.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python (cmrd)",
    "language": "python",
    "name": "cmrd",
}
nb.metadata["generated"] = {
    "purpose": "minimal diagnostic FACED DE 10-fold baseline",
    "protocol": "nine outer folds train, one outer fold monitored as target",
    "target_monitoring": "diagnostic only; never used for gradients or checkpoint selection",
}

cells = []
cells.append(nbf.v4.new_markdown_cell("""# FACED DE: minimal structured CNN, outer 10-fold diagnostic

## Goal

This notebook deliberately removes nearly all modeling and optimization complexity. It asks
one narrow question: can a small CNN learn the existing five-band FACED DE representation
when the data are kept as `[time, channel, band]`?

For every selected outer fold:

- the other nine folds are the complete training source;
- only source subjects fit the channel×band z-score;
- the held-out fold is evaluated every 10 epochs;
- source-train and held-out-target metrics and confusion matrices are saved every 10 epochs;
- training always runs for a fixed number of epochs, and target metrics never select a model.

Because the target is inspected repeatedly, this is a **target-monitored diagnostic**, not an
independent final test. Its purpose is to expose training failures before rebuilding a strict
source-development protocol.
"""))

cells.append(nbf.v4.new_markdown_cell("""## Parameters

For a quick check, set `FOLDS_TO_RUN=(1,)` and `EPOCHS=10`. The default runs all ten outer
folds. Use a fresh `RUN_NAME` for every material run.
"""))

cells.append(nbf.v4.new_code_cell("""# ------------------------------ Protocol ------------------------------
RUN_NAME = "faced_de_simple_structured_cnn_10fold_seed42"
FOLDS_TO_RUN = tuple(range(1, 11))
SEED = 42

# ------------------------------- Model --------------------------------
CONV_CHANNELS = (8, 16, 32)
POOLED_TIME = 4
POOLED_CHANNELS = 4

# ------------------------------ Training ------------------------------
EPOCHS = 100
EVAL_EVERY = 10
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 1e-3

# ------------------------------- Runtime ------------------------------
DEVICE = "auto"
NUM_WORKERS = 0

# Intentionally absent: validation split, dropout, weight decay, label smoothing,
# scheduler, warmup, early stopping, gradient clipping, AMP, class weighting,
# channel-vote head, subject adversary, attention and target-based selection.
"""))

cells.append(nbf.v4.new_markdown_cell("""## Setup

### 1. Imports and deterministic runtime
"""))

cells.append(nbf.v4.new_code_cell("""from __future__ import annotations

import gc
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


def find_repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "src" / "cmrd" / "faced.py").is_file():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the CMRD repository")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


REPO_ROOT = find_repo_root(Path.cwd())
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cmrd.faced import EEG_CHANNEL_NAMES, EMOTION_NAMES, SUBJECTS, VIDEO_LABELS, VIDEOS, official_fold_subjects

if not FOLDS_TO_RUN:
    raise ValueError("FOLDS_TO_RUN must not be empty")
if any(fold not in range(1, 11) for fold in FOLDS_TO_RUN):
    raise ValueError(f"FOLDS_TO_RUN must contain values from 1 to 10: {FOLDS_TO_RUN}")
if len(set(FOLDS_TO_RUN)) != len(FOLDS_TO_RUN):
    raise ValueError(f"Duplicate folds are not allowed: {FOLDS_TO_RUN}")
if EPOCHS <= 0 or EPOCHS % EVAL_EVERY != 0:
    raise ValueError("EPOCHS must be positive and divisible by EVAL_EVERY")

seed_everything(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(
    "cuda" if DEVICE == "auto" and torch.cuda.is_available()
    else DEVICE if DEVICE != "auto" else "cpu"
)
RUN_ROOT = REPO_ROOT / "runs" / RUN_NAME
RUN_ROOT.mkdir(parents=True, exist_ok=True)

print("Repository:", REPO_ROOT)
print("Device:", device)
print("Folds:", FOLDS_TO_RUN)
print("Output:", RUN_ROOT)
"""))

cells.append(nbf.v4.new_markdown_cell("""## Data

### 2. Load the existing five-band DE cache

The cache was computed from official FACED processed signals with fourth-order Butterworth
subband filters and Gaussian differential entropy. Its flattened `[channel, band]` axis is
restored before modeling.
"""))

cells.append(nbf.v4.new_code_cell("""BASE_CACHE = REPO_ROOT / "runs" / "faced_native_compact_base_seed42" / "cache" / "native_spectra"
manifest_paths = sorted(BASE_CACHE.glob("*/manifest.json"))
complete = []
for manifest_path in manifest_paths:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("all_subjects_complete") and len(payload.get("subjects_complete", [])) == SUBJECTS:
        complete.append((manifest_path, payload))
if len(complete) != 1:
    raise RuntimeError(f"Expected exactly one complete FACED DE cache, found {len(complete)}")

MANIFEST_PATH, cache_manifest = complete[0]
CACHE_ROOT = MANIFEST_PATH.parent
BAND_NAMES = tuple(cache_manifest["band_names"])
N_BANDS = len(BAND_NAMES)

all_de = np.empty(
    (SUBJECTS, VIDEOS, 30, len(EEG_CHANNEL_NAMES), N_BANDS),
    dtype=np.float32,
)
for subject in range(SUBJECTS):
    path = CACHE_ROOT / "subjects" / f"sub{subject:03d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        de = np.asarray(archive["de"], dtype=np.float32)
    expected = (VIDEOS, 30, len(EEG_CHANNEL_NAMES) * N_BANDS)
    if de.shape != expected or not np.isfinite(de).all():
        raise ValueError(f"Invalid DE cache for subject {subject}: {de.shape}")
    all_de[subject] = de.reshape(VIDEOS, 30, len(EEG_CHANNEL_NAMES), N_BANDS)

video_labels = np.asarray(VIDEO_LABELS, dtype=np.int64)
assert video_labels.shape == (VIDEOS,)

cache_audit = {
    "cache_manifest": str(MANIFEST_PATH.resolve()),
    "de_estimator": cache_manifest["de_estimator"],
    "tensor_shape": list(all_de.shape),
    "axis_order": ["subject", "video", "time", "channel", "band"],
    "band_names": list(BAND_NAMES),
    "finite": bool(np.isfinite(all_de).all()),
    "target_monitoring": True,
    "target_used_for_gradients": False,
    "target_used_for_checkpoint_selection": False,
}
(RUN_ROOT / "data_and_protocol_audit.json").write_text(
    json.dumps(cache_audit, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("DE tensor:", all_de.shape)
print("Axis order:", cache_audit["axis_order"])
print("Bands:", BAND_NAMES)
print("Memory MB:", all_de.nbytes / 1024**2)
"""))

cells.append(nbf.v4.new_markdown_cell("""### 3. Source-only fold preparation

Every fold fits one mean and standard deviation for each `[channel, band]` position using all
source subjects, videos and time windows. The target receives those frozen statistics.
"""))

cells.append(nbf.v4.new_code_cell("""def prepare_fold(fold: int):
    source_subjects, target_subjects = official_fold_subjects(fold)
    source_subjects = np.asarray(source_subjects, dtype=np.int64)
    target_subjects = np.asarray(target_subjects, dtype=np.int64)

    if np.intersect1d(source_subjects, target_subjects).size:
        raise RuntimeError(f"Fold {fold}: source/target subject overlap")
    if len(source_subjects) + len(target_subjects) != SUBJECTS:
        raise RuntimeError(f"Fold {fold}: incomplete subject partition")

    source_raw = all_de[source_subjects]
    target_raw = all_de[target_subjects]
    mean = source_raw.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = source_raw.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std[std < 1e-7] = 1.0

    source_x = ((source_raw - mean) / std).reshape(
        -1, 30, len(EEG_CHANNEL_NAMES), N_BANDS
    ).astype(np.float32)
    target_x = ((target_raw - mean) / std).reshape(
        -1, 30, len(EEG_CHANNEL_NAMES), N_BANDS
    ).astype(np.float32)
    source_y = np.tile(video_labels, len(source_subjects))
    target_y = np.tile(video_labels, len(target_subjects))

    source_mean_error = float(np.max(np.abs(source_x.mean(axis=(0, 1)))))
    source_std_error = float(np.max(np.abs(source_x.std(axis=(0, 1)) - 1.0)))
    audit = {
        "fold": fold,
        "source_subjects": source_subjects.tolist(),
        "target_subjects": target_subjects.tolist(),
        "source_trials": int(len(source_y)),
        "target_trials": int(len(target_y)),
        "standardizer_fit_scope": "all nine source folds only",
        "maximum_abs_source_zscore_mean": source_mean_error,
        "maximum_abs_source_zscore_std_error": source_std_error,
        "target_used_for_gradients": False,
        "target_used_for_checkpoint_selection": False,
        "target_monitored_every_n_epochs": EVAL_EVERY,
    }
    return source_x, source_y, target_x, target_y, mean, std, audit


first_fold = int(FOLDS_TO_RUN[0])
check_source_x, check_source_y, check_target_x, check_target_y, _, _, check_audit = prepare_fold(first_fold)
display(pd.DataFrame([
    {"split": "source train", "subjects": len(check_audit["source_subjects"]), "trials": len(check_source_y), "shape": str(check_source_x.shape)},
    {"split": "target monitor", "subjects": len(check_audit["target_subjects"]), "trials": len(check_target_y), "shape": str(check_target_x.shape)},
]))
print("Source z-score mean error:", check_audit["maximum_abs_source_zscore_mean"])
print("Source z-score std error:", check_audit["maximum_abs_source_zscore_std_error"])
del check_source_x, check_source_y, check_target_x, check_target_y
gc.collect()
"""))

cells.append(nbf.v4.new_markdown_cell("""## Model

### 4. Minimal structure-preserving 3D CNN

The model receives `[batch, 1, time, channel, band]`. The first convolution mixes local time
and band values within each channel; the second mixes neighboring channels; the third mixes
all three axes. Band resolution is never pooled away. Flattening happens only after the final
adaptive structured pooling. All learned layers use `bias=False`.
"""))

cells.append(nbf.v4.new_code_cell("""class SimpleStructuredDECNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        c1, c2, c3 = CONV_CHANNELS
        self.features = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=(3, 1, 3), padding=(1, 0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(c1, c2, kernel_size=(1, 3, 1), padding=(0, 1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(2, 2, 1)),
            nn.Conv3d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((POOLED_TIME, POOLED_CHANNELS, N_BANDS)),
        )
        flattened = c3 * POOLED_TIME * POOLED_CHANNELS * N_BANDS
        self.classifier = nn.Linear(flattened, classes, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"Expected [B,T,C,Band], got {tuple(value.shape)}")
        value = self.features(value.unsqueeze(1))
        return self.classifier(value.flatten(1))


model_check = SimpleStructuredDECNN(len(EMOTION_NAMES)).to(device)
parameter_count = sum(parameter.numel() for parameter in model_check.parameters())
with torch.no_grad():
    output_check = model_check(torch.zeros(2, 30, len(EEG_CHANNEL_NAMES), N_BANDS, device=device))
assert output_check.shape == (2, len(EMOTION_NAMES))
assert not any(module.bias is not None for module in model_check.modules() if isinstance(module, (nn.Conv3d, nn.Linear)))

print(model_check)
print(f"Trainable parameters: {parameter_count:,}")
print("Output shape:", tuple(output_check.shape))
del model_check, output_check
if device.type == "cuda":
    torch.cuda.empty_cache()
"""))

cells.append(nbf.v4.new_markdown_cell("""## Training and evaluation

### 5. Plain fixed-epoch loop

The only optimization components are ordinary cross-entropy and Adam with a fixed learning
rate. Metrics are emitted every `EVAL_EVERY` epochs. The final fixed-epoch model is saved;
there is no best-target or early-stopping checkpoint.
"""))

cells.append(nbf.v4.new_code_cell("""def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long())
    generator = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    loss_sum = 0.0
    count = 0
    targets = []
    predictions = []
    criterion = nn.CrossEntropyLoss()
    for value, label in loader:
        value = value.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model(value)
        loss_sum += float(criterion(logits, label).cpu()) * len(label)
        count += len(label)
        targets.append(label.cpu().numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return {
        "loss": loss_sum / count,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "predicted_classes": int(np.unique(y_pred).size),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(EMOTION_NAMES))).tolist(),
    }, y_true, y_pred


def save_confusion(matrix, title: str, path: Path):
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(EMOTION_NAMES)),
        yticks=np.arange(len(EMOTION_NAMES)),
        xticklabels=EMOTION_NAMES,
        yticklabels=EMOTION_NAMES,
    )
    ax.tick_params(axis="x", rotation=45)
    threshold = values.max(initial=0) / 2
    for row in range(len(EMOTION_NAMES)):
        for column in range(len(EMOTION_NAMES)):
            ax.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=8,
            )
    fig.savefig(path, dpi=160)
    plt.close(fig)
"""))

cells.append(nbf.v4.new_markdown_cell("""### 6. Run selected outer folds
"""))

cells.append(nbf.v4.new_code_cell("""metric_rows = []
fold_summaries = []
final_target_pairs = []

for fold in FOLDS_TO_RUN:
    fold = int(fold)
    fold_root = RUN_ROOT / f"fold-{fold:02d}"
    fold_root.mkdir(parents=True, exist_ok=True)

    source_x, source_y, target_x, target_y, source_mean, source_std, fold_audit = prepare_fold(fold)
    (fold_root / "protocol_audit.json").write_text(
        json.dumps(fold_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train_loader = make_loader(source_x, source_y, BATCH_SIZE, shuffle=True)
    source_eval_loader = make_loader(source_x, source_y, EVAL_BATCH_SIZE, shuffle=False)
    target_eval_loader = make_loader(target_x, target_y, EVAL_BATCH_SIZE, shuffle=False)

    seed_everything(SEED + fold)
    model = SimpleStructuredDECNN(len(EMOTION_NAMES)).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    final_target_true = None
    final_target_pred = None

    print(f"\\n=== Fold {fold:02d}: source={len(fold_audit['source_subjects'])} subjects, target={len(fold_audit['target_subjects'])} subjects ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0
        for value, label in train_loader:
            value = value.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(value)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(label)
            running_correct += int((logits.argmax(1) == label).sum().detach().cpu())
            running_count += len(label)

        if epoch % EVAL_EVERY == 0:
            source_metrics, _, _ = evaluate(model, source_eval_loader)
            target_metrics, target_true, target_pred = evaluate(model, target_eval_loader)
            if epoch == EPOCHS:
                final_target_true = target_true
                final_target_pred = target_pred

            for split_name, metrics in (("source", source_metrics), ("target", target_metrics)):
                metric_rows.append({
                    "fold": fold,
                    "epoch": epoch,
                    "split": split_name,
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "predicted_classes": metrics["predicted_classes"],
                })
                save_confusion(
                    metrics["confusion_matrix"],
                    f"Fold {fold:02d} epoch {epoch:03d} {split_name}",
                    fold_root / f"epoch-{epoch:03d}_{split_name}_confusion.png",
                )

            print(
                f"epoch {epoch:03d}/{EPOCHS} "
                f"train_loss={running_loss / running_count:.4f} "
                f"train_acc={running_correct / running_count:.3f} | "
                f"source_f1={source_metrics['macro_f1']:.3f} "
                f"source_bacc={source_metrics['balanced_accuracy']:.3f} | "
                f"target_f1={target_metrics['macro_f1']:.3f} "
                f"target_bacc={target_metrics['balanced_accuracy']:.3f} "
                f"target_classes={target_metrics['predicted_classes']}"
            )

    if final_target_true is None or final_target_pred is None:
        raise RuntimeError(f"Fold {fold}: final target evaluation was not produced")
    final_target_pairs.append((final_target_true, final_target_pred))

    checkpoint = {
        "fold": fold,
        "epoch": EPOCHS,
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "source_mean": source_mean,
        "source_std": source_std,
        "model": "SimpleStructuredDECNN",
        "parameter_count": parameter_count,
        "target_used_for_selection": False,
    }
    torch.save(checkpoint, fold_root / "model_final_fixed_epoch.pt")
    fold_summary = {
        "fold": fold,
        "completed_epochs": EPOCHS,
        "source_subjects": fold_audit["source_subjects"],
        "target_subjects": fold_audit["target_subjects"],
        "final_source": source_metrics,
        "final_target_monitored": target_metrics,
        "target_used_for_selection": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    fold_summaries.append(fold_summary)
    (fold_root / "summary.json").write_text(
        json.dumps(fold_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    del source_x, source_y, target_x, target_y, train_loader, source_eval_loader, target_eval_loader
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

results_frame = pd.DataFrame(metric_rows)
results_frame.to_csv(RUN_ROOT / "metrics_every_10_epochs.csv", index=False)
(RUN_ROOT / "fold_summaries.json").write_text(
    json.dumps(fold_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("\\nSaved metrics:", RUN_ROOT / "metrics_every_10_epochs.csv")
"""))

cells.append(nbf.v4.new_markdown_cell("""## Results

### 7. Fixed-final-epoch fold summary

Only the fixed final epoch is aggregated. No target-observed epoch is selected.
"""))

cells.append(nbf.v4.new_code_cell("""final_rows = results_frame[results_frame["epoch"] == EPOCHS].copy()
display(final_rows[[
    "fold", "split", "accuracy", "balanced_accuracy", "macro_f1", "predicted_classes"
]])

target_final = final_rows[final_rows["split"] == "target"]
aggregate_table = pd.DataFrame({
    "metric": ["accuracy", "balanced_accuracy", "macro_f1"],
    "mean": [target_final[column].mean() for column in ("accuracy", "balanced_accuracy", "macro_f1")],
    "std_across_folds": [target_final[column].std(ddof=0) for column in ("accuracy", "balanced_accuracy", "macro_f1")],
    "folds": len(target_final),
})
display(aggregate_table)

pooled_true = np.concatenate([pair[0] for pair in final_target_pairs])
pooled_pred = np.concatenate([pair[1] for pair in final_target_pairs])
pooled_metrics = {
    "accuracy": float(accuracy_score(pooled_true, pooled_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(pooled_true, pooled_pred)),
    "macro_f1": float(f1_score(pooled_true, pooled_pred, average="macro", zero_division=0)),
    "predicted_classes": int(np.unique(pooled_pred).size),
    "confusion_matrix": confusion_matrix(
        pooled_true, pooled_pred, labels=np.arange(len(EMOTION_NAMES))
    ).tolist(),
}
aggregate_result = {
    "status": "target_monitored_diagnostic_complete",
    "folds_run": list(map(int, FOLDS_TO_RUN)),
    "epochs": EPOCHS,
    "evaluation_interval": EVAL_EVERY,
    "model": "SimpleStructuredDECNN",
    "parameter_count": parameter_count,
    "regularization": "none",
    "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE},
    "target_used_for_selection": False,
    "target_is_independent_final_test": False,
    "target_fold_mean_std": aggregate_table.to_dict(orient="records"),
    "pooled_final_target": pooled_metrics,
}
(RUN_ROOT / "aggregate_final_result.json").write_text(
    json.dumps(aggregate_result, indent=2, ensure_ascii=False), encoding="utf-8"
)
save_confusion(
    pooled_metrics["confusion_matrix"],
    f"Pooled fixed-epoch target confusion ({len(FOLDS_TO_RUN)} folds)",
    RUN_ROOT / "pooled_final_target_confusion.png",
)
print(json.dumps(aggregate_result, indent=2, ensure_ascii=False))
"""))

cells.append(nbf.v4.new_markdown_cell("""## Interpretation boundary

- This notebook is intentionally a diagnostic baseline, not a paper protocol.
- Repeated target inspection is allowed here only to diagnose training behavior.
- Compare source and target curves: low source performance indicates optimization/model
  failure; high source but low target performance indicates cross-subject generalization failure.
- If this model learns normally, add complexity one change at a time and keep this baseline.
- Once the pipeline is stable, restore a source-only development split and a never-monitored
  outer target before making formal claims.
"""))

nb.cells = cells
for index, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"{OUTPUT.name}:cell-{index}", "exec")
nbf.write(nb, OUTPUT)
print(OUTPUT)
