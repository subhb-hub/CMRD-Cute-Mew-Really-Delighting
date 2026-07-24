from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUTPUT = Path("notebooks/faced_psd_jsd_legacy_graph_backbone.ipynb")


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    md(
        r"""
# FACED frequency-resolved sqrt-JSD + legacy GraphBackbone（无图偏置）

## Goal

这个 Notebook 把 `C:/Users/Lin/Downloads/Train.py` 当前实际入口使用的 SEED-IV 模型和训练方案迁移到 FACED，用来检验现有的逐频点 PSD-JSD 特征。

保留的旧方案：

- flatten channel×feature → temporal Transformer → masked mean pooling → MLP classifier；
- `d_model=600`、20 heads、5 个时间层、dropout 0.2；
- Adam、100 epochs、cosine learning-rate、label smoothing 0.2、gradient clipping 5.0。

按当前实验要求做的明确修改：

- JSD 使用逐频点 `sqrt(j(f))`，并验证 `sum_f sqrt(j(f))² = JSD`；
- 不使用图偏置；按 `Train.py` 的真实控制流，`chan_bias=None` 时整个 spatial/channel block 都跳过；
- 删除没有进入分类 loss 的 `quality` 和 FCCA；
- 每 10 epoch 只评估 source-validation，外层目标主体默认不加载。

默认 `RUN_TRAINING=True`，运行全部单元会开始完整训练。先保持 `EVALUATE_TARGET_AFTER_LOCK=False`；只有配置完全锁定后才能手动开启最后的目标评估单元。
"""
    ),
    md("## Parameters\n\n所有需要调整的参数集中在这里。每次材料性修改都使用新的 `RUN_NAME`。"),
    code(
        r"""
# ------------------------------ Protocol ------------------------------
RUN_NAME = "faced_sqrt_jsd_legacy_graph_seed42"
FOLD = 1
SEED = 42
SOURCE_DEV_SUBJECTS = 37  # 111 source subjects -> 74 fit / 37 dev
RUN_TRAINING = True
EVALUATE_TARGET_AFTER_LOCK = False
TARGET_CHECKPOINT = "best_source_dev"  # "best_source_dev" or "final"

# ------------------------------ Feature -------------------------------
EPSILON = 1e-12
FEATURE_STORAGE_DTYPE = "float16"
STANDARDIZE_SOURCE_FEATURES = True

# ------------------------ Legacy model (SEED-IV) ---------------------
D_MODEL = 600
NHEAD = 20
TEMPORAL_LAYERS = 5
DROPOUT = 0.20
ATTENTION_BIAS = None  # explicitly unused
USE_CHANNEL_ATTENTION = False  # Train.py skips spatial block when chan_bias=None
USE_QUALITY = False
USE_FCCA = False

# ----------------------- Legacy training scheme ----------------------
EPOCHS = 100
BATCH_SIZE = 24
EVAL_BATCH_SIZE = 48
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 5.0
LABEL_SMOOTHING = 0.20
MIN_LR_RATIO = 0.10
EVAL_EVERY = 10
DEVICE = "auto"
NUM_WORKERS = 0  # deterministic and Windows-safe

# -------------------------- Hard sanity gate --------------------------
RUN_SANITY_GATE = True
SANITY_SAMPLES_PER_CLASS = 2
SANITY_MAX_STEPS = 200
SANITY_LEARNING_RATE = 3e-4
SANITY_TARGET_ACCURACY = 0.99
SANITY_TARGET_LOSS = 0.05
"""
    ),
    md("## Setup\n\n### 1. Imports, paths, deterministic runtime"),
    code(
        r"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


REPO_ROOT = find_repo_root(Path.cwd())
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cmrd.faced import (
    EEG_CHANNEL_NAMES,
    EMOTION_NAMES,
    SUBJECTS,
    VIDEO_LABELS,
    VIDEOS,
    official_fold_subjects,
)
from cmrd.faced_psd_jsd_experiment import (
    SpectraStore,
    fit_reference,
    materialize_split,
)

seed_everything(SEED)
device = torch.device(
    "cuda" if DEVICE == "auto" and torch.cuda.is_available()
    else DEVICE if DEVICE != "auto" else "cpu"
)
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")

RUN_ROOT = REPO_ROOT / "runs" / RUN_NAME
RUN_ROOT.mkdir(parents=True, exist_ok=True)
LEGACY_TRAIN_PATH = Path(r"C:/Users/Lin/Downloads/Train.py")
LEGACY_TRAIN_SHA256 = "422fea4eabe380358784be7e49dc65fad50c17a9c1c7aea53bcf29c936a58ff4"
if LEGACY_TRAIN_PATH.is_file():
    observed_hash = hashlib.sha256(LEGACY_TRAIN_PATH.read_bytes()).hexdigest()
    if observed_hash != LEGACY_TRAIN_SHA256:
        print("Warning: Train.py has changed since this notebook was generated:", observed_hash)

print("Repository:", REPO_ROOT)
print("Device:", device)
print("Output:", RUN_ROOT)
print("Legacy Train.py SHA256:", LEGACY_TRAIN_SHA256)
"""
    ),
    md("## Data\n\n### 2. Locate PSD cache and freeze source/dev/target subjects"),
    code(
        r"""
BASE_CACHE = REPO_ROOT / "runs" / "faced_native_compact_base_seed42" / "cache" / "native_spectra"
manifest_paths = sorted(BASE_CACHE.glob("*/manifest.json"))
complete = []
for manifest_path in manifest_paths:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("all_subjects_complete") and len(payload.get("subjects_complete", [])) == SUBJECTS:
        complete.append((manifest_path, payload))
if len(complete) != 1:
    raise RuntimeError(f"Expected exactly one complete native PSD cache, found {len(complete)}")

SPECTRA_MANIFEST_PATH, spectra_manifest = complete[0]
SPECTRA_ROOT = SPECTRA_MANIFEST_PATH.parent
BAND_NAMES = tuple(spectra_manifest["band_names"])
BAND_SIZES = tuple(map(int, spectra_manifest["band_sizes"]))
NATIVE_FEATURES = sum(BAND_SIZES)  # 3+4+6+16+17 = 46

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
display(pd.DataFrame({"band": BAND_NAMES, "native_frequency_points": BAND_SIZES}))
print("Compact model input per window:", (len(EEG_CHANNEL_NAMES), NATIVE_FEATURES))
"""
    ),
    md("### 3. Fit source-only reference and build frequency-resolved sqrt-JSD"),
    code(
        r"""
def compact_native_bins(padded: np.ndarray) -> np.ndarray:
    # [N,T,C,5,17] -> [N,T,C,46], dropping only deterministic padding.
    return np.concatenate(
        [padded[..., band, :size] for band, size in enumerate(BAND_SIZES)],
        axis=-1,
    )


def fit_compact_standardizer(values: np.ndarray, chunk_trials: int = 32):
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


store = SpectraStore(SPECTRA_ROOT)
reference, reference_windows = fit_reference(store, fit_subjects)

train_raw = materialize_split(store, fit_subjects, reference, EPSILON, FEATURE_STORAGE_DTYPE)
train_features = compact_native_bins(train_raw.pop("x"))
train_labels = np.asarray(train_raw["y"], dtype=np.int64)
train_subject_ids = np.asarray(train_raw["subjects"], dtype=np.int64)
train_jsd_error = float(train_raw["maximum_invariant_error"])
del train_raw
gc.collect()

dev_raw = materialize_split(store, dev_subjects, reference, EPSILON, FEATURE_STORAGE_DTYPE)
dev_features = compact_native_bins(dev_raw.pop("x"))
dev_labels = np.asarray(dev_raw["y"], dtype=np.int64)
dev_subject_ids = np.asarray(dev_raw["subjects"], dtype=np.int64)
dev_jsd_error = float(dev_raw["maximum_invariant_error"])
del dev_raw
gc.collect()

feature_mean, feature_std = fit_compact_standardizer(train_features)
if not STANDARDIZE_SOURCE_FEATURES:
    feature_mean = np.zeros_like(feature_mean)
    feature_std = np.ones_like(feature_std)

target_overlap = set(store.loaded_subjects) & set(outer_target_subjects)
if target_overlap:
    raise RuntimeError(f"Outer target was loaded during source preparation: {sorted(target_overlap)}")

audit = {
    "feature": "frequency_resolved_sqrt_jsd",
    "sqrt_invariant": "sum_frequency(field**2) == JSD",
    "band_sizes": list(BAND_SIZES),
    "compact_feature_count": NATIVE_FEATURES,
    "reference_scope": "source_fit_subjects_only",
    "reference_windows": reference_windows,
    "fit_subjects": list(fit_subjects),
    "development_subjects": list(dev_subjects),
    "outer_target_subjects": list(outer_target_subjects),
    "loaded_subjects": sorted(store.loaded_subjects),
    "target_loaded": False,
    "maximum_sqrt_jsd_reconstruction_error": max(train_jsd_error, dev_jsd_error),
}
(RUN_ROOT / "source_isolation_audit.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("Train shape:", train_features.shape)
print("Dev shape:", dev_features.shape)
print("Max sqrt-JSD reconstruction error:", audit["maximum_sqrt_jsd_reconstruction_error"])
print("Outer target loaded:", audit["target_loaded"])
"""
    ),
    md("### 4. Dataset and loaders"),
    code(
        r"""
class CompactJSDDataset(Dataset):
    def __init__(self, features, labels, subject_ids, mean, std):
        self.features = features
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.subject_ids = np.asarray(subject_ids, dtype=np.int64)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        value = self.features[index].astype(np.float32)
        value = (value - self.mean) / self.std
        return (
            torch.from_numpy(np.ascontiguousarray(value)),
            self.labels[index],
            int(self.subject_ids[index]),
        )


train_dataset = CompactJSDDataset(
    train_features, train_labels, train_subject_ids, feature_mean, feature_std
)
dev_dataset = CompactJSDDataset(
    dev_features, dev_labels, dev_subject_ids, feature_mean, feature_std
)
loader_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=loader_generator,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
)
train_eval_loader = DataLoader(
    train_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
)

sample_x, sample_y, sample_subject = train_dataset[0]
assert sample_x.shape == (30, 30, 46)
print("One trial:", sample_x.shape, "label:", int(sample_y), "subject:", sample_subject)
"""
    ),
    md("## Model\n\n### 5. Effective legacy temporal Transformer, without graph bias/quality/FCCA"),
    code(
        r"""
class ChannelSelfAttentionBlock(nn.Module):
    def __init__(self, feature_dim: int, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(feature_dim, d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, feature_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, feature_dim = value.shape
        channel_tokens = value.reshape(batch * time_steps, channels, feature_dim)
        hidden = self.in_proj(channel_tokens)
        attended, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        hidden = self.norm(hidden + self.drop(attended))
        return self.out_proj(hidden).reshape(batch, time_steps, channels, feature_dim)


class NodeProjection(nn.Module):
    def __init__(self, channels: int, feature_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(channels * feature_dim, d_model)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, feature_dim = value.shape
        return self.proj(value.reshape(batch, time_steps, channels * feature_dim))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout(value + self.encoding[:, :value.size(1)])


class LegacyEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        key_padding_mask = ~valid_mask if valid_mask is not None else None
        attended, _ = self.attn(
            value, value, value, key_padding_mask=key_padding_mask, need_weights=False
        )
        value = self.norm1(value + self.drop(attended))
        value = self.norm2(value + self.drop(self.ffn(value)))
        return value


class LegacyGraphBackboneNoBias(nn.Module):
    # Effective Train.py GraphBackbone when chan_bias=None, with no dead heads.

    def __init__(
        self,
        channels: int,
        feature_dim: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        dropout: float,
        time_steps: int,
        classes: int,
        use_channel_attention: bool = False,
    ):
        super().__init__()
        self.channel_attention = (
            ChannelSelfAttentionBlock(feature_dim, d_model, nhead, dropout)
            if use_channel_attention else None
        )
        self.embed = NodeProjection(channels, feature_dim, d_model)
        self.position = PositionalEncoding(d_model, dropout, time_steps)
        self.layers = nn.ModuleList(
            [LegacyEncoderLayer(d_model, nhead, dropout) for _ in range(nlayers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, classes),
        )

    def forward(self, value: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        if value.ndim != 4 or tuple(value.shape[2:]) != (30, NATIVE_FEATURES):
            raise ValueError(f"Expected [B,T,30,{NATIVE_FEATURES}], got {tuple(value.shape)}")
        if valid_mask is None:
            valid_mask = torch.ones(
                value.shape[:2], dtype=torch.bool, device=value.device
            )
        if self.channel_attention is not None:
            value = self.channel_attention(value)
        hidden = self.position(self.embed(value))
        for layer in self.layers:
            hidden = layer(hidden, valid_mask)
        mask = valid_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.classifier(pooled)


def build_model(dropout: float = DROPOUT):
    return LegacyGraphBackboneNoBias(
        channels=30,
        feature_dim=NATIVE_FEATURES,
        d_model=D_MODEL,
        nhead=NHEAD,
        nlayers=TEMPORAL_LAYERS,
        dropout=dropout,
        time_steps=30,
        classes=len(EMOTION_NAMES),
        use_channel_attention=USE_CHANNEL_ATTENTION,
    )


model = build_model().to(device)
parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
with torch.no_grad():
    shape_check = model(torch.stack([sample_x, sample_x]).to(device))
assert shape_check.shape == (2, len(EMOTION_NAMES))
print(f"Trainable parameters: {parameter_count:,}")
print("Forward shape:", tuple(shape_check.shape))
print(
    "Graph bias:", ATTENTION_BIAS,
    "channel attention:", USE_CHANNEL_ATTENTION,
    "quality:", USE_QUALITY,
    "FCCA:", USE_FCCA,
)
del shape_check
if device.type == "cuda":
    torch.cuda.empty_cache()
"""
    ),
    md("### 6. Strict 18-trial overfit gate"),
    code(
        r"""
def stratified_indices(labels: np.ndarray, samples_per_class: int) -> np.ndarray:
    chosen = []
    for label in range(len(EMOTION_NAMES)):
        matches = np.flatnonzero(labels == label)
        if len(matches) < samples_per_class:
            raise RuntimeError(f"Class {label} has too few sanity samples")
        chosen.extend(matches[:samples_per_class].tolist())
    return np.asarray(chosen, dtype=np.int64)


sanity_result = {"status": "skipped"}
if RUN_SANITY_GATE:
    seed_everything(SEED)
    sanity_indices = stratified_indices(train_labels, SANITY_SAMPLES_PER_CLASS)
    sanity_x = torch.stack([train_dataset[int(i)][0] for i in sanity_indices]).to(device)
    sanity_y = torch.as_tensor(train_labels[sanity_indices], dtype=torch.long, device=device)
    sanity_model = build_model(dropout=0.0).to(device)
    sanity_optimizer = Adam(sanity_model.parameters(), lr=SANITY_LEARNING_RATE)
    initial_loss = None
    final_loss = None
    final_accuracy = 0.0
    completed_steps = 0
    for step in range(1, SANITY_MAX_STEPS + 1):
        sanity_model.train()
        sanity_optimizer.zero_grad(set_to_none=True)
        sanity_logits = sanity_model(sanity_x)
        sanity_loss = nn.functional.cross_entropy(sanity_logits, sanity_y)
        sanity_loss.backward()
        sanity_optimizer.step()
        if initial_loss is None:
            initial_loss = float(sanity_loss.detach())
        if step == 1 or step % 10 == 0:
            sanity_model.eval()
            with torch.no_grad():
                checked_logits = sanity_model(sanity_x)
                final_loss = float(nn.functional.cross_entropy(checked_logits, sanity_y))
                final_accuracy = float((checked_logits.argmax(1) == sanity_y).float().mean())
            if final_accuracy >= SANITY_TARGET_ACCURACY and final_loss <= SANITY_TARGET_LOSS:
                completed_steps = step
                break
    passed = final_accuracy >= SANITY_TARGET_ACCURACY and final_loss <= SANITY_TARGET_LOSS
    sanity_result = {
        "status": "passed" if passed else "failed",
        "samples": len(sanity_indices),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "steps": completed_steps or SANITY_MAX_STEPS,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "gradient_clipping": False,
    }
    (RUN_ROOT / "sanity.json").write_text(
        json.dumps(sanity_result, indent=2), encoding="utf-8"
    )
    print(json.dumps(sanity_result, indent=2))
    del sanity_model, sanity_optimizer, sanity_x, sanity_y
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not passed:
        raise RuntimeError("Hard sanity gate failed; do not interpret full training")
else:
    print("Sanity gate skipped by configuration")
"""
    ),
    md("## Training\n\n### 7. Metrics and plotting helpers"),
    code(
        r"""
@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader):
    model.eval()
    targets, predictions = [], []
    loss_sum = 0.0
    count = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    for value, label, _subject in loader:
        value = value.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model(value)
        loss = criterion(logits, label)
        loss_sum += float(loss) * len(label)
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
        "predicted_classes": int(len(np.unique(y_pred))),
        "prediction_histogram": np.bincount(y_pred, minlength=len(EMOTION_NAMES)).tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(len(EMOTION_NAMES))
        ).tolist(),
    }


def plot_confusion(matrix, title, output_path=None):
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
                column, row, str(values[row, column]), ha="center", va="center",
                color="white" if values[row, column] > threshold else "black", fontsize=8,
            )
    if output_path is not None:
        fig.savefig(output_path, dpi=180)
    plt.show()
    plt.close(fig)
"""
    ),
    md("### 8. Train with the legacy SEED-IV recipe\n\n目标主体不会在这个单元中加载。每 10 epoch 的评估对象是 source-validation。"),
    code(
        r"""
history = []
best_source_dev = None
best_epoch = None
BEST_PATH = RUN_ROOT / "best_source_dev.pt"
FINAL_PATH = RUN_ROOT / "model_final.pt"

if RUN_TRAINING:
    if RUN_SANITY_GATE and sanity_result["status"] != "passed":
        raise RuntimeError("Sanity gate must pass before full training")
    seed_everything(SEED)
    model = build_model().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * MIN_LR_RATIO
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    best_key = (-math.inf, -math.inf, -math.inf)
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        gradient_norms = []
        for value, label, _subject in train_loader:
            value = value.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(value)
            loss = criterion(logits, label)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(label)
            correct += int((logits.argmax(1) == label).sum())
            count += len(label)
            gradient_norms.append(float(gradient_norm.detach().cpu()))
        scheduler.step()

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_mode_loss": loss_sum / count,
            "train_mode_accuracy": correct / count,
            "mean_preclip_gradient_norm": float(np.mean(gradient_norms)),
            "dev_accuracy": np.nan,
            "dev_balanced_accuracy": np.nan,
            "dev_macro_f1": np.nan,
            "dev_predicted_classes": np.nan,
        }
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            dev_metrics = evaluate_model(model, dev_loader)
            row.update({
                "dev_accuracy": dev_metrics["accuracy"],
                "dev_balanced_accuracy": dev_metrics["balanced_accuracy"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_predicted_classes": dev_metrics["predicted_classes"],
            })
            key = (
                dev_metrics["macro_f1"],
                dev_metrics["balanced_accuracy"],
                -dev_metrics["loss"],
            )
            if key > best_key:
                best_key = key
                best_epoch = epoch
                best_source_dev = dev_metrics
                torch.save({
                    "model_state_dict": {
                        name: value.detach().cpu() for name, value in model.state_dict().items()
                    },
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "reference": reference,
                    "best_epoch": best_epoch,
                    "source_dev_metrics": best_source_dev,
                    "target_loaded_during_selection": False,
                }, BEST_PATH)
            print(
                f"epoch {epoch:03d}/{EPOCHS} loss={row['train_mode_loss']:.4f} "
                f"train={row['train_mode_accuracy']:.3f} "
                f"dev_bacc={dev_metrics['balanced_accuracy']:.3f} "
                f"dev_f1={dev_metrics['macro_f1']:.3f} "
                f"classes={dev_metrics['predicted_classes']}"
            )
        elif epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch {epoch:03d}/{EPOCHS} loss={row['train_mode_loss']:.4f} "
                f"train={row['train_mode_accuracy']:.3f}"
            )
        history.append(row)

    final_source_train = evaluate_model(model, train_eval_loader)
    final_source_dev = evaluate_model(model, dev_loader)
    torch.save({
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "reference": reference,
        "epoch": EPOCHS,
        "source_train_metrics": final_source_train,
        "source_dev_metrics": final_source_dev,
        "target_loaded_during_training": False,
    }, FINAL_PATH)
    pd.DataFrame(history).to_csv(RUN_ROOT / "training_history.csv", index=False)
    summary = {
        "status": "source_training_complete",
        "legacy_train_sha256": LEGACY_TRAIN_SHA256,
        "model": "effective Train.py GraphBackbone with chan_bias=None; no quality or FCCA",
        "parameter_count": parameter_count,
        "feature": "frequency_resolved_sqrt_jsd_46_native_bins",
        "best_source_dev_epoch": best_epoch,
        "best_source_dev": best_source_dev,
        "final_source_train": final_source_train,
        "final_source_dev": final_source_dev,
        "target_loaded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (RUN_ROOT / "source_training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
else:
    print("RUN_TRAINING=False: model construction and data-path validation only")
"""
    ),
    md("## Results\n\n### 9. Source-only curves and selected confusion matrix"),
    code(
        r"""
if history:
    history_frame = pd.DataFrame(history)
    evaluated = history_frame.dropna(subset=["dev_macro_f1"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(history_frame["epoch"], history_frame["train_mode_loss"])
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(evaluated["epoch"], evaluated["dev_balanced_accuracy"], marker="o", label="BAcc")
    axes[1].plot(evaluated["epoch"], evaluated["dev_macro_f1"], marker="o", label="Macro-F1")
    axes[1].axhline(1 / len(EMOTION_NAMES), color="gray", linestyle="--", label="chance BAcc")
    axes[1].set(title="Source-validation metrics", xlabel="Epoch", ylabel="Score")
    axes[1].legend()
    fig.savefig(RUN_ROOT / "source_training_curves.png", dpi=180)
    plt.show()
    plt.close(fig)
    plot_confusion(
        best_source_dev["confusion_matrix"],
        f"Best source-validation confusion (epoch {best_epoch})",
        RUN_ROOT / "best_source_dev_confusion.png",
    )
    display(evaluated[[
        "epoch", "train_mode_accuracy", "dev_accuracy",
        "dev_balanced_accuracy", "dev_macro_f1", "dev_predicted_classes"
    ]])
else:
    print("No training history yet")
"""
    ),
    md(
        "## Optional locked target evaluation\n\n"
        "这个单元默认不执行。只有当模型、训练参数和 checkpoint 选择规则已经根据 source 证据锁定后，"
        "才把 `EVALUATE_TARGET_AFTER_LOCK=True`。执行后不要再根据目标结果修改当前协议。"
    ),
    code(
        r"""
if not EVALUATE_TARGET_AFTER_LOCK:
    print("Outer target remains locked and unread.")
else:
    checkpoint_path = BEST_PATH if TARGET_CHECKPOINT == "best_source_dev" else FINAL_PATH
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Train and lock the source checkpoint first: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    locked_reference = checkpoint["reference"]
    locked_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    locked_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)

    target_raw = materialize_split(
        store, outer_target_subjects, locked_reference, EPSILON, FEATURE_STORAGE_DTYPE
    )
    target_features = compact_native_bins(target_raw.pop("x"))
    target_labels = np.asarray(target_raw["y"], dtype=np.int64)
    target_subject_ids = np.asarray(target_raw["subjects"], dtype=np.int64)
    target_error = float(target_raw["maximum_invariant_error"])
    target_dataset = CompactJSDDataset(
        target_features, target_labels, target_subject_ids, locked_mean, locked_std
    )
    target_loader = DataLoader(
        target_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
    )
    locked_model = build_model().to(device)
    locked_model.load_state_dict(checkpoint["model_state_dict"])
    target_metrics = evaluate_model(locked_model, target_loader)
    target_result = {
        "status": "outer_target_evaluated_after_source_lock",
        "checkpoint": TARGET_CHECKPOINT,
        "metrics": target_metrics,
        "target_subjects": list(outer_target_subjects),
        "target_used_for_selection": False,
        "post_target_tuning_permitted": False,
        "maximum_sqrt_jsd_reconstruction_error": target_error,
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
    ),
    md(
        r"""
## Checks & interpretation boundary

- `source_isolation_audit.json` 必须显示 `target_loaded=false`，才能把 source-validation 曲线用于模型选择。
- Sanity gate 失败表示优化路径仍有问题；不要把后续低指标解释为特征无效。
- Source-validation 通过但 target 很低，才支持“跨主体泛化差”的结论。
- 这个 Notebook 是单个 outer fold 的模型试验，不等于完整论文级多 fold 结果。
- 图偏置、随之不生效的 spatial block、quality 和 FCCA 已按要求删除；逐频点 sqrt-JSD 的 46 个原生频点全部保留。
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3 (cmrd)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
