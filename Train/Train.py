# -*- coding: utf-8 -*-
"""
Unified trainer for SEED & SEED-IV (LOOCV / resume) — keep Graph Bias, remove self-supervised (UWM/ING/recon).
- Two independent pipelines (SEED, SEED-IV) live in the SAME file to avoid mixing dataset-specific logic.
- Shared model core keeps the channel graph bias (beta*log(G+eps)).

How to use:
1) Edit the config you need at the bottom (SeedConfig or SeedIVConfig)
2) Run: python train_unified_noSSL_keepGraphBias.py
"""

from __future__ import annotations
import os, sys, math, json, copy, random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================================
# Utils
# ============================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_torch_deterministic(deterministic: bool = True) -> None:
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

class TeeLogger:
    """Mirror stdout to both terminal and a log file."""
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.log_path, "a", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, msg: str):
        self._stdout.write(msg)
        self._fp.write(msg)

    def flush(self):
        self._stdout.flush()
        self._fp.flush()

    def close(self):
        try:
            self._fp.close()
        finally:
            sys.stdout = self._stdout

def _to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


# ============================================================
# Collate: pad / truncate to fixed T_pad
# ============================================================
class PadCollate:
    def __init__(self, T_pad: int):
        self.T_pad = int(T_pad)

    def __call__(self, batch):
        # batch: list of (X, Q, y, path)
        Xs, Qs, ys, paths = zip(*batch)
        Bz = len(Xs)
        T_pad = self.T_pad
        C = Xs[0].shape[1]
        B = Xs[0].shape[2]

        X_out = torch.zeros(Bz, T_pad, C, B, dtype=Xs[0].dtype)
        pad_mask = torch.zeros(Bz, T_pad, dtype=torch.bool)  # True = valid
        for i, x in enumerate(Xs):
            T = x.shape[0]
            if T >= T_pad:
                X_out[i] = x[:T_pad]
                pad_mask[i] = True
            else:
                X_out[i, :T] = x
                pad_mask[i, :T] = True

        Q_out = torch.stack(Qs, dim=0)     # (Bz, C, B)
        y_out = torch.tensor(ys, dtype=torch.long)
        return X_out, Q_out, y_out, pad_mask, list(paths)


# ============================================================
# Graph-bias attention + spatial block
# ============================================================
class GraphBiasMultiheadAttention(nn.MultiheadAttention):
    """
    Same spirit as your original:
    - accept bias_matrix (L,L), set as attn_mask (additive float mask).
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, batch_first: bool = True):
        super().__init__(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, bias_matrix=None):
        if bias_matrix is not None:
            if attn_mask is not None:
                raise RuntimeError("[GraphAttn] do not pass both attn_mask and bias_matrix")
            # bias_matrix: (L,L)
            Lq = query.shape[1]
            if bias_matrix.shape != (Lq, Lq):
                raise RuntimeError(f"[GraphAttn] bias shape {tuple(bias_matrix.shape)} != ({Lq},{Lq})")
            attn_mask = bias_matrix.to(query.device)
        return super().forward(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask
        )

class ChannelGraphBlock(nn.Module):
    """
    Channel-wise attention at each time step:
    X: (Bz,T,C,BandDim)
    bias_matrix: (C,C) where bias_matrix = beta*log(G+eps)
    """
    def __init__(self, C: int, B: int, d_model_chan: int, nhead: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(B, d_model_chan)
        self.attn = GraphBiasMultiheadAttention(d_model_chan, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model_chan)
        self.out_proj = nn.Linear(d_model_chan, B)
        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, bias_matrix: torch.Tensor):
        Bz, T, C, B = X.shape
        Xc = X.view(Bz * T, C, B)
        h = self.in_proj(Xc)
        out, w = self.attn(h, h, h, bias_matrix=bias_matrix)  # w: (Bz*T, C, C) averaged over heads by torch default
        h = self.norm(h + self.drop(out))
        Xg = self.out_proj(h).view(Bz, T, C, B)
        return Xg, w


# ============================================================
# Temporal encoder + backbone (no recon)
# ============================================================
class NodeProjection(nn.Module):
    def __init__(self, C: int, B: int, d_model: int):
        super().__init__()
        self.C, self.B = C, B
        self.proj = nn.Linear(C * B, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bz, T, C, B = x.shape
        return self.proj(x.view(Bz, T, C * B))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        out, w = self.attn(x, x, x, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)
        x = self.norm1(x + self.drop(out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x, w

class GraphBackbone(nn.Module):
    """
    Shared backbone: spatial (channel graph) + temporal Transformer + classifier.
    Self-supervised reconstruction removed: rec is always None, kept only to match old return signature.
    Optionally outputs fcca when enabled (SEED-IV kept this).
    """
    def __init__(self, C: int, B: int, d_model: int, nhead: int, nlayers: int, dropout: float, T_pad: int,
                 num_classes: int, enable_fcca: bool = False):
        super().__init__()
        self.C, self.B = C, B
        self.spatial = ChannelGraphBlock(C, B, d_model_chan=d_model, nhead=nhead, dropout=dropout)
        self.embed = NodeProjection(C, B, d_model)
        self.pos = PositionalEncoding(d_model, dropout, T_pad)
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, dropout) for _ in range(nlayers)])
        self.cls = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, num_classes),
        )
        self.enable_fcca = bool(enable_fcca)
        self.fcca_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        ) if self.enable_fcca else None

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, chan_bias: Optional[torch.Tensor] = None):
        attns: List[torch.Tensor] = []

        # 1) channel graph attention
        if chan_bias is not None:
            x, w_chan = self.spatial(x, chan_bias)
            attns.append(w_chan)

        # 2) temporal transformer
        h = self.pos(self.embed(x))  # (Bz,T,d_model)
        for layer in self.layers:
            h, w = layer(h, key_padding_mask=pad_mask)
            attns.append(w)

        m = pad_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / (m.sum(1) + 1e-6)
        logits = self.cls(pooled)

        rec = None  # removed
        fcca = self.fcca_head(pooled) if self.fcca_head is not None else None
        return logits, rec, attns, h, fcca


# ============================================================
# SEED module (strict)
# ============================================================
def map_seed_label(y_raw: int) -> int:
    """SEED labels in {1,0,-1} -> {0,1,2} (order: -1,0,1)"""
    if y_raw == -1: return 0
    if y_raw == 0:  return 1
    if y_raw == 1:  return 2
    raise ValueError(f"[SEED] unexpected label: {y_raw}")

@dataclass
class SeedConfig:
    # data / io
    data_root: str = "/home/peipei/lzh/Done-Data/SEED/JSD_ZDE"
    out_dir: str = "/home/peipei/lzh/Train-Log/ULTRA/SEED-FULL-"
    graph_bias_root: Optional[str] = None  # folder containing A_spatial.npy; if None uses data_root
    require_chan_graph: bool = True

    # feature
    feature_keys: Tuple[str, ...] = ("jsd_gated",)
    B_per_feature: int = 5
    feature_combine: str = "concat_last"  # concat on last dim

    # quality
    quality_key: str = "quality"
    require_quality: bool = True
    quality_reduce: str = "mean"  # mean / max

    # model
    C: int = 62
    d_model: int = 240
    nhead: int = 12
    nlayers: int = 6
    dropout: float = 0.1
    num_classes: int = 3
    T_pad: int = 256
    attn_bias_beta: float = 1.0

    # train
    device: str = "cuda"
    seed: int = 42
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    label_smoothing: float = 0.0
    MinLrPer: float = 0.05  # keep your style (cosine min lr ratio)
    save_model: bool = True
    save_graph: bool = True
    eval_every: int = 10

    # --- deprecated self-supervised knobs: must be off ---
    ssl_weight: float = 0.0
    uwm_k: int = 0
    is_gamma: float = 0.0
    is_floor: float = 0.0

    def __post_init__(self):
        if self.ssl_weight != 0 or self.uwm_k != 0 or self.is_gamma != 0 or self.is_floor != 0:
            raise RuntimeError(
                "[SEED] Self-supervised parts (UWM/ING/recon) were removed in this unified script.\n"
                "Please set ssl_weight=0, uwm_k=0, is_gamma=0, is_floor=0."
            )
        if self.B_per_feature <= 0:
            raise ValueError("[SEED] B_per_feature must be >0 (strict).")
        self.B_total = int(self.B_per_feature * len(self.feature_keys))

class EEGWindowsSEED(Dataset):
    def __init__(self, files: List[Path], cfg: SeedConfig):
        self.files = list(files)
        self.cfg = cfg
        if len(self.files) == 0:
            raise RuntimeError("[SEED] empty file list")

        # strict: validate a few samples to catch key/shape issues early
        for p in self.files[: min(8, len(self.files))]:
            self._validate_npz(p)

    def _validate_npz(self, path: Path):
        d = np.load(path, allow_pickle=False)
        for k in self.cfg.feature_keys:
            if k not in d:
                raise KeyError(f"[SEED] missing feature '{k}' in {path}")
        if self.cfg.quality_key not in d and self.cfg.require_quality:
            raise KeyError(f"[SEED] missing '{self.cfg.quality_key}' in {path}")
        if "label" not in d:
            raise KeyError(f"[SEED] missing 'label' in {path}")

        # check feature shapes
        T0 = None
        for k in self.cfg.feature_keys:
            x = d[k]
            if x.ndim != 3:
                raise RuntimeError(f"[SEED] {k} must be (T,C,B), got {x.shape} in {path}")
            T, C, B = x.shape
            if C != self.cfg.C:
                raise RuntimeError(f"[SEED] C mismatch: cfg.C={self.cfg.C}, file has {C} in {path}")
            if B != self.cfg.B_per_feature:
                raise RuntimeError(f"[SEED] B mismatch for {k}: cfg={self.cfg.B_per_feature}, file has {B} in {path}")
            if T0 is None: T0 = T
            if T != T0:
                raise RuntimeError(f"[SEED] T mismatch across keys in {path}: {T0} vs {T}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        d = np.load(path, allow_pickle=False)
        feats = [d[k].astype(np.float32) for k in self.cfg.feature_keys]  # list of (T,C,Bk)

        if self.cfg.feature_combine == "concat_last":
            X = np.concatenate(feats, axis=2)  # (T,C,B_total)
        else:
            raise ValueError(f"[SEED] unknown feature_combine={self.cfg.feature_combine}")

        # quality -> (C,B_total) (not used in noSSL, but keep tuple structure)
        if self.cfg.quality_key in d:
            q = d[self.cfg.quality_key].astype(np.float32)
            # allow (T,C,B) or (C,B)
            if q.ndim == 3:
                if self.cfg.quality_reduce == "mean":
                    q = q.mean(axis=0)
                elif self.cfg.quality_reduce == "max":
                    q = q.max(axis=0)
                else:
                    raise ValueError(f"[SEED] unknown quality_reduce={self.cfg.quality_reduce}")
            if q.ndim != 2:
                raise RuntimeError(f"[SEED] quality must be (C,B) after reduce, got {q.shape} in {path}")
            # if quality only has B_per_feature, tile when multi-feature
            if q.shape[1] == self.cfg.B_per_feature and self.cfg.B_total != self.cfg.B_per_feature:
                q = np.tile(q, (1, len(self.cfg.feature_keys)))
        else:
            if self.cfg.require_quality:
                raise KeyError(f"[SEED] missing {self.cfg.quality_key} in {path}")
            q = np.ones((self.cfg.C, self.cfg.B_total), dtype=np.float32)

        y_raw = int(d["label"])
        y = map_seed_label(y_raw)

        X_t = torch.from_numpy(X)  # (T,C,B_total)
        Q_t = torch.from_numpy(q)  # (C,B_total)
        return X_t, Q_t, y, str(path)

def _collect_npz_files(root: Path) -> List[Path]:
    # exclude obvious non-trial artifacts if they exist
    skip_names = {"A_spatial.npy", "channel_graph_init.npy"}
    out = []
    for p in root.rglob("*.npz"):
        if p.name in skip_names: 
            continue
        out.append(p)
    return sorted(out)

def _discover_fold_dirs(data_root: Path) -> List[Path]:
    folds = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    if len(folds) == 0:
        raise RuntimeError(f"[SEED] no fold_* directories under {data_root}")
    return folds

def load_channel_graph_seed(cfg: SeedConfig) -> Optional[np.ndarray]:
    C = cfg.C
    root = Path(cfg.graph_bias_root) if cfg.graph_bias_root is not None else Path(cfg.data_root)
    for name in ["A_spatial.npy", "channel_graph_init.npy"]:
        p = root / name
        if p.exists():
            A = np.load(p)
            if A.shape != (C, C):
                raise RuntimeError(f"[SEED] channel graph {p} shape {A.shape} != ({C},{C})")
            return A
    if cfg.require_chan_graph:
        raise FileNotFoundError(f"[SEED] channel graph not found in {root} (A_spatial.npy / channel_graph_init.npy)")
    return None

@torch.no_grad()
def _eval_model(model: nn.Module, loader: DataLoader, device: torch.device, chan_bias: Optional[torch.Tensor]):
    model.eval()
    ys, ps = [], []
    for X, Q, y, pad_mask, paths in loader:
        X, y, pad_mask = _to_device(X, device), _to_device(y, device), _to_device(pad_mask, device)
        logits, _, _, _, _ = model(X, pad_mask=pad_mask, chan_bias=chan_bias)
        pred = logits.argmax(dim=1)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    acc = float(accuracy_score(y_all, p_all))
    cm = confusion_matrix(y_all, p_all, labels=list(range(model.cls[-1].out_features)))
    return acc, cm

def train_and_eval_one_fold_seed(cfg: SeedConfig, fold_dir: Path, out_fold_dir: Path) -> Dict[str, Any]:
    out_fold_dir.mkdir(parents=True, exist_ok=True)
    log = TeeLogger(out_fold_dir / "train.log")
    sys.stdout = log

    try:
        # build filelists from prebaked structure
        tr_dir = fold_dir / "train_source"
        tt_dir = fold_dir / "test_target"
        if not tr_dir.exists() or not tt_dir.exists():
            raise FileNotFoundError(f"[SEED] fold missing train_source/test_target: {fold_dir}")
        train_files = _collect_npz_files(tr_dir)
        test_files  = _collect_npz_files(tt_dir)
        if len(train_files) == 0 or len(test_files) == 0:
            raise RuntimeError(f"[SEED] empty split in {fold_dir}")

        ds_train = EEGWindowsSEED(train_files, cfg)
        ds_test  = EEGWindowsSEED(test_files, cfg)
        collate = PadCollate(cfg.T_pad)
        dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
        dl_test  = DataLoader(ds_test,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

        # channel graph bias
        A = load_channel_graph_seed(cfg)
        chan_bias = None
        if A is not None:
            eps = 1e-6
            G = np.clip(A.astype(np.float32), eps, None)
            chan_bias = torch.from_numpy(cfg.attn_bias_beta * np.log(G)).float().to(cfg.device)

        # model
        device = torch.device(cfg.device)
        model = GraphBackbone(cfg.C, cfg.B_total, cfg.d_model, cfg.nhead, cfg.nlayers, cfg.dropout, cfg.T_pad, cfg.num_classes, enable_fcca=False).to(device)

        opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * float(cfg.MinLrPer))

        crit = nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))

        
        for ep in range(1, cfg.epochs + 1):
            model.train()
            loss_sum, n = 0.0, 0
            for X, Q, y, pad_mask, paths in dl_train:
                X, y, pad_mask = _to_device(X, device), _to_device(y, device), _to_device(pad_mask, device)
                opt.zero_grad(set_to_none=True)
                logits, _, _, _, _ = model(X, pad_mask=pad_mask, chan_bias=chan_bias)
                loss = crit(logits, y)
                loss.backward()
                if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
                opt.step()
                loss_sum += float(loss.item()) * y.size(0)
                n += y.size(0)

            scheduler.step()
            tr_loss = loss_sum / max(n, 1)

            if ep % cfg.eval_every == 0 or ep == cfg.epochs:
                acc_t, cm_t = _eval_model(model, dl_test, device, chan_bias)
                print(f"[SEED][ep {ep:03d}/{cfg.epochs}] loss={tr_loss:.4f}  acc_tgt={acc_t:.4f}")
                



        acc_src, cm_src = _eval_model(model, dl_train, device, chan_bias)
        acc_tgt, cm_tgt = _eval_model(model, dl_test, device, chan_bias)

        if cfg.save_model:
            torch.save(model.state_dict(), out_fold_dir / "model_final.pth")
        if cfg.save_graph and A is not None:
            np.save(out_fold_dir / "used_channel_graph.npy", A)

        metrics = {
            "acc_source": float(acc_src),
            "acc_target": float(acc_tgt),
            "cm_source": cm_src.tolist(),
            "cm_target": cm_tgt.tolist(),
            
        }
        json_dump(metrics, out_fold_dir / "metrics.json")
        return metrics
    finally:
        sys.stdout = log._stdout
        log.close()

def evaluate_only_seed(cfg: SeedConfig, fold_dir: Path, out_fold_dir: Path) -> Dict[str, Any]:
    out_fold_dir.mkdir(parents=True, exist_ok=True)
    log = TeeLogger(out_fold_dir / "eval.log")
    sys.stdout = log
    try:
        tr_dir = fold_dir / "train_source"
        tt_dir = fold_dir / "test_target"
        train_files = _collect_npz_files(tr_dir)
        test_files  = _collect_npz_files(tt_dir)
        ds_train = EEGWindowsSEED(train_files, cfg)
        ds_test  = EEGWindowsSEED(test_files, cfg)
        collate = PadCollate(cfg.T_pad)
        dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
        dl_test  = DataLoader(ds_test,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

        A = load_channel_graph_seed(cfg)
        chan_bias = None
        if A is not None:
            eps = 1e-6
            G = np.clip(A.astype(np.float32), eps, None)
            chan_bias = torch.from_numpy(cfg.attn_bias_beta * np.log(G)).float().to(cfg.device)

        device = torch.device(cfg.device)
        model = GraphBackbone(cfg.C, cfg.B_total, cfg.d_model, cfg.nhead, cfg.nlayers, cfg.dropout, cfg.T_pad, cfg.num_classes, enable_fcca=False).to(device)

        pth = out_fold_dir / "model_final.pth"
        if not pth.exists():
            raise FileNotFoundError(f"[SEED] missing {pth} for eval-only")
        model.load_state_dict(torch.load(pth, map_location=device))

        acc_src, cm_src = _eval_model(model, dl_train, device, chan_bias)
        acc_tgt, cm_tgt = _eval_model(model, dl_test, device, chan_bias)
        metrics = {
            "acc_source": float(acc_src),
            "acc_target": float(acc_tgt),
            "cm_source": cm_src.tolist(),
            "cm_target": cm_tgt.tolist(),
            
        }
        json_dump(metrics, out_fold_dir / "metrics.json")
        print(f"[SEED][eval-only] acc_tgt={acc_tgt:.4f}")
        return metrics
    finally:
        sys.stdout = log._stdout
        log.close()

def leave_one_out_cv_seed(cfg: SeedConfig) -> Dict[str, Any]:
    seed_everything(cfg.seed)
    set_torch_deterministic(True)

    data_root = Path(cfg.data_root)
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    fold_dirs = _discover_fold_dirs(data_root)
    all_fold_metrics = []
    for fdir in fold_dirs:
        # fold_subj_01 -> sid=1
        name = fdir.name
        sid = None
        for token in name.split("_"):
            if token.isdigit():
                sid = int(token)
        if sid is None:
            # fallback: last 2 digits
            sid = int(name[-2:])
        out_fold = out_root / f"fold_sid{sid:02d}"

        print(f"------------Training on Sid: {sid}----------")

        if (out_fold / "model_final.pth").exists():
            metrics = evaluate_only_seed(cfg, fdir, out_fold)
        else:
            metrics = train_and_eval_one_fold_seed(cfg, fdir, out_fold)

        metrics["fold"] = fdir.name
        all_fold_metrics.append(metrics)

    # summarize
    accs = [m["acc_target"] for m in all_fold_metrics]
    overall = {
        "dataset": "SEED",
        "num_folds": len(all_fold_metrics),
        "acc_target_mean": float(np.mean(accs)) if accs else None,
        "acc_target_std": float(np.std(accs)) if accs else None,
        "per_fold": all_fold_metrics,
        "config": cfg.__dict__,
    }
    json_dump(overall, out_root / "overall_metrics.json")
    return overall


# ============================================================
# SEED-IV module (kept separate; supports auto-infer B)
# ============================================================
FEATURE_KEYS: List[str] = ["jsd_gated"]
FEATURE_FUSE: str = "concat_last"   # concat along last dim

@dataclass
class SeedIVConfig:
    data_root: str = "/home/peipei/lzh/Done-Data/SEED-IV/SEED-IV-JSD_ZDE/_fold_jsd_degate"
    out_dir: str = "/home/peipei/lzh/Train-Log/ULTRA/SEED-IV-FULL-"
    jsd_root: str = "/home/peipei/lzh/Done-Data/SEED-IV/SEED-IV-JSD_ZDE/_fold_jsd_degate"  # where A_spatial.npy lives

    prebaked_folds: bool = True
    require_chan_graph: bool = True

    # feature
    feature_key: str = "" # this is useless, Don`t change!
    feature_keys: List[str] = field(default_factory=lambda: list(FEATURE_KEYS))
    feature_fuse: str = FEATURE_FUSE   # concat_last
    auto_infer_B: bool = True
    B: int = 5  # overwritten when auto_infer_B=True

    # quality
    quality_key: str = "quality"
    require_quality: bool = True
    quality_reduce: str = "mean"

    # model
    C: int = 62
    d_model: int = 600
    nhead: int = 20
    nlayers: int = 5
    dropout: float = 0.2
    num_classes: int = 4
    T_pad: int = 256
    attn_bias_beta: float = 0.5
    save_fcca: bool = True

    # traina 
    device: str = "cuda"
    seed: int = 42
    epochs: int = 100
    batch_size: int = 24
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    label_smoothing: float = 0.2
    MinLrPer: float = 0.1
    save_model: bool = True
    save_graph: bool = True
    eval_every: int = 10

    # --- deprecated self-supervised knobs: must be off ---
    ssl_weight: float = 0.0
    uwm_k: int = 0
    is_gamma: float = 0.0
    is_floor: float = 0.0
    ema_graph_momentum: float = 0.0

    def __post_init__(self):
        if self.ssl_weight != 0 or self.uwm_k != 0 or self.is_gamma != 0 or self.is_floor != 0 or self.ema_graph_momentum != 0:
            raise RuntimeError(
                "[SEED-IV] Self-supervised parts (UWM/ING/recon/EMA-graph) were removed in this unified script.\n"
                "Please set ssl_weight=0, uwm_k=0, is_gamma=0, is_floor=0, ema_graph_momentum=0."
            )

def _effective_feature_keys_iv(cfg: SeedIVConfig) -> List[str]:
    # keep same behavior: feature_key + feature_keys (dedup, preserve order)
    keys = []
    if cfg.feature_key:
        keys.append(cfg.feature_key)
    for k in cfg.feature_keys:
        if k not in keys:
            keys.append(k)
    return keys

def _load_and_fuse_features_iv(npz: Dict[str, Any], keys: List[str], fuse: str) -> np.ndarray:
    feats = [npz[k].astype(np.float32) for k in keys]
    if fuse == "concat_last":
        X = np.concatenate(feats, axis=2)
    else:
        raise ValueError(f"[SEED-IV] unknown feature_fuse={fuse}")
    return X

def _infer_B_from_one_file_iv(path: Path, keys: List[str], fuse: str) -> int:
    d = np.load(path, allow_pickle=False)
    # assume each key has (T,C,Bk) and we concat on last dim
    Bs = []
    for k in keys:
        x = d[k]
        if x.ndim != 3:
            raise RuntimeError(f"[SEED-IV] {k} must be (T,C,B), got {x.shape} in {path}")
        Bs.append(x.shape[2])
    if fuse == "concat_last":
        return int(sum(Bs))
    raise ValueError(f"[SEED-IV] unknown fuse={fuse}")

class EEGWindowsSEEDIV(Dataset):
    def __init__(self, files: List[Path], cfg: SeedIVConfig):
        self.files = list(files)
        self.cfg = cfg
        if len(self.files) == 0:
            raise RuntimeError("[SEED-IV] empty file list")
        self.keys = _effective_feature_keys_iv(cfg)

        # validate a few samples
        for p in self.files[: min(8, len(self.files))]:
            self._validate_npz(p)

    def _validate_npz(self, path: Path):
        d = np.load(path, allow_pickle=False)
        for k in self.keys:
            if k not in d:
                raise KeyError(f"[SEED-IV] missing feature '{k}' in {path}")
            x = d[k]
            if x.ndim != 3:
                raise RuntimeError(f"[SEED-IV] {k} must be (T,C,B), got {x.shape} in {path}")
            if x.shape[1] != self.cfg.C:
                raise RuntimeError(f"[SEED-IV] C mismatch: cfg.C={self.cfg.C}, file has {x.shape[1]} in {path}")
        if "label" not in d:
            raise KeyError(f"[SEED-IV] missing 'label' in {path}")
        if self.cfg.require_quality and self.cfg.quality_key not in d:
            raise KeyError(f"[SEED-IV] missing '{self.cfg.quality_key}' in {path}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        d = np.load(path, allow_pickle=False)
        X = _load_and_fuse_features_iv(d, self.keys, self.cfg.feature_fuse)  # (T,C,B_total)

        # quality -> (C,B_total)
        if self.cfg.quality_key in d:
            q = d[self.cfg.quality_key].astype(np.float32)
            if q.ndim == 3:
                if self.cfg.quality_reduce == "mean":
                    q = q.mean(axis=0)
                elif self.cfg.quality_reduce == "max":
                    q = q.max(axis=0)
                else:
                    raise ValueError(f"[SEED-IV] unknown quality_reduce={self.cfg.quality_reduce}")
            if q.ndim != 2:
                raise RuntimeError(f"[SEED-IV] quality must be (C,B) after reduce, got {q.shape} in {path}")
            # if q only has per-key B, tile/expand is ambiguous; keep strict
        else:
            if self.cfg.require_quality:
                raise KeyError(f"[SEED-IV] missing {self.cfg.quality_key} in {path}")
            q = np.ones((self.cfg.C, X.shape[2]), dtype=np.float32)

        y = int(d["label"])  # assume already 0..num_classes-1
        X_t = torch.from_numpy(X)
        Q_t = torch.from_numpy(q)
        return X_t, Q_t, y, str(path)

def load_channel_graph_seediv(cfg: SeedIVConfig) -> Optional[np.ndarray]:
    C = cfg.C
    root = Path(cfg.jsd_root)
    for name in ["A_spatial.npy", "channel_graph_init.npy"]:
        p = root / name
        if p.exists():
            A = np.load(p)
            if A.shape != (C, C):
                raise RuntimeError(f"[SEED-IV] channel graph {p} shape {A.shape} != ({C},{C})")
            return A
    if cfg.require_chan_graph:
        raise FileNotFoundError(f"[SEED-IV] channel graph not found in {root} (A_spatial.npy / channel_graph_init.npy)")
    return None

def _collect_npz_files_iv(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.npz")])

def train_and_eval_one_fold_seediv(cfg: SeedIVConfig, fold_dir: Path, out_fold_dir: Path) -> Dict[str, Any]:
    out_fold_dir.mkdir(parents=True, exist_ok=True)
    log = TeeLogger(out_fold_dir / "train.log")
    sys.stdout = log

    try:
        tr_dir = fold_dir / "train_source"
        tt_dir = fold_dir / "test_target"
        if not tr_dir.exists() or not tt_dir.exists():
            raise FileNotFoundError(f"[SEED-IV] fold missing train_source/test_target: {fold_dir}")

        train_files = _collect_npz_files_iv(tr_dir)
        test_files  = _collect_npz_files_iv(tt_dir)
        if len(train_files) == 0 or len(test_files) == 0:
            raise RuntimeError(f"[SEED-IV] empty split in {fold_dir}")

        keys = _effective_feature_keys_iv(cfg)
        if cfg.auto_infer_B:
            cfg.B = _infer_B_from_one_file_iv(train_files[0], keys, cfg.feature_fuse)
            print(f"[SEED-IV] auto_infer_B => cfg.B = {cfg.B}")

        ds_train = EEGWindowsSEEDIV(train_files, cfg)
        ds_test  = EEGWindowsSEEDIV(test_files, cfg)
        collate = PadCollate(cfg.T_pad)
        dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
        dl_test  = DataLoader(ds_test,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

        A = load_channel_graph_seediv(cfg)
        chan_bias = None
        if A is not None:
            eps = 1e-6
            G = np.clip(A.astype(np.float32), eps, None)
            chan_bias = torch.from_numpy(cfg.attn_bias_beta * np.log(G)).float().to(cfg.device)

        device = torch.device(cfg.device)
        model = GraphBackbone(cfg.C, cfg.B, cfg.d_model, cfg.nhead, cfg.nlayers, cfg.dropout, cfg.T_pad, cfg.num_classes,
                              enable_fcca=bool(cfg.save_fcca)).to(device)

        opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * float(cfg.MinLrPer))
        crit = nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))

        
        for ep in range(1, cfg.epochs + 1):
            model.train()
            loss_sum, n = 0.0, 0
            for X, Q, y, pad_mask, paths in dl_train:
                X, y, pad_mask = _to_device(X, device), _to_device(y, device), _to_device(pad_mask, device)
                opt.zero_grad(set_to_none=True)
                logits, _, _, _, _ = model(X, pad_mask=pad_mask, chan_bias=chan_bias)
                loss = crit(logits, y)
                loss.backward()
                if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
                opt.step()
                loss_sum += float(loss.item()) * y.size(0)
                n += y.size(0)

            scheduler.step()
            tr_loss = loss_sum / max(n, 1)

            if ep % cfg.eval_every == 0 or ep == cfg.epochs:
                acc_t, cm_t = _eval_model(model, dl_test, device, chan_bias)
                print(f"[SEED-IV][ep {ep:03d}/{cfg.epochs}] loss={tr_loss:.4f}  acc_tgt={acc_t:.4f}")
                


        acc_src, cm_src = _eval_model(model, dl_train, device, chan_bias)
        acc_tgt, cm_tgt = _eval_model(model, dl_test, device, chan_bias)

        if cfg.save_model:
            torch.save(model.state_dict(), out_fold_dir / "model_final.pth")
        if cfg.save_graph and A is not None:
            np.save(out_fold_dir / "used_channel_graph.npy", A)

        metrics = {
            "acc_source": float(acc_src),
            "acc_target": float(acc_tgt),
            "cm_source": cm_src.tolist(),
            "cm_target": cm_tgt.tolist(),
            
        }
        json_dump(metrics, out_fold_dir / "metrics.json")
        return metrics
    finally:
        sys.stdout = log._stdout
        log.close()

def evaluate_only_seediv(cfg: SeedIVConfig, fold_dir: Path, out_fold_dir: Path) -> Dict[str, Any]:
    out_fold_dir.mkdir(parents=True, exist_ok=True)
    log = TeeLogger(out_fold_dir / "eval.log")
    sys.stdout = log
    try:
        tr_dir = fold_dir / "train_source"
        tt_dir = fold_dir / "test_target"
        train_files = _collect_npz_files_iv(tr_dir)
        test_files  = _collect_npz_files_iv(tt_dir)

        keys = _effective_feature_keys_iv(cfg)
        if cfg.auto_infer_B and len(train_files) > 0:
            cfg.B = _infer_B_from_one_file_iv(train_files[0], keys, cfg.feature_fuse)
            print(f"[SEED-IV] auto_infer_B => cfg.B = {cfg.B}")

        ds_train = EEGWindowsSEEDIV(train_files, cfg)
        ds_test  = EEGWindowsSEEDIV(test_files, cfg)
        collate = PadCollate(cfg.T_pad)
        dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
        dl_test  = DataLoader(ds_test,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

        A = load_channel_graph_seediv(cfg)
        chan_bias = None
        if A is not None:
            eps = 1e-6
            G = np.clip(A.astype(np.float32), eps, None)
            chan_bias = torch.from_numpy(cfg.attn_bias_beta * np.log(G)).float().to(cfg.device)

        device = torch.device(cfg.device)
        model = GraphBackbone(cfg.C, cfg.B, cfg.d_model, cfg.nhead, cfg.nlayers, cfg.dropout, cfg.T_pad, cfg.num_classes,
                              enable_fcca=bool(cfg.save_fcca)).to(device)

        pth = out_fold_dir / "model_final.pth"
        if not pth.exists():
            raise FileNotFoundError(f"[SEED-IV] missing {pth} for eval-only")
        model.load_state_dict(torch.load(pth, map_location=device))

        acc_src, cm_src = _eval_model(model, dl_train, device, chan_bias)
        acc_tgt, cm_tgt = _eval_model(model, dl_test, device, chan_bias)
        metrics = {
            "acc_source": float(acc_src),
            "acc_target": float(acc_tgt),
            "cm_source": cm_src.tolist(),
            "cm_target": cm_tgt.tolist(),
            
        }
        json_dump(metrics, out_fold_dir / "metrics.json")
        print(f"[SEED-IV][eval-only] acc_tgt={acc_tgt:.4f}")
        return metrics
    finally:
        sys.stdout = log._stdout
        log.close()

def leave_one_out_cv_seediv(cfg: SeedIVConfig) -> Dict[str, Any]:
    seed_everything(cfg.seed)
    set_torch_deterministic(True)

    data_root = Path(cfg.data_root)
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    fold_dirs = _discover_fold_dirs(data_root)  # same naming: fold_*
    all_fold_metrics = []
    for fdir in fold_dirs:
        name = fdir.name
        sid = None
        for token in name.split("_"):
            if token.isdigit():
                sid = int(token)
        if sid is None:
            sid = int(name[-2:])
        out_fold = out_root / f"fold_sid{sid:02d}"

        print(f"------------Training on Sid: {sid}----------")

        if (out_fold / "model_final.pth").exists():
            metrics = evaluate_only_seediv(cfg, fdir, out_fold)
        else:
            metrics = train_and_eval_one_fold_seediv(cfg, fdir, out_fold)

        metrics["fold"] = fdir.name
        all_fold_metrics.append(metrics)

    accs = [m["acc_target"] for m in all_fold_metrics]
    overall = {
        "dataset": "SEED-IV",
        "num_folds": len(all_fold_metrics),
        "acc_target_mean": float(np.mean(accs)) if accs else None,
        "acc_target_std": float(np.std(accs)) if accs else None,
        "per_fold": all_fold_metrics,
        "config": cfg.__dict__,
    }
    json_dump(overall, out_root / "overall_metrics.json")
    return overall


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    # ====== Choose ONE config to run (comment the other) ======
    # --- SEED ---
    # cfg = SeedConfig()
    # leave_one_out_cv_seed(cfg)

    # --- SEED-IV ---
    cfg = SeedIVConfig()
    leave_one_out_cv_seediv(cfg)
