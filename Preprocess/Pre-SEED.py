# -*- coding: utf-8 -*-
"""
SEED preprocessing utility: fuse JSD and DE into zDE-gated features fold by fold.

This script assumes JSD files and DE files already share the same LOOCV fold
layout. For each fold it estimates mu/std only from ``train_source`` and then
exports:
- ``de``
- ``zde``
- ``gate = tanh(alpha * zde)``
- ``jsd_gated = gate * jsd``
- optional ``feat = concat([jsd, jsd_gated, zde], axis=-1)``

The script is resume-friendly: existing output files are skipped unless
``--overwrite`` is provided.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple, Iterable, List

import numpy as np


# Default paths are intentionally repo-relative so the script can be published
# without carrying the author's local filesystem layout.
JSD_ROOT = "data/SEED/_fold_jsd"
DE_ROOT = "data/SEED/_fold_de"
SAVE_ROOT = "data/SEED/_fold_jsd_degate"

# 如果你的折在子目录里（常见：_fold_jsd），写在这里；不存在则脚本会自动 fallback 到根目录
JSD_FOLD_SUBDIR_CANDIDATES = ["_fold_jsd", ""]
DE_FOLD_SUBDIR_CANDIDATES  = ["_fold_de", "_fold_jsd", ""]

JSD_KEY = "jsd"
DE_KEY  = "de"

ALPHA = 2.0          # tanh(alpha * zde)
EPS_STD = 1e-6       # std 最小值，防止除零
OVERWRITE = False    # 默认不覆盖已存在输出
FEAT_MODE = "cat"    # "cat" -> (T,C,3B) ; "none" -> 不存 feat（只存三路）


# -------------------------
# 工具函数
# -------------------------
def _resolve_fold_root(root: Path, candidates: List[str]) -> Path:
    """从候选子目录中选择存在的 fold_root；否则退回 root。"""
    for sub in candidates:
        p = root / sub if sub else root
        if p.exists() and p.is_dir():
            # 这里不强制里面一定有 fold_*，因为不同数据集命名不同
            return p
    return root


def _list_fold_dirs(fold_root: Path) -> List[Path]:
    """列出 fold 目录（fold_* 或 fold_subj_* 等）。"""
    folds = []
    for p in fold_root.iterdir():
        if p.is_dir() and p.name.lower().startswith("fold"):
            # 需要包含 train_source/test_target 才算有效 fold
            if (p / "train_source").exists() and (p / "test_target").exists():
                folds.append(p)
    folds.sort(key=lambda x: x.name)
    return folds


def _iter_npz_files(dir_path: Path) -> Iterable[Path]:
    """递归遍历 .npz 文件（保持稳定顺序）。"""
    files = sorted(dir_path.rglob("*.npz"))
    return files


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _save_npz(path: Path, data: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def _compute_mu_std_from_train_de(train_de_files: List[Path], de_fold_root: Path, fold_root_jsd: Path,
                                 de_key: str, eps_std: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    用该 fold 的 train_source 计算 mu_src/std_src，统计维度为 (C,B)。
    统计方式：对所有 trial 的所有时间窗 T 累积求和与平方和（不存全量，省内存）。
    """
    sum_cb = None
    sumsq_cb = None
    total_T = 0

    for jsd_fp in train_de_files:
        rel = jsd_fp.relative_to(fold_root_jsd)  # fold_xx/train_source/...
        de_fp = de_fold_root / rel
        if not de_fp.exists():
            raise FileNotFoundError(f"[DE missing] {de_fp}")

        de_npz = _load_npz(de_fp)
        if de_key not in de_npz:
            raise KeyError(f"[DE key missing] {de_fp} has keys={list(de_npz.keys())}")
        de = de_npz[de_key].astype(np.float64)  # (T,C,B)

        if de.ndim != 3:
            raise ValueError(f"[DE shape] expect 3D (T,C,B), got {de.shape} @ {de_fp}")

        T, C, B = de.shape
        if sum_cb is None:
            sum_cb = np.zeros((C, B), dtype=np.float64)
            sumsq_cb = np.zeros((C, B), dtype=np.float64)

        if sum_cb.shape != (C, B):
            raise ValueError(f"[DE shape mismatch] got {de.shape}, expected C,B={sum_cb.shape}")

        sum_cb += de.sum(axis=0)
        sumsq_cb += (de * de).sum(axis=0)
        total_T += T

    if total_T <= 0:
        raise RuntimeError("No training windows found (total_T=0). Check train_source files.")

    mu = sum_cb / float(total_T)
    var = sumsq_cb / float(total_T) - mu * mu
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    std = np.maximum(std, eps_std)

    return mu.astype(np.float32), std.astype(np.float32), total_T


def _make_feat(jsd: np.ndarray, jsd_gated: np.ndarray, zde: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return None
    if mode == "cat":
        # (T,C,B) -> (T,C,3B)
        return np.concatenate([jsd, jsd_gated, zde], axis=-1)
    raise ValueError(f"Unknown FEAT_MODE={mode}")


def process_one_fold(fold_jsd_dir: Path, fold_root_jsd: Path, fold_root_de: Path, save_fold_root: Path,
                     jsd_key: str, de_key: str, alpha: float, eps_std: float,
                     overwrite: bool, feat_mode: str) -> None:
    train_dir = fold_jsd_dir / "train_source"
    test_dir  = fold_jsd_dir / "test_target"

    train_jsd_files = list(_iter_npz_files(train_dir))
    test_jsd_files  = list(_iter_npz_files(test_dir))

    if len(train_jsd_files) == 0 or len(test_jsd_files) == 0:
        raise RuntimeError(f"[Empty fold] {fold_jsd_dir} train={len(train_jsd_files)} test={len(test_jsd_files)}")

    # 1) stats from train_source DE
    mu_src, std_src, count_T = _compute_mu_std_from_train_de(
        train_jsd_files, fold_root_de, fold_root_jsd, de_key, eps_std
    )

    # 保存 stats
    stats_out = save_fold_root / fold_jsd_dir.relative_to(fold_root_jsd) / "stats_src.npz"
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stats_out, mu_src=mu_src, std_src=std_src, count_T=np.int64(count_T),
                        alpha=np.float32(alpha), eps_std=np.float32(eps_std))

    # 2) apply to all files (train + test)
    all_files = train_jsd_files + test_jsd_files
    for jsd_fp in all_files:
        rel = jsd_fp.relative_to(fold_root_jsd)
        de_fp = fold_root_de / rel
        if not de_fp.exists():
            raise FileNotFoundError(f"[DE missing] {de_fp}")

        out_fp = save_fold_root / rel
        if out_fp.exists() and (not overwrite):
            continue

        jsd_npz = _load_npz(jsd_fp)
        de_npz  = _load_npz(de_fp)

        if jsd_key not in jsd_npz:
            raise KeyError(f"[JSD key missing] {jsd_fp} has keys={list(jsd_npz.keys())}")
        if de_key not in de_npz:
            raise KeyError(f"[DE key missing] {de_fp} has keys={list(de_npz.keys())}")

        jsd = jsd_npz[jsd_key].astype(np.float32)  # (T,C,B)
        de  = de_npz[de_key].astype(np.float32)    # (T,C,B)

        if jsd.shape != de.shape:
            raise ValueError(f"[shape mismatch]\n  jsd={jsd.shape} @ {jsd_fp}\n  de ={de.shape} @ {de_fp}")
        if jsd.ndim != 3:
            raise ValueError(f"[JSD shape] expect 3D (T,C,B), got {jsd.shape} @ {jsd_fp}")

        # zde: (T,C,B) with per-(C,B) stats
        zde = (de - mu_src[None, :, :]) / std_src[None, :, :]
        gate = np.tanh(alpha * zde).astype(np.float32)
        jsd_gated = (gate * jsd).astype(np.float32)

        # 3) merge & save
        out = dict(jsd_npz)  # 复制 JSD 文件所有 keys（labels/quality/p_hist/... 都保留）
        out["de"] = de
        out["zde"] = zde.astype(np.float32)
        out["gate"] = gate
        out["jsd_gated"] = jsd_gated

        feat = _make_feat(jsd, jsd_gated, out["zde"], feat_mode)
        if feat is not None:
            out["feat"] = feat.astype(np.float32)

        # 数值检查（可选但强烈建议）
        for k in ["de", "zde", "gate", "jsd_gated"] + (["feat"] if "feat" in out else []):
            arr = out[k]
            if not np.isfinite(arr).all():
                raise FloatingPointError(f"[non-finite] key={k} @ {jsd_fp}")

        _save_npz(out_fp, out)


def main():
    parser = argparse.ArgumentParser(
        description="Fuse SEED JSD and DE folds into zDE-gated features."
    )
    parser.add_argument("--jsd_root", type=str, default=JSD_ROOT)
    parser.add_argument("--de_root",  type=str, default=DE_ROOT)
    parser.add_argument("--save_root", type=str, default=SAVE_ROOT)
    parser.add_argument("--jsd_key", type=str, default=JSD_KEY)
    parser.add_argument("--de_key",  type=str, default=DE_KEY)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--eps_std", type=float, default=EPS_STD)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    parser.add_argument("--feat_mode", type=str, default=FEAT_MODE, choices=["cat", "none"])

    args = parser.parse_args()

    jsd_root = Path(args.jsd_root).expanduser().resolve()
    de_root  = Path(args.de_root).expanduser().resolve()
    save_root = Path(args.save_root).expanduser().resolve()

    if not jsd_root.exists():
        raise FileNotFoundError(f"JSD root not found: {jsd_root}")
    if not de_root.exists():
        raise FileNotFoundError(f"DE root not found: {de_root}")

    fold_root_jsd = _resolve_fold_root(jsd_root, JSD_FOLD_SUBDIR_CANDIDATES)
    fold_root_de  = _resolve_fold_root(de_root,  DE_FOLD_SUBDIR_CANDIDATES)

    folds = _list_fold_dirs(fold_root_jsd)
    if len(folds) == 0:
        raise RuntimeError(f"No folds found under: {fold_root_jsd} (need fold*/train_source & test_target)")

    print(f"[fold_root_jsd] {fold_root_jsd}")
    print(f"[fold_root_de ] {fold_root_de}")
    print(f"[save_root    ] {save_root}")
    print(f"[folds] {len(folds)} -> {[f.name for f in folds]}")

    for fold_dir in folds:
        print(f"\n========== PROCESS {fold_dir.name} ==========")
        process_one_fold(
            fold_jsd_dir=fold_dir,
            fold_root_jsd=fold_root_jsd,
            fold_root_de=fold_root_de,
            save_fold_root=save_root,
            jsd_key=args.jsd_key,
            de_key=args.de_key,
            alpha=float(args.alpha),
            eps_std=float(args.eps_std),
            overwrite=bool(args.overwrite),
            feat_mode=str(args.feat_mode),
        )
        print(f"[OK] {fold_dir.name}")

    print("\nAll folds done ✅")


if __name__ == "__main__":
    main()
