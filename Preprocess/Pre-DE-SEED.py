# de_seed_all_in_one.py
# -*- coding: utf-8 -*-
"""
SEED DE preprocessing pipeline.

Steps:
1) ``Preprocessed_EEG/*.mat`` (excluding ``label.mat``) -> per-trial sliding-window
   ``de`` / ``quality`` files under ``_de/``
2) build LOOCV fold directories under ``_fold_de/``

The numerical logic is kept from the original script; the public-release
version only makes paths and steps configurable from the command line.
"""

import argparse
import os
import re
import glob
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import welch


# Public defaults (CLI-overridable)
BASE_PATH = "data/SEED-RAW"
RAW_DATA = os.path.join(BASE_PATH, "Preprocessed_EEG")
SAVE_ROOT = "data/SEED"
CHANNEL_XLSX = os.path.join(BASE_PATH, "channel-order.xlsx")
LABEL_PATH = os.path.join(RAW_DATA, "label.mat")

# Fold 的搬运方式：建议 copy（最稳，不依赖软链接权限）
FOLD_LINK_MODE = "copy"   # "copy" | "symlink" | "hardlink"


# =========================
# 与你 JSD 脚本保持一致的超参
# =========================
N_CHANNELS = 62
SFREQ = 200.0

WINDOW_SEC = 4.0
STRIDE_SEC = 1.0

WELCH_NPERSEG  = int(SFREQ * 2)   # 2s
WELCH_NOVERLAP = int(SFREQ * 1)   # 1s

EPS = 1e-12

# 频带（同 JSD）
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta":  (14, 31),
    "gamma": (31, 60)
}
BAND_NAMES = list(BANDS.keys())

# 目录名（对齐你的风格）
DE_DIR   = "_de"
FOLD_DIR = "_fold_de"


# =========================
# 工具函数
# =========================
def _safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def discover_subject_files(raw_data_path):
    """
    返回 [[subj1_mat_basename...],[subj2...],...] 按 subject_id 升序
    basenames 来自 Preprocessed_EEG 下的 *.mat（排除 label.mat）
    """
    subject_files = defaultdict(list)
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found: {raw_data_path}")

    all_files = sorted([f for f in os.listdir(raw_data_path)
                        if f.endswith('.mat') and f != 'label.mat'])
    for filename in all_files:
        basename = os.path.splitext(filename)[0]   # e.g., "1_20131027"
        sid = basename.split('_')[0]              # "1"
        subject_files[sid].append(basename)

    sorted_subject_ids = sorted(subject_files.keys(), key=lambda x: int(x))
    return [subject_files[sid] for sid in sorted_subject_ids]

def create_clips(data, clip_len, stride):
    """data: (C,T) -> list[(clip(C,L), s, e)]"""
    clips = []
    n = data.shape[1]
    for s in range(0, n - clip_len + 1, stride):
        e = s + clip_len
        clips.append((data[:, s:e], s, e))
    return clips

def compute_psd(clip, fs):
    """clip:(C,L) -> freqs(F,), psd(C,F)"""
    freqs, psd = welch(
        clip, fs=fs, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP, axis=-1
    )
    return freqs, psd

def quality_components(freqs, psd, clip, frontal_idx):
    """
    复用你 JSD 脚本里的“简易 quality”思想：
    - eog_corr：额叶通道与其他通道的相关性（粗略）
    - line_noise：50Hz 附近占比
    - muscle：45-60Hz 占比
    输出 quality: (C,B) in [0,1]
    """
    C = psd.shape[0]
    B = len(BANDS)

    # eog_corr：与额叶均值信号的绝对相关
    front = clip[frontal_idx, :].mean(axis=0)
    front = (front - front.mean()) / (front.std() + EPS)

    eog_corr = np.zeros((C, B), dtype=np.float32)
    for c in range(C):
        x = clip[c, :]
        x = (x - x.mean()) / (x.std() + EPS)
        corr = np.clip(np.abs(np.dot(x, front) / (len(x) - 1)), 0.0, 1.0)
        eog_corr[c, :] = corr  # 对所有频带同值（与你 JSD 一致风格）

    line_noise = np.zeros((C, B), dtype=np.float32)
    muscle     = np.zeros((C, B), dtype=np.float32)

    for b_idx, (_, (low, high)) in enumerate(BANDS.items()):
        mask_band = (freqs >= low) & (freqs < high)
        denom = psd[:, mask_band].sum(axis=1) + EPS

        mask_ln = (freqs >= max(49.0, low)) & (freqs < min(51.0, high))
        line_noise[:, b_idx] = (psd[:, mask_ln].sum(axis=1) / denom).astype(np.float32)

        mask_emg = (freqs >= max(45.0, low)) & (freqs < min(60.0, high))
        muscle[:, b_idx] = (psd[:, mask_emg].sum(axis=1) / denom).astype(np.float32)

    bad = (np.clip(line_noise, 0, 1) +
           np.clip(muscle,     0, 1) +
           np.clip(eog_corr,   0, 1)) / 3.0
    quality = np.clip(1.0 - bad, 0.0, 1.0).astype(np.float32)
    return quality

def de_from_psd(freqs, psd):
    """
    用 Welch PSD 近似 band 方差：
      var_band ≈ ∫ PSD(f) df ≈ sum(PSD) * df
    Differential Entropy (Gaussian):
      DE = 0.5 * ln(2*pi*e*var)
    返回 (C,B) float32
    """
    C = psd.shape[0]
    B = len(BANDS)
    de = np.zeros((C, B), dtype=np.float32)

    if len(freqs) >= 2:
        df = float(freqs[1] - freqs[0])
        if df <= 0:
            df = 1.0
    else:
        df = 1.0

    for b_idx, (_, (low, high)) in enumerate(BANDS.items()):
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            var = np.full((C,), EPS, dtype=np.float32)
        else:
            power = psd[:, mask].sum(axis=1).astype(np.float32) * df
            var = np.clip(power, EPS, None)
        de[:, b_idx] = 0.5 * np.log((2.0 * np.pi * np.e) * var)
    return de

def load_channel_info_and_labels(channel_xlsx, label_path):
    """
    仅加载：
      - 通道顺序（metadata + quality 的额叶索引）
      - label.mat 标签向量（长度=15）
    与你 JSD 脚本保持一致：不创建 MNE info/montage，不做 ICA。
    """
    channel_order = pd.read_excel(channel_xlsx, header=None)

    def norm(n: str) -> str:
        return str(n).strip().upper()

    ch_names_eeg = [norm(x) for x in channel_order.iloc[:, 0].astype(str).tolist()]
    if len(ch_names_eeg) != N_CHANNELS:
        print(f"[WARN] Excel 列出 {len(ch_names_eeg)} 通道，期望 {N_CHANNELS}")

    frontal_channel_names = ['FP1', 'FP2'] if all(
        x in ch_names_eeg for x in ['FP1', 'FP2']
    ) else ch_names_eeg[:2]
    frontal_idx = [i for i, n in enumerate(ch_names_eeg) if n in frontal_channel_names]
    if not frontal_idx:
        frontal_idx = [0, 1] if len(ch_names_eeg) >= 2 else [0]

    labels_mat = scipy.io.loadmat(label_path)
    lab = None
    for k in ['label', 'labels', 'Label', 'Labels']:
        if k in labels_mat:
            lab = labels_mat[k]
            break
    if lab is None:
        raise KeyError(f"label.mat 中未找到 label/labels 字段：{list(labels_mat.keys())}")

    lab = np.array(lab).squeeze()
    if lab.ndim != 1 or lab.shape[0] != 15:
        print(f"[WARN] label 向量形状为 {lab.shape}，请确认是否为 15 个 trial 标签。")

    return ch_names_eeg, frontal_idx, frontal_channel_names, lab.astype(np.int32)


# =========================
# Step 1：生成每 trial 的 DE 到 _de/
# =========================
def build_de_all(raw_root, save_root, ch_names_eeg, frontal_idx, labels_vec):
    out_root = os.path.join(save_root, DE_DIR)
    _safe_mkdir(out_root)

    subject_groups = discover_subject_files(raw_root)
    clip_len = int(SFREQ * WINDOW_SEC)
    stride   = int(SFREQ * STRIDE_SEC)

    print(f"[DE] raw_root={raw_root}")
    print(f"[DE] out_root={out_root}")
    print(f"[DE] window={WINDOW_SEC}s stride={STRIDE_SEC}s fs={SFREQ}Hz")

    # trial key：只认 *_eeg{1..15}，并按 {数字} 排序，避免错位
    re_eeg = re.compile(r"eeg(\d+)$", re.IGNORECASE)

    for subject_index, mats in enumerate(subject_groups, start=1):
        subj_dir = os.path.join(out_root, f"subj_{subject_index:02d}")
        _safe_mkdir(subj_dir)

        # 每个 subject 一般 3 个 mat（3 次实验/会话）
        for round_num, base in enumerate(mats, start=1):
            mat_path = os.path.join(raw_root, base + ".mat")
            if not os.path.exists(mat_path):
                print(f"[WARN] missing mat: {mat_path}")
                continue

            round_dir = os.path.join(subj_dir, f"round_{round_num}")
            _safe_mkdir(round_dir)

            mat = scipy.io.loadmat(mat_path)

            trial_keys = []
            for k in mat.keys():
                if k.startswith("__"):
                    continue
                m = re_eeg.search(k)
                if m:
                    trial_keys.append((int(m.group(1)), k))
            trial_keys.sort(key=lambda x: x[0])  # 按 eeg序号排序
            if len(trial_keys) != 15:
                print(f"[WARN] {mat_path} trial_keys={len(trial_keys)} (expected 15). keys(example)={trial_keys[:5]}")

            for trial_id, key in trial_keys:
                # label：来自 label.mat（按 trial_id 1..15）
                if trial_id < 1 or trial_id > len(labels_vec):
                    print(f"[WARN] invalid trial_id={trial_id} key={key} in {mat_path}")
                    continue
                lab_val = int(labels_vec[trial_id - 1])

                raw_np = mat[key]
                raw_np = np.squeeze(np.array(raw_np))
                if raw_np.ndim != 2:
                    print(f"[WARN] bad shape {raw_np.shape} for {mat_path}::{key}")
                    continue

                # 统一成 (C,T)
                if raw_np.shape[0] != N_CHANNELS and raw_np.shape[1] == N_CHANNELS:
                    raw_np = raw_np.T
                if raw_np.shape[0] != N_CHANNELS:
                    print(f"[WARN] channels mismatch {raw_np.shape} for {mat_path}::{key}")
                    continue

                subject_name = f's{subject_index:02d}'
                save_path = os.path.join(round_dir, f'{subject_name}_eeg{trial_id}.npz')

                # 断点续跑
                if os.path.exists(save_path):
                    continue

                data = raw_np.astype(np.float32)
                if not np.isfinite(data).all():
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                clips = create_clips(data, clip_len, stride)
                if not clips:
                    print(f"[WARN] No clips for {mat_path}::{key}")
                    continue

                T = len(clips)
                C = N_CHANNELS
                B = len(BANDS)

                de_t   = np.zeros((T, C, B), dtype=np.float16)
                q_t    = np.zeros((T, C, B), dtype=np.float16)
                edges  = np.zeros((T, 2),    dtype=np.int32)

                for t, (clip, s, e) in enumerate(clips):
                    edges[t, :] = [s, e]
                    freqs, psd = compute_psd(clip, SFREQ)  # psd:(C,F)

                    qmerge = quality_components(freqs, psd, clip, frontal_idx)  # (C,B)
                    q_t[t] = qmerge.astype(np.float16)

                    de_cb = de_from_psd(freqs, psd)  # (C,B)
                    de_t[t] = de_cb.astype(np.float16)

                np.savez_compressed(
                    save_path,
                    de=de_t,
                    quality=q_t,
                    clip_edges=edges,
                    fs=np.float32(SFREQ),
                    bands=np.array(BAND_NAMES, dtype="U16"),
                    ch_names=np.array(ch_names_eeg, dtype="U16"),
                    session=np.int32(round_num),
                    label=np.int32(lab_val)
                )
                print(f"[DE] Saved {save_path} | T={T} | label={lab_val}")

    print("[DE] Done.")


# =========================
# Step 2：用 _de/ 生成 _fold_de/（结构对齐 _fold_jsd）
# =========================
def _link_or_copy(src: str, dst: str, mode: str):
    dst_p = Path(dst)
    _safe_mkdir(dst_p.parent)

    if dst_p.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        src_p = Path(src).resolve()
        try:
            rel_src = os.path.relpath(str(src_p), start=str(dst_p.parent.resolve()))
            os.symlink(rel_src, dst)
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode={mode}")

def build_all_folds_from_de(save_root, mode="copy"):
    de_root   = os.path.join(save_root, DE_DIR)
    fold_root = os.path.join(save_root, FOLD_DIR)
    _safe_mkdir(fold_root)

    subs = sorted([p.name for p in Path(de_root).glob("subj_*") if p.is_dir()])
    if not subs:
        raise RuntimeError(f"No subj_* found under {de_root}")

    for target_subj in subs:
        fold_dir = Path(fold_root) / target_subj.replace("subj_", "fold_subj_")
        test_dir  = fold_dir / "test_target"
        train_dir = fold_dir / "train_source"
        _safe_mkdir(test_dir)
        _safe_mkdir(train_dir)

        # test_target：来自 target_subj 的 round_*/ *.npz（不再套 subj_XX）
        target_rounds = sorted(Path(de_root, target_subj).glob("round_*"))
        for rdir in target_rounds:
            rr = rdir.name  # round_1
            for npz in sorted(rdir.glob("*.npz")):
                dst = test_dir / rr / npz.name
                _link_or_copy(str(npz), str(dst), mode)

        # train_source：其余 subj 的 round_*/ *.npz -> train_source/subj_YY/round_R/*.npz
        for s in subs:
            if s == target_subj:
                continue
            for rdir in sorted(Path(de_root, s).glob("round_*")):
                rr = rdir.name
                for npz in sorted(rdir.glob("*.npz")):
                    dst = train_dir / s / rr / npz.name
                    _link_or_copy(str(npz), str(dst), mode)

        print(f"[FOLD_DE] built {fold_dir}")

    print("[FOLD_DE] Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SEED DE preprocessing pipeline."
    )
    parser.add_argument("--base-path", type=str, default=BASE_PATH, help="Directory containing SEED raw metadata and Preprocessed_EEG.")
    parser.add_argument("--raw-data", type=str, default=None, help="Override the Preprocessed_EEG directory.")
    parser.add_argument("--channel-xlsx", type=str, default=None, help="Override the channel-order.xlsx file.")
    parser.add_argument("--label-path", type=str, default=None, help="Override the label.mat path.")
    parser.add_argument("--save-root", type=str, default=SAVE_ROOT, help="Directory used to save all generated artifacts.")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2], choices=[1, 2], help="Pipeline steps to run.")
    parser.add_argument("--fold-link-mode", type=str, default=FOLD_LINK_MODE, choices=["copy", "symlink", "hardlink"], help="How to materialize files in _fold_de.")
    return parser

def main():
    args = build_parser().parse_args()

    base_path = args.base_path
    raw_data = args.raw_data or os.path.join(base_path, "Preprocessed_EEG")
    channel_xlsx = args.channel_xlsx or os.path.join(base_path, "channel-order.xlsx")
    label_path = args.label_path or os.path.join(raw_data, "label.mat")
    save_root = args.save_root
    selected_steps = set(args.steps)

    _safe_mkdir(save_root)
    print("[INFO] Configuration")
    print(f"[INFO] RAW_DATA={raw_data}")
    print(f"[INFO] CHANNEL_XLSX={channel_xlsx}")
    print(f"[INFO] LABEL_PATH={label_path}")
    print(f"[INFO] SAVE_ROOT={save_root}")
    print(f"[INFO] STEPS={sorted(selected_steps)}")
    print(f"[INFO] FOLD_LINK_MODE={args.fold_link_mode}")

    print("[STEP0] Loading channel order + label.mat ...")
    ch_names_eeg, frontal_idx, frontal_names, labels_vec = load_channel_info_and_labels(
        channel_xlsx, label_path
    )
    print(f"  channels={len(ch_names_eeg)} frontal={frontal_names} labels_shape={labels_vec.shape}")

    if 1 in selected_steps:
        print("[STEP1] Building per-trial DE to _de/ ...")
        build_de_all(raw_data, save_root, ch_names_eeg, frontal_idx, labels_vec)

    if 2 in selected_steps:
        print("[STEP2] Building LOOCV folds to _fold_de/ ...")
        build_all_folds_from_de(save_root, mode=args.fold_link_mode)

    print("[ALL DONE]")


if __name__ == "__main__":
    main()
