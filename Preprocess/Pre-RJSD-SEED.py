# jsd_seed_all_in_one_fixed.py
# -*- coding: utf-8 -*-
"""
SEED RJSD/JSD preprocessing pipeline.

Steps:
1) raw ``.mat`` -> per-trial ``p_hist`` / ``quality`` under ``_p_hist/``
2) build per-subject and global reference caches under ``_ref_cache/``
3) emit LOOCV JSD folds under ``_fold_jsd/``

The original feature-construction logic is preserved; this release-oriented
version mainly removes hard-coded local paths and exposes a command-line entry.
"""

import argparse
import os
import re
import glob
import json
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

# 频带
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta":  (14, 31),
    "gamma": (31, 60)
}
BAND_NAMES = list(BANDS.keys())

# 采样与滑窗
SFREQ      = 200.0
WINDOW_SEC = 4.0
STRIDE_SEC = 1.0

# Welch 与直方图
WELCH_NPERSEG       = int(SFREQ * 2)   # 2s
WELCH_NOVERLAP      = int(SFREQ * 1)   # 1s
HIST_BINS_PER_BAND  = 32
EPS                 = 1e-12

# 通道数（SEED 为 62）
N_CHANNELS = 62

# 目录角色
PHIST_DIR = '_p_hist'
CACHE_DIR = '_ref_cache'
FOLD_DIR  = '_fold_jsd'


# =========================
# 工具函数
# =========================

def _safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def natural_key(name: str):
    """
    自然排序 key：
    - 'ww_eeg1','ww_eeg2',...,'ww_eeg9','ww_eeg10','ww_eeg11',...,'ww_eeg15'
    - 避免 Python 默认字符串排序导致 'ww_eeg10' < 'ww_eeg2'
    """
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else name


def discover_subject_files(raw_data_path):
    """
    返回 [[subj1_文件们], [subj2_文件们], ...]（按 subject_id 升序），
    文件基名来自 Preprocessed_EEG 下的 *.mat（排除 label.mat）

    例如：
        [["1_20131027","1_20131030","1_20131107"],
         ["2_20140404","2_20140413",...],
         ...]
    """
    subject_files = defaultdict(list)
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found: {raw_data_path}")

    all_files = sorted(
        [f for f in os.listdir(raw_data_path)
         if f.endswith('.mat') and f != 'label.mat']
    )

    for filename in all_files:
        basename = os.path.splitext(filename)[0]   # e.g., "1_20131027"
        sid = basename.split('_')[0]              # "1"
        subject_files[sid].append(basename)

    sorted_subject_ids = sorted(subject_files.keys(), key=lambda x: int(x))
    return [subject_files[sid] for sid in sorted_subject_ids]


def create_clips(data, clip_len, stride):
    """
    data: (C, T)
    返回列表 [(clip, start_idx, end_idx), ...]
    """
    clips = []
    n = data.shape[1]
    for s in range(0, n - clip_len + 1, stride):
        e = s + clip_len
        clips.append((data[:, s:e], s, e))
    return clips


def compute_psd(clip, fs):
    """
    clip: (C, L)
    返回 freqs: (F,), psd: (C, F)
    """
    freqs, psd = welch(
        clip,
        fs=fs,
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        axis=-1
    )
    return freqs, psd


def band_histogram(freqs, band, psd_row, bins_per_band=HIST_BINS_PER_BAND):
    """
    对单个通道 psd_row 在某个频带内做频率直方图（权重为功率）
    """
    low, high = band
    mask = (freqs >= low) & (freqs < high)
    bins = np.linspace(low, high, bins_per_band + 1)

    if not np.any(mask):
        # 没有频点落入频带，给均匀分布
        return np.full((bins_per_band,), 1.0 / bins_per_band, dtype=np.float32), bins.astype(np.float32)

    fband = freqs[mask]
    pband = psd_row[mask]  # (F_band,)

    hist, _ = np.histogram(fband, bins=bins, weights=pband)
    s = float(hist.sum())
    if s > 0:
        p = (hist / s).astype(np.float32)
    else:
        p = np.full((bins_per_band,), 1.0 / bins_per_band, dtype=np.float32)

    return p, bins.astype(np.float32)


def quality_components(freqs, psd, clip, frontal_idx):
    """
    基于线噪声比例 / 肌电比例 / 与额叶参考的相关系数构造简单质量分数。
    返回 quality: (C, B)
    """
    C = psd.shape[0]
    B = len(BANDS)

    line_noise = np.zeros((C, B), dtype=np.float32)
    muscle     = np.zeros((C, B), dtype=np.float32)
    eog_corr   = np.zeros((C, B), dtype=np.float32)

    if len(frontal_idx) >= 1:
        ref = clip[frontal_idx, :].mean(axis=0)
    else:
        ref = clip.mean(axis=0)

    # EOG 相关（简单 Pearson）
    for c in range(C):
        x = clip[c, :]
        if np.std(x) > 1e-8 and np.std(ref) > 1e-8:
            eog_corr[c, :] = abs(np.corrcoef(x, ref)[0, 1])
        else:
            eog_corr[c, :] = 0.0

    # 线噪 / 肌电比例
    for b_idx, (_, (low, high)) in enumerate(BANDS.items()):
        mask_b = (freqs >= low) & (freqs < high)
        denom = psd[:, mask_b].sum(axis=1) + 1e-12

        # 49–51 Hz 线噪（和带交集）
        mask_ln = (freqs >= max(49.0, low)) & (freqs < min(51.0, high))
        line_noise[:, b_idx] = (psd[:, mask_ln].sum(axis=1) / denom).astype(np.float32)

        # 45–60 Hz 肌电（和带交集）
        mask_emg = (freqs >= max(45.0, low)) & (freqs < min(60.0, high))
        muscle[:, b_idx] = (psd[:, mask_emg].sum(axis=1) / denom).astype(np.float32)

    # 简单组合：bad ∈ [0,1] → quality = 1-bad
    bad = (np.clip(line_noise, 0, 1) +
           np.clip(muscle,     0, 1) +
           np.clip(eog_corr,   0, 1)) / 3.0
    quality = np.clip(1.0 - bad, 0.0, 1.0).astype(np.float32)
    return quality


def jsd_from_phist(p_hist, Q):
    """
    p_hist: (T, C, B, F)，每个时间窗的直方图
    Q     : (C, B, F)，参考分布
    返回 JSD: (T, C, B)
    """
    P = p_hist.astype(np.float32)
    P = np.clip(P, EPS, None)
    P /= P.sum(axis=-1, keepdims=True)

    Qn = np.clip(Q.astype(np.float32), EPS, None)
    Qn /= Qn.sum(axis=-1, keepdims=True)

    M = 0.5 * (P + Qn[None, ...])  # (T,C,B,F)

    kl_PM = (P   * (np.log(P)   - np.log(M))).sum(axis=-1)
    kl_QM = (Qn  * (np.log(Qn)  - np.log(M))).sum(axis=-1)   # 广播到 T

    jsd = 0.5 * kl_PM + 0.5 * kl_QM
    return jsd.astype(np.float32)


# =========================
# Step 0：加载通道 / 蒙太奇 / 标签
# =========================

def load_channel_info_and_labels(channel_xlsx, label_path):
    """
    仅加载：
      - 通道顺序（用于 metadata 与 quality 的额叶索引）
      - label.mat 标签向量（长度通常为 15）
    注意：不再创建 MNE info / montage，因为 .mat 已经做过 ICA 等预处理，我们不重复预处理。
    """
    channel_order = pd.read_excel(channel_xlsx, header=None)

    def norm(n: str) -> str:
        return str(n).strip().upper()

    ch_names_eeg = [norm(x) for x in channel_order.iloc[:, 0].astype(str).tolist()]
    if len(ch_names_eeg) != N_CHANNELS:
        print(f"[WARN] Excel 列出 {len(ch_names_eeg)} 通道，期望 {N_CHANNELS}")

    # 额叶通道（用于简易 EOG 参考的 quality 组件；若缺失则退化为前两个通道）
    frontal_channel_names = ['FP1', 'FP2'] if all(
        x in ch_names_eeg for x in ['FP1', 'FP2']
    ) else ch_names_eeg[:2]
    frontal_idx = [i for i, n in enumerate(ch_names_eeg) if n in frontal_channel_names]

    # --- 标签 ---
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


def build_phist_all(raw_root, save_root, ch_names_eeg, frontal_idx, labels_vec):
    """
    对所有 subject 的所有 session：
    - 读取 *.mat
    - 对每个 trial (ww_eeg1..ww_eeg15) 做自然排序
    - 为每个 trial 生成滑窗 p_hist / quality / label
    - 存到 SAVE_ROOT/_p_hist/subj_xx/round_y/ 下
    """

    out_root = os.path.join(save_root, PHIST_DIR)
    _safe_mkdir(out_root)

    clip_len = int(WINDOW_SEC * SFREQ)
    stride   = int(STRIDE_SEC * SFREQ)

    subs = discover_subject_files(raw_root)
    if not subs:
        raise RuntimeError('No subject files found in RAW_DATA.')

    C = len(ch_names_eeg)
    B = len(BANDS)

    for subject_index, sess_list in enumerate(subs, 1):
        subj_tag = f"subj_{subject_index:02d}"
        subj_dir = os.path.join(out_root, subj_tag)
        _safe_mkdir(subj_dir)

        for round_num, base in enumerate(sess_list, 1):
            mat_path = os.path.join(raw_root, f'{base}.mat')
            if not os.path.exists(mat_path):
                print(f'[WARN] missing {mat_path}')
                continue

            round_dir = os.path.join(subj_dir, f'round_{round_num}')
            _safe_mkdir(round_dir)

            mat = scipy.io.loadmat(mat_path)
            trial_keys = [k for k in mat.keys() if not k.startswith('__')]

            # ✨ 关键修正：trial 使用“自然排序”，保证 ww_eeg1..15 顺序对应 label[0..14]
            trial_keys = sorted(trial_keys, key=natural_key)

            if len(trial_keys) != len(labels_vec):
                print(f"[WARN] {mat_path} trial 数 {len(trial_keys)} != label 数 {len(labels_vec)}")

            for trial_idx, key in enumerate(trial_keys):
                if trial_idx >= len(labels_vec):
                    print(f"[WARN] label 索引越界：trial_idx={trial_idx}, labels_len={len(labels_vec)}")
                    lab_val = -999
                else:
                    lab_val = int(labels_vec[trial_idx])

                raw_np = mat[key]  # (C, T) 或 (T, C)
                if raw_np.shape[0] != N_CHANNELS:
                    raw_np = raw_np.T
                if raw_np.shape[0] != N_CHANNELS:
                    print(f'  Skip {mat_path}::{key}: bad shape {raw_np.shape}')
                    continue

                subject_name = f's{subject_index:02d}'
                save_path = os.path.join(round_dir, f'{subject_name}_eeg{trial_idx+1}.npz')

                # 断点续跑：如果已经存在则跳过
                if os.path.exists(save_path):
                    continue

                # ✅ .mat 已完成 ICA/滤波等预处理：这里不重复任何预处理
                data = raw_np.astype(np.float32)  # (C, T)
                if not np.isfinite(data).all():
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                clips = create_clips(data, clip_len, stride)
                if not clips:
                    print(f'  No clips for {mat_path}::{key}')
                    continue

                T = len(clips)

                p_hist = np.zeros((T, C, B, HIST_BINS_PER_BAND), dtype=np.float32)
                q_t    = np.zeros((T, C, B),                    dtype=np.float32)
                edges  = np.zeros((T, 2),                       dtype=np.int32)

                for t, (clip, s, e) in enumerate(clips):
                    edges[t, :] = [s, e]
                    freqs, psd = compute_psd(clip, SFREQ)

                    qmerge = quality_components(freqs, psd, clip, frontal_idx)  # (C,B)
                    q_t[t] = qmerge.astype(np.float32)

                    for b_idx, (_, band) in enumerate(BANDS.items()):
                        for c in range(C):
                            p, _ = band_histogram(freqs, band, psd[c, :])
                            p_hist[t, c, b_idx, :] = p.astype(np.float32)

                np.savez_compressed(
                    save_path,
                    p_hist=p_hist,
                    quality=q_t,
                    clip_edges=edges,
                    fs=np.float32(SFREQ),
                    bands=np.array(BAND_NAMES, dtype="U16"),
                    ch_names=np.array(ch_names_eeg, dtype="U16"),
                    session=np.int32(round_num),
                    label=np.int32(lab_val)
                )
                print(f'[PHIST] Saved {save_path} | T={T} | label={lab_val}')

    print('[PHIST] Done.')


# ======================================
# Step 2：汇总 per-subject / global 累积
# ======================================

def build_accum_all(save_root):
    phist_root = os.path.join(save_root, PHIST_DIR)
    cache_root = os.path.join(save_root, CACHE_DIR)
    _safe_mkdir(cache_root)

    subs = sorted([p.name for p in Path(phist_root).glob("subj_*") if p.is_dir()])
    if not subs:
        raise RuntimeError(f'No subjects found in {phist_root}')

    global_accum = None

    for s in subs:
        files = sorted(glob.glob(f"{phist_root}/{s}/round_*/*.npz"))
        if not files:
            print(f"[WARN] {s} no npz, skip")
            continue

        subj_accum = None
        for fp in files:
            with np.load(fp) as d:
                p_hist = d["p_hist"].astype(np.float32)   # (T,C,B,F)
            w = p_hist.sum(axis=0)                        # (C,B,F)
            subj_accum = w if subj_accum is None else (subj_accum + w)

        out_subj = f"{cache_root}/accum_{s}.npz"
        np.savez_compressed(out_subj, accum=subj_accum.astype(np.float32))
        print(f"[ACCUM] {out_subj} shape={tuple(subj_accum.shape)}")

        global_accum = subj_accum if global_accum is None else (global_accum + subj_accum)

    out_global = f"{cache_root}/accum_global.npz"
    np.savez_compressed(out_global, accum=global_accum.astype(np.float32))
    print(f"[ACCUM] {out_global} shape={tuple(global_accum.shape)}")
    print('[ACCUM] Done.')


# ===========================================
# Step 3：为所有 LOOCV 折生成 JSD 特征
# ===========================================

def make_ref_excluding(cache_root, target_sid_tag):
    """
    global_accum - target_subject_accum → 归一化 → (C,B,F) 作为参考分布 Q
    """
    g = np.load(f"{cache_root}/accum_global.npz")["accum"].astype(np.float64)
    a = np.load(f"{cache_root}/accum_{target_sid_tag}.npz")["accum"].astype(np.float64)

    g -= a
    g = np.clip(g, 0.0, None)

    denom = g.sum(axis=-1, keepdims=True) + EPS
    Q = (g / denom).astype(np.float32)
    Q = np.clip(Q, 1e-12, None)
    Q /= Q.sum(axis=-1, keepdims=True)
    return Q


def write_fold_trial(in_path, out_path, Q):
    d = np.load(in_path, allow_pickle=True)
    p_hist  = d["p_hist"]
    quality = d["quality"]

    # 保留其余 meta 信息
    def sanitize(x):
        arr = np.array(x)
        if arr.dtype == object:
            return np.array(list(map(str, arr.tolist())), dtype="U64")
        return arr

    meta = {k: sanitize(d[k]) for k in d.files if k not in ("p_hist", "quality")}

    jsd = jsd_from_phist(p_hist, Q)  # (T,C,B)

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        jsd=jsd.astype(np.float32),
        quality=quality.astype(np.float32),
        **meta
    )


def build_all_folds(save_root):
    phist_root = os.path.join(save_root, PHIST_DIR)
    cache_root = os.path.join(save_root, CACHE_DIR)
    fold_root  = os.path.join(save_root, FOLD_DIR)
    _safe_mkdir(fold_root)

    # 先拿一个样本检查形状
    sample = None
    subs = sorted([p.name for p in Path(phist_root).glob("subj_*") if p.is_dir()])
    for s in subs:
        cand = sorted(glob.glob(f"{phist_root}/{s}/round_*/*.npz"))
        if cand:
            sample = cand[0]
            break
    if sample is None:
        raise RuntimeError("No sample npz found under _p_hist/")

    with np.load(sample, allow_pickle=True) as d:
        _, C_chk, B_chk, F_chk = d["p_hist"].shape

    for target_sid_tag in subs:
        Q = make_ref_excluding(cache_root, target_sid_tag)
        assert Q.shape == (C_chk, B_chk, F_chk), f"Q shape {Q.shape} != {(C_chk,B_chk,F_chk)}"

        fold_dir = os.path.join(fold_root, f"fold_{target_sid_tag}")
        src_root = os.path.join(fold_dir, "train_source")
        tgt_root = os.path.join(fold_dir, "test_target")
        _safe_mkdir(src_root)
        _safe_mkdir(tgt_root)

        # 目标域
        tgt_files = sorted(glob.glob(f"{phist_root}/{target_sid_tag}/round_*/*.npz"))
        for fp in tgt_files:
            rel = fp.split(f"{phist_root}/")[1]
            outp = os.path.join(tgt_root, rel.replace(target_sid_tag + '/', ''))
            if os.path.exists(outp):
                continue
            write_fold_trial(fp, outp, Q)

        # 源域
        for s in subs:
            if s == target_sid_tag:
                continue
            files = sorted(glob.glob(f"{phist_root}/{s}/round_*/*.npz"))
            for fp in files:
                rel = fp.split(f"{phist_root}/")[1]
                outp = os.path.join(src_root, rel)
                if os.path.exists(outp):
                    continue
                write_fold_trial(fp, outp, Q)

        sig = {
            "target_sid": target_sid_tag,
            "source_sids": [s for s in subs if s != target_sid_tag],
            "ref_cache": {
                "global": f"{cache_root}/accum_global.npz",
                target_sid_tag: f"{cache_root}/accum_{target_sid_tag}.npz"
            }
        }
        with open(os.path.join(fold_dir, "REF_SIGNATURE.json"), "w", encoding="utf-8") as f:
            json.dump(sig, f, indent=2, ensure_ascii=False)
        print(f"[FOLD] built {fold_dir}")

    print('[FOLD] Done.')


# ============
# 主流程
# ============

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SEED RJSD/JSD preprocessing pipeline."
    )
    parser.add_argument("--base-path", type=str, default=BASE_PATH, help="Directory containing SEED raw metadata and Preprocessed_EEG.")
    parser.add_argument("--raw-data", type=str, default=None, help="Override the Preprocessed_EEG directory.")
    parser.add_argument("--channel-xlsx", type=str, default=None, help="Override the channel-order.xlsx file.")
    parser.add_argument("--label-path", type=str, default=None, help="Override the label.mat path.")
    parser.add_argument("--save-root", type=str, default=SAVE_ROOT, help="Directory used to save all generated artifacts.")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3], help="Pipeline steps to run.")
    return parser

def main():
    args = build_parser().parse_args()

    base_path = args.base_path
    raw_data = args.raw_data or os.path.join(base_path, "Preprocessed_EEG")
    channel_xlsx = args.channel_xlsx or os.path.join(base_path, "channel-order.xlsx")
    label_path = args.label_path or os.path.join(raw_data, "label.mat")
    save_root = args.save_root
    selected_steps = set(args.steps)

    print("[INFO] Configuration")
    print(f"[INFO] RAW_DATA={raw_data}")
    print(f"[INFO] CHANNEL_XLSX={channel_xlsx}")
    print(f"[INFO] LABEL_PATH={label_path}")
    print(f"[INFO] SAVE_ROOT={save_root}")
    print(f"[INFO] STEPS={sorted(selected_steps)}")

    print('[INFO] Loading channels/labels...')
    ch_names_eeg, frontal_idx, frontal_names, labels_vec = load_channel_info_and_labels(
        channel_xlsx, label_path
    )
    print(f'[INFO] Channels={len(ch_names_eeg)} | Frontal={frontal_names} | labels_len={len(labels_vec)}')

    if 1 in selected_steps:
        print('[STEP1] Building p_hist/quality per trial ...')
        build_phist_all(
            raw_data,
            save_root,
            ch_names_eeg,
            frontal_idx,
            labels_vec
        )

    if 2 in selected_steps:
        print('[STEP2] Building per-subject/global accum ...')
        build_accum_all(save_root)

    if 3 in selected_steps:
        print('[STEP3] Emitting ALL LOOCV folds (JSD features) ...')
        build_all_folds(save_root)

    print('[ALL DONE]')


if __name__ == "__main__":
    main()
