# -*- coding: utf-8 -*-
"""
SEED-IV end-to-end preprocessing pipeline.

The original implementation is kept intact, but the public-release workflow is
made configurable from the command line instead of requiring source edits.

Pipeline steps:
1) raw ``.mat`` -> per-trial ``p_hist`` / ``quality`` / ``de`` in ``_p_hist``
2) build per-subject and global reference caches in ``_ref_cache``
3) emit LOOCV folds with ``jsd`` + ``de`` in ``_fold_jsd_de``
4) compute train-only zDE statistics and export gated features to ``_fold_jsd_degate``
"""

from __future__ import annotations
import argparse
import os, re, glob, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.io
import mne
from scipy.signal import welch


# =========================
# Public defaults (CLI-overridable)
# =========================
BASE_PATH = "data/SEED-IV-RAW"
RAW_DATA = os.path.join(BASE_PATH, "eeg_raw_data")
SAVE_ROOT = "data/SEED-IV"
CHANNEL_XLSX = os.path.join(BASE_PATH, "Zehn-Channel Order.xlsx")
MONTAGE_LOCS = os.path.join(BASE_PATH, "channel_62_pos.locs")

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
SFREQ = 200.0
WINDOW_SEC = 4.0
STRIDE_SEC = 1.0

# Welch 与直方图
WELCH_NPERSEG = int(SFREQ * 2)   # 2s
WELCH_NOVERLAP = int(SFREQ * 1)  # 1s
HIST_BINS_PER_BAND = 32

# 数值稳定
EPS = 1e-12          # JSD / normalize
DE_EPS = 1e-12       # log( band_power + DE_EPS )

# 门控强度
ALPHA = 0.5

# 输出目录角色
PHIST_DIR = '_p_hist'
CACHE_DIR = '_ref_cache'
FOLD_DIR_RAW   = '_fold_jsd_de'        # jsd + de + quality
FOLD_DIR_FUSED = '_fold_jsd_degate'    # jsd + de + zde + gated + feat

# 是否保存拼接后的 feat
SAVE_FEAT = True


# 受试者名（与训练脚本保持一致，索引 1..15；SUBJECT_NAMES[1] 对应 subj_01）
SUBJECT_NAMES = [
    '-1', 'cz', 'ha', 'hql', 'ldy', 'ly', 'mhw', 'mz', 'qyt', 'rx',
    'tyc', 'whh', 'wll', 'wq', 'zjd', 'zjy'
]

# 标签表（兼容 24/25）
LABELS_BY_SESSION = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
}


# ================
# 基础工具
# ================
def _safe_mkdir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def discover_subject_files(raw_data_path: str):
    """返回 [[s1_r1_base, s1_r2_base, s1_r3_base], [s2_r1_base, ...], ...]（按sid升序）"""
    subject_files = defaultdict(list)
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found: {raw_data_path}")
    for session_num in range(1, 3+1):
        session_path = os.path.join(raw_data_path, str(session_num))
        if not os.path.isdir(session_path):
            continue
        for filename in sorted(os.listdir(session_path)):
            if filename.endswith('.mat'):
                basename = os.path.splitext(filename)[0]
                sid = basename.split('_')[0]  # '1_201311'
                if basename not in subject_files[sid]:
                    subject_files[sid].append(basename)
    sorted_subject_ids = sorted(subject_files.keys(), key=lambda x: int(x))
    return [subject_files[sid] for sid in sorted_subject_ids]

def create_clips(data: np.ndarray, clip_len: int, stride: int):
    clips = []
    n = data.shape[1]
    for s in range(0, n - clip_len + 1, stride):
        e = s + clip_len
        clips.append((data[:, s:e], s, e))
    return clips

def compute_psd(clip: np.ndarray, fs: float):
    freqs, psd = welch(
        clip, fs=fs,
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        axis=-1
    )
    return freqs, psd  # (F,), (C,F)

def band_histogram(freqs: np.ndarray, band: tuple[float,float], psd_row: np.ndarray,
                   bins_per_band: int = HIST_BINS_PER_BAND):
    low, high = band
    mask = (freqs >= low) & (freqs < high)
    bins = np.linspace(low, high, bins_per_band + 1)
    if not np.any(mask):
        return np.full((bins_per_band,), 1.0/bins_per_band, dtype=np.float32), bins.astype(np.float32)
    fband = freqs[mask]
    pband = psd_row[mask]
    hist, _ = np.histogram(fband, bins=bins, weights=pband)
    s = float(hist.sum())
    p = (hist / s).astype(np.float32) if s > 0 else np.full((bins_per_band,), 1.0/bins_per_band, dtype=np.float32)
    return p, bins.astype(np.float32)

def extract_trial_id(mat_key: str) -> int:
    m = re.findall(r'\d+', str(mat_key))
    return int(m[-1]) if m else -1

def get_label(round_id: int, trial_id: int) -> int:
    labels = LABELS_BY_SESSION.get(round_id, None)
    if labels is None:
        return -1
    if len(labels) == 24:
        idx = trial_id - 1
    elif len(labels) == 25:
        idx = trial_id
    else:
        return -1
    if idx < 0 or idx >= len(labels):
        return -1
    return int(labels[idx])

def quality_components(freqs: np.ndarray, psd: np.ndarray, clip: np.ndarray, frontal_idx: list[int]):
    """简单质量分：线噪、肌电、高相关(近似EOG) -> 合成 quality in [0,1]"""
    C = psd.shape[0]; B = len(BANDS)
    line_noise = np.zeros((C,B), dtype=np.float32)
    muscle     = np.zeros((C,B), dtype=np.float32)
    eog_corr   = np.zeros((C,B), dtype=np.float32)

    ref = clip[frontal_idx, :].mean(axis=0) if len(frontal_idx)>=1 else clip.mean(axis=0)
    for c in range(C):
        x = clip[c, :]
        if np.std(x)>1e-8 and np.std(ref)>1e-8:
            eog_corr[c,:] = abs(np.corrcoef(x, ref)[0,1])
        else:
            eog_corr[c,:] = 0.0

    for b_idx, (_bn,(low,high)) in enumerate(BANDS.items()):
        mask_b = (freqs>=low)&(freqs<high)
        denom = psd[:,mask_b].sum(axis=1)+1e-12

        mask_ln = (freqs>=max(49.0,low))&(freqs<min(51.0,high))
        line_noise[:,b_idx] = (psd[:,mask_ln].sum(axis=1)/denom).astype(np.float32)

        mask_emg = (freqs>=max(45.0,low))&(freqs<min(60.0,high))
        muscle[:,b_idx] = (psd[:,mask_emg].sum(axis=1)/denom).astype(np.float32)

    bad = (np.clip(line_noise,0,1)+np.clip(muscle,0,1)+np.clip(eog_corr,0,1))/3.0
    return np.clip(1.0-bad, 0.0, 1.0).astype(np.float32)

def jsd_from_phist(p_hist: np.ndarray, Q: np.ndarray):
    """ p_hist: (T,C,B,F) float16/32, Q: (C,B,F) float32 -> (T,C,B) float32 """
    P = p_hist.astype(np.float32)
    P = np.clip(P, EPS, None)
    P /= P.sum(axis=-1, keepdims=True)

    Qn = np.clip(Q.astype(np.float32), EPS, None)
    Qn /= Qn.sum(axis=-1, keepdims=True)

    M = 0.5*(P + Qn[None, ...])      # (T,C,B,F)
    kl_PM = (P * (np.log(P) - np.log(M))).sum(axis=-1)
    kl_QM = (Qn * (np.log(Qn) - np.log(M))).sum(axis=-1)  # broadcast over T
    jsd = 0.5*kl_PM + 0.5*kl_QM
    return jsd.astype(np.float32)     # (T,C,B)

def de_from_psd_bandpower(freqs: np.ndarray, psd: np.ndarray):
    """
    以 PSD band power 近似方差尺度：DE = log(power + eps)
    输入 psd: (C,F) -> 输出 de_cb: (C,B)
    """
    C = psd.shape[0]
    B = len(BANDS)
    out = np.zeros((C,B), dtype=np.float32)
    for b_idx, (_bn,(low,high)) in enumerate(BANDS.items()):
        mask_b = (freqs>=low)&(freqs<high)
        power = psd[:,mask_b].sum(axis=1)  # (C,)
        out[:,b_idx] = np.log(power + DE_EPS).astype(np.float32)
    return out  # (C,B)


# =========================
# Step 0：加载通道信息/蒙太奇
# =========================
def load_channel_info(channel_xlsx: str, montage_locs: str):
    channel_order = pd.read_excel(channel_xlsx, header=None)

    def norm(n: str) -> str: return str(n).strip().upper()

    ch_names_eeg = [norm(x) for x in channel_order.iloc[:,0].astype(str).tolist()]
    ch_names_eeg = [n for n in ch_names_eeg if n not in ('CHANNEL_NAME','', 'NAN')]

    montage = mne.channels.read_custom_montage(montage_locs)
    info_eeg = mne.create_info(ch_names=ch_names_eeg, ch_types=['eeg']*len(ch_names_eeg), sfreq=SFREQ)
    info_eeg.set_montage(montage)

    frontal_channel_names = ['FP1','FP2'] if all(x in ch_names_eeg for x in ['FP1','FP2']) else ch_names_eeg[:2]
    frontal_idx = [i for i,n in enumerate(ch_names_eeg) if n in frontal_channel_names]
    return info_eeg, ch_names_eeg, frontal_idx, frontal_channel_names


# ===============================================
# Step 1：生成每个 trial 的 p_hist/quality/de（可续跑）
# ===============================================
def build_phist_all(raw_root: str, save_root: str,
                    info_eeg, ch_names_eeg: list[str], frontal_idx: list[int], frontal_names: list[str]):
    out_root = os.path.join(save_root, PHIST_DIR)
    _safe_mkdir(out_root)

    clip_len = int(WINDOW_SEC*SFREQ)
    stride   = int(STRIDE_SEC*SFREQ)

    subs = discover_subject_files(raw_root)
    if not subs:
        raise RuntimeError('No subject files found.')

    C = len(ch_names_eeg)
    B = len(BANDS)

    for subject_index, sess_list in enumerate(subs, 1):
        subj_tag = f"subj_{subject_index:02d}"
        subj_dir = os.path.join(out_root, subj_tag)
        _safe_mkdir(subj_dir)

        for round_num, base in enumerate(sess_list, 1):
            mat_path = os.path.join(raw_root, str(round_num), f'{base}.mat')
            if not os.path.exists(mat_path):
                print(f'[WARN] missing {mat_path}')
                continue

            round_dir = os.path.join(subj_dir, f'round_{round_num}')
            _safe_mkdir(round_dir)

            mat = scipy.io.loadmat(mat_path)
            for key in mat:
                if key.startswith('__'):
                    continue
                trial_id = extract_trial_id(key)
                if trial_id < 1 or trial_id > 24:
                    continue

                label_val = get_label(round_num, trial_id)
                if label_val == -1:
                    continue

                subject_name = SUBJECT_NAMES[subject_index]
                save_path = os.path.join(round_dir, f'{subject_name}_eeg{trial_id}.npz')
                if os.path.exists(save_path):
                    # 断点续跑：已存在则跳过
                    continue

                # notch → 高通 → ICA → 1–60Hz 带通
                raw = mne.io.RawArray(mat[key], info_eeg, verbose=False)
                raw.notch_filter(freqs=50, verbose=False)
                raw.filter(l_freq=1.0, h_freq=None, verbose=False)

                try:
                    ica = mne.preprocessing.ICA(n_components=0.999, random_state=97, max_iter=1000)
                    ica.fit(raw, reject_by_annotation=False)
                except Exception:
                    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=1000)
                    ica.fit(raw, reject_by_annotation=False)

                eog_idx,_ = ica.find_bads_eog(raw, ch_name=frontal_names, verbose=False)
                mus_idx,_ = ica.find_bads_muscle(raw, verbose=False)
                ica.exclude = sorted(set((eog_idx or []) + (mus_idx or [])))
                raw = ica.apply(raw, exclude=ica.exclude, verbose=False)

                raw.filter(l_freq=1.0, h_freq=60.0, verbose=False)

                data = raw.get_data()  # (C, N)
                clips = create_clips(data, clip_len, stride)
                if not clips:
                    print(f'  No clips for {key}')
                    continue

                T = len(clips)
                p_hist = np.zeros((T, C, B, HIST_BINS_PER_BAND), dtype=np.float16)
                q_t    = np.zeros((T, C, B), dtype=np.float16)
                de_t   = np.zeros((T, C, B), dtype=np.float16)
                edges  = np.zeros((T, 2), dtype=np.int32)

                for t,(clip,s,e) in enumerate(clips):
                    edges[t,:] = [s,e]
                    freqs, psd = compute_psd(clip, SFREQ)         # psd: (C,F)

                    # quality
                    qmerge = quality_components(freqs, psd, clip, frontal_idx)  # (C,B)
                    q_t[t] = qmerge.astype(np.float16)

                    # DE (log band power)
                    de_cb = de_from_psd_bandpower(freqs, psd)                   # (C,B)
                    de_t[t] = de_cb.astype(np.float16)

                    # p_hist
                    for b_idx, (_bn, band) in enumerate(BANDS.items()):
                        for c in range(C):
                            p,_ = band_histogram(freqs, band, psd[c,:])
                            p_hist[t, c, b_idx, :] = p.astype(np.float16)

                np.savez_compressed(
                    save_path,
                    p_hist=p_hist,
                    quality=q_t,
                    de=de_t,
                    clip_edges=edges,
                    fs=np.float32(SFREQ),
                    bands=np.array(BAND_NAMES, dtype=object),
                    ch_names=np.array(ch_names_eeg, dtype=object),
                    session=np.int32(round_num),
                    label=np.int32(label_val)
                )
                print(f'[PHIST] Saved {save_path} | T={T}')

    print('[PHIST] Done.')


# ======================================
# Step 2：汇总 per-subject / global 累积
# ======================================
def build_accum_all(save_root: str):
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
            d = np.load(fp, allow_pickle=True)
            p_hist = d["p_hist"].astype(np.float32)   # (T,C,B,F)
            w = p_hist.sum(axis=0)                    # (C,B,F)
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
# Step 3：为所有 LOOCV 折生成 JSD + DE
# ===========================================
def make_ref_excluding(cache_root: str, target_sid_tag: str):
    g = np.load(f"{cache_root}/accum_global.npz")["accum"].astype(np.float64)
    a = np.load(f"{cache_root}/accum_{target_sid_tag}.npz")["accum"].astype(np.float64)
    g -= a
    g = np.clip(g, 0.0, None)
    denom = g.sum(axis=-1, keepdims=True) + EPS
    return (g / denom).astype(np.float32)  # (C,B,F)

def _sanitize_meta_dict(d: np.lib.npyio.NpzFile, exclude: set[str]):
    def sanitize(x):
        arr = np.array(x)
        if arr.dtype == object:
            return np.array(list(map(str, arr.tolist())), dtype="U64")
        return arr
    return {k: sanitize(d[k]) for k in d.files if k not in exclude}

def write_fold_trial_jsd_de(in_path: str, out_path: str, Q: np.ndarray):
    d = np.load(in_path, allow_pickle=True)
    p_hist = d["p_hist"]
    quality = d["quality"]
    de = d["de"]

    meta = _sanitize_meta_dict(d, exclude={"p_hist","quality","de"})

    jsd = jsd_from_phist(p_hist, Q)  # (T,C,B)

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        jsd=jsd.astype(np.float32),
        de=de.astype(np.float32),
        quality=quality.astype(np.float16),
        **meta
    )

def build_all_folds_jsd_de(save_root: str):
    phist_root = os.path.join(save_root, PHIST_DIR)
    cache_root = os.path.join(save_root, CACHE_DIR)
    fold_root  = os.path.join(save_root, FOLD_DIR_RAW)
    _safe_mkdir(fold_root)

    subs = sorted([p.name for p in Path(phist_root).glob("subj_*") if p.is_dir()])
    if not subs:
        raise RuntimeError(f'No subjects found in {phist_root}')

    for target_sid_tag in subs:
        Q = make_ref_excluding(cache_root, target_sid_tag)
        fold_dir = os.path.join(fold_root, f"fold_{target_sid_tag}")
        src_root = os.path.join(fold_dir, "train_source")
        tgt_root = os.path.join(fold_dir, "test_target")
        _safe_mkdir(src_root); _safe_mkdir(tgt_root)

        # 目标域：保持与你原脚本一致（去掉 subj_xx/ 前缀）
        tgt_files = sorted(glob.glob(f"{phist_root}/{target_sid_tag}/round_*/*.npz"))
        for fp in tgt_files:
            rel = fp.split(f"{phist_root}/")[1]
            outp = os.path.join(tgt_root, rel.replace(target_sid_tag + '/', ''))
            if os.path.exists(outp):
                continue
            write_fold_trial_jsd_de(fp, outp, Q)

        # 源域：保留 subj_xx/ 结构
        for s in subs:
            if s == target_sid_tag:
                continue
            files = sorted(glob.glob(f"{phist_root}/{s}/round_*/*.npz"))
            for fp in files:
                rel = fp.split(f"{phist_root}/")[1]
                outp = os.path.join(src_root, rel)
                if os.path.exists(outp):
                    continue
                write_fold_trial_jsd_de(fp, outp, Q)

        sig = {
            "target_sid": target_sid_tag,
            "source_sids": [s for s in subs if s != target_sid_tag],
            "ref_cache": {
                "global": f"{cache_root}/accum_global.npz",
                target_sid_tag: f"{cache_root}/accum_{target_sid_tag}.npz"
            }
        }
        with open(os.path.join(fold_dir, "REF_SIGNATURE.json"), "w") as f:
            json.dump(sig, f, indent=2, ensure_ascii=False)
        print(f"[FOLD_RAW] built {fold_dir}")

    print('[FOLD_RAW] Done.')


# ===========================================
# Step 4：每折用 train_source 统计 mu/sigma，生成 zDE-gated JSD + feat
# ===========================================
def _iter_npz_files(root: str):
    return sorted([str(p) for p in Path(root).rglob("*.npz")])

def _compute_mu_sigma_from_train_source(train_root: str):
    files = _iter_npz_files(train_root)
    if not files:
        raise RuntimeError(f"No train_source npz under: {train_root}")

    sum_de = None
    sumsq_de = None
    n_total = 0

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        de = d["de"].astype(np.float32)      # (T,C,B)
        # sum over T
        s1 = de.sum(axis=0).astype(np.float64)         # (C,B)
        s2 = (de * de).sum(axis=0).astype(np.float64)  # (C,B)
        if sum_de is None:
            sum_de = s1
            sumsq_de = s2
        else:
            sum_de += s1
            sumsq_de += s2
        n_total += de.shape[0]

    mu = (sum_de / max(n_total, 1)).astype(np.float32)
    var = (sumsq_de / max(n_total, 1) - (mu.astype(np.float64) ** 2)).astype(np.float64)
    var = np.maximum(var, 1e-8)  # 防止负数与除零
    sigma = np.sqrt(var).astype(np.float32)
    return mu, sigma, int(n_total)

def _augment_one_file(in_fp: str, out_fp: str, mu: np.ndarray, sigma: np.ndarray):
    d = np.load(in_fp, allow_pickle=True)
    jsd = d["jsd"].astype(np.float32)      # (T,C,B)
    de  = d["de"].astype(np.float32)       # (T,C,B)
    quality = d["quality"].astype(np.float16)

    meta = _sanitize_meta_dict(d, exclude={"jsd","de","quality"})

    # zDE: (T,C,B)
    sigma_safe = np.where(sigma < 1e-6, 1.0, sigma).astype(np.float32)  # (C,B)
    zde = (de - mu[None, ...]) / sigma_safe[None, ...]

    # gate + gated jsd
    gate = np.tanh(ALPHA * zde).astype(np.float32)         # (T,C,B)
    jsd_gated = (gate * jsd).astype(np.float32)            # (T,C,B)

    out_dir = os.path.dirname(out_fp)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    payload = dict(
        jsd=jsd,
        de=de,
        zde=zde.astype(np.float32),
        gate=gate,
        jsd_gated=jsd_gated,
        quality=quality,
        alpha=np.float32(ALPHA),
        **meta
    )

    if SAVE_FEAT:
        # feat: concat on last dim -> (T,C,3B)
        feat = np.concatenate([jsd, jsd_gated, zde.astype(np.float32)], axis=-1).astype(np.float32)
        payload["feat"] = feat

    np.savez_compressed(out_fp, **payload)

def build_all_folds_degate(save_root: str):
    fold_in_root  = os.path.join(save_root, FOLD_DIR_RAW)
    fold_out_root = os.path.join(save_root, FOLD_DIR_FUSED)
    _safe_mkdir(fold_out_root)

    fold_dirs = sorted([p for p in Path(fold_in_root).glob("fold_subj_*") if p.is_dir()])
    if not fold_dirs:
        raise RuntimeError(f"No folds found under: {fold_in_root}")

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        in_fold = str(fold_dir)
        out_fold = os.path.join(fold_out_root, fold_name)
        _safe_mkdir(out_fold)

        train_root = os.path.join(in_fold, "train_source")
        test_root  = os.path.join(in_fold, "test_target")
        if (not os.path.isdir(train_root)) or (not os.path.isdir(test_root)):
            print(f"[WARN] skip {in_fold}: missing train_source/test_target")
            continue

        # 1) 统计 mu/sigma（仅 train_source）
        stats_fp = os.path.join(out_fold, "DE_STATS.npz")
        stats_json = os.path.join(out_fold, "DE_STATS.json")

        if os.path.exists(stats_fp):
            st = np.load(stats_fp, allow_pickle=True)
            mu = st["mu"].astype(np.float32)
            sigma = st["sigma"].astype(np.float32)
            n_total = int(st["n_total"])
            print(f"[DE_STATS] reuse {stats_fp} n_total={n_total}")
        else:
            mu, sigma, n_total = _compute_mu_sigma_from_train_source(train_root)
            np.savez_compressed(stats_fp, mu=mu, sigma=sigma, n_total=np.int64(n_total))
            with open(stats_json, "w") as f:
                json.dump(
                    {"alpha": ALPHA, "n_total": n_total, "note": "mu/sigma computed from train_source only"},
                    f, indent=2, ensure_ascii=False
                )
            print(f"[DE_STATS] saved {stats_fp} n_total={n_total}")

        # 2) 逐文件生成 degate 输出（断点续跑）
        for split in ["train_source", "test_target"]:
            in_root = os.path.join(in_fold, split)
            out_root = os.path.join(out_fold, split)
            _safe_mkdir(out_root)

            in_files = _iter_npz_files(in_root)
            for fp in in_files:
                rel = os.path.relpath(fp, in_root)
                out_fp = os.path.join(out_root, rel)
                if os.path.exists(out_fp):
                    continue
                _augment_one_file(fp, out_fp, mu, sigma)

        # 复制 REF_SIGNATURE.json（可选）
        src_sig = os.path.join(in_fold, "REF_SIGNATURE.json")
        if os.path.exists(src_sig):
            dst_sig = os.path.join(out_fold, "REF_SIGNATURE.json")
            if not os.path.exists(dst_sig):
                try:
                    with open(src_sig, "r", encoding="utf-8") as f:
                        sig = json.load(f)
                    with open(dst_sig, "w", encoding="utf-8") as f:
                        json.dump(sig, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

        print(f"[FOLD_FUSED] built {out_fold}")

    print("[FOLD_FUSED] Done.")


# ============
# 主流程
# ============
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SEED-IV preprocessing pipeline for CMRD."
    )
    parser.add_argument("--base-path", type=str, default=BASE_PATH, help="Directory containing raw SEED-IV metadata and EEG files.")
    parser.add_argument("--raw-data", type=str, default=None, help="Override the raw eeg directory (defaults to <base-path>/eeg_raw_data).")
    parser.add_argument("--channel-xlsx", type=str, default=None, help="Override the channel order xlsx file.")
    parser.add_argument("--montage-locs", type=str, default=None, help="Override the channel location file.")
    parser.add_argument("--save-root", type=str, default=SAVE_ROOT, help="Directory used to save all generated artifacts.")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Gating strength for tanh(alpha * zde).")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4], choices=[1, 2, 3, 4], help="Pipeline steps to run.")
    parser.add_argument("--save-feat", dest="save_feat", action="store_true", help="Save the concatenated feat tensor.")
    parser.add_argument("--no-save-feat", dest="save_feat", action="store_false", help="Do not save the concatenated feat tensor.")
    parser.set_defaults(save_feat=SAVE_FEAT)
    return parser

def main():
    global RAW_DATA, SAVE_ROOT, CHANNEL_XLSX, MONTAGE_LOCS, ALPHA, SAVE_FEAT

    args = build_parser().parse_args()
    base_path = args.base_path
    RAW_DATA = args.raw_data or os.path.join(base_path, "eeg_raw_data")
    CHANNEL_XLSX = args.channel_xlsx or os.path.join(base_path, "Zehn-Channel Order.xlsx")
    MONTAGE_LOCS = args.montage_locs or os.path.join(base_path, "channel_62_pos.locs")
    SAVE_ROOT = args.save_root
    ALPHA = float(args.alpha)
    SAVE_FEAT = bool(args.save_feat)

    selected_steps = set(args.steps)

    print("[INFO] Configuration")
    print(f"[INFO] RAW_DATA={RAW_DATA}")
    print(f"[INFO] CHANNEL_XLSX={CHANNEL_XLSX}")
    print(f"[INFO] MONTAGE_LOCS={MONTAGE_LOCS}")
    print(f"[INFO] SAVE_ROOT={SAVE_ROOT}")
    print(f"[INFO] ALPHA={ALPHA}")
    print(f"[INFO] SAVE_FEAT={SAVE_FEAT}")
    print(f"[INFO] STEPS={sorted(selected_steps)}")

    print('[INFO] Loading channel/montage...')
    info_eeg, ch_names_eeg, frontal_idx, frontal_names = load_channel_info(
        CHANNEL_XLSX, MONTAGE_LOCS
    )
    print(f'[INFO] Channels={len(ch_names_eeg)} | Frontal={frontal_names}')

    if 1 in selected_steps:
        print('[STEP1] Building p_hist/quality/de per trial ...')
        build_phist_all(RAW_DATA, SAVE_ROOT, info_eeg, ch_names_eeg, frontal_idx, frontal_names)

    if 2 in selected_steps:
        print('[STEP2] Building per-subject/global accum ...')
        build_accum_all(SAVE_ROOT)

    if 3 in selected_steps:
        print('[STEP3] Emitting ALL LOOCV folds (JSD+DE) ...')
        build_all_folds_jsd_de(SAVE_ROOT)

    if 4 in selected_steps:
        print('[STEP4] Per-fold zDE stats + tanh-gated JSD + feat ...')
        build_all_folds_degate(SAVE_ROOT)

    print('[ALL DONE] ✅')

if __name__ == "__main__":
    main()
