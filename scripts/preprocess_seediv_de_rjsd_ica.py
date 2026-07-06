from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmrd.config import ExperimentConfig, load_config
from cmrd.data import iter_trials, subject_loso_split, validate_dataset
from cmrd.features.rd import fit_reference, transform_rd
from cmrd.io import read_json, write_json, write_npz


LOGGER = logging.getLogger("seediv.de_rjsd_ica")

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 31.0),
    "gamma": (31.0, 50.0),
}

SAMPLING_RATE = 200.0
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5
NOTCH_HZ = 50.0
BROAD_BAND_HZ = (1.0, 75.0)
HIST_BINS_PER_BAND = 32
MIN_SPECTRAL_NFFT = 512
EPS = 1e-12


def _spectral_nfft(window_seconds: float) -> int:
    """Keep the old 512-point FFT and grow to the next power of two when needed."""
    window_samples = int(round(window_seconds * SAMPLING_RATE))
    if window_samples <= 0:
        raise ValueError("window_seconds must produce at least one sample")
    return max(MIN_SPECTRAL_NFFT, 1 << (window_samples - 1).bit_length())


def _window_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "")


def _output_family(window_seconds: float, hop_seconds: float) -> str:
    return f"de_rjsd_ica_{_window_tag(window_seconds)}s_hop{_window_tag(hop_seconds)}"


def _signature_payload(args: argparse.Namespace, config: ExperimentConfig, channel_names: list[str]) -> dict[str, Any]:
    window_seconds = float(args.window_seconds)
    hop_seconds = float(args.hop_seconds)
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    spectral_nfft = _spectral_nfft(window_seconds)
    dataset_name = "SEED-IV" if config.dataset == "seediv" else "SEED"
    return {
        "schema_version": 1,
        "dataset": dataset_name,
        "raw_dir": str((config.data_root / config.raw["dataset"]["raw_dir"]).resolve()),
        "channels": int(config.raw["dataset"]["channels"]),
        "channel_names": channel_names,
        "sampling_rate": SAMPLING_RATE,
        "input_unit": "microvolt",
        "mne_internal_unit": "volt",
        "notch_hz": NOTCH_HZ,
        "ica_highpass_hz": BROAD_BAND_HZ[0],
        "final_bandpass_hz": list(BROAD_BAND_HZ),
        "ica": {
            "n_components": args.ica_n_components,
            "fallback_n_components": args.ica_fallback_components,
            "random_state": args.ica_seed,
            "max_iter": args.ica_max_iter,
            "decim": args.ica_decim,
            "find_bads_eog_channels": ["Fp1", "Fp2"],
            "find_bads_muscle": True,
            "strict_detection": args.strict_ica,
            "bad_channel_std_ratio": args.bad_channel_std_ratio,
        },
        "window_seconds": window_seconds,
        "hop_seconds": hop_seconds,
        "welch": {
            "window": "hann",
            "nperseg": int(round(SAMPLING_RATE * window_seconds)),
            "noverlap": 0,
            "nfft": spectral_nfft,
            "detrend": "constant",
            "scaling": "density",
        },
        "bands_hz": {name: list(limits) for name, limits in BANDS.items()},
        "hist_bins_per_band": HIST_BINS_PER_BAND,
        "de_definition": "log(sum(Welch PSD bins within band) + eps)",
        "rjsd_definition": "Jensen-Shannon divergence(P_window, Q_source_train)",
        "p_hist_storage_dtype": "float16",
        "de_storage_dtype": "float32",
        "fold_protocol": {
            "outer": "15-subject LOSO",
            "source_validation_subjects": int(config.raw["split"]["validation_subjects"]),
            "split_seed": int(config.raw["split"]["seed"]),
            "reference_source": "source_train_only",
        },
        "mne_version": mne.__version__,
    }


def _cleaning_signature_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Select only settings that can change the ICA-cleaned continuous signal."""
    keys = (
        "dataset",
        "raw_dir",
        "channels",
        "channel_names",
        "sampling_rate",
        "input_unit",
        "mne_internal_unit",
        "notch_hz",
        "ica_highpass_hz",
        "final_bandpass_hz",
        "ica",
        "mne_version",
        "official_upstream_preprocessing",
        "montage_source",
        "channel_order_verification",
    )
    return {
        "schema_version": 1,
        **{key: payload[key] for key in keys if key in payload},
    }


def _signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _score_summary(scores: np.ndarray | list[float]) -> dict[str, float | None]:
    value = np.asarray(scores, dtype=np.float64)
    value = value[np.isfinite(value)]
    if value.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {"min": float(value.min()), "max": float(value.max()), "mean": float(value.mean())}


def clean_signal_with_mne(
    signal_microvolt: np.ndarray,
    channel_names: list[str],
    montage: mne.channels.DigMontage,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Notch -> 1 Hz high-pass -> ICA artifact rejection -> 1-75 Hz final band-pass."""
    if signal_microvolt.shape[0] != len(channel_names):
        raise ValueError(f"Signal channels {signal_microvolt.shape[0]} != montage channels {len(channel_names)}")

    # MNE requires EEG in volts. Official SEED/SEED-IV arrays are represented in microvolts.
    signal_volt = np.asarray(signal_microvolt, dtype=np.float64) * 1e-6
    info = mne.create_info(channel_names, sfreq=SAMPLING_RATE, ch_types=["eeg"] * len(channel_names))
    raw = mne.io.RawArray(signal_volt, info, verbose="ERROR")
    raw.set_montage(montage, on_missing="raise", verbose="ERROR")
    raw.notch_filter(freqs=[NOTCH_HZ], picks="eeg", phase="zero", verbose="ERROR")
    # ICA should be fitted to high-pass-filtered data. The final low-pass is applied after ICA.
    raw.filter(l_freq=BROAD_BAND_HZ[0], h_freq=None, picks="eeg", phase="zero", verbose="ERROR")

    interpolated_bad_channels: list[str] = []
    channel_std_before_interpolation = np.std(raw.get_data(), axis=1)
    if args.bad_channel_std_ratio is not None:
        ratio = float(args.bad_channel_std_ratio)
        if ratio <= 1.0:
            raise ValueError("bad_channel_std_ratio must be > 1 when enabled")
        median_std = float(np.median(channel_std_before_interpolation))
        lower = median_std / ratio
        upper = median_std * ratio
        interpolated_bad_channels = [
            name
            for name, std in zip(channel_names, channel_std_before_interpolation, strict=True)
            if float(std) < lower or float(std) > upper
        ]
        if interpolated_bad_channels:
            raw.info["bads"] = interpolated_bad_channels
            raw.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")

    fit_errors: list[str] = []
    requested_components: float | int = args.ica_n_components
    if isinstance(requested_components, float) and requested_components >= 1.0 and requested_components.is_integer():
        requested_components = int(requested_components)
    try:
        ica = mne.preprocessing.ICA(
            n_components=requested_components,
            random_state=args.ica_seed,
            max_iter=args.ica_max_iter,
            method="fastica",
        )
        ica.fit(raw, picks="eeg", decim=args.ica_decim, reject_by_annotation=False, verbose="ERROR")
    except Exception as exc:
        fit_errors.append(f"primary ICA failed: {type(exc).__name__}: {exc}")
        fallback = min(int(args.ica_fallback_components), len(channel_names) - 1)
        ica = mne.preprocessing.ICA(
            n_components=fallback,
            random_state=args.ica_seed,
            max_iter=args.ica_max_iter,
            method="fastica",
        )
        ica.fit(raw, picks="eeg", decim=args.ica_decim, reject_by_annotation=False, verbose="ERROR")

    detection_errors: list[str] = []
    eog_indices: list[int] = []
    muscle_indices: list[int] = []
    eog_scores: np.ndarray | list[float] = []
    muscle_scores: np.ndarray | list[float] = []

    try:
        found, scores = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"], verbose="ERROR")
        eog_indices = sorted(set(map(int, found)))
        eog_scores = np.asarray(scores)
    except Exception as exc:
        detection_errors.append(f"find_bads_eog failed: {type(exc).__name__}: {exc}")

    try:
        found, scores = ica.find_bads_muscle(raw, verbose="ERROR")
        muscle_indices = sorted(set(map(int, found)))
        muscle_scores = np.asarray(scores)
    except Exception as exc:
        detection_errors.append(f"find_bads_muscle failed: {type(exc).__name__}: {exc}")

    if detection_errors and args.strict_ica:
        raise RuntimeError("; ".join(detection_errors))

    excluded = sorted(set(eog_indices + muscle_indices))
    ica.exclude = excluded
    if excluded:
        ica.apply(raw, exclude=excluded, verbose="ERROR")
    raw.filter(
        l_freq=BROAD_BAND_HZ[0],
        h_freq=BROAD_BAND_HZ[1],
        picks="eeg",
        phase="zero",
        verbose="ERROR",
    )
    cleaned_microvolt = np.asarray(raw.get_data() * 1e6, dtype=np.float32)
    if not np.isfinite(cleaned_microvolt).all():
        raise FloatingPointError("MNE cleaning produced NaN or infinite values")

    metadata = {
        "requested_n_components": requested_components,
        "fitted_n_components": int(ica.n_components_),
        "excluded_components": excluded,
        "eog_components": eog_indices,
        "muscle_components": muscle_indices,
        "eog_score_summary": _score_summary(eog_scores),
        "muscle_score_summary": _score_summary(muscle_scores),
        "fit_errors": fit_errors,
        "detection_errors": detection_errors,
        "interpolated_bad_channels": interpolated_bad_channels,
        "channel_std_median_before_interpolation_microvolt": float(np.median(channel_std_before_interpolation) * 1e6),
        "channel_std_max_before_interpolation_microvolt": float(np.max(channel_std_before_interpolation) * 1e6),
        "input_std_microvolt": float(np.std(signal_microvolt)),
        "cleaned_std_microvolt": float(np.std(cleaned_microvolt)),
    }
    return cleaned_microvolt, metadata


def extract_de_and_phist(
    signal: np.ndarray,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    window_size = int(round(window_seconds * SAMPLING_RATE))
    hop_size = int(round(hop_seconds * SAMPLING_RATE))
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    spectral_nfft = _spectral_nfft(window_seconds)
    starts = np.arange(0, signal.shape[-1] - window_size + 1, hop_size, dtype=np.int64)
    if starts.size == 0:
        raise ValueError(f"Signal with {signal.shape[-1]} samples is shorter than one window")

    channels = signal.shape[0]
    de = np.empty((starts.size, channels, len(BANDS)), dtype=np.float32)
    p_hist = np.zeros((starts.size, channels, len(BANDS), HIST_BINS_PER_BAND), dtype=np.float32)

    for window_index, start in enumerate(starts):
        clip = signal[:, start : start + window_size]
        frequencies, psd = welch(
            clip,
            fs=SAMPLING_RATE,
            window="hann",
            nperseg=window_size,
            noverlap=0,
            nfft=spectral_nfft,
            detrend="constant",
            scaling="density",
            axis=-1,
        )
        for band_index, (name, (low, high)) in enumerate(BANDS.items()):
            selected = (frequencies >= low) & (frequencies < high)
            if not np.any(selected):
                raise ValueError(f"No Welch frequencies in {name}={low}-{high} Hz")
            band_frequencies = frequencies[selected]
            band_psd = psd[:, selected]

            # Preserve the old log-bandpower definition used for zDE gating.
            de[window_index, :, band_index] = np.log(band_psd.sum(axis=-1) + EPS)

            bin_indices = np.floor(
                (band_frequencies - low) * HIST_BINS_PER_BAND / (high - low)
            ).astype(np.int64)
            bin_indices = np.clip(bin_indices, 0, HIST_BINS_PER_BAND - 1)
            histogram = p_hist[window_index, :, band_index]
            for bin_index in np.unique(bin_indices):
                histogram[:, bin_index] = band_psd[:, bin_indices == bin_index].sum(axis=-1)
            denominator = histogram.sum(axis=-1, keepdims=True)
            if np.any(denominator <= EPS):
                raise FloatingPointError(f"Near-zero {name} power at window {window_index}")
            histogram /= denominator

    if not np.isfinite(de).all() or not np.isfinite(p_hist).all():
        raise FloatingPointError("DE/p_hist extraction produced non-finite values")
    return de, p_hist


def _valid_cleaned_file(path: Path, cleaning_signature: str, channels: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            cleaned = archive["cleaned"]
            return (
                str(archive["cleaning_signature"].item()) == cleaning_signature
                and "ica_metadata_json" in archive
                and cleaned.ndim == 2
                and cleaned.shape[0] == channels
                and cleaned.shape[1] > 0
            )
    except (KeyError, OSError, ValueError):
        return False


def _load_or_build_cleaned_signal(
    record: Any,
    cache_root: Path,
    cleaning_signature: str,
    channel_names: list[str],
    montage: mne.channels.DigMontage,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], bool]:
    signal_path = cache_root / "trials" / f"{record.trial_id}.npz"
    metadata_path = cache_root / "trial_metadata" / f"{record.trial_id}.json"
    reusable = (
        args.reuse_ica_cache
        and not args.force_ica
        and _valid_cleaned_file(signal_path, cleaning_signature, len(channel_names))
    )
    if reusable:
        with np.load(signal_path, allow_pickle=False) as archive:
            cleaned = np.asarray(archive["cleaned"], dtype=np.float32)
            ica_metadata = json.loads(str(archive["ica_metadata_json"].item()))
        if not np.isfinite(cleaned).all():
            raise FloatingPointError(f"Cached ICA signal contains NaN or infinite values: {signal_path}")
        return cleaned, ica_metadata, True

    cleaned, ica_metadata = clean_signal_with_mne(record.signal, channel_names, montage, args)
    write_npz(
        signal_path,
        cleaned=cleaned.astype(np.float32),
        label=np.int64(record.label),
        subject=np.int64(record.subject),
        session=np.int64(record.session),
        trial=np.int64(record.trial),
        cleaning_signature=np.asarray(cleaning_signature),
        ica_metadata_json=np.asarray(json.dumps(ica_metadata, sort_keys=True, separators=(",", ":"))),
    )
    write_json(
        metadata_path,
        {
            "trial_id": record.trial_id,
            "source_file": record.source_file,
            "source_key": record.source_key,
            "cleaned_shape": list(cleaned.shape),
            "cleaning_signature": cleaning_signature,
            "ica": ica_metadata,
        },
    )
    return cleaned, ica_metadata, False


def _valid_trial_file(path: Path, signature: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            stored = str(archive["preprocessing_signature"].item())
            return (
                stored == signature
                and archive["de"].ndim == 3
                and archive["p_hist"].ndim == 4
                and archive["de"].shape[:2] == archive["p_hist"].shape[:2]
            )
    except (KeyError, OSError, ValueError):
        return False


def _entry_from_files(root: Path, record: Any, source_index: int, signature: str) -> dict[str, Any]:
    relative_npz = f"trials/{record.trial_id}.npz"
    relative_json = f"trial_metadata/{record.trial_id}.json"
    with np.load(root / relative_npz, allow_pickle=False) as archive:
        de_shape = list(archive["de"].shape)
        phist_shape = list(archive["p_hist"].shape)
    metadata = read_json(root / relative_json)
    return {
        "trial_id": record.trial_id,
        "path": relative_npz,
        "metadata_path": relative_json,
        "label": int(record.label),
        "subject": int(record.subject),
        "session": int(record.session),
        "trial": int(record.trial),
        "source_index": int(source_index),
        "source_file": record.source_file,
        "source_key": record.source_key,
        "de_shape": de_shape,
        "p_hist_shape": phist_shape,
        "preprocessing_signature": signature,
        "ica": metadata["ica"],
    }


def build_trial_features(
    config: ExperimentConfig,
    root: Path,
    signature: str,
    signature_payload: dict[str, Any],
    ica_cache_root: Path,
    cleaning_signature: str,
    montage: mne.channels.DigMontage,
    channel_names: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    dataset_manifest = validate_dataset(config)
    expected_trials = int(dataset_manifest["expected_trials"])
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()

    for source_index, record in enumerate(iter_trials(config)):
        if args.max_trials is not None and len(entries) >= args.max_trials:
            break
        trial_path = root / "trials" / f"{record.trial_id}.npz"
        metadata_path = root / "trial_metadata" / f"{record.trial_id}.json"
        try:
            reusable = args.resume and not args.force and _valid_trial_file(trial_path, signature) and metadata_path.is_file()
            if not reusable:
                cleaned, ica_metadata, reused_cleaned = _load_or_build_cleaned_signal(
                    record,
                    ica_cache_root,
                    cleaning_signature,
                    channel_names,
                    montage,
                    args,
                )
                if reused_cleaned:
                    LOGGER.info("Reusing ICA-cleaned signal for %s", record.trial_id)
                else:
                    LOGGER.info("Cached newly ICA-cleaned signal for %s", record.trial_id)
                de, p_hist = extract_de_and_phist(
                    cleaned,
                    window_seconds=float(args.window_seconds),
                    hop_seconds=float(args.hop_seconds),
                )
                write_npz(
                    trial_path,
                    de=de.astype(np.float32),
                    p_hist=p_hist.astype(np.float16),
                    label=np.int64(record.label),
                    subject=np.int64(record.subject),
                    session=np.int64(record.session),
                    trial=np.int64(record.trial),
                    source_index=np.int64(source_index),
                    preprocessing_signature=np.asarray(signature),
                )
                write_json(
                    metadata_path,
                    {
                        "trial_id": record.trial_id,
                        "source_file": record.source_file,
                        "source_key": record.source_key,
                        "ica": ica_metadata,
                        "ica_cache_path": str(ica_cache_root / "trials" / f"{record.trial_id}.npz"),
                        "cleaning_signature": cleaning_signature,
                        "de_shape": list(de.shape),
                        "p_hist_shape": list(p_hist.shape),
                    },
                )
            entries.append(_entry_from_files(root, record, source_index, signature))
        except Exception as exc:
            failure = {
                "trial_id": record.trial_id,
                "source_file": record.source_file,
                "source_key": record.source_key,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            write_json(root / "failures.json", failures)
            LOGGER.exception("Failed %s", record.trial_id)
            if not args.continue_on_error:
                raise

        if len(entries) % 10 == 0:
            write_json(
                root / "trials_manifest.partial.json",
                {
                    "complete": False,
                    "expected_trials": expected_trials,
                    "processed_trials": len(entries),
                    "failures": failures,
                    "preprocessing_signature": signature,
                    "trials": entries,
                },
            )
            LOGGER.info("Processed %d/%d trial features", len(entries), expected_trials)

    complete = len(entries) == expected_trials and not failures
    manifest = {
        "schema_version": 1,
        "complete": complete,
        "expected_trials": expected_trials,
        "processed_trials": len(entries),
        "failures": failures,
        "preprocessing_signature": signature,
        "signature_payload": signature_payload,
        "dataset_manifest": dataset_manifest,
        "elapsed_seconds": time.perf_counter() - started,
        "trials": entries,
    }
    write_json(root / "trials_manifest.json", manifest)
    LOGGER.info("Trial stage complete=%s processed=%d failures=%d", complete, len(entries), len(failures))
    return manifest


def _load_histograms(root: Path, entries: Iterable[dict[str, Any]]) -> Iterable[np.ndarray]:
    for entry in entries:
        with np.load(root / entry["path"], allow_pickle=False) as archive:
            yield np.asarray(archive["p_hist"], dtype=np.float32)


def _valid_rjsd_file(path: Path, signature: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return str(archive["preprocessing_signature"].item()) == signature and archive["rjsd"].ndim == 3
    except (KeyError, OSError, ValueError):
        return False


def build_one_fold(
    config: ExperimentConfig,
    root: Path,
    trials: list[dict[str, Any]],
    target_subject: int,
    signature: str,
    args: argparse.Namespace,
) -> Path:
    fold_root = root / "folds" / f"fold-{target_subject:02d}"
    manifest_path = fold_root / "manifest.json"
    if args.resume and not args.force and manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("complete") and existing.get("preprocessing_signature") == signature:
            LOGGER.info("Reusing complete fold %02d", target_subject)
            return fold_root

    subjects = np.asarray([entry["subject"] for entry in trials], dtype=np.int64)
    split = subject_loso_split(
        subjects,
        target_subject,
        int(config.raw["split"]["validation_subjects"]),
        int(config.raw["split"]["seed"]),
    )
    groups = {
        "train": [entry for entry in trials if int(entry["subject"]) in split.train_subjects],
        "validation": [entry for entry in trials if int(entry["subject"]) in split.validation_subjects],
        "test": [entry for entry in trials if int(entry["subject"]) == target_subject],
    }
    reference, reference_window_count = fit_reference(_load_histograms(root, groups["train"]))
    train_indices = np.asarray([entry["source_index"] for entry in groups["train"]], dtype=np.int64)
    write_npz(
        fold_root / "rjsd_reference.npz",
        Q=reference.astype(np.float32),
        source_train_subjects=np.asarray(split.train_subjects, dtype=np.int64),
        source_train_indices=train_indices,
        preprocessing_signature=np.asarray(signature),
    )

    output_groups: dict[str, list[dict[str, Any]]] = {name: [] for name in groups}
    for group_name, entries in groups.items():
        for index, entry in enumerate(entries, 1):
            relative_rjsd = f"folds/fold-{target_subject:02d}/rjsd/{entry['trial_id']}.npz"
            destination = root / relative_rjsd
            if not (args.resume and not args.force and _valid_rjsd_file(destination, signature)):
                with np.load(root / entry["path"], allow_pickle=False) as archive:
                    histogram = np.asarray(archive["p_hist"], dtype=np.float32)
                flat = transform_rd(histogram, reference)
                rjsd = flat.reshape(flat.shape[0], int(config.raw["dataset"]["channels"]), len(BANDS))
                write_npz(
                    destination,
                    rjsd=rjsd.astype(np.float32),
                    label=np.int64(entry["label"]),
                    subject=np.int64(entry["subject"]),
                    session=np.int64(entry["session"]),
                    trial=np.int64(entry["trial"]),
                    source_index=np.int64(entry["source_index"]),
                    preprocessing_signature=np.asarray(signature),
                )
            output_groups[group_name].append(
                {
                    "trial_id": entry["trial_id"],
                    "label": entry["label"],
                    "subject": entry["subject"],
                    "session": entry["session"],
                    "trial": entry["trial"],
                    "source_index": entry["source_index"],
                    "de_phist_path": entry["path"],
                    "rjsd_path": relative_rjsd,
                }
            )
            if index % 100 == 0:
                LOGGER.info("Fold %02d %s transformed %d/%d", target_subject, group_name, index, len(entries))

    provenance = {
        "target_subject": target_subject,
        "train_subjects": list(split.train_subjects),
        "validation_subjects": list(split.validation_subjects),
        "reference_source": "source_train_only",
        "reference_window_count": int(reference_window_count),
        "reference_shape": list(reference.shape),
        "source_train_trial_indices": train_indices.tolist(),
        "source_validation_trial_indices": [entry["source_index"] for entry in groups["validation"]],
        "target_test_trial_indices": [entry["source_index"] for entry in groups["test"]],
        "preprocessing_signature": signature,
    }
    write_json(fold_root / "provenance.json", provenance)
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "complete": True,
            "dataset": "SEED-IV" if config.dataset == "seediv" else "SEED",
            "features": ["de", "rjsd"],
            "target_subject": target_subject,
            "preprocessing_signature": signature,
            "feature_shapes": {"de": "[T,62,5]", "rjsd": "[T,62,5]"},
            "split": split.as_dict(),
            "provenance": provenance,
            "groups": output_groups,
        },
    )
    LOGGER.info(
        "Completed fold %02d train=%d validation=%d test=%d Q_windows=%d",
        target_subject,
        len(groups["train"]),
        len(groups["validation"]),
        len(groups["test"]),
        reference_window_count,
    )
    return fold_root


def build_folds(
    config: ExperimentConfig,
    root: Path,
    signature: str,
    args: argparse.Namespace,
) -> None:
    manifest_path = root / "trials_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing trial manifest: {manifest_path}. Run --stage trials first.")
    manifest = read_json(manifest_path)
    if manifest.get("preprocessing_signature") != signature:
        raise ValueError("Trial manifest signature differs from the active preprocessing settings")
    if not manifest.get("complete"):
        raise RuntimeError("Trial feature stage is incomplete; folds cannot be built safely")
    trials = list(manifest["trials"])
    targets = [args.fold] if args.fold is not None else list(range(1, 16))
    for target in targets:
        if not 1 <= int(target) <= 15:
            raise ValueError(f"Fold must be in 1..15, got {target}")
        build_one_fold(config, root, trials, int(target), signature, args)
    complete_folds = sorted(path.name for path in (root / "folds").glob("fold-*") if (path / "manifest.json").is_file())
    write_json(
        root / "pipeline_manifest.json",
        {
            "schema_version": 1,
            "dataset": "SEED-IV" if config.dataset == "seediv" else "SEED",
            "features": ["de", "rjsd"],
            "preprocessing_signature": signature,
            "trial_stage_complete": True,
            "complete_folds": complete_folds,
            "all_15_folds_complete": len(complete_folds) == 15,
        },
    )


def build_parser(
    default_window_seconds: float = WINDOW_SECONDS,
    default_hop_seconds: float = HOP_SECONDS,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SEED-IV DE+RJSD preprocessing with reusable ICA-cleaned time-series, 1-75 Hz band-pass, and strict fold provenance."
    )
    parser.add_argument("--config", default="configs/seediv/rd.yaml")
    parser.add_argument("--output-root", default=None, help="Parent output directory; a signature subdirectory is added")
    parser.add_argument("--channel-locs", default=None, help="62-channel .locs file; defaults to the SEED-IV data directory")
    parser.add_argument("--stage", choices=("all", "trials", "folds"), default="all")
    parser.add_argument("--fold", type=int, help="Build one RJSD fold during the folds stage")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Recompute existing matching trial/fold files")
    parser.add_argument("--max-trials", type=int, default=None, help="Diagnostic trial-stage limit; produces an incomplete manifest")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failed trials and continue; folds remain blocked")
    parser.add_argument("--window-seconds", type=float, default=default_window_seconds)
    parser.add_argument("--hop-seconds", type=float, default=default_hop_seconds)
    parser.add_argument(
        "--ica-cache-root",
        default=None,
        help="ICA cache parent; defaults to <processed>/<dataset>/ica_cleaned (a cleaning signature is added)",
    )
    parser.add_argument(
        "--reuse-ica-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse matching ICA-cleaned continuous signals when available",
    )
    parser.add_argument("--force-ica", action="store_true", help="Recompute ICA even when a matching time-series cache exists")
    parser.add_argument("--strict-ica", action="store_true", help="Abort if EOG or muscle component detection fails")
    parser.add_argument(
        "--bad-channel-std-ratio",
        type=float,
        default=None,
        help="Interpolate channels whose post-high-pass std is outside median/ratio..median*ratio; disabled by default",
    )
    parser.add_argument("--ica-n-components", type=float, default=0.999)
    parser.add_argument("--ica-fallback-components", type=int, default=20)
    parser.add_argument("--ica-seed", type=int, default=97)
    parser.add_argument("--ica-max-iter", type=int, default=1000)
    parser.add_argument("--ica-decim", type=int, default=1)
    parser.add_argument("--mne-log-level", default="WARNING", choices=("ERROR", "WARNING", "INFO"))
    return parser


def main(
    default_window_seconds: float = WINDOW_SECONDS,
    default_hop_seconds: float = HOP_SECONDS,
) -> None:
    args = build_parser(default_window_seconds, default_hop_seconds).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    mne.set_log_level(args.mne_log_level)
    config = load_config(args.config, expected_feature="rd")
    if config.dataset != "seediv":
        raise ValueError("This script only supports SEED-IV")

    locs_path = Path(args.channel_locs).expanduser() if args.channel_locs else config.data_root / "Ori" / "SEED-IV" / "channel_62_pos.locs"
    locs_path = locs_path.resolve()
    if not locs_path.is_file():
        raise FileNotFoundError(f"Channel montage does not exist: {locs_path}")
    montage = mne.channels.read_custom_montage(locs_path)
    channel_names = list(montage.ch_names)
    if len(channel_names) != int(config.raw["dataset"]["channels"]):
        raise ValueError(f"Montage contains {len(channel_names)} channels, expected 62")
    if not {"Fp1", "Fp2"}.issubset(channel_names):
        raise ValueError("Montage must contain Fp1 and Fp2 for find_bads_eog")

    payload = _signature_payload(args, config, channel_names)
    signature = _signature(payload)
    cleaning_payload = _cleaning_signature_payload(payload)
    cleaning_signature = _signature(cleaning_payload)
    output_parent = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else config.processed_root / "seediv" / _output_family(args.window_seconds, args.hop_seconds)
    )
    output_root = output_parent / signature
    output_root.mkdir(parents=True, exist_ok=True)
    ica_cache_parent = (
        Path(args.ica_cache_root).expanduser().resolve()
        if args.ica_cache_root
        else config.processed_root / "seediv" / "ica_cleaned"
    )
    ica_cache_root = ica_cache_parent / cleaning_signature
    ica_cache_root.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_root / "preprocessing.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(file_handler)

    write_json(
        output_root / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "mne": mne.__version__,
            "command": sys.argv,
            "channel_locs": str(locs_path),
            "preprocessing_signature": signature,
            "signature_payload": payload,
            "cleaning_signature": cleaning_signature,
            "cleaning_signature_payload": cleaning_payload,
            "ica_cache_root": str(ica_cache_root),
        },
    )
    write_json(
        ica_cache_root / "cache_manifest.json",
        {
            "schema_version": 1,
            "dataset": "SEED-IV",
            "cleaning_signature": cleaning_signature,
            "cleaning_signature_payload": cleaning_payload,
            "signal_shape": "[62,samples]",
            "signal_unit": "microvolt",
            "storage_dtype": "float32",
        },
    )
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Preprocessing signature: %s", signature)
    LOGGER.info("ICA time-series cache: %s", ica_cache_root)

    if args.stage in {"all", "trials"}:
        build_trial_features(
            config,
            output_root,
            signature,
            payload,
            ica_cache_root,
            cleaning_signature,
            montage,
            channel_names,
            args,
        )
    if args.stage in {"all", "folds"}:
        if args.max_trials is not None:
            raise ValueError("--max-trials is diagnostic-only and cannot be combined with the folds stage")
        build_folds(config, output_root, signature, args)
    LOGGER.info("Finished requested stage=%s", args.stage)


if __name__ == "__main__":
    main()
