"""DEAP original-BDF preprocessing aligned with the formal CMRD 1 s/1 s protocol."""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmrd.config import load_config
from cmrd.data.deap import DEAP_EEG_CHANNELS, DEAP_EOG_CHANNELS
from cmrd.io import write_json

from preprocess_seediv_de_rjsd_ica import (
    WINDOW_SECONDS,
    _cleaning_signature_payload,
    _output_family,
    _signature,
    _signature_payload,
    build_folds,
    build_parser as _shared_parser,
    build_trial_features,
)


LOGGER = logging.getLogger("deap.de_rjsd_ica")


def build_parser(default_window_seconds: float = WINDOW_SECONDS, default_hop_seconds: float = 1.0):
    parser = _shared_parser(default_window_seconds, default_hop_seconds)
    parser.description = (
        "DEAP DE+RJSD preprocessing from original 512 Hz BDF: status-event trial cuts, "
        "200 Hz resampling, 50 Hz notch, EOG-informed ICA, final 1-75 Hz filtering, "
        "non-overlapping windows, and source-training-only fold references."
    )
    parser.set_defaults(config="configs/deap/rd.yaml")
    parser.set_defaults(bad_channel_std_ratio=10.0)
    for action in parser._actions:
        if action.dest == "channel_locs":
            action.help = "Unused for DEAP; the official MNE biosemi32 montage is required"
    return parser


def main(default_window_seconds: float = WINDOW_SECONDS, default_hop_seconds: float = 1.0) -> None:
    args = build_parser(default_window_seconds, default_hop_seconds).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    mne.set_log_level(args.mne_log_level)
    config = load_config(args.config, expected_feature="rd")
    if config.dataset != "deap":
        raise ValueError("This script only supports DEAP")
    if args.channel_locs:
        raise ValueError("DEAP uses the fixed official biosemi32 montage; --channel-locs is not supported")

    montage = mne.channels.make_standard_montage("biosemi32")
    channel_names = list(montage.ch_names)
    if tuple(channel_names) != DEAP_EEG_CHANNELS:
        raise ValueError("Installed MNE biosemi32 order differs from the official DEAP EEG order")

    raw_archive = (config.data_root / str(config.raw["dataset"]["raw_dir"])).resolve()
    ratings_archive = (config.data_root / str(config.raw["dataset"]["label_file"])).resolve()
    payload = _signature_payload(args, config, channel_names)
    payload.update(
        {
            "eog_channels": list(DEAP_EOG_CHANNELS),
            "montage_source": "mne.channels.make_standard_montage('biosemi32')",
            "label_target": str(config.raw["dataset"]["label_target"]).lower(),
            "label_threshold": 5.0,
            "label_source": f"{ratings_archive}::participant_ratings.csv",
            "raw_archive": {
                "path": str(raw_archive),
                "size": raw_archive.stat().st_size,
                "mtime_ns": raw_archive.stat().st_mtime_ns,
            },
            "ratings_archive": {
                "path": str(ratings_archive),
                "size": ratings_archive.stat().st_size,
                "mtime_ns": ratings_archive.stat().st_mtime_ns,
            },
            "official_upstream_preprocessing": {
                "original_sampling_rate_hz": 512,
                "resampled_to_hz": 200,
                "source_format": "BioSemi BDF",
                "trial_start_status_code": 4,
                "trial_end_status_code": 5,
                "stimulus_seconds_used": 60,
                "pre_trial_baseline_excluded": True,
                "prior_eog_removal": False,
                "prior_bandpass_filter": False,
                "eeg_channels": 32,
                "auxiliary_eog_channels_for_ica": list(DEAP_EOG_CHANNELS),
            },
        }
    )
    signature = _signature(payload)
    cleaning_payload = _cleaning_signature_payload(payload)
    cleaning_signature = _signature(cleaning_payload)

    output_parent = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else config.processed_root / "deap" / _output_family(args.window_seconds, args.hop_seconds)
    )
    output_root = output_parent / signature
    output_root.mkdir(parents=True, exist_ok=True)
    ica_cache_parent = (
        Path(args.ica_cache_root).expanduser().resolve()
        if args.ica_cache_root
        else config.processed_root / "deap" / "ica_cleaned"
    )
    ica_cache_root = ica_cache_parent / cleaning_signature
    ica_cache_root.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(output_root / "preprocessing.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(file_handler)
    logging.getLogger("seediv.de_rjsd_ica").addHandler(file_handler)

    write_json(
        output_root / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "mne": mne.__version__,
            "command": sys.argv,
            "channel_montage": "biosemi32",
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
            "dataset": "DEAP",
            "cleaning_signature": cleaning_signature,
            "cleaning_signature_payload": cleaning_payload,
            "signal_shape": "[32,samples]",
            "signal_unit": "microvolt",
            "storage_dtype": "float32",
        },
    )
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Preprocessing signature: %s", signature)
    LOGGER.info("ICA time-series cache: %s", ica_cache_root)
    LOGGER.info("Using DEAP original BDF stimulus intervals at 512 Hz, resampled to 200 Hz")

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
