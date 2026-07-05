from __future__ import annotations

import hashlib
import json
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
from cmrd.io import write_json

from preprocess_seediv_de_rjsd_ica import (
    _signature,
    _signature_payload,
    build_folds,
    build_parser as _shared_parser,
    build_trial_features,
)


LOGGER = logging.getLogger("seed.de_rjsd_ica")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_montage_and_verify_order(config, supplied: str | None) -> tuple[Path, dict[str, str]]:
    seed_root = config.data_root / "Ori" / "SEED"
    seed_order = seed_root / "channel-order.xlsx"
    seediv_root = config.data_root / "Ori" / "SEED-IV"
    seediv_order = seediv_root / "Channel Order.xlsx"

    if supplied:
        locs_path = Path(supplied).expanduser().resolve()
    else:
        own_locs = seed_root / "channel_62_pos.locs"
        locs_path = own_locs if own_locs.is_file() else seediv_root / "channel_62_pos.locs"
        locs_path = locs_path.resolve()
    if not locs_path.is_file():
        raise FileNotFoundError(
            "SEED has no channel_62_pos.locs and the SEED-IV fallback is unavailable; pass --channel-locs"
        )
    if not seed_order.is_file():
        raise FileNotFoundError(f"SEED channel order is missing: {seed_order}")

    hashes = {"seed_channel_order_sha256": _sha256(seed_order)}
    if seediv_order.is_file():
        hashes["seediv_channel_order_sha256"] = _sha256(seediv_order)
        if not supplied and hashes["seed_channel_order_sha256"] != hashes["seediv_channel_order_sha256"]:
            raise ValueError(
                "SEED and SEED-IV channel orders differ; refusing to reuse the SEED-IV montage automatically"
            )
    return locs_path, hashes


def build_parser():
    parser = _shared_parser()
    parser.description = (
        "SEED DE+RJSD preprocessing from official Preprocessed_EEG: preserve the official 200 Hz/trial cuts, "
        "then add 50 Hz notch, ICA artifact removal, final 1-75 Hz filtering, and strict fold provenance."
    )
    parser.set_defaults(config="configs/seed/rd.yaml")
    parser.set_defaults(bad_channel_std_ratio=10.0)
    for action in parser._actions:
        if action.dest == "channel_locs":
            action.help = "62-channel .locs file; defaults to the verified SEED-IV montage when orders match"
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    mne.set_log_level(args.mne_log_level)
    config = load_config(args.config, expected_feature="rd")
    if config.dataset != "seed":
        raise ValueError("This script only supports SEED")

    locs_path, order_hashes = _resolve_montage_and_verify_order(config, args.channel_locs)
    montage = mne.channels.read_custom_montage(locs_path)
    channel_names = list(montage.ch_names)
    if len(channel_names) != int(config.raw["dataset"]["channels"]):
        raise ValueError(f"Montage contains {len(channel_names)} channels, expected 62")
    if not {"Fp1", "Fp2"}.issubset(channel_names):
        raise ValueError("Montage must contain Fp1 and Fp2 for find_bads_eog")

    payload = _signature_payload(args, config, channel_names)
    payload["official_upstream_preprocessing"] = {
        "already_downsampled_to_hz": 200,
        "already_bandpass_filtered_hz": [0, 75],
        "already_segmented_by_movie_trial": True,
        "repeat_resampling": False,
        "repeat_official_0_75_filter_before_ica": False,
    }
    payload["montage_source"] = str(locs_path)
    payload["channel_order_verification"] = order_hashes
    signature = _signature(payload)

    output_parent = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else config.processed_root / "seed" / "de_rjsd_ica_1s_hop05"
    )
    output_root = output_parent / signature
    output_root.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_root / "preprocessing.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(file_handler)
    # Shared functions log under their own logger; attach the same persistent handler.
    logging.getLogger("seediv.de_rjsd_ica").addHandler(file_handler)

    write_json(
        output_root / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "mne": mne.__version__,
            "command": sys.argv,
            "channel_locs": str(locs_path),
            "channel_order_verification": order_hashes,
            "preprocessing_signature": signature,
            "signature_payload": payload,
        },
    )
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Preprocessing signature: %s", signature)
    LOGGER.info("Using official SEED 200 Hz, 0-75 Hz, movie-trial input without resampling")

    if args.stage in {"all", "trials"}:
        build_trial_features(config, output_root, signature, payload, montage, channel_names, args)
    if args.stage in {"all", "folds"}:
        if args.max_trials is not None:
            raise ValueError("--max-trials is diagnostic-only and cannot be combined with the folds stage")
        build_folds(config, output_root, signature, args)
    LOGGER.info("Finished requested stage=%s", args.stage)


if __name__ == "__main__":
    main()
