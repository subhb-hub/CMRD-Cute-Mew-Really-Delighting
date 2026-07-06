"""Train the SEED-IV models on the ICA-cleaned 4 s window / 1 s hop cache."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import train_seediv_de_rjsd_ica as pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PARENT = (
    ROOT.parent / "Dataset" / "Processed" / "CMRD" / "seediv" / "de_rjsd_ica_4s_hop1"
)
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "seediv" / "de_rjsd_ica_4s_hop1"
ALL_SOURCE_STATS_ROOT = (
    ROOT / "runs" / "diagnostics" / "seediv_feature_tuning_4s_hop1" / "_all_source_statistics"
)


def __getattr__(name: str):
    """Expose the shared training helpers to notebooks without duplicating them."""
    return getattr(pipeline, name)


def _resolve_data_root(value: str | None) -> Path:
    requested = str(DEFAULT_DATA_PARENT) if value is None else value
    return pipeline._resolve_data_root(requested)


def build_parser():
    parser = pipeline.build_parser()
    parser.description = (
        "Train SEED-IV LOSO on ICA-cleaned DE/RJSD features extracted with "
        "4-second windows and a 1-second hop."
    )
    parser.set_defaults(data_root=str(DEFAULT_DATA_PARENT), output_root=str(DEFAULT_OUTPUT_ROOT))
    for action in parser._actions:
        if action.dest == "data_root":
            action.help = (
                "4s/1s signature cache or parent "
                f"(default: {DEFAULT_DATA_PARENT})"
            )
        elif action.dest == "output_root":
            action.help = f"Training output parent (default: {DEFAULT_OUTPUT_ROOT})"
    return parser


def main() -> None:
    args = pipeline.parse_args_with_config(parser=build_parser())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    root = _resolve_data_root(args.data_root)
    pipeline.LOGGER.info("4s/1s data root: %s", root)
    validation = pipeline.validate_cache(root, deep=args.deep_validate)
    pipeline.LOGGER.info("Selected training feature: %s", args.feature.upper())
    pipeline.LOGGER.info(
        "Cache valid: trials=%d folds=%d labels=%s windows=%d..%d",
        validation["trials"],
        validation["folds"],
        validation["labels"],
        validation["windows"]["min"],
        validation["windows"]["max"],
    )
    pipeline.LOGGER.info("ICA audit: %s", validation["ica"])
    if args.validate_only:
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        return
    pipeline.ALL_SOURCE_STATS_ROOT = ALL_SOURCE_STATS_ROOT
    output = pipeline.train(args, root, validation)
    pipeline.LOGGER.info("Training outputs: %s", output)


if __name__ == "__main__":
    main()
