"""Train SEED models on the completed ICA-cleaned 1 s / 0.5 s-hop cache."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import train_seediv_de_rjsd_ica as pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PARENT = (
    ROOT.parent / "Dataset" / "Processed" / "CMRD" / "seed" / "de_rjsd_ica_1s_hop05"
)
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "seed" / "de_rjsd_ica"
ALL_SOURCE_STATS_ROOT = (
    ROOT / "runs" / "diagnostics" / "seed_hierarchical_attention" / "_all_source_statistics"
)
LOGGER = logging.getLogger("seed.cmrd_ica.train")


def _configure_pipeline() -> None:
    pipeline.DATASET_NAME = "SEED"
    pipeline.EXPECTED_SUBJECTS = 15
    pipeline.EXPECTED_CLASSES = 3
    pipeline.EXPECTED_TRIALS_PER_SUBJECT = 45
    pipeline.EXPECTED_TRIALS = 675
    pipeline.EXPECTED_GROUP_SIZES = {"train": 540, "validation": 90, "test": 45}
    pipeline.EXPECTED_SOURCE_TRIALS = 630
    pipeline.EXPECTED_TARGET_TRIALS = 45
    pipeline.DEFAULT_DATA_PARENT = DEFAULT_DATA_PARENT
    pipeline.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    pipeline.ALL_SOURCE_STATS_ROOT = ALL_SOURCE_STATS_ROOT
    pipeline.LOGGER = LOGGER


_configure_pipeline()


def __getattr__(name: str):
    """Expose shared helpers/constants to notebooks without duplicating the engine."""
    return getattr(pipeline, name)


def _resolve_data_root(value: str | None) -> Path:
    return pipeline._resolve_data_root(value or str(DEFAULT_DATA_PARENT))


def build_parser():
    _configure_pipeline()
    parser = pipeline.build_parser()
    parser.description = (
        "Train SEED LOSO on ICA-cleaned DE/RJSD features extracted with "
        "1-second windows and a 0.5-second hop."
    )
    parser.set_defaults(data_root=str(DEFAULT_DATA_PARENT), output_root=str(DEFAULT_OUTPUT_ROOT))
    for action in parser._actions:
        if action.dest == "data_root":
            action.help = f"SEED signature cache or parent (default: {DEFAULT_DATA_PARENT})"
        elif action.dest == "output_root":
            action.help = f"Training output parent (default: {DEFAULT_OUTPUT_ROOT})"
    return parser


def main() -> None:
    _configure_pipeline()
    args = pipeline.parse_args_with_config(parser=build_parser())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    root = _resolve_data_root(args.data_root)
    LOGGER.info("SEED data root: %s", root)
    validation = pipeline.validate_cache(root, deep=args.deep_validate)
    LOGGER.info("Selected training feature: %s", args.feature.upper())
    LOGGER.info(
        "Cache valid: trials=%d folds=%d labels=%s windows=%d..%d",
        validation["trials"],
        validation["folds"],
        validation["labels"],
        validation["windows"]["min"],
        validation["windows"]["max"],
    )
    LOGGER.info("ICA audit: %s", validation["ica"])
    if args.validate_only:
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        return
    output = pipeline.train(args, root, validation)
    LOGGER.info("Training outputs: %s", output)


if __name__ == "__main__":
    main()
