from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmrd.config import load_config
from cmrd.fixed_protocol import load_representation_samples, scaling_statistics
from cmrd.io import write_json
from cmrd.native_compact_runner import (
    FIXED_SEED,
    FOLD,
    _fold_groups,
    _json_hash,
    run_task,
    utc_now,
    validate_native_sources,
)
from cmrd.training.runtime import select_device


CONDITION = "z_de_zscore_base_v2"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="DEAP fold-1 full-DE baseline using the Native-Compact Base/v2 trainer"
    )
    value.add_argument("--config", default="configs/native_compact/deap_v1.yaml")
    value.add_argument("--cache-parent")
    value.add_argument("--run-root", default="runs/deap_de_baseline_v1_seed42")
    value.add_argument("--smoke-epochs", type=int)
    return value


def main() -> int:
    args = parser().parse_args()
    config = load_config(Path(args.config))
    if config.dataset != "deap":
        raise ValueError("The DE baseline runner requires dataset=deap")
    if args.smoke_epochs is not None and not 1 <= args.smoke_epochs <= 200:
        raise ValueError("--smoke-epochs must be within 1..200")

    run_root = Path(args.run_root).resolve()
    cache_parent = Path(args.cache_parent).resolve() if args.cache_parent else None
    audit = validate_native_sources(config, cache_parent)
    train_entries, validation_entries, target_entries = _fold_groups(
        Path(audit["cache_root"]), config
    )
    channels = int(config.raw["dataset"]["channels"])
    cache_root = Path(audit["cache_root"])

    # Match the compact-feature protocol: feature state is fitted on the 29
    # source-train subjects, while model training uses all 31 source subjects.
    train_samples = load_representation_samples(
        cache_root, train_entries, "de_zscore", channels=channels
    )
    validation_samples = load_representation_samples(
        cache_root, validation_entries, "de_zscore", channels=channels
    )
    normalization = scaling_statistics(train_samples, True)
    source_locked_at = utc_now()
    target_samples = load_representation_samples(
        cache_root, target_entries, "de_zscore", channels=channels
    )
    source_samples = train_samples + validation_samples

    protocol = {
        "schema_version": 1,
        "family": "DEAP-DE-Baseline-v1",
        "dataset": "deap",
        "condition": CONDITION,
        "representation": "de_zscore",
        "architecture": "base",
        "training_method": "v2",
        "fold": FOLD,
        "seed": FIXED_SEED,
        "source_model_subjects": sorted({sample.subject for sample in source_samples}),
        "source_normalization_subjects": sorted({sample.subject for sample in train_samples}),
        "target_subject": FOLD,
        "target_monitor_interval": 10,
        "target_metrics_affect_training": False,
        "checkpoint_selection": "fixed_final_epoch_only",
        "preprocessing_signature": audit["preprocessing_signature"],
        "cleaning_signature": audit["cleaning_signature"],
    }
    protocol_hash = _json_hash(protocol)
    task = {
        "task_id": f"deap__{CONDITION}__fold-01__seed-{FIXED_SEED}",
        "dataset": "deap",
        "condition": CONDITION,
        "representation": "de_zscore",
        "architecture": "base",
        "fold": FOLD,
        "seed": FIXED_SEED,
        "protocol_hash": protocol_hash,
        "status": "running",
        "attempts": 1,
    }
    output = run_root / "deap" / CONDITION / "fold-01" / f"seed-{FIXED_SEED}"
    output.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "protocol.json", protocol)
    write_json(output / "status.json", {"status": "running", "task": task})

    bundle = {
        "source_samples": source_samples,
        "target_samples": target_samples,
        "normalization": normalization,
        "source_subjects": sorted({sample.subject for sample in source_samples}),
        "normalization_subjects": sorted({sample.subject for sample in train_samples}),
        "source_locked_at": source_locked_at,
        "target_loaded_at": utc_now(),
        "feature_cache": str(cache_root.resolve()),
    }
    try:
        result = run_task(
            config,
            audit,
            task,
            run_root,
            select_device(str(config.raw["training"].get("device", "auto"))),
            bundle,
            args.smoke_epochs,
        )
    except BaseException as exc:
        task.update({
            "status": "failed",
            "failed_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        write_json(output / "status.json", {
            "status": "failed",
            "task": task,
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        })
        raise

    task.update({
        "status": "complete",
        "completed_at": result["completed_at"],
        "elapsed_seconds": result["elapsed_seconds"],
    })
    write_json(output / "status.json", {"status": "complete", "task": task})
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
