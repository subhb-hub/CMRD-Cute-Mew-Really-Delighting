from __future__ import annotations

import copy
import csv
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

from cmrd.config import ExperimentConfig
from cmrd.io import read_json, write_json
from cmrd.processed import load_split

from .artifacts import create_run, latest_selection, set_run_status
from .engine import train_once
from .runtime import environment_manifest, select_device

LOGGER = logging.getLogger("cmrd.training")


def candidates(config: ExperimentConfig) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for architecture_index, architecture in enumerate(config.raw["tuning"]["architectures"], 1):
        for optimization_index, optimization in enumerate(config.raw["tuning"]["optimizations"], 1):
            model = copy.deepcopy(config.raw["model"])
            training = copy.deepcopy(config.raw["training"])
            model.update({key: architecture[key] for key in ("d_model", "nhead", "layers", "feedforward")})
            model["dropout"] = optimization["dropout"]
            training.update({key: optimization[key] for key in ("learning_rate", "weight_decay", "label_smoothing")})
            values.append({
                "candidate_id": f"a{architecture_index:02d}_o{optimization_index:02d}",
                "model": model,
                "training": training,
            })
    if len(values) != 12:
        raise ValueError(f"The medium sweep requires exactly 12 candidates, got {len(values)}")
    return values


def _write_flat_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_or_train(path: Path, resume: bool, force: bool, **kwargs):
    result_path = path / "result.json"
    if result_path.is_file() and resume and not force:
        return read_json(result_path)
    if force and path.exists():
        shutil.rmtree(path)
    return train_once(output_dir=path, **kwargs)


def _run_tuning(config: ExperimentConfig, run: Path, targets: list[int], resume: bool, force: bool, device) -> dict[str, object]:
    candidate_values = candidates(config)
    existing_path = run / "selected_by_fold.json"
    selections: dict[str, object] = read_json(existing_path).get("folds", {}) if existing_path.is_file() and resume else {}
    tuning_seed = int(config.raw["tuning"]["seed"])
    for target in targets:
        target_key = f"{target:02d}"
        if target_key in selections and resume and not force:
            LOGGER.info("Reusing tuned fold %s", target_key)
            continue
        LOGGER.info("Tuning target fold %02d without loading target trials", target)
        train, validation, test, split = load_split(config, target, include_test=False)
        if test:
            raise AssertionError("Tuning must not load target test trials")
        scored: list[dict[str, object]] = []
        for candidate in candidate_values:
            identifier = str(candidate["candidate_id"])
            output = run / "folds" / f"fold-{target:02d}" / "candidates" / identifier
            result = _load_or_train(
                output,
                resume,
                force,
                train_samples=train,
                validation_samples=validation,
                test_samples=None,
                model_config=copy.deepcopy(candidate["model"]),
                training=copy.deepcopy(candidate["training"]),
                classes=int(config.raw["dataset"]["classes"]),
                seed=tuning_seed,
                device=device,
                context={
                    "mode": "tune",
                    "dataset": config.dataset,
                    "feature": config.feature,
                    "target_subject": target,
                    "candidate_id": identifier,
                    "split": split.as_dict(),
                    "target_loaded": False,
                },
            )
            validation_metrics = result["validation"]
            scored.append({
                "candidate_id": identifier,
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_accuracy": validation_metrics["accuracy"],
                "best_epoch": result["best_epoch"],
                "model": candidate["model"],
                "training": candidate["training"],
            })
        scored.sort(key=lambda item: (float(item["validation_macro_f1"]), float(item["validation_accuracy"])), reverse=True)
        best = scored[0]
        selections[target_key] = {
            "target_subject": target,
            "split": split.as_dict(),
            "selected": best,
            "ranking": scored,
            "selection_metric": "validation_macro_f1",
            "tie_breaker": "validation_accuracy",
            "target_loaded": False,
        }
        write_json(existing_path, {
            "dataset": config.dataset,
            "feature": config.feature,
            "tuning_seed": tuning_seed,
            "folds": selections,
        })
        LOGGER.info("Fold %02d selected %s (macro_f1=%.4f)", target, best["candidate_id"], best["validation_macro_f1"])
    rows = []
    for fold, selection in sorted(selections.items()):
        selected = selection["selected"]
        rows.append({"target_subject": int(fold), **selected})
    _write_flat_csv(run / "tuning_summary.csv", rows, ["target_subject", "candidate_id", "validation_macro_f1", "validation_accuracy", "best_epoch"])
    return {"folds_completed": len(selections), "expected_folds": int(config.raw["dataset"]["subjects"])}


def _summarize_final(config: ExperimentConfig, run: Path, results: list[dict[str, object]]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for result in results:
        test = result["test"]
        validation = result["validation"]
        rows.append({
            "target_subject": result["target_subject"],
            "seed": result["seed"],
            "candidate_id": result["candidate_id"],
            "accuracy": test["accuracy"],
            "balanced_accuracy": test["balanced_accuracy"],
            "macro_f1": test["macro_f1"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "best_epoch": result["best_epoch"],
            "elapsed_seconds": result["elapsed_seconds"],
        })
    _write_flat_csv(run / "fold_results.csv", rows, list(rows[0]) if rows else ["target_subject"])
    subject_summary: dict[str, object] = {}
    for target in sorted({int(row["target_subject"]) for row in rows}):
        subset = [row for row in rows if int(row["target_subject"]) == target]
        subject_summary[f"{target:02d}"] = {
            metric: {"mean": float(np.mean([row[metric] for row in subset])), "std_across_seeds": float(np.std([row[metric] for row in subset]))}
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
    seed_summary: dict[str, object] = {}
    for seed in sorted({int(row["seed"]) for row in rows}):
        subset = [row for row in rows if int(row["seed"]) == seed]
        seed_summary[str(seed)] = {
            metric: {"mean_across_subjects": float(np.mean([row[metric] for row in subset])), "std_across_subjects": float(np.std([row[metric] for row in subset]))}
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
    aggregate: dict[str, object] = {}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        subject_means = [value[metric]["mean"] for value in subject_summary.values()]
        seed_means = [value[metric]["mean_across_subjects"] for value in seed_summary.values()]
        aggregate[metric] = {
            "mean_of_subject_means": float(np.mean(subject_means)),
            "std_across_subjects": float(np.std(subject_means)),
            "mean_of_seed_means": float(np.mean(seed_means)),
            "std_across_seeds": float(np.std(seed_means)),
        }
    payload = {
        "dataset": config.dataset,
        "feature": config.feature,
        "model": "plain_transformer",
        "protocol": "15-subject LOSO with two source-validation subjects",
        "seeds": list(map(int, config.raw["training"]["seeds"])),
        "subject_summary": subject_summary,
        "seed_summary": seed_summary,
        "aggregate": aggregate,
        "runs": results,
    }
    write_json(run / "summary.json", payload)
    return payload


def _run_final(config: ExperimentConfig, run: Path, targets: list[int], resume: bool, force: bool, device) -> dict[str, object]:
    selection_path = latest_selection(config)
    require_selection = bool(config.raw["tuning"].get("require_selection", True))
    selection = read_json(selection_path) if selection_path else {"folds": {}}
    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidates(config)}
    results: list[dict[str, object]] = []
    for target in targets:
        key = f"{target:02d}"
        if key in selection.get("folds", {}):
            selected = selection["folds"][key]["selected"]
            candidate_id = str(selected["candidate_id"])
            candidate = candidate_map[candidate_id]
        elif require_selection:
            raise FileNotFoundError(f"No source-only tuning selection for fold {target}; run --mode tune first")
        else:
            candidate = candidates(config)[0]
            candidate_id = str(candidate["candidate_id"])
        # Target trials are loaded only after the candidate has been fixed above.
        train, validation, test, split = load_split(config, target, include_test=True)
        for seed in map(int, config.raw["training"]["seeds"]):
            output = run / "folds" / f"fold-{target:02d}" / f"seed-{seed}"
            result = _load_or_train(
                output,
                resume,
                force,
                train_samples=train,
                validation_samples=validation,
                test_samples=test,
                model_config=copy.deepcopy(candidate["model"]),
                training=copy.deepcopy(candidate["training"]),
                classes=int(config.raw["dataset"]["classes"]),
                seed=seed,
                device=device,
                context={
                    "mode": "final",
                    "dataset": config.dataset,
                    "feature": config.feature,
                    "target_subject": target,
                    "candidate_id": candidate_id,
                    "selection_file": str(selection_path) if selection_path else None,
                    "split": split.as_dict(),
                    "target_loaded_after_selection": True,
                },
            )
            results.append(result)
            LOGGER.info("Final fold=%02d seed=%d accuracy=%.4f macro_f1=%.4f", target, seed, result["test"]["accuracy"], result["test"]["macro_f1"])
    summary = _summarize_final(config, run, results)
    return {"runs_completed": len(results), "aggregate": summary["aggregate"]}


def run_experiment(config: ExperimentConfig, mode: str, fold: int | None, resume: bool, force: bool, command: list[str] | None = None) -> Path:
    if mode not in {"tune", "final"}:
        raise ValueError("mode must be tune or final")
    subjects = int(config.raw["dataset"]["subjects"])
    targets = [int(fold)] if fold is not None else list(range(1, subjects + 1))
    if any(target < 1 or target > subjects for target in targets):
        raise ValueError(f"Fold must be in 1..{subjects}")
    command = command or sys.argv
    environment = environment_manifest(command)
    run = create_run(config, mode, resume, command, environment)
    device = select_device(str(config.raw["training"].get("device", "auto")))
    try:
        details = _run_tuning(config, run, targets, resume, force, device) if mode == "tune" else _run_final(config, run, targets, resume, force, device)
        expected = subjects if mode == "tune" else subjects * len(config.raw["training"]["seeds"])
        completed = details.get("folds_completed", details.get("runs_completed", 0))
        status = "complete" if int(completed) == expected else "partial"
        set_run_status(run, status, **details)
    except BaseException as exc:
        set_run_status(run, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    return run

