from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .faced import EMOTION_NAMES, official_fold_subjects
from .faced_psd_jsd_experiment import (
    CLASSES,
    SpectraStore,
    TrialDataset,
    _autocast,
    _file_hash,
    _json_hash,
    _lr_factor,
    _now,
    _write_json,
    build_model,
    evaluate,
    fit_reference,
    fit_standardizer,
    materialize_split,
    parameter_count,
    resolve_device,
    seed_everything,
)


def fixed_epoch_from_inner_cv(best_epochs: Sequence[int]) -> int:
    """Choose the integer median without consulting the held-out target."""
    if not best_epochs or any(int(epoch) <= 0 for epoch in best_epochs):
        raise ValueError("Inner-CV best epochs must be positive")
    return int(np.median(np.asarray(best_epochs, dtype=np.int64)))


def _plot_confusion(matrix: Sequence[Sequence[int]], output: Path) -> None:
    values = np.asarray(matrix, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xlabel="Predicted class",
        ylabel="True class",
        title="FACED locked outer-target confusion matrix",
        xticks=np.arange(CLASSES),
        yticks=np.arange(CLASSES),
        xticklabels=EMOTION_NAMES,
        yticklabels=EMOTION_NAMES,
    )
    axis.tick_params(axis="x", rotation=45)
    threshold = values.max(initial=0) / 2
    for row in range(CLASSES):
        for column in range(CLASSES):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=8,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _recover_after_plot_failure(
    *,
    output: Path,
    config: dict[str, Any],
    spectra_root: Path,
    lock: dict[str, Any],
    summary: dict[str, Any],
    existing_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Persist the locked result without repeating source training.

    The initial attempt computed target metrics but failed before persistence
    when pyplot selected Tk.  Recovery reloads the already-fixed source model;
    it never changes weights, epochs, preprocessing, or configuration.
    """
    checkpoint_path = output / "locked_source_model.pt"
    history_path = output / "source_training_history.csv"
    if not checkpoint_path.is_file() or not history_path.is_file():
        raise RuntimeError("Incomplete outer-test run is not safely recoverable")
    started = time.perf_counter()
    candidate_name = str(lock["selected_candidate"])
    candidate = config["candidates"][candidate_name]
    training = config["training"]
    protocol = config["protocol"]
    seed = int(protocol["seed"])
    outer_fold = int(protocol["outer_fold"])
    source_subjects, target_subjects = official_fold_subjects(outer_fold)
    target_set = set(map(int, target_subjects))
    best_epochs = summary[candidate_name]["best_epochs"]
    fixed_epochs = fixed_epoch_from_inner_cv(best_epochs)
    device = resolve_device(str(training["device"]))
    seed_everything(seed)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint["fixed_epochs"]) != fixed_epochs:
        raise RuntimeError("Saved model epoch count differs from the source-only rule")
    model = build_model(candidate).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)

    store = SpectraStore(spectra_root)
    reference, _ = fit_reference(store, source_subjects)
    if store.loaded_subjects & target_set:
        raise RuntimeError("Outer target loaded while reconstructing the source reference")
    target_split = materialize_split(
        store,
        target_subjects,
        reference,
        float(config["feature"]["epsilon"]),
        str(config["feature"]["storage_dtype"]),
    )
    target_loader = DataLoader(
        TrialDataset(target_split, mean, std),
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training["label_smoothing"]))
    target_metrics = evaluate(
        model, target_loader, device, str(training["precision"]), criterion
    )
    _plot_confusion(target_metrics["confusion_matrix"], output / "target_confusion.png")
    with history_path.open("r", encoding="utf-8", newline="") as stream:
        final_source_row = list(csv.DictReader(stream))[-1]
    final_source_metrics = {
        "epoch": int(final_source_row["epoch"]),
        "source_eval_loss": float(final_source_row["source_eval_loss"]),
        "source_eval_accuracy": float(final_source_row["source_eval_accuracy"]),
        "source_eval_balanced_accuracy": float(
            final_source_row["source_eval_balanced_accuracy"]
        ),
        "source_eval_macro_f1": float(final_source_row["source_eval_macro_f1"]),
    }
    result = {
        "status": "locked_outer_target_recovered_after_artifact_failure",
        "evaluation_protocol_hash": existing_manifest["evaluation_protocol_hash"],
        "source_protocol_hash": lock["protocol_hash"],
        "candidate": candidate_name,
        "parameter_count": parameter_count(model),
        "inner_best_epochs": best_epochs,
        "fixed_epoch_rule": "integer median of three source-only best epochs",
        "fixed_epochs": fixed_epochs,
        "source_trials": len(source_subjects) * 28,
        "target_trials": len(target_split["y"]),
        "final_source_metrics": final_source_metrics,
        "target_metrics": target_metrics,
        "target_used_for_selection": False,
        "post_target_tuning_permitted": False,
        "target_forward_passes": 2,
        "protocol_incident": {
            "first_pass": "metrics computed but not persisted because pyplot selected unavailable Tk",
            "recovery_pass": "same immutable source model reloaded; no tuning or weight update",
            "recovery_evaluator_sha256": _file_hash(Path(__file__).resolve()),
        },
        "loaded_subjects_after_evaluation": sorted(store.loaded_subjects),
        "maximum_target_jsd_invariant_error": float(
            target_split["maximum_invariant_error"]
        ),
        "recovery_elapsed_seconds": time.perf_counter() - started,
        "completed_at": _now(),
    }
    _write_json(output / "target_result.json", result)
    _write_json(
        output / "outer_test_manifest.json",
        {
            **existing_manifest,
            "status": result["status"],
            "target_loaded": True,
            "target_forward_passes": 2,
            "post_target_tuning_permitted": False,
            "recovered_at": _now(),
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def run_locked_outer_test(config_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_run = (root / config["paths"]["run_root"]).resolve()
    spectra_root = (root / config["paths"]["spectra_root"]).resolve()
    output = root / "runs" / "faced_psd_jsd_flatten_locked_outer_test_seed42"
    result_path = output / "target_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    lock_path = source_run / "source_selection_lock.json"
    summary_path = source_run / "inner_cv_summary.json"
    source_manifest_path = source_run / "experiment_manifest.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if lock.get("status") != "source_configuration_locked":
        raise RuntimeError("Outer target cannot be evaluated before source configuration lock")
    if lock.get("outer_target_evaluated") is not False:
        raise RuntimeError("Source lock is not in the expected pre-target state")
    if lock.get("protocol_hash") != source_manifest.get("protocol_hash"):
        raise RuntimeError("Source lock and manifest protocol hashes differ")

    runner_path = Path(__file__).resolve().parent / "faced_psd_jsd_experiment.py"
    model_path = Path(__file__).resolve().parent / "models" / "faced_psd_jsd.py"
    recorded_hashes = source_manifest["protocol"]["code_hashes"]
    current_hashes = {"runner": _file_hash(runner_path), "models": _file_hash(model_path)}
    if current_hashes != recorded_hashes:
        raise RuntimeError("Locked source code changed after configuration selection")

    candidate_name = str(lock["selected_candidate"])
    candidate = config["candidates"][candidate_name]
    best_epochs = summary[candidate_name]["best_epochs"]
    fixed_epochs = fixed_epoch_from_inner_cv(best_epochs)
    protocol = config["protocol"]
    training = config["training"]
    seed = int(protocol["seed"])
    outer_fold = int(protocol["outer_fold"])
    source_subjects, target_subjects = official_fold_subjects(outer_fold)
    target_set = set(map(int, target_subjects))
    evaluation_payload = {
        "family": "FACED-PSD-JSD-Locked-Outer-Test-v1",
        "source_protocol_hash": lock["protocol_hash"],
        "source_lock_sha256": _file_hash(lock_path),
        "source_summary_sha256": _file_hash(summary_path),
        "locked_config_sha256": _file_hash(config_path),
        "evaluator_sha256": _file_hash(Path(__file__).resolve()),
        "candidate": candidate_name,
        "fixed_epoch_rule": "integer median of three source-only best epochs",
        "inner_best_epochs": best_epochs,
        "fixed_epochs": fixed_epochs,
        "target_use": "exactly one evaluation after fixed source-only training",
    }
    evaluation_hash = _json_hash(evaluation_payload)
    if output.exists():
        existing_manifest = json.loads(
            (output / "outer_test_manifest.json").read_text(encoding="utf-8")
        )
        return _recover_after_plot_failure(
            output=output,
            config=config,
            spectra_root=spectra_root,
            lock=lock,
            summary=summary,
            existing_manifest=existing_manifest,
        )
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "outer_test_manifest.json",
        {
            "status": "training_source_before_target_load",
            "evaluation_protocol_hash": evaluation_hash,
            "protocol": evaluation_payload,
            "created_at": _now(),
            "target_loaded": False,
        },
    )

    device = resolve_device(str(training["device"]))
    seed_everything(seed)
    store = SpectraStore(spectra_root)
    started = time.perf_counter()
    reference, reference_windows = fit_reference(store, source_subjects)
    source_split = materialize_split(
        store,
        source_subjects,
        reference,
        float(config["feature"]["epsilon"]),
        str(config["feature"]["storage_dtype"]),
    )
    mean, std = fit_standardizer(source_split["x"])
    if store.loaded_subjects & target_set:
        raise RuntimeError("Outer-target data loaded before final source model was fixed")
    _write_json(
        output / "pre_target_isolation_audit.json",
        {
            "reference_scope": "all_outer_source_subjects_only",
            "standardizer_scope": "all_outer_source_subjects_only",
            "reference_windows": reference_windows,
            "source_subjects": list(map(int, source_subjects)),
            "outer_target_subjects": list(map(int, target_subjects)),
            "loaded_subjects": sorted(store.loaded_subjects),
            "target_loaded": False,
            "fixed_epochs": fixed_epochs,
            "maximum_jsd_invariant_error": float(source_split["maximum_invariant_error"]),
        },
    )

    dataset = TrialDataset(source_split, mean, std)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    train_eval_loader = DataLoader(
        dataset,
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    model = build_model(candidate).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    total_steps = fixed_epochs * len(train_loader)
    warmup_steps = round(total_steps * float(training["warmup_fraction"]))
    minimum_ratio = float(training["minimum_learning_rate"]) / float(training["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(step, total_steps, warmup_steps, minimum_ratio),
    )
    precision = str(training["precision"])
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "float16"
    )
    clip_norm = float(training["gradient_clip_norm"])
    history: list[dict[str, Any]] = []
    for epoch in range(1, fixed_epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for value, label in train_loader:
            value = value.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                logits = model(value)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(label)
            correct += int((logits.argmax(1) == label).sum())
            count += len(label)
        source_metrics = evaluate(model, train_eval_loader, device, precision, criterion)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_mode_loss": loss_sum / count,
                "train_mode_accuracy": correct / count,
                "source_eval_loss": source_metrics["loss"],
                "source_eval_accuracy": source_metrics["accuracy"],
                "source_eval_balanced_accuracy": source_metrics["balanced_accuracy"],
                "source_eval_macro_f1": source_metrics["macro_f1"],
            }
        )
        print(
            f"locked source epoch {epoch:03d}/{fixed_epochs} "
            f"accuracy={source_metrics['accuracy']:.3f} macro_f1={source_metrics['macro_f1']:.3f}",
            flush=True,
        )
    with (output / "source_training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    if store.loaded_subjects & target_set:
        raise RuntimeError("Outer target was loaded during final source training")
    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "feature_mean": mean,
            "feature_std": std,
            "candidate": candidate,
            "source_protocol_hash": lock["protocol_hash"],
            "outer_test_protocol_hash": evaluation_hash,
            "fixed_epochs": fixed_epochs,
            "target_loaded_during_training": False,
        },
        output / "locked_source_model.pt",
    )

    target_split = materialize_split(
        store,
        target_subjects,
        reference,
        float(config["feature"]["epsilon"]),
        str(config["feature"]["storage_dtype"]),
    )
    target_loader = DataLoader(
        TrialDataset(target_split, mean, std),
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    target_metrics = evaluate(model, target_loader, device, precision, criterion)
    _plot_confusion(target_metrics["confusion_matrix"], output / "target_confusion.png")
    result = {
        "status": "locked_outer_target_evaluated_once",
        "evaluation_protocol_hash": evaluation_hash,
        "source_protocol_hash": lock["protocol_hash"],
        "candidate": candidate_name,
        "parameter_count": parameter_count(model),
        "inner_best_epochs": best_epochs,
        "fixed_epoch_rule": "integer median of three source-only best epochs",
        "fixed_epochs": fixed_epochs,
        "source_trials": len(dataset),
        "target_trials": len(target_split["y"]),
        "final_source_metrics": history[-1],
        "target_metrics": target_metrics,
        "target_used_for_selection": False,
        "post_target_tuning_permitted": False,
        "loaded_subjects_after_evaluation": sorted(store.loaded_subjects),
        "maximum_jsd_invariant_error": max(
            float(source_split["maximum_invariant_error"]),
            float(target_split["maximum_invariant_error"]),
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at": _now(),
    }
    _write_json(result_path, result)
    _write_json(
        output / "outer_test_manifest.json",
        {
            "status": result["status"],
            "evaluation_protocol_hash": evaluation_hash,
            "protocol": evaluation_payload,
            "created_at": _now(),
            "target_loaded": True,
            "target_evaluation_count": 1,
            "post_target_tuning_permitted": False,
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/faced/psd_jsd_flatten_inner_cv.yaml"),
    )
    arguments = parser.parse_args(argv)
    run_locked_outer_test(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
