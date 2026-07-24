from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from .faced import EMOTION_NAMES, SUBJECTS, VIDEO_LABELS, VIDEOS, official_fold_subjects


LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProbeConfig:
    path: Path
    base_run_root: Path
    run_root: Path
    fold: int
    seed: int
    development_subjects: tuple[int, ...]
    pseudo_reference_windows: int
    response_start_window: int
    prototype_counts: tuple[int, ...]
    shrinkage_weights: tuple[float, ...]
    bootstrap_repeats: int
    evidence_label: str


@dataclass(frozen=True)
class AtlasState:
    scaler: StandardScaler
    centroids_scaled: np.ndarray
    centroids_original: np.ndarray
    temperature: float
    subject_cluster_counts: tuple[int, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_probe_config(path: str | Path) -> ProbeConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.parents[2]
    probe = raw["probe"]
    result = ProbeConfig(
        path=config_path,
        base_run_root=_resolve(repo_root, raw["paths"]["base_run_root"]),
        run_root=_resolve(repo_root, raw["paths"]["run_root"]),
        fold=int(probe["fold"]),
        seed=int(probe["seed"]),
        development_subjects=tuple(map(int, probe["development_subjects"])),
        pseudo_reference_windows=int(probe["pseudo_reference_windows"]),
        response_start_window=int(probe["response_start_window"]),
        prototype_counts=tuple(map(int, probe["prototype_counts"])),
        shrinkage_weights=tuple(map(float, probe["shrinkage_weights"])),
        bootstrap_repeats=int(probe["bootstrap_repeats"]),
        evidence_label=str(raw["experiment"]["evidence_label"]),
    )
    source, target = official_fold_subjects(result.fold)
    if set(result.development_subjects) & set(target):
        raise ValueError("Development subjects overlap outer target subjects")
    if not set(result.development_subjects) <= set(source):
        raise ValueError("Development subjects must be outer-fold sources")
    if not 0 < result.pseudo_reference_windows <= result.response_start_window < 30:
        raise ValueError("Pseudo-reference/response windows must satisfy 0 < reference <= response < 30")
    if any(k < 2 for k in result.prototype_counts):
        raise ValueError("Prototype counts must be at least two; K=1 is the explicit A2 condition")
    if any(not 0.0 < alpha < 1.0 for alpha in result.shrinkage_weights):
        raise ValueError("A5 shrinkage weights must be strictly between zero and one")
    return result


def heldout_videos() -> np.ndarray:
    """Return one deterministic (last) video index for every emotion."""
    return np.asarray([
        int(np.flatnonzero(VIDEO_LABELS == label)[-1]) for label in range(len(EMOTION_NAMES))
    ], dtype=np.int64)


def _native_spectra_root(base_run_root: Path) -> Path:
    roots = sorted((base_run_root / "cache" / "native_spectra").glob("*/manifest.json"))
    complete = []
    for manifest in roots:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("all_subjects_complete") and len(payload.get("subjects_complete", [])) == SUBJECTS:
            complete.append(manifest.parent)
    if len(complete) != 1:
        raise RuntimeError(f"Expected exactly one complete FACED native spectra cache, found {complete}")
    return complete[0]


def load_de_cache(config: ProbeConfig) -> np.ndarray:
    root = _native_spectra_root(config.base_run_root) / "subjects"
    values: list[np.ndarray] = []
    for subject in range(SUBJECTS):
        path = root / f"sub{subject:03d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            de = np.asarray(archive["de"], dtype=np.float32)
        if de.shape != (VIDEOS, 30, 150) or not np.isfinite(de).all():
            raise ValueError(f"Invalid DE cache for subject {subject}: {de.shape}")
        values.append(de)
    return np.stack(values)


def trial_summaries(de: np.ndarray, config: ProbeConfig) -> tuple[np.ndarray, np.ndarray]:
    # Processed_data is the final 30 seconds of each video. The first five
    # windows below are therefore a within-video pseudo-reference, never a
    # pre-stimulus baseline. Median reduces sensitivity to one transient window.
    reference = np.median(de[:, :, : config.pseudo_reference_windows], axis=2)
    response = np.mean(de[:, :, config.response_start_window :], axis=2)
    return reference.astype(np.float64), response.astype(np.float64)


def fit_atlas(reference: np.ndarray, subjects: Sequence[int], k: int, seed: int) -> AtlasState:
    subject_reference = np.median(reference[np.asarray(subjects)], axis=1)
    scaler = StandardScaler().fit(subject_reference)
    standardized = scaler.transform(subject_reference)
    if k == 1:
        centroids_scaled = standardized.mean(axis=0, keepdims=True)
        labels = np.zeros(len(subjects), dtype=np.int64)
    else:
        estimator = KMeans(n_clusters=k, random_state=seed, n_init=20, algorithm="lloyd")
        labels = estimator.fit_predict(standardized)
        centroids_scaled = estimator.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    distances = squared_distances(standardized, centroids_scaled)
    positive = distances[distances > 1e-12]
    temperature = float(np.median(positive)) if positive.size else 1.0
    temperature = max(temperature, 1e-6)
    counts = tuple(int(np.sum(labels == index)) for index in range(k))
    return AtlasState(scaler, centroids_scaled, centroids_original, temperature, counts)


def squared_distances(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(values * values, axis=1, keepdims=True)
        + np.sum(centroids * centroids, axis=1)[None]
        - 2.0 * values @ centroids.T,
        0.0,
    )


def route_atlas(reference: np.ndarray, state: AtlasState) -> tuple[np.ndarray, np.ndarray]:
    flat = reference.reshape(-1, reference.shape[-1])
    scaled = state.scaler.transform(flat)
    distances = squared_distances(scaled, state.centroids_scaled)
    weights = softmax(-distances / state.temperature, axis=1)
    atlas = weights @ state.centroids_original
    return atlas.reshape(reference.shape), weights.reshape(*reference.shape[:-1], -1)


def source_video_reference(reference: np.ndarray, subjects: Sequence[int]) -> np.ndarray:
    return reference[np.asarray(subjects)].mean(axis=0)


def build_features(
    name: str,
    reference: np.ndarray,
    response: np.ndarray,
    *,
    atlas: np.ndarray | None = None,
    alpha: float | None = None,
    video_reference: np.ndarray | None = None,
) -> np.ndarray:
    if name == "A0_absolute_de":
        return response
    if name == "A1_paired_pseudo_brde":
        return response - reference
    if name in {"A2_global_atlas", "A4_soft_atlas"}:
        if atlas is None:
            raise ValueError(f"{name} requires atlas features")
        return response - atlas
    if name == "A5_shrink_pseudo_brde":
        if atlas is None or alpha is None:
            raise ValueError("A5 requires atlas and alpha")
        posterior = alpha * reference + (1.0 - alpha) * atlas
        return response - posterior
    if name == "N1_source_video_reference":
        if video_reference is None:
            raise ValueError("N1 requires source-only per-video references")
        return response - video_reference[None]
    raise KeyError(name)


def _flatten(
    features: np.ndarray,
    subjects: Sequence[int],
    videos: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = np.asarray(subjects, dtype=np.int64)
    v = np.asarray(videos, dtype=np.int64)
    selected = features[np.ix_(s, v)]
    x = selected.reshape(len(s) * len(v), -1)
    y = np.tile(VIDEO_LABELS[v], len(s))
    subject_ids = np.repeat(s, len(v))
    video_ids = np.tile(v, len(s))
    return x, y, subject_ids, video_ids


def fit_classifier(x: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LinearDiscriminantAnalysis]:
    scaler = StandardScaler().fit(x)
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(scaler.transform(x), y)
    return scaler, model


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=np.arange(9)).tolist(),
    }


def evaluate_condition(
    features: np.ndarray,
    train_subjects: Sequence[int],
    test_subjects: Sequence[int],
    train_videos: Sequence[int],
    test_videos: Sequence[int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    x_train, y_train, _, _ = _flatten(features, train_subjects, train_videos)
    x_test, y_test, subject_ids, video_ids = _flatten(features, test_subjects, test_videos)
    scaler, model = fit_classifier(x_train, y_train)
    prediction = model.predict(scaler.transform(x_test))
    result = metrics(y_test, prediction)
    result["train_samples"] = int(len(y_train))
    result["test_samples"] = int(len(y_test))
    result["feature_dimension"] = int(x_train.shape[1])
    return result, {
        "y": y_test,
        "prediction": prediction,
        "subjects": subject_ids,
        "videos": video_ids,
    }


def _candidate_key(row: dict[str, Any]) -> tuple[float, float, int, float]:
    alpha = row.get("alpha")
    return (
        float(row["macro_f1"]),
        float(row["accuracy"]),
        -int(row.get("k", 1)),
        float(alpha) if alpha is not None else 0.0,
    )


def select_representation(
    condition: str,
    reference: np.ndarray,
    response: np.ndarray,
    fit_subjects: Sequence[int],
    dev_subjects: Sequence[int],
    train_videos: Sequence[int],
    dev_videos: Sequence[int],
    config: ProbeConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[tuple[int, float | None]]
    if condition == "A2_global_atlas":
        candidates = [(1, 0.0)]
    elif condition == "A4_soft_atlas":
        candidates = [(k, 0.0) for k in config.prototype_counts]
    elif condition == "A5_shrink_pseudo_brde":
        candidates = [(k, alpha) for k in config.prototype_counts for alpha in config.shrinkage_weights]
    else:
        candidates = [(0, None)]
    rows: list[dict[str, Any]] = []
    for k, alpha in candidates:
        if k:
            state = fit_atlas(reference, fit_subjects, k, config.seed)
            atlas, weights = route_atlas(reference, state)
            entropy = float(np.mean(-np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=-1)))
        else:
            atlas = None
            state = None
            entropy = 0.0
        if condition == "N1_source_video_reference":
            value = build_features(
                condition, reference, response,
                video_reference=source_video_reference(reference, fit_subjects),
            )
        else:
            value = build_features(condition, reference, response, atlas=atlas, alpha=alpha)
        score, _ = evaluate_condition(value, fit_subjects, dev_subjects, train_videos, dev_videos)
        rows.append({
            "condition": condition,
            "k": k,
            "alpha": alpha,
            "route_entropy": entropy,
            "cluster_counts": list(state.subject_cluster_counts) if state else [],
            **{key: score[key] for key in ("accuracy", "balanced_accuracy", "macro_f1")},
        })
    return max(rows, key=_candidate_key), rows


def final_features(
    condition: str,
    selected: dict[str, Any],
    reference: np.ndarray,
    response: np.ndarray,
    source_subjects: Sequence[int],
    config: ProbeConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    k = int(selected.get("k", 0))
    alpha = selected.get("alpha")
    diagnostics: dict[str, Any] = {}
    if condition == "N1_source_video_reference":
        value = build_features(
            condition, reference, response,
            video_reference=source_video_reference(reference, source_subjects),
        )
    elif k:
        state = fit_atlas(reference, source_subjects, k, config.seed)
        atlas, weights = route_atlas(reference, state)
        value = build_features(condition, reference, response, atlas=atlas, alpha=alpha)
        diagnostics = {
            "k": k,
            "alpha": alpha,
            "temperature": state.temperature,
            "source_subject_cluster_counts": list(state.subject_cluster_counts),
            "mean_route_entropy": float(np.mean(-np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=-1))),
            "mean_max_route_weight": float(np.mean(weights.max(axis=-1))),
        }
    else:
        value = build_features(condition, reference, response)
    return value, diagnostics


def bootstrap_subjects(
    records: dict[str, dict[str, np.ndarray]],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    subject_values = np.unique(next(iter(records.values()))["subjects"])
    condition_names = list(records)
    draws: dict[str, dict[str, list[float]]] = {
        name: {"accuracy": [], "macro_f1": []} for name in condition_names
    }
    differences: dict[str, list[float]] = {name: [] for name in condition_names if name != "A0_absolute_de"}
    for _ in range(repeats):
        sampled = rng.choice(subject_values, size=len(subject_values), replace=True)
        for name, record in records.items():
            indices = np.concatenate([np.flatnonzero(record["subjects"] == subject) for subject in sampled])
            y = record["y"][indices]
            prediction = record["prediction"][indices]
            draws[name]["accuracy"].append(float(accuracy_score(y, prediction)))
            draws[name]["macro_f1"].append(float(f1_score(y, prediction, average="macro", zero_division=0)))
        base = draws["A0_absolute_de"]["accuracy"][-1]
        for name in differences:
            differences[name].append(draws[name]["accuracy"][-1] - base)
    result: dict[str, Any] = {"conditions": {}, "accuracy_difference_vs_A0": {}}
    for name, by_metric in draws.items():
        result["conditions"][name] = {
            metric: [float(x) for x in np.quantile(values, [0.025, 0.975])]
            for metric, values in by_metric.items()
        }
    for name, values in differences.items():
        result["accuracy_difference_vs_A0"][name] = {
            "mean": float(np.mean(values)),
            "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            "probability_greater_than_zero": float(np.mean(np.asarray(values) > 0)),
        }
    return result


def nearest_centroid_accuracy(
    features: np.ndarray,
    train_subjects: Sequence[int],
    test_subjects: Sequence[int],
    train_videos: Sequence[int],
    test_videos: Sequence[int],
    target: str,
) -> float:
    x_train, _, subjects_train, videos_train = _flatten(features, train_subjects, train_videos)
    x_test, _, subjects_test, videos_test = _flatten(features, test_subjects, test_videos)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    labels_train = subjects_train if target == "subject" else videos_train
    labels_test = subjects_test if target == "subject" else videos_test
    classes = np.unique(labels_train)
    centroids = np.stack([x_train[labels_train == label].mean(axis=0) for label in classes])
    prediction = classes[np.argmin(squared_distances(x_test, centroids), axis=1)]
    return float(accuracy_score(labels_test, prediction))


def prototype_stability_audit(
    reference: np.ndarray,
    fit_subjects: Sequence[int],
    final_subjects: Sequence[int],
    prototype_counts: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    """Check whether adding held-out source-development subjects changes the atlas."""
    fit_subjects = list(map(int, fit_subjects))
    final_subjects = list(map(int, final_subjects))
    fit_summary = np.median(reference[np.asarray(fit_subjects)], axis=1)
    final_summary = np.median(reference[np.asarray(final_subjects)], axis=1)
    rows: dict[str, Any] = {}
    for k in prototype_counts:
        fit_state = fit_atlas(reference, fit_subjects, int(k), seed)
        final_state = fit_atlas(reference, final_subjects, int(k), seed)
        fit_labels = np.argmin(
            squared_distances(fit_state.scaler.transform(fit_summary), fit_state.centroids_scaled), axis=1
        )
        final_common_labels = np.argmin(
            squared_distances(final_state.scaler.transform(fit_summary), final_state.centroids_scaled), axis=1
        )
        final_labels = np.argmin(
            squared_distances(final_state.scaler.transform(final_summary), final_state.centroids_scaled), axis=1
        )
        singletons = [
            final_subjects[index]
            for index, label in enumerate(final_labels)
            if int(np.sum(final_labels == label)) == 1
        ]
        rows[f"k={int(k)}"] = {
            "adjusted_rand_fit_vs_refit_on_common_subjects": float(
                adjusted_rand_score(fit_labels, final_common_labels)
            ),
            "fit_cluster_counts": list(fit_state.subject_cluster_counts),
            "refit_cluster_counts": list(final_state.subject_cluster_counts),
            "refit_singleton_subjects": singletons,
        }
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def protocol_hash(config: ProbeConfig) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fold": config.fold,
        "seed": config.seed,
        "development_subjects": config.development_subjects,
        "pseudo_reference_windows": config.pseudo_reference_windows,
        "response_start_window": config.response_start_window,
        "prototype_counts": config.prototype_counts,
        "shrinkage_weights": config.shrinkage_weights,
        "heldout_videos": heldout_videos().tolist(),
        "classifier": "StandardScaler + shrinkage LDA(lsqr, auto)",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def run_probe(config: ProbeConfig, *, force: bool = False) -> dict[str, Any]:
    output = config.run_root
    result_path = output / "result.json"
    if result_path.is_file() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    log_handler = logging.FileHandler(output / "experiment.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(log_handler)
    LOGGER.setLevel(logging.INFO)
    started = utc_now()
    source, target = official_fold_subjects(config.fold)
    dev = list(config.development_subjects)
    fit = [subject for subject in source if subject not in set(dev)]
    all_videos = np.arange(VIDEOS, dtype=np.int64)
    isolated_test_videos = heldout_videos()
    isolated_train_videos = np.asarray([v for v in all_videos if v not in set(isolated_test_videos)], dtype=np.int64)
    LOGGER.info("Loading complete source-only FACED DE cache")
    de = load_de_cache(config)
    reference, response = trial_summaries(de, config)
    del de

    conditions = (
        "A0_absolute_de",
        "A1_paired_pseudo_brde",
        "A2_global_atlas",
        "A4_soft_atlas",
        "A5_shrink_pseudo_brde",
        "N1_source_video_reference",
    )
    protocols = {
        "conventional_subject_holdout": (all_videos, all_videos),
        "subject_and_stimulus_holdout": (isolated_train_videos, isolated_test_videos),
    }
    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {"representations": {}}
    diagnostics["prototype_stability"] = prototype_stability_audit(
        reference, fit, source, config.prototype_counts, config.seed
    )
    for protocol_name, (train_videos, test_videos) in protocols.items():
        LOGGER.info("Selecting and evaluating protocol %s", protocol_name)
        protocol_results: dict[str, Any] = {}
        records: dict[str, dict[str, np.ndarray]] = {}
        final_values: dict[str, np.ndarray] = {}
        for condition in conditions:
            selected, rows = select_representation(
                condition, reference, response, fit, dev,
                train_videos, test_videos if protocol_name.endswith("stimulus_holdout") else train_videos,
                config,
            )
            selection_rows.extend({"protocol": protocol_name, **row} for row in rows)
            values, atlas_diagnostics = final_features(
                condition, selected, reference, response, source, config
            )
            score, record = evaluate_condition(values, source, target, train_videos, test_videos)
            protocol_results[condition] = {
                "selected_on_source_development": selected,
                "target_test": score,
                "atlas_diagnostics": atlas_diagnostics,
            }
            records[condition] = record
            final_values[condition] = values
            metric_rows.append({
                "protocol": protocol_name,
                "condition": condition,
                "selected_k": selected.get("k", 0),
                "selected_alpha": selected.get("alpha"),
                **{key: score[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "train_samples", "test_samples", "feature_dimension")},
            })
            LOGGER.info("%s %s target ACC=%.4f Macro-F1=%.4f", protocol_name, condition, score["accuracy"], score["macro_f1"])
        results[protocol_name] = {
            "train_videos_zero_based": train_videos.tolist(),
            "test_videos_zero_based": test_videos.tolist(),
            "conditions": protocol_results,
        }
        # Diagnostic-only dual representation: retain absolute response while
        # adding the source-selected relative residual. This tests whether a
        # relative-only representation discarded emotion signal.
        dual = np.concatenate([final_values["A0_absolute_de"], final_values["A5_shrink_pseudo_brde"]], axis=-1)
        dual_score, dual_record = evaluate_condition(dual, source, target, train_videos, test_videos)
        results[protocol_name]["conditions"]["D1_absolute_plus_A5"] = {
            "selected_on_source_development": protocol_results["A5_shrink_pseudo_brde"]["selected_on_source_development"],
            "target_test": dual_score,
            "atlas_diagnostics": protocol_results["A5_shrink_pseudo_brde"]["atlas_diagnostics"],
        }
        records["D1_absolute_plus_A5"] = dual_record
        results[protocol_name]["bootstrap_subject_ci"] = bootstrap_subjects(
            records, config.bootstrap_repeats, config.seed
        )
        metric_rows.append({"protocol": protocol_name, "condition": "D1_absolute_plus_A5", **{key: dual_score[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "train_samples", "test_samples", "feature_dimension")}})
        # Identity and stimulus probes use sources only. The stimulus probe is
        # trained on fit subjects and evaluated on disjoint development subjects.
        for condition in ("A0_absolute_de", "A1_paired_pseudo_brde", "A5_shrink_pseudo_brde"):
            value = final_values[condition]
            diagnostics["representations"].setdefault(condition, {})[protocol_name] = {
                "source_subject_id_nearest_centroid_accuracy": nearest_centroid_accuracy(
                    value, source, source, isolated_train_videos, isolated_test_videos, "subject"
                ),
                "source_dev_video_id_nearest_centroid_accuracy": nearest_centroid_accuracy(
                    value, fit, dev, all_videos, all_videos, "video"
                ),
            }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset": "FACED",
        "fold": config.fold,
        "seed": config.seed,
        "protocol_hash": protocol_hash(config),
        "source_subjects": source,
        "development_subjects": dev,
        "final_fit_subjects": source,
        "target_subjects": target,
        "target_used_for_representation_or_model_selection": False,
        "target_used_for_gradients": False,
        "classifier": "StandardScaler fitted on source train + shrinkage LDA(lsqr, auto)",
        "evidence_label": config.evidence_label,
        "data_fact": "Official Processed_data contains only the final 30 seconds of each video and no pre-stimulus fixation baseline.",
        "interpretation_limit": "A1/A5 are within-video early-versus-late pseudo-reference diagnostics; they do not validate pre-stimulus baseline-relative recognition. A6 BR-GES is not run because the required true baseline covariance is unavailable.",
        "started_at": started,
        "completed_at": utc_now(),
        "results": results,
        "diagnostics": diagnostics,
    }
    _write_json(result_path, payload)
    _write_csv(output / "condition_metrics.csv", metric_rows)
    _write_csv(output / "source_selection.csv", selection_rows)
    _write_json(output / "protocol.json", {
        "protocol_hash": payload["protocol_hash"],
        "config": str(config.path),
        "heldout_videos_zero_based": isolated_test_videos.tolist(),
        "heldout_emotions": list(EMOTION_NAMES),
        "target_isolation": True,
        "limitations": payload["interpretation_limit"],
    })
    LOGGER.info("Probe complete: %s", result_path)
    LOGGER.removeHandler(log_handler)
    log_handler.close()
    return payload


def status(config: ProbeConfig) -> dict[str, Any]:
    result = config.run_root / "result.json"
    return {
        "run_root": str(config.run_root),
        "complete": result.is_file(),
        "result": str(result),
        "log": str(config.run_root / "experiment.log"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lightweight FACED normative pseudo-reference probe")
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--config", default="configs/faced/normative_probe_fold1.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = load_probe_config(args.config)
    payload = run_probe(config, force=args.force) if args.command == "run" else status(config)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
