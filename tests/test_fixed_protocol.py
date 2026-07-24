from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmrd.config import ExperimentConfig, load_config
from cmrd.data.records import TrialSample
from cmrd.fixed_protocol import (
    HistogramHierarchicalTransformer,
    SmallPooledMLP,
    _loader,
    _shutdown_persistent_loader,
    assert_fixed_protocol,
    build_model,
    clear_feature_cache,
    feature_cache_info,
    load_representation_samples,
    pooled_vectors,
)
from cmrd.training.engine import SequenceDataset, collate_sequences
from torch.utils.data import DataLoader
from cmrd.fixed_protocol_runner import (
    _neural_loader_settings,
    declared_tasks,
    invalidate_model_results,
    lock_dataset_epoch,
    matrix_status,
    run_fold_task,
    run_matrix,
    summarize_matrix,
    validate_fixed_cache,
)
from cmrd.io import read_json, write_json


def signature_payload() -> dict[str, object]:
    return {
        "window_seconds": 1.0,
        "hop_seconds": 1.0,
        "sampling_rate": 200,
        "hist_bins_per_band": 32,
        "welch": {"noverlap": 0},
        "rjsd_definition": "Jensen-Shannon divergence(P_window, Q_source_train)",
    }


def entry(subject: int, path: str) -> dict[str, object]:
    return {
        "trial_id": f"sub-{subject:02d}",
        "label": (subject - 1) % 3,
        "subject": subject,
        "session": 1,
        "trial": 1,
        "source_index": subject - 1,
        "de_phist_path": path,
    }


def make_cache(parent: Path) -> Path:
    cache = parent / "fixed-cache"
    (cache / "trials").mkdir(parents=True)
    for subject in range(1, 16):
        rng = np.random.default_rng(subject)
        histogram = rng.random((3, 62, 5, 32), dtype=np.float32)
        histogram /= histogram.sum(axis=-1, keepdims=True)
        np.savez_compressed(
            cache / "trials" / f"sub-{subject:02d}.npz",
            de=rng.normal(size=(3, 62, 5)).astype(np.float32),
            p_hist=histogram.astype(np.float16),
        )
    write_json(cache / "environment.json", {"signature_payload": signature_payload()})
    write_json(cache / "pipeline_manifest.json", {
        "dataset": "SEED",
        "preprocessing_signature": "fixed-test",
        "all_15_folds_complete": True,
    })
    subjects = set(range(1, 16))
    for target in range(1, 16):
        sources = sorted(subjects - {target})
        validation = sources[:2]
        training = sources[2:]
        groups = {
            "train": [entry(subject, f"trials/sub-{subject:02d}.npz") for subject in training],
            "validation": [entry(subject, f"trials/sub-{subject:02d}.npz") for subject in validation],
            "test": [entry(target, f"trials/sub-{target:02d}.npz")],
        }
        write_json(cache / "folds" / f"fold-{target:02d}" / "manifest.json", {
            "preprocessing_signature": "fixed-test",
            "split": {
                "train_subjects": training,
                "validation_subjects": validation,
                "target_subject": target,
            },
            "groups": groups,
        })
    return cache


class FixedProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs" / "fixed_protocol" / "seed_rjsd_1s1s.yaml")

    def config_with_fixed_epoch(self, epoch: int) -> ExperimentConfig:
        raw = copy.deepcopy(self.config.raw)
        raw["matrix"]["fixed_epoch"] = epoch
        return ExperimentConfig(path=self.config.path, raw=raw)

    def test_protocol_rejects_overlap(self) -> None:
        payload = signature_payload()
        assert_fixed_protocol(payload)
        payload["hop_seconds"] = 0.5
        with self.assertRaisesRegex(ValueError, "hop_seconds"):
            assert_fixed_protocol(payload)

    def test_declared_matrix_has_300_tasks_per_dataset_and_classical_first(self) -> None:
        tasks = declared_tasks(self.config, "protocol")
        self.assertEqual(len(tasks), 300)
        self.assertEqual(tasks[0]["model"], "logistic_regression")
        self.assertEqual(tasks[59]["model"], "logistic_regression")
        self.assertEqual(tasks[60]["model"], "linear_svm")
        self.assertEqual(tasks[120]["model"], "small_mlp")
        self.assertEqual(len({task["task_id"] for task in tasks}), 300)

    def test_histogram_hierarchical_adapter_uses_complete_input(self) -> None:
        model = build_model({
            "name": "hierarchical_attention",
            "d_model": 8,
            "heads": 2,
            "layers": 1,
            "feedforward": 16,
            "dropout": 0.0,
        }, input_dim=62 * 5 * 32, classes=3, max_length=4)
        self.assertIsInstance(model, HistogramHierarchicalTransformer)
        data = torch.rand(2, 4, 62 * 5 * 32)
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        self.assertEqual(tuple(model(data, mask).shape), (2, 3))

    def test_small_mlp_uses_trial_mean_and_std_features(self) -> None:
        samples = [
            type("Sample", (), {"x": np.asarray([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32)})(),
            type("Sample", (), {"x": np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)})(),
        ]
        features = pooled_vectors(samples)
        self.assertEqual(features.shape, (2, 4))
        np.testing.assert_allclose(features[:, :2], np.ones((2, 2)), atol=1e-6)
        self.assertFalse(np.allclose(features[0, 2:], features[1, 2:]))
        model = SmallPooledMLP(4, 3, 8, 0.0)
        self.assertEqual(tuple(model(torch.from_numpy(features)).shape), (2, 3))

    def test_cache_audit_confirms_14_source_one_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = make_cache(Path(temporary))
            audit = validate_fixed_cache(self.config, cache)
        self.assertEqual(len(audit["folds"]), 15)
        self.assertEqual(audit["folds"][0]["target_subject"], 1)
        self.assertEqual(len(audit["folds"][0]["source_train_subjects"]), 12)
        self.assertEqual(len(audit["folds"][0]["source_validation_subjects"]), 2)

    def test_feature_archive_cache_avoids_repeated_decompression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = make_cache(Path(temporary))
            item = entry(1, "trials/sub-01.npz")
            clear_feature_cache()
            load_representation_samples(cache, [item], "de_raw")
            load_representation_samples(cache, [item], "de_raw")
            info = feature_cache_info()
        self.assertEqual(info["de"]["misses"], 1)
        self.assertGreaterEqual(info["de"]["hits"], 1)

    def test_de_loader_accepts_deap_32_channel_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            trial = cache / "trials" / "sub-01.npz"
            trial.parent.mkdir(parents=True)
            np.savez_compressed(trial, de=np.zeros((60, 32, 5), dtype=np.float32))
            item = entry(1, "trials/sub-01.npz")
            clear_feature_cache()
            samples = load_representation_samples(
                cache, [item], "de_raw", channels=32
            )
        self.assertEqual(samples[0].x.shape, (60, 160))

    def test_two_persistent_workers_preserve_old_batches(self) -> None:
        rng = np.random.default_rng(99)
        source = [
            TrialSample(
                rng.normal(size=(2 + index % 4, 6)).astype(np.float32),
                index,
                1,
                1,
                index + 1,
                index,
            )
            for index in range(17)
        ]
        mean = np.linspace(-0.2, 0.3, 6, dtype=np.float32)
        std = np.linspace(0.7, 1.2, 6, dtype=np.float32)
        old = DataLoader(
            SequenceDataset(source, mean, std),
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_sequences,
            generator=torch.Generator().manual_seed(42),
        )
        optimized = _loader(
            source,
            mean,
            std,
            4,
            True,
            42,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=1,
            cache_normalized=True,
        )
        expected = [torch.cat([labels for _, _, labels in old]).tolist() for _ in range(3)]
        actual = [torch.cat([labels for _, _, labels in optimized]).tolist() for _ in range(3)]
        self.assertEqual(actual, expected)
        self.assertIsNotNone(optimized._iterator)
        _shutdown_persistent_loader(optimized)
        self.assertIsNone(optimized._iterator)

    def test_histogram_loader_avoids_windows_shared_mappings(self) -> None:
        matrix = {
            "dataloader_workers": 2,
            "persistent_workers": True,
        }
        histogram = _neural_loader_settings(matrix, "histogram", "hierarchical_attention")
        compact = _neural_loader_settings(matrix, "rjsd_zscore", "hierarchical_attention")
        self.assertEqual(histogram["num_workers"], 0)
        self.assertFalse(histogram["persistent_workers"])
        self.assertIn("no_shared_mapping", histogram["loader_policy"])
        self.assertEqual(compact["num_workers"], 2)
        self.assertTrue(compact["persistent_workers"])

    def test_fold_task_locks_model_before_target_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = make_cache(root / "cache-parent")
            run_root = root / "run"
            task = declared_tasks(
                self.config,
                "protocol",
                representations=["rjsd_zscore"],
                models=["small_mlp"],
                folds=[1],
            )[0]
            result = run_fold_task(
                self.config,
                {"cache_root": str(cache), "preprocessing_signature": "fixed-test"},
                "protocol",
                task,
                run_root,
                torch.device("cpu"),
                fixed_epoch_override=1,
            )
            output = run_root / "seed" / "rjsd_zscore" / "small_mlp" / "fold-01" / "seed-42"
            audit = read_json(output / "protocol_audit.json")
            checkpoint = torch.load(output / "locked_source_model.pt", map_location="cpu", weights_only=False)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["fixed_epoch"], 1)
        self.assertEqual(result["formal_source_subject_count"], 14)
        events = audit["events"]
        self.assertEqual(events[0]["event"], "dataset_epoch_declared")
        locked = next(index for index, event in enumerate(events) if event["event"] == "locked_source_model_complete")
        target = next(index for index, event in enumerate(events) if event["event"] == "target_final_evaluation_started")
        self.assertLess(locked, target)
        self.assertFalse(events[locked]["target_arrays_loaded"])
        self.assertEqual(audit["formal_source_subject_count"], 14)
        self.assertEqual(audit["formal_source_validation_subjects"], [])
        self.assertEqual(result["model_input_adapter"], "trial_mean_std_source_zscore_v1")
        self.assertEqual(checkpoint["input_adapter"], "trial_mean_std_source_zscore_v1")
        self.assertEqual(checkpoint["input_dim"], 62 * 5 * 2)
        self.assertEqual(checkpoint["training"]["batch_size"], 64)
        self.assertEqual(checkpoint["training"]["precision"], "float32")
        self.assertEqual(result["dataloader_workers"], 0)
        self.assertFalse(result["persistent_workers"])
        self.assertEqual(result["normalization_cache"], "trial_pool_once_source_zscore_v1")
        self.assertEqual(result["loader_policy"], "pooled_features_main_process")

    def test_formal_matrix_refuses_to_run_without_dataset_epoch_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = make_cache(root / "cache-parent")
            with self.assertRaisesRegex(FileNotFoundError, "lock-epoch"):
                run_matrix(
                    self.config_with_fixed_epoch(1),
                    root / "run",
                    cache,
                    ["rjsd_zscore"],
                    ["small_mlp"],
                    [1],
                    resume=True,
                    retry_failed=True,
                    max_tasks=1,
                )

    def test_status_and_strict_summary_reject_incomplete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            write_json(run_root / "matrix_manifest.json", {
                "expected_full_matrix_tasks": 600,
                "tasks": {
                    "one": {"task_id": "one", "status": "failed", "error": "boom"},
                    "two": {"task_id": "two", "status": "pending"},
                },
            })
            status = matrix_status(run_root)
            with self.assertRaisesRegex(RuntimeError, "600 complete"):
                summarize_matrix(run_root)
        self.assertEqual(status["failed"], 1)
        self.assertEqual(status["pending"], 1)

    def test_matrix_resume_reuses_matching_complete_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = make_cache(root / "cache-parent")
            run_root = root / "run"
            config = self.config_with_fixed_epoch(1)
            lock = lock_dataset_epoch(config, run_root, cache)
            self.assertEqual(lock["fixed_epoch"], 1)
            self.assertFalse(lock["target_metrics_used"])
            first = run_matrix(
                config,
                run_root,
                cache,
                ["rjsd_zscore"],
                ["small_mlp"],
                [1],
                resume=True,
                retry_failed=True,
                max_tasks=1,
            )
            second = run_matrix(
                config,
                run_root,
                cache,
                ["rjsd_zscore"],
                ["small_mlp"],
                [1],
                resume=True,
                retry_failed=False,
                max_tasks=1,
            )
            manifest = read_json(run_root / "matrix_manifest.json")
            task = next(iter(manifest["tasks"].values()))
        self.assertEqual(first["complete"], 1)
        self.assertEqual(second["complete"], 1)
        self.assertEqual(task["attempts"], 1)

    def test_classical_folds_complete_with_parallel_manifest_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = make_cache(root / "cache-parent")
            run_root = root / "run"
            config = self.config_with_fixed_epoch(1)
            config.raw["matrix"]["classical_workers"] = 4
            config.raw["matrix"]["classical_threads_per_worker"] = 1
            lock_dataset_epoch(config, run_root, cache)
            status = run_matrix(
                config,
                run_root,
                cache,
                ["de_raw"],
                ["logistic_regression", "linear_svm"],
                [1, 2, 3, 4],
                resume=True,
                retry_failed=True,
                max_tasks=None,
            )
            manifest = read_json(run_root / "matrix_manifest.json")
            selected = list(manifest["tasks"].values())
            svm_options = [
                read_json(run_root / task["result_path"])["classical_estimator_options"]
                for task in selected if task["model"] == "linear_svm"
            ]
        self.assertEqual(status["complete"], 8)
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(task["status"] == "complete" for task in selected))
        self.assertEqual(svm_options, [{"tol": 0.001, "max_iter": 5000}] * 4)

    def test_invalidate_model_preserves_other_completed_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = {
                "mlp": {"task_id": "mlp", "dataset": "seed", "representation": "histogram", "model": "small_mlp", "fold": 1, "seed": 42, "status": "complete", "attempts": 1},
                "svm": {"task_id": "svm", "dataset": "seed", "representation": "histogram", "model": "linear_svm", "fold": 1, "seed": 42, "status": "complete", "attempts": 1},
            }
            write_json(root / "matrix_manifest.json", {"tasks": tasks})
            mlp_output = root / "seed" / "histogram" / "small_mlp" / "fold-01" / "seed-42"
            mlp_output.mkdir(parents=True)
            write_json(mlp_output / "result.json", {"invalid": True})
            audit = invalidate_model_results(root, "seed", "small_mlp", "test")
            manifest = read_json(root / "matrix_manifest.json")
        self.assertEqual(len(audit["reset_tasks"]), 1)
        self.assertFalse(mlp_output.exists())
        self.assertEqual(manifest["tasks"]["mlp"]["status"], "pending")
        self.assertEqual(manifest["tasks"]["svm"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
