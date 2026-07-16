from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmrd.config import ExperimentConfig, load_config
from cmrd.data.records import TrialSample
from cmrd.io import read_json
from cmrd.io import write_json, write_npz
from cmrd.preprocessing import cache_root
from cmrd.training.artifacts import create_run
from cmrd.training.engine import (
    LegacyDataLoaderRandomSampler,
    SequenceDataset,
    train_once,
)
from cmrd.training.experiment import run_experiment


def samples(subject: int, count: int, seed: int) -> list[TrialSample]:
    rng = np.random.default_rng(seed)
    result = []
    for index in range(count):
        label = index % 3
        x = rng.normal(loc=label * 0.4, size=(3 + index % 3, 6)).astype(np.float32)
        result.append(TrialSample(x, label, subject, 1, index + 1, seed * 100 + index))
    return result


class TrainingTests(unittest.TestCase):
    def test_cached_normalization_is_identical_to_lazy_normalization(self) -> None:
        source = samples(1, 7, 9)
        mean = np.linspace(-0.2, 0.3, 6, dtype=np.float32)
        std = np.linspace(0.5, 1.5, 6, dtype=np.float32)
        lazy = SequenceDataset(source, mean, std)
        cached = SequenceDataset(source, mean, std, cache_normalized=True)
        self.assertIsNone(cached.samples)
        for index in range(len(source)):
            lazy_value, lazy_label = lazy[index]
            cached_value, cached_label = cached[index]
            torch.testing.assert_close(cached_value, lazy_value, rtol=0.0, atol=0.0)
            self.assertEqual(cached_label, lazy_label)

    def test_persistent_worker_sampler_preserves_old_epoch_orders(self) -> None:
        dataset = TensorDataset(torch.arange(17))
        old_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(42),
        )
        expected = [
            torch.cat([batch[0] for batch in old_loader]).tolist()
            for _ in range(4)
        ]
        sampler = LegacyDataLoaderRandomSampler(dataset, 42)
        compatible_loader = DataLoader(
            dataset,
            batch_size=4,
            sampler=sampler,
            num_workers=0,
            generator=torch.Generator().manual_seed(1_000_045),
        )
        actual = [
            torch.cat([batch[0] for batch in compatible_loader]).tolist()
            for _ in range(4)
        ]
        self.assertEqual(actual, expected)

    def test_one_epoch_writes_metrics_checkpoint_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "job"
            result = train_once(
                train_samples=samples(1, 9, 1),
                validation_samples=samples(2, 6, 2),
                test_samples=samples(3, 6, 3),
                model_config={"d_model": 8, "nhead": 2, "layers": 1, "feedforward": 16, "dropout": 0.0},
                training={
                    "batch_size": 3, "learning_rate": 1e-3, "minimum_learning_rate": 1e-5,
                    "weight_decay": 0.0, "label_smoothing": 0.0, "epochs": 1,
                    "early_stopping_patience": 1, "gradient_clip_norm": 1.0,
                    "num_workers": 0, "pin_memory": False, "deterministic": True,
                },
                classes=3,
                seed=7,
                device=torch.device("cpu"),
                output_dir=output,
                context={"target_subject": 3, "mode": "final"},
            )
            self.assertIn("test", result)
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "epochs.csv").is_file())
            self.assertEqual(read_json(output / "result.json")["seed"], 7)

    def test_run_directory_is_unique_and_resumable(self) -> None:
        base = load_config(ROOT / "configs" / "seed" / "de.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            raw = copy.deepcopy(base.raw)
            raw["paths"]["run_root"] = temporary
            config = ExperimentConfig(base.path, raw)
            first = create_run(config, "tune", False, ["test"], {"python": "test"})
            resumed = create_run(config, "tune", True, ["test"], {"python": "test"})
            second = create_run(config, "tune", False, ["test"], {"python": "test"})
            self.assertEqual(first, resumed)
            self.assertNotEqual(first, second)

    def test_end_to_end_tune_final_and_resume_never_tests_candidates(self) -> None:
        base = load_config(ROOT / "configs" / "seed" / "de.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            raw = copy.deepcopy(base.raw)
            raw["paths"]["processed_root"] = str(temporary_path / "processed")
            raw["paths"]["run_root"] = str(temporary_path / "runs")
            raw["training"].update({"seeds": [7], "device": "cpu", "epochs": 1, "batch_size": 8, "early_stopping_patience": 1, "pin_memory": False})
            raw["tuning"]["seed"] = 5
            raw["tuning"]["architectures"] = [
                {"d_model": 8, "nhead": 2, "layers": 1, "feedforward": 16},
                {"d_model": 8, "nhead": 2, "layers": 1, "feedforward": 24},
                {"d_model": 8, "nhead": 2, "layers": 2, "feedforward": 16},
            ]
            config = ExperimentConfig(base.path, raw)
            root = cache_root(config)
            entries = []
            rng = np.random.default_rng(11)
            source_index = 0
            for subject in range(1, 16):
                for trial in range(1, 4):
                    label = trial - 1
                    relative = f"trials/sub-{subject:02d}_ses-01_trial-{trial:02d}.npz"
                    x = rng.normal(loc=label * 0.2, size=(3 + trial, 6)).astype(np.float32)
                    write_npz(root / relative, X=x)
                    entries.append({
                        "trial_id": f"sub-{subject:02d}_ses-01_trial-{trial:02d}", "path": relative,
                        "label": label, "subject": subject, "session": 1, "trial": trial,
                        "source_index": source_index, "shape": list(x.shape),
                    })
                    source_index += 1
            write_json(root / "manifest.json", {
                "preprocessing_signature": config.preprocessing_signature(), "feature": "de", "trials": entries,
            })

            tune_run = run_experiment(config, "tune", 1, False, False, ["synthetic", "tune"])
            selection = read_json(tune_run / "selected_by_fold.json")
            self.assertIn("01", selection["folds"])
            self.assertFalse(selection["folds"]["01"]["target_loaded"])
            candidate_result = next((tune_run / "folds" / "fold-01" / "candidates").glob("*/result.json"))
            candidate_payload = read_json(candidate_result)
            self.assertNotIn("test", candidate_payload)
            self.assertFalse(candidate_payload["target_loaded"])

            final_run = run_experiment(config, "final", 1, False, False, ["synthetic", "final"])
            self.assertTrue((final_run / "summary.json").is_file())
            self.assertTrue((final_run / "folds" / "fold-01" / "seed-7" / "best.pt").is_file())
            resumed = run_experiment(config, "final", 1, True, False, ["synthetic", "final", "--resume"])
            self.assertEqual(final_run, resumed)


if __name__ == "__main__":
    unittest.main()
