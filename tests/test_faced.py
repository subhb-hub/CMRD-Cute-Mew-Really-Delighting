from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cmrd.config import load_config
from cmrd.faced import (
    CRITICAL_METADATA,
    EMOTION_NAMES,
    SUBJECTS,
    VIDEO_LABELS,
    official_fold_subjects,
    validate_faced_data,
)
from cmrd.faced_runner import CONDITIONS, _declared_tasks, experiment_settings, protocol_payload


ROOT = Path(__file__).resolve().parents[1]


class FacedMetadataTests(unittest.TestCase):
    def test_video_labels_match_official_nine_class_order(self) -> None:
        self.assertEqual(len(EMOTION_NAMES), 9)
        self.assertEqual(VIDEO_LABELS.tolist(), [
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
            4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8,
        ])

    def test_official_contiguous_subject_folds_cover_each_subject_once(self) -> None:
        targets = []
        for fold in range(1, 11):
            source, target = official_fold_subjects(fold)
            self.assertFalse(set(source) & set(target))
            self.assertEqual(len(source) + len(target), SUBJECTS)
            self.assertEqual(len(target), 12 if fold < 10 else 15)
            targets.extend(target)
        self.assertEqual(targets, list(range(SUBJECTS)))

    def test_shallow_audit_checks_names_recording_subjects_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "Processed_data"
            processed.mkdir()
            for subject in range(SUBJECTS):
                (processed / f"sub{subject:03d}.pkl").touch()
            for name in CRITICAL_METADATA:
                (root / name).touch()
            with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=["name", "dataFileMD5Hex"]).writeheader()
            with (root / "Recording_info.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["sub", "Cohort ", "Sample_rate", "Unit"]
                )
                writer.writeheader()
                for subject in range(SUBJECTS):
                    writer.writerow({
                        "sub": f"sub{subject:03d}",
                        "Cohort ": 1 if subject <= 60 else 2,
                        "Sample_rate": 250 if subject < 55 else 1000,
                        "Unit": "uV" if subject < 33 else "V",
                    })
            fake = np.zeros((28, 32, 7500), dtype=np.float64)
            with patch("cmrd.faced.load_processed_subject", return_value=fake) as loader:
                audit = validate_faced_data(processed, root)
            self.assertEqual(audit["official_fold_sizes"], [12] * 9 + [15])
            self.assertEqual(audit["checked_subjects"], [0, 36, 60, 61, 122])
            self.assertEqual(loader.call_count, 5)


class FacedProtocolTests(unittest.TestCase):
    def test_config_locks_base_architecture_and_adapted_training(self) -> None:
        config = load_config(ROOT / "configs" / "faced" / "native_compact_base.yaml")
        settings = experiment_settings(config)
        self.assertEqual(tuple(settings["conditions"]), tuple(CONDITIONS))
        self.assertEqual(CONDITIONS["de_base"], "de")
        self.assertEqual(settings["architecture"]["d_model"], 128)
        self.assertEqual(config.raw["dataset"]["channels"], 30)
        self.assertEqual(config.raw["dataset"]["recorded_channels"], 32)
        self.assertEqual(config.raw["training"]["epochs"], 100)
        self.assertEqual(config.raw["training"]["batch_size"], 64)
        self.assertEqual(config.raw["training"]["label_smoothing"], 0.05)

    def test_protocol_declares_source_only_state_and_no_target_training(self) -> None:
        config = load_config(ROOT / "configs" / "faced" / "native_compact_base.yaml")
        payload = protocol_payload(config, {"metadata_md5": {}})
        self.assertEqual(payload["reference_fit_scope"], "outer-fold source subjects only")
        self.assertFalse(payload["target_used_during_training"])
        self.assertEqual(payload["processed_shape"], [28, 32, 7500])
        self.assertEqual(payload["feature_shape"], [30, 30, 5])
        self.assertEqual(len(payload["fold_subjects"]["fold-10"]), 15)

    def test_full_matrix_has_ten_folds_by_three_representations(self) -> None:
        tasks = _declared_tasks(list(range(1, 11)), list(CONDITIONS))
        self.assertEqual(len(tasks), 30)
        self.assertEqual(len({task["task_id"] for task in tasks}), 30)

    def test_powershell_entrypoint_exposes_safe_stages(self) -> None:
        script = (ROOT / "scripts" / "run_faced_native_compact.ps1").read_text(encoding="utf-8")
        for stage in ("Validate", "Spectra", "Lock", "Smoke", "DE", "SqrtJsd", "FisherRao", "Summarize"):
            self.assertIn(f'"{stage}"', script)
        self.assertIn("conda run --no-capture-output", script)


if __name__ == "__main__":
    unittest.main()
