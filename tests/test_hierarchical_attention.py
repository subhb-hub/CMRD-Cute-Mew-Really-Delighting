from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cmrd.data.records import TrialSample
from cmrd.models import HierarchicalChannelBandTransformer
from train_seediv_de_rjsd_ica import (
    _build_model,
    _train_all_source_once,
    build_parser,
    parse_args_with_config,
)


class HierarchicalAttentionTests(unittest.TestCase):
    def make_model(self) -> HierarchicalChannelBandTransformer:
        return HierarchicalChannelBandTransformer(
            input_dim=12,
            channels=3,
            classes=4,
            max_length=6,
            d_model=8,
            channel_heads=2,
            temporal_heads=2,
            temporal_layers=1,
            feedforward=16,
            dropout=0.0,
        ).eval()

    def test_shapes_and_normalized_attention(self) -> None:
        model = self.make_model()
        data = torch.randn(2, 6, 12)
        mask = torch.tensor(
            [[True, True, True, False, False, False], [True, True, True, True, True, False]]
        )
        with torch.no_grad():
            logits, attention = model(data, mask, return_attention=True)
        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(attention["band"].shape, (2, 6, 3, 4))
        self.assertEqual(attention["channel"].shape, (2, 6, 3))
        torch.testing.assert_close(
            attention["band"].sum(dim=-1)[mask], torch.ones(8, 3)
        )
        torch.testing.assert_close(
            attention["channel"].sum(dim=-1)[mask], torch.ones(8)
        )
        self.assertEqual(attention["band"][~mask].count_nonzero().item(), 0)
        self.assertEqual(attention["channel"][~mask].count_nonzero().item(), 0)

    def test_flat_and_structured_inputs_are_equivalent(self) -> None:
        model = self.make_model()
        data = torch.randn(2, 6, 12)
        mask = torch.ones(2, 6, dtype=torch.bool)
        with torch.no_grad():
            flat = model(data, mask)
            structured = model(data.reshape(2, 6, 3, 4), mask)
        torch.testing.assert_close(flat, structured)

    def test_padding_values_do_not_change_logits(self) -> None:
        model = self.make_model()
        data = torch.randn(2, 6, 12)
        mask = torch.tensor(
            [[True, True, False, False, False, False], [True, True, True, True, False, False]]
        )
        changed = data.clone()
        changed[~mask] = torch.randn_like(changed[~mask]) * 10_000
        with torch.no_grad():
            original_logits = model(data, mask)
            changed_logits = model(changed, mask)
        torch.testing.assert_close(original_logits, changed_logits)

    def test_precomputed_valid_indices_preserve_logits_and_attention(self) -> None:
        model = self.make_model()
        data = torch.randn(2, 6, 12)
        mask = torch.tensor(
            [[True, True, False, False, False, False], [True, True, True, True, False, False]]
        )
        valid_indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        with torch.no_grad():
            fallback_logits, fallback_attention = model(data, mask, return_attention=True)
            indexed_logits, indexed_attention = model(
                data,
                mask,
                return_attention=True,
                valid_indices=valid_indices,
            )
        torch.testing.assert_close(indexed_logits, fallback_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            indexed_attention["band"], fallback_attention["band"], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            indexed_attention["channel"], fallback_attention["channel"], rtol=0.0, atol=0.0
        )

    def test_training_script_builds_notebook_equivalent_model(self) -> None:
        args = build_parser().parse_args([
            "--model", "hierarchical_attention",
            "--d-model", "64",
            "--channel-heads", "4",
            "--temporal-heads", "4",
            "--temporal-layers", "3",
            "--feedforward", "256",
            "--dropout", "0.2",
        ])
        config = {
            "name": args.model,
            "channels": 62,
            "d_model": args.d_model,
            "channel_heads": args.channel_heads,
            "temporal_heads": args.temporal_heads,
            "temporal_layers": args.temporal_layers,
            "feedforward": args.feedforward,
            "dropout": args.dropout,
        }
        model = _build_model(input_dim=310, max_length=20, model_config=config)
        self.assertIsInstance(model, HierarchicalChannelBandTransformer)
        self.assertEqual(model.channels, 62)
        self.assertEqual(model.feature_slots, 5)

    def test_yaml_config_and_cli_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiment.yaml"
            path.write_text(
                """experiment:
  feature: RJSD
  folds: [1, 3]
  seeds: [42, 2026]
model:
  name: hierarchical_attention
  d_model: 64
  channel_heads: 4
  temporal_heads: 4
  temporal_layers: 3
  feedforward: 256
  dropout: 0.2
training:
  epochs: 100
  batch_size: 8
runtime:
  device: cpu
  deterministic: true
  pin_memory: false
  resume: true
""",
                encoding="utf-8",
            )
            args = parse_args_with_config(["--config", str(path), "--epochs", "2"])
        self.assertEqual(args.feature, "rjsd")
        self.assertEqual(args.fold, [1, 3])
        self.assertEqual(args.seed, [42, 2026])
        self.assertEqual(args.model, "hierarchical_attention")
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.batch_size, 8)
        self.assertTrue(args.resume)
        self.assertTrue(args.no_pin_memory)
        self.assertFalse(args.non_deterministic)

    def test_periodic_evaluation_is_monitoring_only(self) -> None:
        rng = np.random.default_rng(8)

        def samples(subject: int, count: int) -> list[TrialSample]:
            return [
                TrialSample(
                    rng.normal(size=(3 + index % 2, 6)).astype(np.float32),
                    index % 4,
                    subject,
                    1,
                    index + 1,
                    subject * 100 + index,
                )
                for index in range(count)
            ]

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "job"
            result = _train_all_source_once(
                train_samples=samples(1, 8),
                test_samples=samples(2, 4),
                model_config={
                    "name": "plain_transformer",
                    "d_model": 8,
                    "nhead": 2,
                    "layers": 1,
                    "feedforward": 16,
                    "dropout": 0.0,
                },
                training={
                    "epochs": 2,
                    "evaluation_interval": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-3,
                    "minimum_learning_rate": 1e-5,
                    "weight_decay": 0.0,
                    "label_smoothing": 0.0,
                    "gradient_clip_norm": 1.0,
                    "num_workers": 0,
                    "pin_memory": False,
                    "deterministic": True,
                },
                seed=42,
                device=torch.device("cpu"),
                output_dir=output,
                context={"target_subject": 2, "feature": "de"},
            )
            with (output / "evaluations.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue((output / "final.pt").is_file())
        self.assertEqual([int(row["epoch"]) for row in rows], [1, 2])
        self.assertTrue(result["target_evaluated_during_training"])
        self.assertEqual(result["final_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
