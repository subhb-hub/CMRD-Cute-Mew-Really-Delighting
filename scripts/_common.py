from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmrd.config import load_config
from cmrd.preprocessing import preprocess_de, preprocess_rd
from cmrd.training import run_experiment


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def preprocess_cli(feature: str) -> None:
    parser = argparse.ArgumentParser(description=f"Preprocess {feature.upper()} trials with reproducible metadata and caching.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--fold", type=int, help="RD fold to build; accepted but unnecessary for DE")
    parser.add_argument("--force", action="store_true", help="Rebuild matching cached artifacts")
    parser.add_argument("--resume", action="store_true", help="Reuse complete per-trial artifacts")
    args = parser.parse_args()
    _logging()
    config = load_config(args.config, args.overrides, expected_feature=feature)
    if feature == "de":
        if args.fold is not None:
            logging.warning("DE features are fold-independent; --fold is ignored")
        output = preprocess_de(config, args.force, args.resume)
    else:
        output = preprocess_rd(config, args.fold, args.force, args.resume)
    logging.info("Processed artifacts: %s", output)


def train_cli(feature: str) -> None:
    parser = argparse.ArgumentParser(description=f"Tune or train the {feature.upper()} masked-Transformer LOSO baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=("tune", "final"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--fold", type=int, help="Run one outer target fold")
    parser.add_argument("--force", action="store_true", help="Recompute artifacts inside a resumed run")
    parser.add_argument("--resume", action="store_true", help="Resume the latest matching run")
    args = parser.parse_args()
    _logging()
    config = load_config(args.config, args.overrides, expected_feature=feature)
    output = run_experiment(config, args.mode, args.fold, args.resume, args.force, sys.argv)
    logging.info("Run directory: %s", output)

