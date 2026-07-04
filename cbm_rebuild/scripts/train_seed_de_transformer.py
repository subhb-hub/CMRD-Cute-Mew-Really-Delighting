from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.training.train_loop import run_loso


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SEED DE + plain Transformer LOSO baseline.")
    parser.add_argument("--config", required=True, help="Path to the SEED experiment config")
    args = parser.parse_args()
    config = load_config(args.config)
    run_loso(config, "SEED")


if __name__ == "__main__":
    main()

