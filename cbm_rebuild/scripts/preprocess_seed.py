from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.preprocessing.pipeline import run_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SEED time-domain EEG into padded DE trials.")
    parser.add_argument("--config", required=True, help="Path to the SEED experiment config")
    args = parser.parse_args()
    config = load_config(args.config)
    run_preprocessing(config, "SEED")


if __name__ == "__main__":
    main()

