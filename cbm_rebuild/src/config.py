from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML when available, with a dependency-free JSON-compatible fallback."""
    config_path = Path(path).resolve()
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        config = yaml.safe_load(text)
    except ModuleNotFoundError:
        try:
            config = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{config_path} requires PyYAML, which is not installed, and is not "
                "JSON-compatible YAML. Install pyyaml or keep the config JSON-compatible."
            ) from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()

