from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from cmrd.config import ExperimentConfig
from cmrd.io import read_json, write_json


def create_run(config: ExperimentConfig, mode: str, resume: bool, command: list[str], environment: dict[str, object]) -> Path:
    parent = config.run_root / config.dataset / config.feature
    parent.mkdir(parents=True, exist_ok=True)
    config_hash = config.hash()
    if resume:
        for candidate in sorted(parent.glob(f"*_{mode}_{config_hash}"), reverse=True):
            manifest_path = candidate / "manifest.json"
            if manifest_path.is_file():
                manifest = read_json(manifest_path)
                if manifest.get("config_hash") == config_hash and manifest.get("mode") == mode:
                    return candidate
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run = parent / f"{timestamp}_{mode}_{config_hash}"
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "resolved_config.json", config.canonical())
    write_json(run / "environment.json", environment)
    write_json(run / "manifest.json", {
        "schema_version": 1,
        "mode": mode,
        "dataset": config.dataset,
        "feature": config.feature,
        "config_hash": config_hash,
        "preprocessing_signature": config.preprocessing_signature(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "status": "running",
    })
    return run


def set_run_status(run: Path, status: str, **extra: object) -> None:
    manifest = read_json(run / "manifest.json")
    manifest.update(extra)
    manifest["status"] = status
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run / "manifest.json", manifest)


def latest_selection(config: ExperimentConfig) -> Path | None:
    configured = str(config.raw["tuning"].get("selection_file", "")).strip()
    if configured:
        path = config.resolve_path(configured)
        return path if path.is_file() else None
    parent = config.run_root / config.dataset / config.feature
    if not parent.is_dir():
        return None
    for run in sorted(parent.glob("*_tune_*"), reverse=True):
        selection = run / "selected_by_fold.json"
        manifest_path = run / "manifest.json"
        if selection.is_file() and manifest_path.is_file() and read_json(manifest_path).get("config_hash") == config.hash():
            return selection
    return None
