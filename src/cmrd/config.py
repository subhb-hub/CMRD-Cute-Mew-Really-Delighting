from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        value = yaml.safe_load(text)
    except ModuleNotFoundError:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = _simple_yaml(text)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def _scalar(text: str) -> Any:
    value = text.strip()
    if not value:
        return None
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        return [] if not body else [_scalar(part) for part in body.split(",")]
    if value.startswith("{") and value.endswith("}"):
        body = value[1:-1].strip()
        result: dict[str, Any] = {}
        if body:
            for part in body.split(","):
                key, item = part.split(":", 1)
                result[key.strip().strip("\"'")] = _scalar(item)
        return result
    if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value):
        return float(value)
    return value


def _simple_yaml(text: str) -> dict[str, Any]:
    """Parse the deliberately small YAML subset used by bundled configs."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, stripped))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        is_list = lines[index][1].startswith("- ")
        container: Any = [] if is_list else {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent != indent:
                raise ValueError(f"Invalid YAML indentation near {content!r}")
            if is_list:
                if not content.startswith("- "):
                    break
                item = content[2:].strip()
                if not item:
                    if index + 1 >= len(lines) or lines[index + 1][0] <= indent:
                        container.append(None)
                        index += 1
                    else:
                        child, index = parse_block(index + 1, lines[index + 1][0])
                        container.append(child)
                else:
                    container.append(_scalar(item))
                    index += 1
            else:
                if content.startswith("- ") or ":" not in content:
                    break
                key, raw_value = content.split(":", 1)
                key = key.strip()
                raw_value = raw_value.strip()
                if raw_value:
                    container[key] = _scalar(raw_value)
                    index += 1
                elif index + 1 < len(lines) and lines[index + 1][0] > indent:
                    child, index = parse_block(index + 1, lines[index + 1][0])
                    container[key] = child
                else:
                    container[key] = {}
                    index += 1
        return container, index

    if not lines:
        return {}
    value, final = parse_block(0, lines[0][0])
    if final != len(lines) or not isinstance(value, dict):
        raise ValueError("Configuration is not a supported YAML mapping")
    return value


def _parse_value(text: str) -> Any:
    try:
        import yaml

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    result = copy.deepcopy(data)
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError(f"Override must be dotted.key=value, got {expression!r}")
        dotted, raw = expression.split("=", 1)
        keys = [part for part in dotted.split(".") if part]
        if not keys:
            raise ValueError(f"Empty override key: {expression!r}")
        cursor: dict[str, Any] = result
        for key in keys[:-1]:
            child = cursor.get(key)
            if not isinstance(child, dict):
                raise KeyError(f"Unknown/non-mapping override path: {dotted}")
            cursor = child
        if keys[-1] not in cursor:
            raise KeyError(f"Unknown override key: {dotted}")
        cursor[keys[-1]] = _parse_value(raw)
    return result


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def dataset(self) -> str:
        return str(self.raw["experiment"]["dataset"]).lower()

    @property
    def feature(self) -> str:
        return str(self.raw["experiment"]["feature"]).lower()

    @property
    def data_root(self) -> Path:
        return self.resolve_path(self.raw["paths"]["data_root"])

    @property
    def processed_root(self) -> Path:
        return self.resolve_path(self.raw["paths"]["processed_root"])

    @property
    def run_root(self) -> Path:
        return self.resolve_path(self.raw["paths"]["run_root"])

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def canonical(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)

    def hash(self, section: str | None = None, length: int = 12) -> str:
        value: Any = self.raw if section is None else self.raw[section]
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:length]

    def preprocessing_signature(self) -> str:
        payload = {
            "dataset": self.raw["dataset"],
            "signal": self.raw["signal"],
            "feature": self.raw["feature"],
            "feature_name": self.feature,
            "split": self.raw["split"],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def _require(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing {context} keys: {missing}")


def validate_config(raw: dict[str, Any], expected_feature: str | None = None) -> None:
    _require(raw, ("experiment", "paths", "dataset", "signal", "feature", "split", "model", "training", "tuning", "output"), "top-level")
    _require(raw["experiment"], ("dataset", "feature", "model"), "experiment")
    dataset = str(raw["experiment"]["dataset"]).lower()
    feature = str(raw["experiment"]["feature"]).lower()
    if dataset not in {"seed", "seediv", "deap", "faced"}:
        raise ValueError(f"dataset must be seed, seediv, deap, or faced, got {dataset!r}")
    if feature not in {"de", "rd"}:
        raise ValueError(f"feature must be de or rd, got {feature!r}")
    if expected_feature and feature != expected_feature:
        raise ValueError(f"This entry point requires feature={expected_feature}, config has {feature}")
    _require(raw["paths"], ("data_root", "processed_root", "run_root"), "paths")
    _require(raw["dataset"], ("raw_dir", "channels", "subjects", "classes"), "dataset")
    _require(raw["signal"], ("original_rate", "target_rate", "broad_band_hz", "filter_order", "window_seconds", "hop_seconds", "bands_hz"), "signal")
    expected_shape = {
        "seed": (15, 62),
        "seediv": (15, 62),
        "deap": (32, 32),
        "faced": (123, 30),
    }[dataset]
    actual_shape = (int(raw["dataset"]["subjects"]), int(raw["dataset"]["channels"]))
    if actual_shape != expected_shape:
        raise ValueError(
            f"{dataset.upper()} requires subjects/channels={expected_shape}, got {actual_shape}"
        )
    if dataset == "deap":
        _require(raw["dataset"], ("label_file", "label_target"), "DEAP dataset")
        if str(raw["dataset"]["label_target"]).lower() not in {
            "quadrant", "valence", "arousal"
        }:
            raise ValueError("DEAP label_target must be quadrant, valence, or arousal")
        expected_classes = 4 if str(raw["dataset"]["label_target"]).lower() == "quadrant" else 2
        if int(raw["dataset"]["classes"]) != expected_classes:
            raise ValueError(
                f"DEAP label_target={raw['dataset']['label_target']} requires classes={expected_classes}"
            )
    if dataset == "faced":
        _require(raw["dataset"], ("metadata_dir", "label_target", "recorded_channels"), "FACED dataset")
        if str(raw["dataset"]["label_target"]).lower() != "emotion_9":
            raise ValueError("FACED label_target must be emotion_9")
        if int(raw["dataset"]["classes"]) != 9:
            raise ValueError("FACED emotion_9 requires classes=9")
        if int(raw["dataset"]["recorded_channels"]) != 32:
            raise ValueError("FACED Processed_data must declare recorded_channels=32")
    bands = raw["signal"]["bands_hz"]
    if not isinstance(bands, dict) or len(bands) != 5:
        raise ValueError("Exactly five ordered frequency bands are required")
    if dataset == "faced":
        if str(raw["split"].get("protocol", "")).lower() != "official_subject_10fold":
            raise ValueError("FACED requires split.protocol=official_subject_10fold")
        if int(raw["split"].get("folds", 0)) != 10:
            raise ValueError("FACED requires split.folds=10")
    elif int(raw["split"].get("validation_subjects", 0)) != 2:
        raise ValueError("Protocol requires exactly two complete source validation subjects")
    if int(raw["model"]["d_model"]) % int(raw["model"]["nhead"]) != 0:
        raise ValueError("model.d_model must be divisible by model.nhead")
    if not raw["training"].get("seeds"):
        raise ValueError("training.seeds must not be empty")
    if feature == "rd":
        _require(raw["feature"], ("hist_bins_per_band", "spectral_nfft", "storage_dtype"), "RD feature")


def load_config(path: str | Path, overrides: list[str] | None = None, expected_feature: str | None = None) -> ExperimentConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    raw = apply_overrides(_read_mapping(config_path), overrides)
    validate_config(raw, expected_feature)
    return ExperimentConfig(config_path, raw)
