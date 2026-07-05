from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from cmrd.config import PROJECT_ROOT


def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Fused/flash attention is faster but PyTorch explicitly marks its CUDA
        # backward pass as non-deterministic. The math kernel is reproducible.
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_manifest(command: list[str]) -> dict[str, object]:
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    status = _git("status", "--short")
    return {
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": cuda_name,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(status),
        "git_status": status.splitlines() if status else [],
    }
