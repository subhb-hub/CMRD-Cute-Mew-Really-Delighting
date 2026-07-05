from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrialRecord:
    signal: np.ndarray
    label: int
    subject: int
    session: int
    trial: int
    source_file: str
    source_key: str

    @property
    def trial_id(self) -> str:
        return f"sub-{self.subject:02d}_ses-{self.session:02d}_trial-{self.trial:02d}"


@dataclass(frozen=True)
class TrialSample:
    x: np.ndarray
    label: int
    subject: int
    session: int
    trial: int
    source_index: int

