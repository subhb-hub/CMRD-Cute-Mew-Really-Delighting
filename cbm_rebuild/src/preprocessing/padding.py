from __future__ import annotations

import numpy as np


def pad_trials(trials: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not trials:
        raise ValueError("Cannot pad an empty trial list")
    if any(trial.ndim != 2 for trial in trials):
        raise ValueError("All trials must be 2-D [windows, features]")
    feature_dims = {trial.shape[1] for trial in trials}
    if len(feature_dims) != 1:
        raise ValueError("All trials must be 2-D and have the same feature dimension")
    lengths = np.asarray([trial.shape[0] for trial in trials], dtype=np.int32)
    max_length = int(lengths.max())
    feature_dim = next(iter(feature_dims))
    data = np.zeros((len(trials), max_length, feature_dim), dtype=np.float32)
    mask = np.zeros((len(trials), max_length), dtype=np.bool_)
    for index, trial in enumerate(trials):
        length = trial.shape[0]
        data[index, :length] = trial
        mask[index, :length] = True
    return data, mask, lengths
