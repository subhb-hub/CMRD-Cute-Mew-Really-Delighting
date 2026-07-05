from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubjectSplit:
    train_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]
    target_subject: int

    def as_dict(self) -> dict[str, object]:
        return {
            "train_subjects": list(self.train_subjects),
            "validation_subjects": list(self.validation_subjects),
            "target_subject": self.target_subject,
        }


def subject_loso_split(all_subjects: np.ndarray, target_subject: int, validation_subjects: int, split_seed: int) -> SubjectSplit:
    unique = np.unique(np.asarray(all_subjects, dtype=np.int64))
    if target_subject not in unique:
        raise ValueError(f"Target subject {target_subject} is absent")
    source = unique[unique != target_subject]
    if not 0 < validation_subjects < source.size:
        raise ValueError("validation_subjects must leave at least one source-training subject")
    rng = np.random.default_rng(int(split_seed) + int(target_subject))
    validation = np.sort(rng.choice(source, validation_subjects, replace=False))
    train = np.sort(np.setdiff1d(source, validation))
    if set(train) & set(validation) or target_subject in train or target_subject in validation:
        raise AssertionError("Subject leakage in LOSO split")
    return SubjectSplit(tuple(map(int, train)), tuple(map(int, validation)), int(target_subject))

