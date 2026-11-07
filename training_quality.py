"""Shared, deterministic preparation for notebook baselines and transformers."""
import re
import unicodedata

import numpy as np
import pandas as pd


def text_key(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def clean_task(frame, target, min_count=10, excluded=()):
    """Exclude invalid/conflicting/rare rows from training, never edit source files."""
    if min_count < 3:
        raise ValueError("Use at least 3 examples for train/validation/test coverage.")
    work = frame.copy()
    work["_text_key"] = work["text"].map(text_key)
    valid = work["_text_key"].str.contains(r"[^\W\d_]", regex=True) & work[target].notna()
    valid &= ~work[target].astype(str).str.casefold().isin(["", "nan", "none", "unknown", *excluded])
    rejected = work.loc[~valid].assign(exclusion_reason="invalid_text_or_label")
    work = work.loc[valid].copy()
    conflicts = work.groupby("_text_key")[target].nunique()
    conflict = work["_text_key"].isin(conflicts[conflicts > 1].index)
    rejected = pd.concat([rejected, work.loc[conflict].assign(exclusion_reason="conflicting_labels")])
    work = work.loc[~conflict].copy()
    duplicate = work.duplicated("_text_key")
    rejected = pd.concat([rejected, work.loc[duplicate].assign(exclusion_reason="duplicate_text")])
    work = work.loc[~duplicate].copy()
    counts = work[target].value_counts()
    rare = work[target].isin(counts[counts < min_count].index)
    rejected = pd.concat([rejected, work.loc[rare].assign(exclusion_reason="rare_class")])
    work = work.loc[~rare].copy()
    if work[target].nunique() < 2:
        raise ValueError(f"{target}: fewer than two supported classes remain; collect more data.")
    return work.reset_index(drop=True), rejected.drop(columns="_text_key").reset_index(drop=True)


def split_task(frame, target, seed=42, test_size=.15, valid_size=.15):
    """Per-class allocation avoids global stratification rounding failures."""
    if not 0 < test_size < 1 or not 0 < valid_size < 1 or test_size + valid_size >= 1:
        raise ValueError("Invalid split fractions")
    parts = {key: [] for key in ("train", "val", "test")}
    rng = np.random.default_rng(seed)
    for label, group in frame.groupby(target, sort=True):
        n = len(group)
        if n < 3:
            raise ValueError(f"{target}/{label}: not enough rows for three splits")
        order = rng.permutation(n)
        nt, nv = max(1, int(n * test_size)), max(1, int(n * valid_size))
        if nt + nv >= n:
            raise ValueError("Split leaves no training rows")
        for name, positions in (("test", order[:nt]), ("val", order[nt:nt+nv]), ("train", order[nt+nv:])):
            parts[name].append(group.iloc[positions])
    result = {key: pd.concat(value).reset_index(drop=True) for key, value in parts.items()}
    keys = {key: set(value["text"].map(text_key)) for key, value in result.items()}
    if any(keys[a] & keys[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Duplicate text crossed split boundaries")
    return result
