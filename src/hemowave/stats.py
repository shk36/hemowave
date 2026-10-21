"""Dataset-level waveform R² with a patient-cluster bootstrap CI.

Segments from one patient are correlated, so resampling segments would give a
CI that is too narrow. Instead each patient is reduced to four sufficient
statistics ``[n, Σy, Σy², Σ(y − ŷ)²]`` and *patients* are resampled; every
bootstrap replicate is then a sum of rows, which keeps millions of segments
tractable.

Point estimate::

    R² = 1 − Σ(y − ŷ)² / (Σy² − (Σy)² / n)     over every (segment, sample)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["R2Result", "cluster_bootstrap_r2", "global_r2", "patient_sums", "r2_from_sums",
           "segment_sums"]


@dataclass(frozen=True)
class R2Result:
    r2: float
    ci_low: float
    ci_high: float
    n_segments: int
    n_patients: int

    def __str__(self) -> str:
        return (f"R² = {self.r2:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}] "
                f"({self.n_segments:,} segments, {self.n_patients:,} patients)")


def segment_sums(y_true, y_pred, batch: int = 4000, progress: bool = False):
    """Per-segment ``(n, Σy, Σy², Σ(y−ŷ)²)``. Accepts (N, L), (N, L, 1) or a list of equal-length arrays."""
    N = len(y_true)
    is_nd = isinstance(y_true, np.ndarray)
    first = y_true[0]
    T = int(np.asarray(first).reshape(-1).shape[0])

    n_arr = np.full(N, T, dtype=np.int64)
    sy_arr = np.empty(N, dtype=np.float64)
    sy2_arr = np.empty(N, dtype=np.float64)
    se2_arr = np.empty(N, dtype=np.float64)

    starts = range(0, N, batch)
    if progress:
        from tqdm.auto import tqdm
        starts = tqdm(starts, total=(N + batch - 1) // batch, desc='segment sums', unit='batch')
    for s in starts:
        e = min(s + batch, N)
        if is_nd:
            y = y_true[s:e].astype(np.float64, copy=False)
            p = y_pred[s:e].astype(np.float64, copy=False)
        else:
            y = np.stack([np.asarray(y_true[i], dtype=np.float64) for i in range(s, e)])
            p = np.stack([np.asarray(y_pred[i], dtype=np.float64) for i in range(s, e)])
        if y.ndim > 2:   # (batch, T, 1) → (batch, T)
            y = y.reshape(y.shape[0], -1)
        if p.ndim > 2:
            p = p.reshape(p.shape[0], -1)
        sy_arr[s:e] = y.sum(axis=1)
        sy2_arr[s:e] = (y * y).sum(axis=1)
        se2_arr[s:e] = ((y - p) ** 2).sum(axis=1)
    return n_arr, sy_arr, sy2_arr, se2_arr


def patient_sums(groups, n_arr, sy_arr, sy2_arr, se2_arr) -> np.ndarray:
    """Collapse segment sums to one row per patient → array (G, 4)."""
    df = pd.DataFrame({'rid': groups, 'n': n_arr, 'sy': sy_arr, 'sy2': sy2_arr, 'se2': se2_arr})
    g = df.groupby('rid', sort=True).sum()
    return g[['n', 'sy', 'sy2', 'se2']].to_numpy(dtype=np.float64)


def r2_from_sums(s) -> float:
    """R² from summed ``[n, Σy, Σy², Σ(y−ŷ)²]``. NaN if the reference has no variance."""
    n, sy, sy2, se2 = s
    ss_tot = sy2 - (sy * sy) / n
    if ss_tot <= 0:
        return float('nan')
    return 1.0 - se2 / ss_tot


def global_r2(y_true, y_pred) -> float:
    """R² pooled over every sample of every segment (no CI)."""
    sums = np.stack(segment_sums(y_true, y_pred), axis=1).astype(np.float64)
    return r2_from_sums(sums.sum(axis=0))


def cluster_bootstrap_r2(y_true, y_pred, groups, *, n_boot: int = 2000, seed: int = 42,
                         ci: tuple[float, float] = (2.5, 97.5), batch: int = 4000,
                         progress: bool = False) -> R2Result:
    """Global R² with a percentile CI from resampling ``groups`` (e.g. patient IDs).

    Parameters
    ----------
    y_true, y_pred : array_like, shape (N, L) or (N, L, 1)
        Reference and estimated segments.
    groups : array_like, shape (N,)
        Cluster label of each segment (patient ID).
    n_boot, seed : int
        Bootstrap replicates and RNG seed.
    ci : (float, float)
        Percentiles of the bootstrap distribution to report.
    """
    if len(groups) != len(y_true):
        raise ValueError(f"groups length {len(groups):,} != segments {len(y_true):,}")
    stats = patient_sums(groups, *segment_sums(y_true, y_pred, batch=batch, progress=progress))
    G = stats.shape[0]
    r2_point = r2_from_sums(stats.sum(axis=0))

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    reps = range(n_boot)
    if progress:
        from tqdm.auto import tqdm
        reps = tqdm(reps, desc='bootstrap', unit='iter')
    for b in reps:
        idx = rng.integers(0, G, G)
        boot[b] = r2_from_sums(stats[idx].sum(axis=0))
    valid = boot[~np.isnan(boot)]
    lo, hi = np.percentile(valid, ci)
    return R2Result(float(r2_point), float(lo), float(hi), len(y_true), G)
