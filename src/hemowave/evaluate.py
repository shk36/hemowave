"""One call for all nine metrics, on a single pair or a whole test set."""
from __future__ import annotations

import csv
import os
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .agreement import AGREEMENT_KEYS, evaluate_agreement
from .config import DEFAULT_CONFIG, HemodynamicConfig
from .hemodynamics import extract_hemodynamics

__all__ = ["evaluate", "evaluate_dataset", "result_columns"]

_INDICES = ("PPV", "dPdt_max", "tau")
_COUNTS = ("n_beats", "n_ppv", "n_dpdt", "n_tau")


def result_columns(agreement: bool = True, hemodynamics: bool = True) -> list[str]:
    """Column order of :func:`evaluate_dataset` output."""
    cols = ['idx']
    if agreement:
        cols += list(AGREEMENT_KEYS)
    if hemodynamics:
        cols += [f'{k}_{s}' for k in _INDICES for s in ('true', 'pred', 'diff')]
        cols += [f'{k}_{s}' for k in _COUNTS for s in ('true', 'pred')]
    return cols


def evaluate(y_true, y_pred, fs: float, *, agreement: bool = True, hemodynamics: bool = True,
             config: HemodynamicConfig = DEFAULT_CONFIG) -> dict:
    """All metrics for one reference/estimate pair, as a flat dict.

    * agreement: ``MAE, R2, PCC, CCC, Cross_Corr, NDTW`` — NaN for all six if the
      input fails validation (e.g. non-finite samples)
    * hemodynamics: ``{PPV, dPdt_max, tau}_{true, pred, diff}`` with
      ``diff = pred − true`` (positive ⇒ overestimation) and the beat counts
      ``{n_beats, n_ppv, n_dpdt, n_tau}_{true, pred}``

    Examples
    --------
    >>> from hemowave import evaluate, synthetic_abp
    >>> r = evaluate(synthetic_abp(seed=0), synthetic_abp(seed=1), fs=100)
    >>> [k for k in r][:6]
    ['MAE', 'R2', 'PCC', 'CCC', 'Cross_Corr', 'NDTW']
    """
    row = {}
    if agreement:
        try:
            row.update(evaluate_agreement(y_true, y_pred))
        except ValueError:
            row.update(dict.fromkeys(AGREEMENT_KEYS, np.nan))
    if hemodynamics:
        ft = extract_hemodynamics(y_true, fs, config=config)
        fp = extract_hemodynamics(y_pred, fs, config=config)
        for k in _INDICES:
            row[f'{k}_true'], row[f'{k}_pred'] = ft[k], fp[k]
            row[f'{k}_diff'] = fp[k] - ft[k]      # NaN propagates
        for k in _COUNTS:
            row[f'{k}_true'], row[f'{k}_pred'] = ft[k], fp[k]
    return row


# ============================================================================
# Dataset runner
# ============================================================================
def _worker(task):
    """Module-level so it pickles under the ``spawn`` start method (macOS, Windows)."""
    i, t, p, fs, agreement, hemodynamics, config = task
    return {'idx': i, **evaluate(t, p, fs, agreement=agreement, hemodynamics=hemodynamics,
                                 config=config)}


def _resume_from(out_csv: Path, columns: list[str]) -> int:
    """Next idx to compute. Refuses to append to a file whose header differs."""
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return 0
    with open(out_csv, newline='', encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n')
    if header != ','.join(columns):
        raise RuntimeError(f"{out_csv} has different columns; move it away or pass resume=False.\n"
                           f"  file:     {header}\n  expected: {','.join(columns)}")
    done = pd.read_csv(out_csv, usecols=['idx'])['idx']
    return int(done.max()) + 1 if len(done) else 0


def _rows(y_true, y_pred, fs, start, agreement, hemodynamics, config, n_jobs, chunksize,
          progress) -> Iterator[dict]:
    n = len(y_true)
    tasks = ((i, np.asarray(y_true[i]), np.asarray(y_pred[i]), fs, agreement, hemodynamics, config)
             for i in range(start, n))
    with (Pool(n_jobs) if n_jobs > 1 else nullcontext()) as pool:
        it = pool.imap(_worker, tasks, chunksize=chunksize) if pool else map(_worker, tasks)
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(it, total=n, initial=start, desc='hemowave', unit='seg')
        yield from it


def evaluate_dataset(y_true, y_pred, fs: float, *, agreement: bool = True,
                     hemodynamics: bool = True, config: HemodynamicConfig = DEFAULT_CONFIG,
                     n_jobs: int = 1, chunksize: int = 200, out_csv: str | os.PathLike | None = None,
                     resume: bool = True, flush_every: int = 1000,
                     progress: bool = True) -> pd.DataFrame | None:
    """Evaluate every segment pair of a test set.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (N, L) or (N, L, 1), or lists of arrays
        Reference and estimated segments, paired by index.
    fs : float
        Sampling rate in Hz.
    agreement, hemodynamics : bool
        Which metric families to compute. NDTW dominates the run time; set
        ``agreement=False`` if only PPV/dP/dt_max/τ are needed.
    n_jobs : int
        Worker processes (1 = run in-process). Results keep input order. On
        macOS and Windows, call from under ``if __name__ == "__main__":`` in
        scripts, as with any :mod:`multiprocessing` code.
    out_csv : path, optional
        Stream rows to this CSV instead of holding them in memory — recommended
        for millions of segments. With ``resume=True`` an interrupted run
        continues after the last written ``idx``.

    Returns
    -------
    pandas.DataFrame or None
        One row per segment (columns from :func:`result_columns`), or ``None``
        when streaming to ``out_csv``.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"segment count mismatch: y_true {len(y_true):,} vs y_pred {len(y_pred):,}")
    if not (agreement or hemodynamics):
        raise ValueError("enable at least one of agreement / hemodynamics")
    columns = result_columns(agreement, hemodynamics)
    args = (agreement, hemodynamics, config, n_jobs, chunksize, progress)

    if out_csv is None:
        rows = list(_rows(y_true, y_pred, fs, 0, *args))
        return pd.DataFrame(rows, columns=columns)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    start = _resume_from(out_csv, columns) if resume else 0
    if start >= len(y_true):
        return None
    with open(out_csv, 'a' if start else 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not start:
            writer.writeheader()
        for k, row in enumerate(_rows(y_true, y_pred, fs, start, *args), 1):
            writer.writerow(row)
            if k % flush_every == 0:
                f.flush()
    return None
