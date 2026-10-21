"""Segment-level agreement between a reference and an estimated waveform.

==========  =========  ============================================  ===========
Metric      Unit       Definition                                    Better
==========  =========  ============================================  ===========
MAE         signal     mean |y_true − y_pred|                        → 0
R2          —          1 − SS_res / SS_tot within the segment        → 1
PCC         —          Pearson correlation                           → 1
CCC         —          Lin's concordance correlation (ddof = 0)      → 1
Cross_Corr  —          max normalised cross-correlation over lags    → 1
NDTW        —          FastDTW distance / (N × range(y_true))        → 0
==========  =========  ============================================  ===========

.. note::
   ``R2`` here is computed *within one segment* and is unbounded below when the
   reference barely moves. To report a dataset-level R², pool all samples with
   :func:`hemowave.stats.global_r2` (and :func:`~hemowave.stats.cluster_bootstrap_r2`
   for a patient-level CI) instead of averaging segment values.
"""
from __future__ import annotations

import numpy as np
from fastdtw import fastdtw
from scipy.signal import correlate
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

__all__ = ["AGREEMENT_KEYS", "ccc", "cross_corr", "evaluate_agreement",
           "mae", "ndtw", "pcc", "r2"]

#: Keys returned by :func:`evaluate_agreement`.
AGREEMENT_KEYS = ("MAE", "R2", "PCC", "CCC", "Cross_Corr", "NDTW")


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in the unit of the signal (mmHg for ABP)."""
    return mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred) -> float:
    """Coefficient of determination within one segment (``y_true`` = reference)."""
    return r2_score(y_true, y_pred)


def pcc(y_true, y_pred) -> float:
    """Pearson correlation coefficient."""
    return pearsonr(y_true, y_pred)[0]


def ccc(y_true, y_pred) -> float:
    """Lin's concordance correlation coefficient (population variances). NaN if undefined."""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator == 0:
        return np.nan
    return (2 * covariance) / denominator


def cross_corr(y_true, y_pred) -> float:
    """Maximum of the mean-centred, norm-scaled full cross-correlation over all lags.

    Equals PCC at zero lag, so ``Cross_Corr ≥ PCC``; the gap shows how much a
    time shift alone would improve agreement. NaN if either signal is constant.
    """
    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)

    corr = correlate(y_pred_centered, y_true_centered, mode='full')
    norm = np.linalg.norm(y_true_centered) * np.linalg.norm(y_pred_centered)
    if norm == 0:
        return np.nan
    return np.max(corr / norm)


def ndtw(y_true, y_pred) -> float:
    """FastDTW distance normalised by length and by the reference amplitude range."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    dtw_distance, _ = fastdtw(y_true, y_pred)

    amplitude_range = np.max(y_true) - np.min(y_true)
    return dtw_distance / (len(y_true) * amplitude_range)


def evaluate_agreement(y_true, y_pred) -> dict:
    """All six agreement metrics for one waveform pair.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (L,) or (L, 1)
        Reference and estimated waveforms of equal length.

    Returns
    -------
    dict
        ``{'MAE', 'R2', 'PCC', 'CCC', 'Cross_Corr', 'NDTW'}``.

    Raises
    ------
    ValueError
        On invalid input such as non-finite values or mismatched lengths.

    Examples
    --------
    >>> from hemowave import evaluate_agreement, synthetic_abp
    >>> y = synthetic_abp(seed=0)
    >>> m = evaluate_agreement(y, y + 2.0)
    >>> round(float(m['MAE']), 6), round(float(m['PCC']), 6)
    (2.0, 1.0)
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return {
        'MAE': mae(y_true, y_pred),
        'R2': r2(y_true, y_pred),
        'PCC': pcc(y_true, y_pred),
        'CCC': ccc(y_true, y_pred),
        'Cross_Corr': cross_corr(y_true, y_pred),
        'NDTW': ndtw(y_true, y_pred),
    }
