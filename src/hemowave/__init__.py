"""hemowave — hemodynamic indices and agreement metrics for arterial pressure waveforms.

Quick start::

    import hemowave as hw

    hw.extract_hemodynamics(abp, fs=100)      # PPV, dPdt_max, tau (+ beat counts)
    hw.evaluate_agreement(abp_ref, abp_est)   # MAE, R2, PCC, CCC, Cross_Corr, NDTW
    hw.evaluate(abp_ref, abp_est, fs=100)     # all nine, one pair
    hw.evaluate_dataset(Y, Y_hat, fs=100, n_jobs=8, out_csv="metrics.csv")
"""
from .agreement import (AGREEMENT_KEYS, ccc, cross_corr, evaluate_agreement, mae, ndtw, pcc,
                        r2)
from .beats import Beats, detect_peaks, pair_beats, segment_beats
from .config import DEFAULT_CONFIG, HemodynamicConfig
from .evaluate import evaluate, evaluate_dataset, result_columns
from .hemodynamics import HEMODYNAMIC_KEYS, dpdt_max, extract_hemodynamics, ppv, tau
from .stats import R2Result, cluster_bootstrap_r2, global_r2
from .synthetic import synthetic_abp

__version__ = "1.0.0"

__all__ = [
    # hemodynamic indices
    "extract_hemodynamics", "ppv", "dpdt_max", "tau", "HEMODYNAMIC_KEYS",
    "HemodynamicConfig", "DEFAULT_CONFIG",
    # waveform agreement
    "evaluate_agreement", "mae", "r2", "pcc", "ccc", "cross_corr", "ndtw", "AGREEMENT_KEYS",
    # all-in-one
    "evaluate", "evaluate_dataset", "result_columns",
    # dataset-level statistics
    "global_r2", "cluster_bootstrap_r2", "R2Result",
    # beats
    "detect_peaks", "pair_beats", "segment_beats", "Beats",
    # utilities
    "synthetic_abp",
]
