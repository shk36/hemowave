"""Waveform-derived hemodynamic indices from one arterial pressure segment.

==========  ======  ==========================================================
Index       Unit    Definition
==========  ======  ==========================================================
PPV         %       (PPmax − PPmin) / mean(PPmax, PPmin) × 100 over QC beats
dPdt_max    mmHg/s  max Savitzky–Golay derivative on each systolic upstroke
tau         s       −1 / slope of ln[P(t) − P∞] over 55–92 % of each beat
==========  ======  ==========================================================

dP/dt_max and τ follow Ensarioğlu & Yıldırım, *J Clin Med* 2026;15(17):6871
(Methods §2.5–2.6, https://doi.org/10.3390/jcm15176871), adapted as follows —
this is **not** an identical re-implementation:

* sampling rate 500 Hz → 100 Hz; SG derivative 21 → 5 points (both span 40 ms,
  but kernels and frequency responses differ)
* aggregation over a 10.24 s segment instead of a 1-minute median
* beats from pyvital ``detect_peaks`` + foot–peak–next-foot pairing
* beat QC (length 0.3–3.0 s, PP 20–100 mmHg) and the per-beat τ range
  0.12–1.50 s are study rules (see :class:`~hemowave.config.HemodynamicConfig`)

PPV follows Michard et al. 2000 with the beat QC of pyvital ``abp_ppv``.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .beats import Beats, segment_beats
from .config import DEFAULT_CONFIG, HemodynamicConfig

__all__ = ["HEMODYNAMIC_KEYS", "dpdt_max", "extract_hemodynamics", "ppv", "tau"]

#: Keys returned by :func:`extract_hemodynamics`.
HEMODYNAMIC_KEYS = ("PPV", "dPdt_max", "tau", "n_beats", "n_ppv", "n_dpdt", "n_tau")

_MIN_SAMPLES = 8   # shorter inputs are rejected outright


# ============================================================================
# Per-index estimators (operate on pre-segmented beats)
# ============================================================================
def _aggregate(values, beat_aggregator):
    """Beat values → one window value (median | mean). NaN if empty."""
    if not values:
        return np.nan
    return float(np.median(values) if beat_aggregator == 'median' else np.mean(values))


def _ppv(data, fs, foots, next_foots, pp_all, beatlens, qc, cfg):
    """Michard PPV (%) and number of beats used. ``(nan, 0)`` on failure.

    The whole window is a single analysis period (respiratory-cycle splitting
    needs ≥ 30 s). Beat QC follows pyvital ``abp_ppv`` / Aboy 2009:

    1. beat length and PP within physiological range (``qc`` mask)
    2. regular rhythm: beat-length std/mean ≤ ``ppv_rhythm_cv_max``
    3. correlation with the mean beat shape ≥ ``ppv_shape_corr_min``

    Minimum beat count = ceil(40 bpm × window_s / 60) − 2 (5 for 10.24 s). The −2
    offsets the two edge peaks that :func:`~hemowave.beats.pair_beats` always
    drops, so the effective heart-rate floor is ≈ 42 bpm rather than ≈ 53 bpm.
    """
    min_beats = max(int(np.ceil(cfg.ppv_min_hr_bpm * (data.size / fs) / 60.0)) - cfg.ppv_pairing_beat_loss,
                    cfg.ppv_min_beats_floor)
    idxs = np.where(qc)[0]
    if idxs.size < min_beats:
        return np.nan, 0
    bl = beatlens[idxs]
    if not (bl.mean() > 0 and (bl.std() / bl.mean()) <= cfg.ppv_rhythm_cv_max):
        return np.nan, 0

    grid = np.linspace(0.0, 1.0, cfg.ppv_shape_grid)
    shapes, kept_idx = [], []
    for j in idxs:
        b = data[foots[j]:next_foots[j] + 1].astype(np.float64)
        if len(b) >= 4 and np.ptp(b) > 0:
            shapes.append(np.interp(grid, np.linspace(0.0, 1.0, len(b)), b))
            kept_idx.append(j)
    if len(shapes) < min_beats:
        return np.nan, 0
    shapes = np.asarray(shapes)
    avgbeat = shapes.mean(axis=0)
    keep = [k for k in range(len(shapes))
            if np.corrcoef(avgbeat, shapes[k])[0, 1] >= cfg.ppv_shape_corr_min]
    if len(keep) < min_beats:
        return np.nan, 0

    pp_keep = pp_all[np.asarray(kept_idx)[keep]]
    pp_max, pp_min = float(pp_keep.max()), float(pp_keep.min())
    s = pp_max + pp_min
    val = (pp_max - pp_min) / s * 200.0 if s > 0 else np.nan
    if np.isfinite(val) and 0.0 < val < cfg.ppv_valid_max:
        return val, len(keep)
    return np.nan, 0


def _dpdt_max(data, fs, foots, peaks, qc, cfg):
    """Arterial dP/dt_max (mmHg/s) and number of beats used. ``(nan, 0)`` on failure.

    Reference §2.5: per beat, the maximum Savitzky–Golay first derivative on the
    foot → peak upstroke; beat values outside 30–5000 mmHg/s are discarded and
    the rest aggregated. The SG filter (``delta = 1/fs`` → mmHg/s) runs once over
    the whole window; inputs shorter than the SG window return NaN (no
    finite-difference fallback).
    """
    if data.size < cfg.dpdt_sg_window_points:
        return np.nan, 0
    dpdt_signal = savgol_filter(data, cfg.dpdt_sg_window_points, cfg.dpdt_sg_polyorder,
                                deriv=1, delta=1.0 / fs, mode='interp')
    values = []
    for j, (foot, peak) in enumerate(zip(foots, peaks)):
        if not (0 <= foot < peak < data.size) or not qc[j]:
            continue
        v = float(np.max(dpdt_signal[foot:peak + 1]))
        if np.isfinite(v) and cfg.dpdt_min_mmhg_s <= v <= cfg.dpdt_max_mmhg_s:
            values.append(v)
    return _aggregate(values, cfg.beat_aggregator), len(values)


def _tau(data, fs, foots, next_foots, qc, cfg):
    """Diastolic decay time constant τ (s) and number of beats used. ``(nan, 0)`` on failure.

    Reference §2.6: over 55–92 % of each beat, with P∞ fixed at the beat's
    diastolic minimum − 5 mmHg, regress ln[P(t) − P∞] on t (s); τ = −1/slope for
    a negative slope. No dicrotic-notch detection, no nonlinear optimisation.

    Rules not specified by the reference:

    * fit window: ``foot + round(0.55·L)`` … ``foot + round(0.92·L)`` inclusive,
      L = next_foot − foot
    * diastolic minimum taken over [foot, next_foot], so P − P∞ ≥ 5 mmHg in the window
    * a beat is skipped if it fails QC, has < 3 fit samples, non-finite values,
      any P − P∞ ≤ 0, or a slope ≥ 0
    * beat τ outside 0.12–1.50 s (inclusive) is excluded as an artifact. As the
      slope → 0⁻, τ diverges to 10²–10³ s; with ~10 beats per window a single
      such beat would dominate the window median.
    """
    taus = []
    for j, (foot, next_foot) in enumerate(zip(foots, next_foots)):
        if not qc[j] or next_foot <= foot:
            continue
        beat_len = next_foot - foot
        i0 = foot + int(round(cfg.tau_fit_start_frac * beat_len))
        i1 = foot + int(round(cfg.tau_fit_end_frac * beat_len))
        seg = data[i0:i1 + 1].astype(np.float64)
        if seg.size < cfg.tau_min_fit_samples or not np.all(np.isfinite(seg)):
            continue
        p_inf = float(np.min(data[foot:next_foot + 1])) - cfg.tau_pinf_offset_mmhg
        if not np.isfinite(p_inf):
            continue
        y = seg - p_inf
        if np.any(y <= 0.0):
            continue
        t = np.arange(seg.size, dtype=np.float64) / fs
        slope = float(np.polyfit(t, np.log(y), 1)[0])
        if not np.isfinite(slope) or slope >= 0.0:
            continue
        tau_beat = -1.0 / slope
        if not (cfg.tau_min_s <= tau_beat <= cfg.tau_max_s):
            continue
        taus.append(tau_beat)
    return _aggregate(taus, cfg.beat_aggregator), len(taus)


# ============================================================================
# Public API
# ============================================================================
def _prepare(waveform, mmhg_min, mmhg_max):
    """Copy to 1-D float64 and undo [0, 1] scaling if requested. ``None`` if unusable.

    Non-finite samples make the segment unusable: NaN already yields no beats in
    the detector, while ±inf either crashes the SG edge fit or passes through a
    failed heart-rate estimate.
    """
    data = np.squeeze(np.asarray(waveform, dtype=np.float64).copy())
    if data.ndim != 1 or data.size < _MIN_SAMPLES:
        return None
    if mmhg_min is not None and mmhg_max is not None:
        data = data * (mmhg_max - mmhg_min) + mmhg_min
    if not np.all(np.isfinite(data)):
        return None
    return data


def extract_hemodynamics(waveform, fs: float, *, mmhg_min: float | None = None,
                         mmhg_max: float | None = None,
                         config: HemodynamicConfig = DEFAULT_CONFIG) -> dict:
    """PPV, dP/dt_max and τ from one arterial pressure segment.

    Parameters
    ----------
    waveform : array_like, shape (L,) or (L, 1)
        Arterial pressure in mmHg (or [0, 1]-scaled, see ``mmhg_min``/``mmhg_max``).
    fs : float
        Sampling rate in Hz.
    mmhg_min, mmhg_max : float, optional
        Map a [0, 1]-normalised input back to mmHg. Ignored unless both are given.
    config : HemodynamicConfig
        Thresholds. Defaults reproduce the published analysis.

    Returns
    -------
    dict
        ``PPV`` (%), ``dPdt_max`` (mmHg/s), ``tau`` (s) — NaN when not computable —
        plus beat counts ``n_beats`` (paired beats) and ``n_ppv``/``n_dpdt``/``n_tau``
        (beats that contributed to each index).

    Examples
    --------
    >>> from hemowave import extract_hemodynamics, synthetic_abp
    >>> f = extract_hemodynamics(synthetic_abp(seed=0), fs=100)
    >>> sorted(f)
    ['PPV', 'dPdt_max', 'n_beats', 'n_dpdt', 'n_ppv', 'n_tau', 'tau']
    """
    result = {'PPV': np.nan, 'dPdt_max': np.nan, 'tau': np.nan,
              'n_beats': 0, 'n_ppv': 0, 'n_dpdt': 0, 'n_tau': 0}

    data = _prepare(waveform, mmhg_min, mmhg_max)
    if data is None:
        return result
    beats = segment_beats(data, fs, config)
    if beats is None:
        return result
    result['n_beats'] = len(beats)

    result['PPV'], result['n_ppv'] = _ppv(
        data, fs, beats.foots, beats.next_foots, beats.pulse_pressure, beats.lengths, beats.qc, config)
    result['dPdt_max'], result['n_dpdt'] = _dpdt_max(
        data, fs, beats.foots, beats.peaks, beats.qc, config)
    result['tau'], result['n_tau'] = _tau(
        data, fs, beats.foots, beats.next_foots, beats.qc, config)
    return result


def _segment(waveform, fs, config) -> tuple[np.ndarray, Beats] | None:
    data = _prepare(waveform, None, None)
    if data is None:
        return None
    beats = segment_beats(data, fs, config)
    return None if beats is None else (data, beats)


def ppv(waveform, fs: float, *, config: HemodynamicConfig = DEFAULT_CONFIG) -> float:
    """Pulse pressure variation (%) of one segment. See :func:`extract_hemodynamics`."""
    seg = _segment(waveform, fs, config)
    if seg is None:
        return np.nan
    data, b = seg
    return _ppv(data, fs, b.foots, b.next_foots, b.pulse_pressure, b.lengths, b.qc, config)[0]


def dpdt_max(waveform, fs: float, *, config: HemodynamicConfig = DEFAULT_CONFIG) -> float:
    """Arterial dP/dt_max (mmHg/s) of one segment. See :func:`extract_hemodynamics`."""
    seg = _segment(waveform, fs, config)
    if seg is None:
        return np.nan
    data, b = seg
    return _dpdt_max(data, fs, b.foots, b.peaks, b.qc, config)[0]


def tau(waveform, fs: float, *, config: HemodynamicConfig = DEFAULT_CONFIG) -> float:
    """Diastolic decay time constant τ (s) of one segment. See :func:`extract_hemodynamics`."""
    seg = _segment(waveform, fs, config)
    if seg is None:
        return np.nan
    data, b = seg
    return _tau(data, fs, b.foots, b.next_foots, b.qc, config)[0]
