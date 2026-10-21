"""Synthetic arterial pressure for examples and tests (not a physiological model)."""
from __future__ import annotations

import numpy as np

__all__ = ["synthetic_abp"]


def synthetic_abp(duration: float = 10.24, fs: float = 100, *, hr: float = 72.0,
                  sbp: float = 120.0, dbp: float = 75.0, ppv: float = 12.0,
                  resp_rate: float = 14.0, tau: float = 0.45, hr_jitter: float = 0.02,
                  noise: float = 0.3, seed: int | None = None) -> np.ndarray:
    """Beat-by-beat ABP-like waveform with respiratory pulse-pressure variation.

    Each beat is a half-cosine upstroke, a late-systolic decline to a dicrotic
    notch and an exponential diastolic run-off with time constant ``tau``. Pulse
    pressure is modulated sinusoidally at ``resp_rate`` so that the peak-to-peak
    swing is roughly ``ppv`` %.

    Parameters
    ----------
    duration, fs : float
        Length in seconds and sampling rate in Hz (defaults: 1,024 samples at 100 Hz).
    hr, sbp, dbp : float
        Mean heart rate (bpm), systolic and diastolic pressure (mmHg).
    ppv : float
        Target pulse pressure variation (%).
    resp_rate : float
        Breaths per minute.
    tau : float
        Diastolic decay time constant (s).
    hr_jitter : float
        Relative beat-to-beat SD of the RR interval.
    noise : float
        SD of additive white noise (mmHg).
    seed : int, optional
        RNG seed.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * fs))
    out = np.empty(0)
    pp0 = sbp - dbp
    phase = rng.uniform(0, 2 * np.pi)
    t0 = 0.0
    while out.size < n + int(fs):
        rr = 60.0 / hr * (1 + hr_jitter * rng.standard_normal())
        pp = pp0 * (1 + ppv / 200.0 * np.sin(2 * np.pi * resp_rate / 60.0 * t0 + phase))
        t = np.arange(int(round(rr * fs))) / fs
        tp, tn = 0.12 * rr / 0.83, 0.32 * rr / 0.83          # scale landmarks with RR
        peak, notch = dbp + pp, dbp + 0.62 * pp
        e = np.exp(-(rr - tn) / tau)
        p_inf = (dbp - notch * e) / (1 - e)                    # decay lands on dbp at the next foot
        late = np.clip((t - tp) / (tn - tp), 0.0, None)
        beat = np.where(
            t < tp, dbp + pp * 0.5 * (1 - np.cos(np.pi * t / tp)),
            np.where(t < tn, peak - (peak - notch) * late ** 1.5,
                     p_inf + (notch - p_inf) * np.exp(-(t - tn) / tau)))
        beat -= 2.5 * np.exp(-((t - tn) / 0.015) ** 2)         # dicrotic notch
        out = np.concatenate([out, beat])
        t0 += rr
    return out[:n] + noise * rng.standard_normal(n)
