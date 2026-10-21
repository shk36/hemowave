"""Analysis parameters for waveform-derived hemodynamic indices.

Every threshold lives in one frozen dataclass so a study can state its exact
settings in one line and override any of them without touching the algorithms::

    from hemowave import HemodynamicConfig, extract_hemodynamics
    cfg = HemodynamicConfig(beat_aggregator="mean", tau_max_s=2.0)
    extract_hemodynamics(abp, fs=100, config=cfg)

The defaults reproduce the published analysis (100 Hz, 10.24 s segments).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class HemodynamicConfig:
    """Thresholds used by :func:`hemowave.extract_hemodynamics`.

    Sources are marked per group: *[ref]* = Ensarioğlu & Yıldırım, J Clin Med 2026
    (dP/dt_max, τ); *[pyvital]* = pyvital ``abp_ppv`` / Aboy 2009; *[study]* = rule
    chosen for this study, not taken from a reference.
    """

    # --- beat → window aggregation (dP/dt_max, τ) [ref: median] ---
    beat_aggregator: Literal["median", "mean"] = "median"

    # --- beat QC shared by all three indices [study, adapted from pyvital] ---
    beat_len_min_s: float = 0.3          # beat length 0.3–3.0 s → HR 20–200 bpm
    beat_len_max_s: float = 3.0
    pp_min_mmhg: float = 20.0            # physiological pulse pressure
    pp_max_mmhg: float = 100.0

    # --- PPV [pyvital / Michard 2000] ---
    ppv_min_hr_bpm: float = 40.0         # min beats = ceil(40 × window_s / 60) − pairing loss
    ppv_pairing_beat_loss: int = 2       # pairing drops the first and last peak
    ppv_min_beats_floor: int = 2         # need ≥ 2 beats for PP max/min
    ppv_rhythm_cv_max: float = 0.2       # beat-length std/mean > 0.2 → irregular rhythm, PPV invalid
    ppv_shape_corr_min: float = 0.9      # beats with r < 0.9 to the mean beat are artifacts
    ppv_shape_grid: int = 128            # resample points for shape comparison
    ppv_valid_max: float = 50.0          # accept 0 < PPV < 50 %

    # --- dP/dt_max [ref §2.5] ---
    dpdt_sg_window_points: int = 5       # 5-point quadratic Savitzky–Golay derivative (50 ms @ 100 Hz)
    dpdt_sg_polyorder: int = 2
    dpdt_min_mmhg_s: float = 30.0        # beat values outside 30–5000 mmHg/s are discarded
    dpdt_max_mmhg_s: float = 5000.0

    # --- τ (diastolic decay) [ref §2.6] ---
    tau_fit_start_frac: float = 0.55     # fit window: 55–92 % of the beat from its foot
    tau_fit_end_frac: float = 0.92
    tau_pinf_offset_mmhg: float = 5.0    # P∞ = beat diastolic minimum − 5 mmHg
    tau_min_fit_samples: int = 3         # [study] ≥ 3 points for the log-linear fit
    tau_min_s: float = 0.12              # [study] beats outside 0.12–1.50 s are excluded
    tau_max_s: float = 1.50              # (the reference uses this range only as a case-level flag)

    def __post_init__(self):
        if self.beat_aggregator not in ("median", "mean"):
            raise ValueError(f"beat_aggregator must be 'median' or 'mean', got {self.beat_aggregator!r}")

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = HemodynamicConfig()
