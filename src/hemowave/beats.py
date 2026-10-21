"""Beat segmentation: detect peaks/feet, pair them into beats, apply beat QC."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._vendor.pyvital_arr import detect_peaks
from .config import DEFAULT_CONFIG, HemodynamicConfig

__all__ = ["Beats", "detect_peaks", "pair_beats", "segment_beats"]


def pair_beats(minlist, maxlist) -> list[tuple[int, int, int]]:
    """Match every peak with its preceding and following foot → ``[(foot, peak, next_foot), ...]``.

    Each peak looks up its own neighbours instead of assuming the two lists
    alternate, so one missing or extra extremum cannot shift every later beat.
    Peaks without a foot on both sides (always the first and last) are dropped;
    duplicate peaks keep their first entry.
    """
    foots = np.sort(np.asarray(minlist, dtype=int))
    peaks = np.sort(np.asarray(maxlist, dtype=int))
    if foots.size == 0 or peaks.size == 0:
        return []
    beats, seen = [], set()
    for pk in peaks:
        if pk in seen:
            continue
        prev, nxt = foots[foots < pk], foots[foots > pk]
        if prev.size and nxt.size:
            seen.add(pk)
            beats.append((int(prev[-1]), int(pk), int(nxt[0])))
    return beats


@dataclass(frozen=True)
class Beats:
    """Per-beat landmarks (sample indices) and the shared QC mask.

    ``qc`` is True when beat length is within ``beat_len_min_s``–``beat_len_max_s``
    and pulse pressure within ``pp_min_mmhg``–``pp_max_mmhg`` (both exclusive).
    """

    foots: np.ndarray
    peaks: np.ndarray
    next_foots: np.ndarray
    pulse_pressure: np.ndarray   # mmHg, data[peak] − data[foot]
    lengths: np.ndarray          # samples, next_foot − foot
    qc: np.ndarray               # bool

    def __len__(self) -> int:
        return int(self.peaks.size)


def segment_beats(data: np.ndarray, fs: float,
                  config: HemodynamicConfig = DEFAULT_CONFIG) -> Beats | None:
    """Detect, pair and QC beats in a 1-D pressure signal. ``None`` if no beat is found.

    The pyvital detector raises on degenerate input (a flat line has no maxima;
    a signal shorter than the filter padding cannot be band-passed). Those
    segments are treated as having no beats instead of aborting a whole run.
    """
    try:
        minlist, maxlist = detect_peaks(data, fs)
    except (IndexError, ValueError):
        return None
    beats = pair_beats(minlist, maxlist)
    if not beats:
        return None

    foots, peaks, next_foots = (np.array(col, dtype=int) for col in zip(*beats))
    pp_all = data[peaks] - data[foots]
    beatlens = (next_foots - foots).astype(float)
    qc = ((beatlens > config.beat_len_min_s * fs) & (beatlens < config.beat_len_max_s * fs)
          & (pp_all > config.pp_min_mmhg) & (pp_all < config.pp_max_mmhg))
    return Beats(foots, peaks, next_foots, pp_all, beatlens, qc)
