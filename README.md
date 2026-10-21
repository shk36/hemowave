<div align="center">

# hemowave

**Hemodynamic indices and waveform agreement metrics for arterial line (A-line) blood pressure waveforms**

[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](#installation)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](tests)

</div>

`hemowave` works on **arterial line (A-line) blood pressure waveforms in mmHg**.
It does two things:

1. **Characterise one A-line waveform** → PPV, dP/dt_max, τ
2. **Compare two A-line waveforms** — e.g. a measured one and one estimated by a
   model (such as ABP reconstructed from PPG) → MAE, R², PCC, CCC, Cross_Corr, NDTW

| Family | Metric | Input | Unit | What it captures |
|---|---|---|---|---|
| **Hemodynamic indices** | `PPV` | 1 A-line | % | Pulse pressure variation (fluid responsiveness) |
| | `dPdt_max` | 1 A-line | mmHg/s | Steepest systolic upstroke (contractility) |
| | `tau` | 1 A-line | s | Diastolic decay time constant (arterial run-off) |
| **Waveform agreement** | `MAE` | 2 A-lines | mmHg | Mean absolute error |
| | `R2` | 2 A-lines | – | Coefficient of determination |
| | `PCC` | 2 A-lines | – | Pearson correlation (shape) |
| | `CCC` | 2 A-lines | – | Lin's concordance (shape **and** level) |
| | `Cross_Corr` | 2 A-lines | – | Best correlation over all time lags |
| | `NDTW` | 2 A-lines | – | Dynamic time warping distance, normalised |

These are the implementations used in our study, packaged as a library. A
regression test checks that they reproduce the original analysis scripts
([Reproducibility](#reproducibility)).

---

## Installation

```bash
pip install git+https://github.com/shk36/hemowave.git
# or, from a clone
pip install -e ".[dev]"
```

Requires `numpy`, `scipy`, `pandas`, `scikit-learn` and `fastdtw`. No deep-learning
frameworks, and no pyvital pin: the beat detector from pyvital 0.4.0 is bundled
(MIT).

## Inputs and outputs

### What goes in

| | Requirement |
|---|---|
| **Signal** | Invasive arterial pressure (A-line) in **mmHg** |
| **Shape** | One segment: `(L,)`, `(L, 1)`, `(1, L)` or a list. Many segments: `(N, L)`, `(N, L, 1)` or a list of 1-D arrays |
| **Length** | Any length ≥ 8 samples; the defaults were tuned on 10.24 s (1,024 samples at 100 Hz). A few beats are needed: PPV needs ≥ ⌈40 bpm × duration / 60⌉ − 2 clean beats |
| **Sampling rate** | `fs` in Hz, any value (beat detection runs internally at 100 Hz) |
| **Pairs** | For agreement metrics, `y_true` and `y_pred` must have the same length and be time-aligned |
| **Normalised input** | If the waveform was scaled to [0, 1], pass `mmhg_min=`, `mmhg_max=` to restore mmHg |

> **mmHg matters.** Beat QC accepts pulse pressures of 20–100 mmHg and
> dP/dt_max of 30–5000 mmHg/s, and τ uses `P∞ = min − 5 mmHg`. A waveform in other
> units (e.g. PPG in a.u.) still gets beats detected, but every beat fails QC and
> all three indices come back NaN.

### What comes out

**One A-line → hemodynamic indices**

```python
hw.extract_hemodynamics(aline, fs=100)
```

| Key | Type | Meaning |
|---|---|---|
| `PPV` | float | Pulse pressure variation, % |
| `dPdt_max` | float | Median over beats, mmHg/s |
| `tau` | float | Median over beats, s |
| `n_beats` | int | Beats detected and paired |
| `n_ppv`, `n_dpdt`, `n_tau` | int | Beats that passed QC and contributed to each index |

A value that cannot be computed (too few clean beats, irregular rhythm, flat or
non-finite segment) is `NaN`, with its beat count `0`. `hw.ppv`, `hw.dpdt_max`
and `hw.tau` return just the float.

**Two A-lines → agreement**

```python
hw.evaluate_agreement(y_true, y_pred)
```

| Key | Range | Best |
|---|---|---|
| `MAE` | ≥ 0, mmHg | 0 |
| `R2` | ≤ 1 (can be very negative on flat segments) | 1 |
| `PCC` | −1 … 1 | 1 |
| `CCC` | −1 … 1 | 1 |
| `Cross_Corr` | −1 … 1, always ≥ `PCC` | 1 |
| `NDTW` | ≥ 0 | 0 |

Raises `ValueError` on non-finite samples or length mismatch. `hw.mae`, `hw.r2`,
`hw.pcc`, `hw.ccc`, `hw.cross_corr`, `hw.ndtw` return one metric each.

**Two A-lines → everything**

```python
hw.evaluate(y_true, y_pred, fs=100)              # one pair  → flat dict
hw.evaluate_dataset(Y_true, Y_pred, fs=100)      # N pairs   → DataFrame, one row per pair
```

| Columns | Meaning |
|---|---|
| `idx` | Segment index (`evaluate_dataset` only) |
| `MAE R2 PCC CCC Cross_Corr NDTW` | Agreement (NaN for a pair that raises `ValueError`) |
| `PPV_true`, `PPV_pred`, `PPV_diff` | Index of each waveform and `pred − true` (positive ⇒ overestimation) |
| `dPdt_max_*`, `tau_*` | Same for dP/dt_max and τ |
| `n_beats_*`, `n_ppv_*`, `n_dpdt_*`, `n_tau_*` | Beat counts of each waveform (`_true`, `_pred`) |

**N pairs + patient IDs → dataset R²**

```python
hw.cluster_bootstrap_r2(Y_true, Y_pred, groups=patient_id)
# R2Result(r2, ci_low, ci_high, n_segments, n_patients)
```

### Which function do I need?

| I have… | I want… | Use |
|---|---|---|
| one A-line segment | PPV / dP/dt_max / τ | `extract_hemodynamics` |
| a measured and an estimated A-line | how similar they are | `evaluate_agreement` |
| a measured and an estimated A-line | similarity **and** whether the estimate preserves PPV / dP/dt_max / τ | `evaluate` |
| a whole test set of pairs | all of the above per segment | `evaluate_dataset` |
| a whole test set + patient IDs | one R² with a 95 % CI | `cluster_bootstrap_r2` |

## Quick start

```python
import hemowave as hw

aline = hw.synthetic_abp(seed=0)        # 10.24 s @ 100 Hz demo A-line waveform (mmHg)

hw.extract_hemodynamics(aline, fs=100)
# {'PPV': 11.6, 'dPdt_max': 562.3, 'tau': 0.28,
#  'n_beats': 11, 'n_ppv': 11, 'n_dpdt': 11, 'n_tau': 11}

hw.ppv(aline, fs=100)                    # or one index at a time
```

Compare a measured A-line (`ref`) with an estimated one (`est`):

```python
ref, est = hw.synthetic_abp(seed=0), hw.synthetic_abp(seed=1)

hw.evaluate_agreement(ref, est)
# {'MAE': 2.64, 'R2': 0.89, 'PCC': 0.947, 'CCC': 0.947, 'Cross_Corr': 0.955, 'NDTW': 0.0093}

hw.evaluate(ref, est, fs=100)            # all nine metrics + PPV/dPdt_max/tau of both, and their difference
```

### Whole test sets

```python
# Y, Y_hat: (N, L) or (N, L, 1) arrays, or lists of 1-D arrays
df = hw.evaluate_dataset(Y, Y_hat, fs=100, n_jobs=8)        # → pandas.DataFrame, one row per segment

# Millions of segments: stream to disk; rerunning resumes after the last written row
hw.evaluate_dataset(Y, Y_hat, fs=100, n_jobs=8, out_csv="metrics.csv")
```

Columns are listed under [What comes out](#what-comes-out). NDTW dominates run time. Pass `agreement=False` if you only need the
hemodynamic indices. With `n_jobs > 1` in a script on macOS or Windows, call it
under `if __name__ == "__main__":`, as with any `multiprocessing` code.

### Dataset-level R² with a patient-cluster CI

Segment-level R² is unbounded below for flat segments, so averaging it is
misleading. Report the pooled R² with a CI from resampling **patients**:

```python
r = hw.cluster_bootstrap_r2(Y, Y_hat, groups=patient_id, n_boot=2000, seed=42)
print(r)   # e.g. R² = 0.957 [0.956, 0.957] (50 segments, 10 patients)  — from examples/quickstart.py
r.r2, r.ci_low, r.ci_high
```

---

## Method

### Beat segmentation (shared by all three indices)

1. Peaks and feet are found with pyvital's `detect_peaks`. The band-pass filter
   is used only to locate beats; every index is computed on the **raw** mmHg
   signal.
2. Each peak is paired with its preceding and following foot →
   `(foot, peak, next_foot)`. Edge peaks without both feet are dropped.
3. Beat QC: length 0.3–3.0 s (20–200 bpm) and pulse pressure 20–100 mmHg.

### Hemodynamic indices

| Index | Per-beat value | Exclusion | Window value |
|---|---|---|---|
| **PPV** | pulse pressure `P[peak] − P[foot]` | irregular rhythm (CV of beat length > 0.2), beat shape r < 0.9 vs. mean beat, fewer than ⌈40 bpm × window⌉ − 2 beats | `(PPmax − PPmin) / mean(PPmax, PPmin) × 100`, accepted if 0 < PPV < 50 |
| **dP/dt_max** | max of a 5-point quadratic Savitzky–Golay derivative over foot → peak | outside 30–5000 mmHg/s | median |
| **τ** | `−1/slope` of `ln[P(t) − P∞]` vs. t over 55–92 % of the beat, `P∞ = min(beat) − 5 mmHg` | < 3 fit points, slope ≥ 0, τ outside 0.12–1.50 s | median |

Every threshold lives in `HemodynamicConfig`, so you can change one in a line:

```python
cfg = hw.HemodynamicConfig(beat_aggregator="mean", tau_max_s=2.0)
hw.extract_hemodynamics(abp, fs=100, config=cfg)
```

> **Adapted, not identical.** dP/dt_max and τ follow Ensarioğlu & Yıldırım
> (2026), adapted to 100 Hz / 10.24 s segments: a 5-point instead of a 21-point
> SG kernel (same 40 ms span, different kernel), aggregation over one segment
> instead of one minute, a different beat detector, and the τ range applied per
> beat as an artifact filter rather than as a case-level flag. See the docstrings
> in [`hemodynamics.py`](src/hemowave/hemodynamics.py) for every rule the
> reference leaves unspecified.

### Waveform agreement

| Metric | Formula |
|---|---|
| MAE | `mean(|y − ŷ|)` |
| R2 | `1 − Σ(y − ŷ)² / Σ(y − ȳ)²` |
| PCC | `cov(y, ŷ) / (σ_y σ_ŷ)` |
| CCC | `2 cov(y, ŷ) / (σ_y² + σ_ŷ² + (ȳ − ŷ̄)²)` (population variances) |
| Cross_Corr | `max_k Σ (y_c ⋆ ŷ_c)[k] / (‖y_c‖ ‖ŷ_c‖)`, with `y_c`, `ŷ_c` mean-centred |
| NDTW | `FastDTW(y, ŷ) / (N × (max y − min y))` |

`Cross_Corr` equals `PCC` at zero lag, so `Cross_Corr − PCC` measures how much of
the disagreement is a pure time shift.

---

## Reproducibility

`tests/test_paper_equivalence.py` compares `hemowave` against reference outputs
produced by the original analysis scripts (with the real pyvital 0.4.0) on 20
segment pairs, including artifact-laden ones. Beat counts and NaN patterns must
match exactly, and all nine metrics plus the bootstrap R² CI to a relative
1e-12. On the numpy/scipy versions that produced the reference file they are
**bit-identical**; older stacks differ by ~1e-15 inside scipy's filters.

```bash
pytest          # unit tests + doctests + paper equivalence
```

Intentional differences, limited to input the original code could not handle:

- **Degenerate segments** (a flat line, or fewer samples than the filter
  needs): pyvital raises an exception. `hemowave` returns NaN so a long run
  is not aborted.
- **Non-finite samples**: the original returns NaN for NaN input, but `±inf`
  either crashes or yields values from a failed heart-rate estimate.
  `hemowave` returns NaN for any non-finite segment.

## API

| Function | Returns |
|---|---|
| `extract_hemodynamics(waveform, fs, *, config)` | dict: `PPV dPdt_max tau n_beats n_ppv n_dpdt n_tau` |
| `ppv / dpdt_max / tau(waveform, fs, *, config)` | float |
| `evaluate_agreement(y_true, y_pred)` | dict: `MAE R2 PCC CCC Cross_Corr NDTW` |
| `mae / r2 / pcc / ccc / cross_corr / ndtw(y_true, y_pred)` | float |
| `evaluate(y_true, y_pred, fs)` | flat dict with all of the above for one pair |
| `evaluate_dataset(Y, Y_hat, fs, *, n_jobs, out_csv)` | `DataFrame`, or `None` when streaming |
| `global_r2(Y, Y_hat)` | float |
| `cluster_bootstrap_r2(Y, Y_hat, groups, *, n_boot, seed)` | `R2Result(r2, ci_low, ci_high, n_segments, n_patients)` |
| `segment_beats(data, fs)` | `Beats(foots, peaks, next_foots, pulse_pressure, lengths, qc)` |
| `synthetic_abp(duration, fs, *, hr, sbp, dbp, ppv, tau, …)` | demo waveform |

## References

- Ensarioğlu M, Yıldırım AK. Intraoperative arterial waveform morphology trajectories and prediction of postoperative acute kidney injury: a latent-class analysis of the VitalDB registry. *J Clin Med.* 2026;15(17):6871. [doi:10.3390/jcm15176871](https://doi.org/10.3390/jcm15176871)
- Michard F, et al. Relation between respiratory changes in arterial pulse pressure and fluid responsiveness in septic patients with acute circulatory failure. *Am J Respir Crit Care Med.* 2000;162(1):134–138.
- Aboy M, et al. An automatic beat detection algorithm for pressure signals. *IEEE Trans Biomed Eng.* 2005;52(10):1662–1670.
- Lin LI. A concordance correlation coefficient to evaluate reproducibility. *Biometrics.* 1989;45(1):255–268.
- Salvador S, Chan P. FastDTW: toward accurate dynamic time warping in linear time and space. *Intell Data Anal.* 2007;11(5):561–580.
- pyvital — https://github.com/vitaldb/pyvital (MIT)

## License

MIT — see [LICENSE](LICENSE). The bundled beat detector is © 2021 VitalDB, MIT.
