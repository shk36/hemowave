"""hemowave must reproduce the original paper scripts exactly.

``data/reference_outputs_synthetic.npz`` holds reference outputs of the scripts
used for the publication (with the real pyvital 0.4.0 detector). The inputs are
entirely synthetic — no patient data:

* ``y_true`` — 20 waveforms from :func:`hemowave.synthetic_abp` (seeds 100–119,
  random HR / pressure / PPV / τ, some with spike artifacts)
* ``y_pred`` — ``y_true`` with added noise, offset and time shift
* ``rid`` — random pseudo-patient IDs (0–7)
* ``hemo_true`` / ``hemo_pred`` / ``hemo_true_mean`` — columns follow
  :data:`hemowave.HEMODYNAMIC_KEYS` (median, and mean beat aggregation)
* ``agreement`` — columns follow :data:`hemowave.AGREEMENT_KEYS`
* ``r2_boot`` — global R² and its 95 % patient-cluster bootstrap CI
  (n_boot = 300, seed = 42) for clusters ``rid``

Beat counts and the NaN pattern must match exactly. Values must agree to a
relative 1e-12: they are bit-identical on the numpy/scipy versions used to make
the file, and differ by ~1e-15 on older stacks (SG filter / Pearson internals).
"""
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import hemowave as hw

REF = np.load(Path(__file__).parent / "data" / "reference_outputs_synthetic.npz")
RTOL = 1e-12


def assert_matches(got, expected):
    got = np.asarray(got, dtype=float)
    assert_array_equal(np.isnan(got), np.isnan(expected))
    assert_allclose(got, expected, rtol=RTOL, atol=0, equal_nan=True)


def _hemo_table(waves, config=hw.DEFAULT_CONFIG):
    rows = [hw.extract_hemodynamics(w, fs=100, config=config) for w in waves]
    return np.array([[r[k] for k in hw.HEMODYNAMIC_KEYS] for r in rows], dtype=float)


@pytest.mark.parametrize("which", ["true", "pred"])
def test_hemodynamics_match_paper(which):
    got, expected = _hemo_table(REF[f"y_{which}"]), REF[f"hemo_{which}"]
    assert_array_equal(got[:, 3:], expected[:, 3:])          # beat counts
    assert_matches(got, expected)


def test_hemodynamics_mean_aggregation_match_paper():
    cfg = hw.HemodynamicConfig(beat_aggregator="mean")
    assert_matches(_hemo_table(REF["y_true"], cfg), REF["hemo_true_mean"])


def test_reference_set_exercises_every_index():
    finite = np.isfinite(REF["hemo_true"][:, :3]).sum(axis=0)
    assert (finite >= 10).all(), finite


def test_agreement_match_paper():
    got = [[hw.evaluate_agreement(t, p)[k] for k in hw.AGREEMENT_KEYS]
           for t, p in zip(REF["y_true"], REF["y_pred"])]
    assert_matches(got, REF["agreement"])


def test_cluster_bootstrap_r2_match_paper():
    r = hw.cluster_bootstrap_r2(REF["y_true"], REF["y_pred"], REF["rid"],
                                n_boot=300, seed=42, batch=8)
    assert_matches([r.r2, r.ci_low, r.ci_high], REF["r2_boot"])


def test_single_index_functions_match_extract():
    w = REF["y_true"][0]
    full = hw.extract_hemodynamics(w, fs=100)
    assert_array_equal([hw.ppv(w, 100), hw.dpdt_max(w, 100), hw.tau(w, 100)],
                       [full["PPV"], full["dPdt_max"], full["tau"]])
