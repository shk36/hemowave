import numpy as np
import pytest

import hemowave as hw
from hemowave.beats import pair_beats

FS = 100


def test_recovers_synthetic_ground_truth():
    f = hw.extract_hemodynamics(hw.synthetic_abp(seed=3, ppv=15, tau=0.6, noise=0.0), FS)
    assert 10 < f["PPV"] < 20
    assert f["dPdt_max"] > 300
    assert 0.12 <= f["tau"] <= 1.5
    assert f["n_beats"] >= 8


@pytest.mark.parametrize("bad", [np.zeros(5), np.full(1024, 80.0), np.zeros((2, 512))])
def test_degenerate_input_returns_nan(bad):
    f = hw.extract_hemodynamics(bad, FS)
    assert np.isnan([f["PPV"], f["dPdt_max"], f["tau"]]).all()
    assert f["n_beats"] == 0


@pytest.mark.parametrize("pos", [0, 500, 1023])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_samples_return_nan(pos, value):
    y = hw.synthetic_abp(seed=0)
    y[pos] = value
    f = hw.extract_hemodynamics(y, FS)
    assert np.isnan([f["PPV"], f["dPdt_max"], f["tau"]]).all() and f["n_beats"] == 0


def test_column_vector_input_is_accepted():
    y = hw.synthetic_abp(seed=0)
    a, b = hw.extract_hemodynamics(y, FS), hw.extract_hemodynamics(y[:, None], FS)
    assert a == b or all(np.isclose(a[k], b[k], equal_nan=True) for k in a)


def test_input_is_not_modified():
    y = hw.synthetic_abp(seed=0)
    before = y.copy()
    hw.extract_hemodynamics(y, FS)
    np.testing.assert_array_equal(y, before)


def test_unit_rescaling():
    y = hw.synthetic_abp(seed=0)
    lo, hi = 20.0, 200.0
    scaled = (y - lo) / (hi - lo)
    a = hw.extract_hemodynamics(y, FS)
    b = hw.extract_hemodynamics(scaled, FS, mmhg_min=lo, mmhg_max=hi)
    assert np.isclose(a["dPdt_max"], b["dPdt_max"])


def test_tau_range_is_enforced():
    cfg = hw.HemodynamicConfig(tau_min_s=0.12, tau_max_s=0.13)
    t = hw.tau(hw.synthetic_abp(seed=1, tau=1.0), FS, config=cfg)
    assert np.isnan(t) or 0.12 <= t <= 0.13


def test_irregular_rhythm_invalidates_ppv():
    y = hw.synthetic_abp(seed=2, hr_jitter=0.45)
    assert np.isnan(hw.ppv(y, FS))


def test_config_validation():
    with pytest.raises(ValueError):
        hw.HemodynamicConfig(beat_aggregator="max")


def test_pair_beats_is_robust_to_missing_extrema():
    # foot at 30 missing: the peak at 40 should pair with 10 → 50, nothing shifts
    assert pair_beats([10, 50, 90], [20, 40, 70, 100]) == [(10, 20, 50), (10, 40, 50), (50, 70, 90)]
    assert pair_beats([], [1, 2]) == []
