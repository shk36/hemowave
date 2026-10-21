import numpy as np
import pytest

import hemowave as hw


@pytest.fixture
def y():
    return hw.synthetic_abp(seed=0)


def test_identical_waveforms_are_perfect(y):
    m = hw.evaluate_agreement(y, y)
    assert m["MAE"] == 0
    assert np.isclose(m["R2"], 1) and np.isclose(m["PCC"], 1) and np.isclose(m["CCC"], 1)
    assert np.isclose(m["Cross_Corr"], 1)
    assert m["NDTW"] == 0


def test_offset_hurts_ccc_not_pcc(y):
    m = hw.evaluate_agreement(y, y + 10)
    assert np.isclose(m["MAE"], 10)
    assert np.isclose(m["PCC"], 1)
    assert m["CCC"] < 0.9


def test_time_shift_is_recovered_by_cross_corr(y):
    shifted = np.roll(y, 8)
    assert hw.cross_corr(y, shifted) > hw.pcc(y, shifted) + 0.05


def test_ccc_matches_closed_form(y):
    p = 0.8 * y + 12 + np.random.default_rng(0).normal(0, 2, y.size)
    cov = np.mean((y - y.mean()) * (p - p.mean()))
    expected = 2 * cov / (y.var() + p.var() + (y.mean() - p.mean()) ** 2)
    assert np.isclose(hw.ccc(y, p), expected)


def test_constant_signals_give_nan():
    c = np.full(100, 80.0)
    assert np.isnan(hw.ccc(c, c))
    assert np.isnan(hw.cross_corr(c, c))


def test_non_finite_input_raises(y):
    bad = y.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError):
        hw.evaluate_agreement(y, bad)


def test_keys(y):
    assert tuple(hw.evaluate_agreement(y, y)) == hw.AGREEMENT_KEYS
