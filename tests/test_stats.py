import numpy as np
import pytest
from sklearn.metrics import r2_score

import hemowave as hw


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    Y = np.stack([hw.synthetic_abp(seed=s) for s in range(40)])
    P = Y + rng.normal(0, 4, Y.shape)
    rid = rng.integers(0, 12, len(Y))
    return Y, P, rid


def test_global_r2_equals_pooled_r2(data):
    Y, P, _ = data
    assert np.isclose(hw.global_r2(Y, P), r2_score(Y.ravel(), P.ravel()))


def test_bootstrap_ci_brackets_point(data):
    r = hw.cluster_bootstrap_r2(*data, n_boot=200)
    assert r.ci_low <= r.r2 <= r.ci_high
    assert r.n_segments == 40 and r.n_patients == len(np.unique(data[2]))
    assert "R²" in str(r)


def test_bootstrap_is_reproducible_and_shape_agnostic(data):
    Y, P, rid = data
    a = hw.cluster_bootstrap_r2(Y, P, rid, n_boot=100, seed=7)
    b = hw.cluster_bootstrap_r2(Y[..., None], P[..., None], rid, n_boot=100, seed=7)
    c = hw.cluster_bootstrap_r2(list(Y), list(P), rid, n_boot=100, seed=7)
    assert a == b == c


def test_group_length_checked(data):
    Y, P, rid = data
    with pytest.raises(ValueError):
        hw.cluster_bootstrap_r2(Y, P, rid[:-1])
