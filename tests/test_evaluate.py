import numpy as np
import pandas as pd
import pytest

import hemowave as hw

FS = 100


@pytest.fixture(scope="module")
def dataset():
    Y = np.stack([hw.synthetic_abp(seed=s) for s in range(6)])
    P = Y + np.random.default_rng(0).normal(0, 3, Y.shape)
    return Y, P


def test_evaluate_row_layout(dataset):
    Y, P = dataset
    row = hw.evaluate(Y[0], P[0], FS)
    assert ["idx", *row] == hw.result_columns()
    assert row["PPV_diff"] == pytest.approx(row["PPV_pred"] - row["PPV_true"], nan_ok=True)


def test_evaluate_nan_agreement_on_invalid_pair(dataset):
    Y, _ = dataset
    bad = Y[0].copy()
    bad[0] = np.inf
    row = hw.evaluate(Y[0], bad, FS)
    assert all(np.isnan(row[k]) for k in hw.AGREEMENT_KEYS)


def test_dataset_matches_single_calls(dataset):
    Y, P = dataset
    df = hw.evaluate_dataset(Y, P, FS, progress=False)
    assert list(df.columns) == hw.result_columns() and len(df) == len(Y)
    expected = pd.DataFrame([{"idx": i, **hw.evaluate(Y[i], P[i], FS)} for i in range(len(Y))])
    pd.testing.assert_frame_equal(df, expected[df.columns], check_dtype=False)


def test_parallel_equals_serial(dataset):
    Y, P = dataset
    a = hw.evaluate_dataset(Y, P, FS, progress=False)
    b = hw.evaluate_dataset(Y, P, FS, n_jobs=2, chunksize=2, progress=False)
    pd.testing.assert_frame_equal(a, b)


def test_family_selection(dataset):
    Y, P = dataset
    df = hw.evaluate_dataset(Y, P, FS, agreement=False, progress=False)
    assert "NDTW" not in df and "tau_true" in df
    with pytest.raises(ValueError):
        hw.evaluate_dataset(Y, P, FS, agreement=False, hemodynamics=False)


def test_streaming_csv_resumes(dataset, tmp_path):
    Y, P = dataset
    out = tmp_path / "m.csv"
    hw.evaluate_dataset(Y[:3], P[:3], FS, out_csv=out, progress=False)
    hw.evaluate_dataset(Y, P, FS, out_csv=out, progress=False)        # picks up at idx 3
    df = pd.read_csv(out)
    assert df["idx"].tolist() == list(range(len(Y)))
    full = hw.evaluate_dataset(Y, P, FS, progress=False)
    pd.testing.assert_frame_equal(df, full, check_dtype=False)


def test_streaming_refuses_foreign_csv(dataset, tmp_path):
    Y, P = dataset
    out = tmp_path / "m.csv"
    out.write_text("idx,other\n0,1\n")
    with pytest.raises(RuntimeError):
        hw.evaluate_dataset(Y, P, FS, out_csv=out, progress=False)


def test_length_mismatch(dataset):
    Y, P = dataset
    with pytest.raises(ValueError):
        hw.evaluate_dataset(Y, P[:-1], FS)
