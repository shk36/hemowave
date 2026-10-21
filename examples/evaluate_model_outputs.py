"""Score a model's test-set predictions and write one CSV row per segment.

The input file holds ``{'test_Y': (N, L[, 1]), 'pred': (N, L[, 1])}`` in mmHg —
a joblib/pickle dump or an ``.npz``. Optionally pass a CSV with one patient ID
per segment to also report the global R² with a patient-cluster bootstrap CI.

    python examples/evaluate_model_outputs.py preds.pkl --fs 100 --jobs 8 \
        --out metrics.csv --groups segments.csv --group-col rid

Interrupted runs resume from the last row written to ``--out``.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import hemowave as hw


def load(path: Path):
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=False)
    else:
        import joblib
        d = joblib.load(path)
    if isinstance(d, (tuple, list)):
        return d[0], d[1]
    return d["test_Y"], d["pred"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--fs", type=float, default=100)
    ap.add_argument("--out", type=Path, default=Path("metrics.csv"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-agreement", action="store_true", help="skip MAE…NDTW (NDTW is slow)")
    ap.add_argument("--no-hemodynamics", action="store_true", help="skip PPV, dPdt_max, tau")
    ap.add_argument("--groups", type=Path, help="CSV with one patient ID per segment, same order")
    ap.add_argument("--group-col", default="rid")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    y_true, y_pred = load(args.predictions)
    print(f"{len(y_true):,} segments → {args.out}")
    hw.evaluate_dataset(y_true, y_pred, args.fs, n_jobs=args.jobs, out_csv=args.out,
                        agreement=not args.no_agreement, hemodynamics=not args.no_hemodynamics)

    df = pd.read_csv(args.out)
    print(df.drop(columns="idx").mean(numeric_only=True).round(4).to_string())

    if args.groups:
        groups = pd.read_csv(args.groups, usecols=[args.group_col])[args.group_col].to_numpy()
        print(hw.cluster_bootstrap_r2(y_true, y_pred, groups, n_boot=args.n_boot, progress=True))


if __name__ == "__main__":
    main()
