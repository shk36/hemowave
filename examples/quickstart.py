"""Tour of the hemowave API on synthetic waveforms. Run: python examples/quickstart.py"""
import numpy as np

import hemowave as hw

FS = 100


def main():
    # --- one waveform → hemodynamic indices ----------------------------------
    abp = hw.synthetic_abp(seed=0, ppv=15, tau=0.5)
    f = hw.extract_hemodynamics(abp, FS)
    print(f"PPV {f['PPV']:.1f} %   dP/dt_max {f['dPdt_max']:.0f} mmHg/s   tau {f['tau']:.2f} s"
          f"   ({f['n_beats']} beats)\n")

    # --- reference vs estimate → agreement -----------------------------------
    estimate = np.roll(abp, 3) + np.random.default_rng(0).normal(0, 2, abp.size) + 4
    for k, v in hw.evaluate_agreement(abp, estimate).items():
        print(f"{k:>10}: {v:.4f}")

    # --- a small test set ----------------------------------------------------
    rng = np.random.default_rng(1)
    Y = np.stack([hw.synthetic_abp(seed=s, hr=rng.uniform(55, 110)) for s in range(50)])
    Y_hat = Y + rng.normal(0, 3, Y.shape)
    patient = np.repeat(np.arange(10), 5)

    df = hw.evaluate_dataset(Y, Y_hat, FS, n_jobs=4, progress=False)
    print()
    print(df[["MAE", "PCC", "CCC", "NDTW", "PPV_diff", "dPdt_max_diff", "tau_diff"]]
          .describe().loc[["mean", "std", "min", "max"]].round(3))
    print()
    print(hw.cluster_bootstrap_r2(Y, Y_hat, patient, n_boot=500))


if __name__ == "__main__":   # required for n_jobs > 1 on macOS/Windows (spawn start method)
    main()
