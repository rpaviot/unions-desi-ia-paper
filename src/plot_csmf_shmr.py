#!/usr/bin/env python3
"""Plot the central stellar-to-halo mass relation (SHMR) from a CSMF fit.

Reads a saved CSMF fit (``<dest>/ggl/csmf_fit/csmf_fit_<label>_<method>.npz``,
written by fit_csmf.py) and draws the best-fit central relation

    M*_c(Mh) = M0 * x^gamma1 / (1 + x)^(gamma1 - gamma2),   x = Mh / M1

exactly as ``HOD_NRV.HOD_analytical.hod_analytical.csmf_Mstar_central`` defines
it, with M0, M1 converted from the fitted log10 values (the model uses
masses_are_log10=True). A 1-sigma band is propagated from the minuit parameter
covariance by Monte-Carlo sampling the (M0, M1, gamma1, gamma2) sub-covariance.

The stellar-mass range actually probed by the fitted lens bins is shaded, so the
extrapolated (unconstrained) part of the curve is visually distinct.

Run from the NRV venv (needs jax via the package import):
    /home/rpaviot/NRV_HOD/.venv_hod/bin/python scripts/plot_csmf_shmr.py --label bgs_lrg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402

from HOD_NRV.HOD_analytical.hod_analytical import csmf_Mstar_central  # noqa: E402

# Maps the convenience --label back to the sample names whose stellar-mass
# windows define the constrained region (only used for the shaded coverage).
LABEL_SAMPLES = {
    "bgs": ["BGS_RED_GMM"], "bgs_lrg": ["BGS_RED_GMM", "LRG"],
    "bgs_nohi": ["BGS_RED_GMM"], "bgs_lrg_nohi": ["BGS_RED_GMM", "LRG"],
}

# Per-label mass bins excluded from the fit (mirrors fit_csmf.py --drop-massbins),
# so the shaded coverage band stops at the highest *fitted* bin, not the dropped one.
LABEL_DROP = {
    "bgs_nohi": {"BGS_RED_GMM": {9}},
    "bgs_lrg_nohi": {"BGS_RED_GMM": {9}, "LRG": {3}},
}


def mstar_central(logMh, p):
    """log10 M*_c on a log10 halo-mass grid, from a param dict (log10 M0/M1)."""
    M0 = 10.0 ** p["M0"]
    M1 = 10.0 ** p["M1"]
    Mh = 10.0 ** np.asarray(logMh)
    Msc = np.asarray(csmf_Mstar_central(Mh, M0, M1, p["gamma1"], p["gamma2"]))
    return np.log10(Msc)


def count_data_points(in_dir, sample_names, cfg, drop=None):
    """Number of ΔΣ points entering the fit (input bins under the rp cut).

    Used to form ndof for the nautilus path, whose npz stores only the posterior
    (no precomputed chi2/ndof, unlike the minuit/de results)."""
    drop = drop or {}
    rp_min = cfg["csmf"].get("rp_min")
    rp_max = cfg["csmf"].get("rp_max")
    n = 0
    for f in sorted(in_dir.glob("csmf_*_massbin*.npz")):
        d = np.load(f, allow_pickle=True)
        sample = str(d["sample"])
        if sample not in sample_names:
            continue
        idx = int(f.stem.split("massbin")[1])
        if idx in drop.get(sample, set()):
            continue
        rp = np.asarray(d["rp"])
        m = np.ones(len(rp), dtype=bool)
        if rp_min is not None:
            m &= rp >= rp_min
        if rp_max is not None:
            m &= rp <= rp_max
        n += int(m.sum())
    return n


def stellar_coverage(in_dir, sample_names, drop=None):
    """[min, max] log10 M* across the built input bins for these samples.

    ``drop`` maps sample name -> set of mass-bin indices to exclude (so the band
    reflects only the bins actually used in the fit)."""
    drop = drop or {}
    lo, hi = np.inf, -np.inf
    for f in sorted(in_dir.glob("csmf_*_massbin*.npz")):
        d = np.load(f, allow_pickle=True)
        sample = str(d["sample"])
        if sample not in sample_names:
            continue
        idx = int(f.stem.split("massbin")[1])
        if idx in drop.get(sample, set()):
            continue
        lo = min(lo, float(d["logmstar_min"]))
        hi = max(hi, float(d["logmstar_max"]))
    return (lo, hi) if np.isfinite(lo) else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(gu.CONFIG))
    p.add_argument("--label", default="bgs_lrg", help="fit label (default: bgs_lrg)")
    p.add_argument("--method", default="minuit", choices=["minuit", "de", "nautilus"])
    p.add_argument("--samples", nargs="+", default=None,
                   help="Override the samples whose M* coverage is shaded")
    p.add_argument("--n-mc", type=int, default=4000, help="MC draws for the 1-sigma band")
    p.add_argument("--logMh", nargs=3, type=float, default=(11.0, 15.5, 300),
                   metavar=("MIN", "MAX", "N"))
    p.add_argument("--mstar-range", nargs=2, type=float, default=(9.0, 12.0),
                   metavar=("MIN", "MAX"), help="x-axis (stellar mass) limits")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = gu.load_config(args.config)
    dest = Path(cfg["dest"])
    fit_npz = dest / "ggl" / "csmf_fit" / f"csmf_fit_{args.label}_{args.method}.npz"
    if not fit_npz.exists():
        sys.exit(f"fit file not found: {fit_npz}")

    d = np.load(fit_npz, allow_pickle=True)
    samples = args.samples or LABEL_SAMPLES.get(args.label, [])
    drop = LABEL_DROP.get(args.label)
    rng = np.random.default_rng(42)
    logMh = np.linspace(args.logMh[0], args.logMh[1], int(args.logMh[2]))

    if args.method == "nautilus":
        # Posterior npz: best-fit = MAP (max data log-L, which is pure -chi2/2),
        # credible band = the SHMR curve marginalised over the posterior cloud.
        names = [str(s) for s in d["param_names"]]
        pts = np.asarray(d["points"], float)
        logw = np.asarray(d["log_w"], float)
        logl = np.asarray(d["log_l"], float)
        imap = int(np.argmax(logl))
        best = pts[imap]
        chi2 = float(-2.0 * logl[imap])
        ndof = count_data_points(dest / cfg["csmf"]["out_subdir"], set(samples),
                                 cfg, drop) - len(names)
        # weighted resample of the posterior to propagate the band
        w = np.exp(logw - logw.max()); w /= w.sum()
        idx = rng.choice(pts.shape[0], size=args.n_mc, p=w)
        draws = pts[idx]
    else:
        names = [str(s) for s in d[f"{args.method}_param_names"]]
        best = np.asarray(d[f"{args.method}_best_fit"], float)
        cov = np.asarray(d[f"{args.method}_covariance"], float)
        chi2, ndof = float(d[f"{args.method}_chi2"]), int(d[f"{args.method}_ndof"])
        draws = rng.multivariate_normal(best, cov, size=args.n_mc)

    pbest = dict(zip(names, best))
    y_best = mstar_central(logMh, pbest)

    curves = np.empty((args.n_mc, logMh.size))
    for i, dr in enumerate(draws):
        curves[i] = mstar_central(logMh, dict(zip(names, dr)))
    lo, hi = np.nanpercentile(curves, [16, 84], axis=0)
    cover = (stellar_coverage(dest / cfg["csmf"]["out_subdir"], set(samples), drop)
             if samples else None)

    # Plot halo mass (y) against stellar mass (x): the curve is parametric in
    # logMh, so just put M* on x and Mh on y (band is horizontal -> fill_betweenx).
    fig, ax = plt.subplots(figsize=(7, 6))
    band_label = "68% (posterior)" if args.method == "nautilus" else "68% (param cov)"
    best_label = "MAP central SHMR" if args.method == "nautilus" else "best-fit central SHMR"
    ax.fill_betweenx(logMh, lo, hi, color="C0", alpha=0.25, lw=0, label=band_label)
    ax.plot(y_best, logMh, "C0-", lw=2.2, label=best_label)
    if cover is not None:
        ax.axvspan(cover[0], cover[1], color="0.6", alpha=0.18, lw=0,
                   label=f"M$_*$ probed [{cover[0]:.2f}, {cover[1]:.2f}]")
    ax.set_xlabel(r"$\log_{10}(M_*^{\rm cen}\,/\,h^{-2}M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(M_{\rm h}\,/\,h^{-1}M_\odot)$")
    ax.set_xlim(*args.mstar_range)
    # Focus the y-range on the halo masses that map into the chosen M* window.
    ylo = float(np.interp(args.mstar_range[0], y_best, logMh))
    yhi = float(np.interp(args.mstar_range[1], y_best, logMh))
    ax.set_ylim(ylo - 0.2, yhi + 0.2)
    ax.set_title(f"CSMF central SHMR — {args.label} ({args.method})\n"
                 rf"$\chi^2/{{\rm ndof}}={chi2:.1f}/{ndof}={chi2/ndof:.2f}$")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "plots" / f"csmf_shmr_{args.label}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    print(f"  best-fit M0={pbest['M0']:.3f} M1={pbest['M1']:.3f} "
          f"gamma1={pbest['gamma1']:.3f} gamma2={pbest['gamma2']:.3f}")


if __name__ == "__main__":
    main()
