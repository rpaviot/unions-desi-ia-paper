#!/usr/bin/env python3
"""Overlay the CSMF central SHMR from one or more fits against Dvornik+23.

For each ``--label`` (a nautilus or minuit CSMF fit written by fit_csmf.py) the
MAP/best-fit central relation

    M*_c(Mh) = M0 * x^gamma1 / (1 + x)^(gamma1 - gamma2),   x = Mh / M1

is drawn with its 68% band (posterior cloud for nautilus, parameter covariance
for minuit), then the Dvornik+23 (A&A 675 A189, Table 2 fiducial 2x2pt+SMF MMAX
column) central SHMR is overlaid for reference. Same h-convention throughout
(M* in h^-2 Msun, Mh in h^-1 Msun), so the curves are directly comparable.

Run from the NRV venv:
    /home/rpaviot/NRV_HOD/.venv_hod/bin/python scripts/plot_csmf_shmr_vs_dvornik.py \
        --labels vlim_ngal_rmin1 vlim_ngal_rmin2 --method nautilus \
        --samples BGS_RED_GMM_VLIM
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

# Dvornik+23 Table 2 fiducial (2x2pt+SMF, MMAX). Asymmetric errors -> mean of the
# two sides for the MC band. M0 [h^-2 Msun], M1 [h^-1 Msun], gamma1/2 dimensionless.
DVORNIK = dict(M0=10.519, M1=11.138, gamma1=7.096, gamma2=0.201)
DVORNIK_ERR = dict(M0=0.050, M1=0.115, gamma1=1.78, gamma2=0.010)


def mstar_central(logMh, p):
    """log10 M*_c on a log10 halo-mass grid, from a param dict (log10 M0/M1)."""
    M0 = 10.0 ** p["M0"]
    M1 = 10.0 ** p["M1"]
    Mh = 10.0 ** np.asarray(logMh)
    Msc = np.asarray(csmf_Mstar_central(Mh, M0, M1, p["gamma1"], p["gamma2"]))
    return np.log10(Msc)


def load_curve(fit_npz, method, logMh, n_mc, rng):
    """Marginalised SHMR for one fit: (median-curve, lo16, hi84, summary dict).

    The plotted curve is the per-logMh posterior MEDIAN of log10 M*_c, not the
    MAP point -- on these data the MAP sits on the curved M0-gamma2 degeneracy
    ridge (its gamma2 can rail to ~0 and the curve goes near-vertical), whereas
    the marginal median + 68% band is the honest, degeneracy-averaged relation.
    The summary reports the MARGINAL gamma2 mean +/- std (not the MAP value)."""
    d = np.load(fit_npz, allow_pickle=True)
    if method == "nautilus":
        names = [str(s) for s in d["param_names"]]
        pts = np.asarray(d["points"], float)
        logw = np.asarray(d["log_w"], float)
        w = np.exp(logw - logw.max()); w /= w.sum()
        idx = rng.choice(pts.shape[0], size=n_mc, p=w)
        draws = pts[idx]
    else:
        names = [str(s) for s in d[f"{method}_param_names"]]
        best = np.asarray(d[f"{method}_best_fit"], float)
        cov = np.asarray(d[f"{method}_covariance"], float)
        draws = rng.multivariate_normal(best, cov, size=n_mc)
    curves = np.empty((n_mc, logMh.size))
    for i, dr in enumerate(draws):
        curves[i] = mstar_central(logMh, dict(zip(names, dr)))
    lo, med, hi = np.nanpercentile(curves, [16, 50, 84], axis=0)
    summ = {n: (float(draws[:, j].mean()), float(draws[:, j].std()))
            for j, n in enumerate(names)}
    return med, lo, hi, summ


def stellar_coverage(in_dir, sample_names):
    lo, hi = np.inf, -np.inf
    for f in sorted(in_dir.glob("csmf_*_massbin*.npz")):
        dd = np.load(f, allow_pickle=True)
        if str(dd["sample"]) not in sample_names:
            continue
        lo = min(lo, float(dd["logmstar_min"]))
        hi = max(hi, float(dd["logmstar_max"]))
    return (lo, hi) if np.isfinite(lo) else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(gu.CONFIG))
    p.add_argument("--labels", nargs="+", required=True, help="fit labels to overlay")
    p.add_argument("--method", default="nautilus", choices=["minuit", "de", "nautilus"])
    p.add_argument("--samples", nargs="+", default=["BGS_RED_GMM_VLIM"])
    p.add_argument("--n-mc", type=int, default=4000)
    p.add_argument("--logMh", nargs=3, type=float, default=(11.0, 15.0, 300),
                   metavar=("MIN", "MAX", "N"))
    p.add_argument("--mstar-range", nargs=2, type=float, default=(9.8, 11.6),
                   metavar=("MIN", "MAX"))
    p.add_argument("--mstar-cap", type=float, default=11.8,
                   help="cap the shaded M* coverage top edge (FSF outlier tail)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = gu.load_config(args.config)
    dest = Path(cfg["dest"])
    rng = np.random.default_rng(42)
    logMh = np.linspace(args.logMh[0], args.logMh[1], int(args.logMh[2]))

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for k, label in enumerate(args.labels):
        fit_npz = dest / "ggl" / "csmf_fit" / f"csmf_fit_{label}_{args.method}.npz"
        if not fit_npz.exists():
            print(f"  SKIP (missing): {fit_npz}")
            continue
        y, lo, hi, pb = load_curve(fit_npz, args.method, logMh, args.n_mc, rng)
        c = colors[k % len(colors)]
        ax.fill_betweenx(logMh, lo, hi, color=c, alpha=0.22, lw=0)
        ax.plot(y, logMh, c, lw=2.2,
                label=f"{label}  (M0={pb['M0'][0]:.2f}, M1={pb['M1'][0]:.2f}, "
                      rf"$\gamma_1$={pb['gamma1'][0]:.1f}$\pm${pb['gamma1'][1]:.1f}, "
                      rf"$\gamma_2$={pb['gamma2'][0]:.2f}$\pm${pb['gamma2'][1]:.2f})")

    # Dvornik+23 reference: central + MC band from the (symmetrised) errors.
    yd = mstar_central(logMh, DVORNIK)
    dd = rng.normal(0, 1, size=(args.n_mc, 4))
    dcurves = np.empty((args.n_mc, logMh.size))
    keys = ["M0", "M1", "gamma1", "gamma2"]
    for i in range(args.n_mc):
        pi = {kk: DVORNIK[kk] + dd[i, j] * DVORNIK_ERR[kk] for j, kk in enumerate(keys)}
        dcurves[i] = mstar_central(logMh, pi)
    dlo, dhi = np.nanpercentile(dcurves, [16, 84], axis=0)
    ax.fill_betweenx(logMh, dlo, dhi, color="k", alpha=0.12, lw=0)
    ax.plot(yd, logMh, "k--", lw=2.0,
            label=r"Dvornik+23 ($\gamma_1$=7.1, $\gamma_2$=0.20)")

    cover = stellar_coverage(dest / cfg["csmf"]["out_subdir"], set(args.samples))
    if cover is not None:
        # Cap the top edge at the highest physical bin window; the raw bin7 edge
        # (~12.76) is an FSF outlier tail, not real M* coverage.
        chi = min(cover[1], args.mstar_cap)
        ax.axvspan(cover[0], chi, color="0.6", alpha=0.15, lw=0,
                   label=f"M$_*$ probed [{cover[0]:.2f}, {chi:.2f}]")

    ax.set_xlabel(r"$\log_{10}(M_*^{\rm cen}\,/\,h^{-2}M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(M_{\rm h}\,/\,h^{-1}M_\odot)$")
    ax.set_xlim(*args.mstar_range)
    ylo = float(np.interp(args.mstar_range[0], yd, logMh))
    yhi = float(np.interp(args.mstar_range[1], yd, logMh))
    ax.set_ylim(ylo - 0.3, yhi + 0.5)
    ax.set_title("CSMF central SHMR vs Dvornik+23 — anchored (n_gal) VLIM fit")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "plots" / "csmf_shmr_vs_dvornik_ngal.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
