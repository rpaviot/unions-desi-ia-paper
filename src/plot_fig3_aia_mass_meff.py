#!/usr/bin/env python
"""Paper Fig. 3 (2x1): A_IA for the red samples vs stellar mass and halo mass.

Left : A_IA vs median log M* (BGS_RED_GMM 12 S/N bins + LRG 2x5 bins, nautilus
       NLA multipole fits).
Right: the same A_IA vs effective halo mass M_eff, obtained by interpolating
       each bin's median log M* through the final CSMF SHMR relation
       (bgs_snr_dr6_rmin05_fhcap_g1prior_z5, carried in the saved
       plots/aia_vs_meff_snr_z5.npz table; log-log linear extrapolation beyond
       the relation's range). A continuous broken power law with a free
       grid-scanned pivot is overlaid.

Usage:
    python scripts/plot_fig3_aia_mass_meff.py \
        [--fits-dir results/fits/nautilus_split_rmin30-10_zerocross] \
        [--relation-npz plots/aia_vs_meff_snr_z5.npz] \
        [--out plot_for_paper/fig3_aia_vs_mass_red.png]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_style


def collect(fits_dir, pattern):
    """(logM*_median, a1, a1_err) from every fit npz matching pattern."""
    pts = []
    for f in sorted(glob.glob(os.path.join(fits_dir, pattern))):
        d = np.load(f, allow_pickle=True)
        names = list(d["param_names"].astype(str))
        i = names.index("a1")
        pts.append((float(d["prop_logmstar_median"]),
                    float(d["bestfit"][i]), float(d["errors"][i])))
    pts.sort()
    return np.array(pts).reshape(-1, 3)


def lin_extrap(x, xp, fp):
    """np.interp with linear extrapolation beyond the node range."""
    y = np.interp(x, xp, fp)
    below, above = x < xp[0], x > xp[-1]
    if np.any(below):
        s = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y = np.where(below, fp[0] + s * (x - xp[0]), y)
    if np.any(above):
        s = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y = np.where(above, fp[-1] + s * (x - xp[-1]), y)
    return y


def fit_broken_powerlaw(logMeff, a1, a1_err, pivot_grid=None, sigma_log=None):
    """Weighted continuous broken power law in log-log space (see plot_aia_meff).

    ``sigma_log`` overrides the linearised log-space errors. It MUST be supplied
    when this routine is re-run on Monte Carlo realisations: recomputing
    sig = a1_err/(a1 ln10) from a *noisy* a1 up-weights draws that scattered high
    and down-weights those that scattered low, which biases every realisation
    upwards (the MC band then sits above the best-fit curve instead of
    bracketing it). Freezing the weights at the measured a1 removes that bias.
    """
    x = np.asarray(logMeff, float)
    y = np.log10(np.asarray(a1, float))
    if sigma_log is None:
        sig = np.asarray(a1_err, float) / (np.asarray(a1, float) * np.log(10.0))
    else:
        sig = np.asarray(sigma_log, float)
    w = 1.0 / sig**2
    if pivot_grid is None:
        pivot_grid = np.linspace(x.min() + 0.05, x.max() - 0.05, 200)

    def solve(p):
        lo, hi = x < p, x >= p
        if lo.sum() < 2 or hi.sum() < 2:
            return None
        X = np.column_stack([np.ones_like(x),
                             np.where(lo, x - p, 0.0),
                             np.where(hi, x - p, 0.0)])
        A = X.T @ (w[:, None] * X)
        b = X.T @ (w * y)
        try:
            theta = np.linalg.solve(A, b)
            cov = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return None
        chi2 = float((w * (y - X @ theta) ** 2).sum())
        return theta, cov, chi2

    prof = [(p, r) for p in pivot_grid if (r := solve(p)) is not None]
    chi2s = np.array([r[2] for _, r in prof])
    ps = np.array([p for p, _ in prof])
    j = int(np.argmin(chi2s))
    p_best, (theta, cov, chi2) = ps[j], prof[j][1]
    c0, b1, b2 = theta
    c0e, b1e, b2e = np.sqrt(np.diag(cov))
    return dict(A0=10.0**c0, beta1=b1, beta1_err=b1e, beta2=b2, beta2_err=b2e,
                logMb=p_best, chi2=chi2, dof=len(x) - 4, cov=cov)


def minuit_broken_powerlaw(logMeff, a1, a1_err, seed_fit=None):
    """4-parameter MINUIT fit of the broken power law, with the pivot FREE.

    fit_broken_powerlaw locates the pivot by scanning a grid and returns the
    (c0, beta1, beta2) covariance *conditional* on that pivot -- there is no
    logMb row, so a band propagated from it is too narrow and the slope errors
    are conditional rather than marginal. Letting MIGRAD vary logMb as a real
    parameter gives the full 4x4, which is what the appendix SHMR band uses.

    The log-space weights are frozen at the measured a1 (identical to the grid
    estimator evaluated on the data), so the point estimate is unchanged; only
    the error treatment differs.
    """
    from iminuit import Minuit

    x = np.asarray(logMeff, float)
    y = np.log10(np.asarray(a1, float))
    sig = np.asarray(a1_err, float) / (np.asarray(a1, float) * np.log(10.0))
    w = 1.0 / sig**2

    def chi2(c0, beta1, beta2, logMb):
        model = c0 + np.where(x < logMb, beta1, beta2) * (x - logMb)
        return float((w * (y - model) ** 2).sum())

    seed = seed_fit or fit_broken_powerlaw(x, a1, a1_err)
    m = Minuit(chi2, c0=np.log10(seed["A0"]), beta1=seed["beta1"],
               beta2=seed["beta2"], logMb=seed["logMb"])
    m.errordef = 1.0
    m.limits["logMb"] = (x.min() + 0.05, x.max() - 0.05)
    m.migrad()
    m.hesse()
    cov = np.array(m.covariance)
    out = dict(A0=10.0 ** m.values["c0"], c0=m.values["c0"],
               beta1=m.values["beta1"], beta2=m.values["beta2"],
               logMb=m.values["logMb"],
               beta1_err=m.errors["beta1"], beta2_err=m.errors["beta2"],
               logMb_err=m.errors["logMb"],
               chi2=float(m.fval), dof=len(x) - 4, cov=cov, valid=bool(m.valid))
    try:
        m.minos()
        out["minos"] = {k: (-m.merrors[k].lower, m.merrors[k].upper)
                        for k in ("c0", "beta1", "beta2", "logMb")}
    except Exception:
        out["minos"] = None
    return out


def minuit_band(res, mgrid_log):
    """1-sigma band from the 4x4 covariance, including the pivot column.

    d(model)/d(logMb) = -beta on each side of the break, so the pivot freedom
    contributes where it should: strongly on the steep side, weakly on the flat
    side. Returns (band_lo, band_hi) in A_IA.
    """
    xb = np.asarray(mgrid_log, float) - res["logMb"]
    beta = np.where(xb < 0, res["beta1"], res["beta2"])
    yb = res["c0"] + beta * xb
    J = np.column_stack([np.ones_like(xb),
                         np.where(xb < 0, xb, 0.0),
                         np.where(xb >= 0, xb, 0.0),
                         -beta])
    sig_y = np.sqrt(np.einsum("ij,jk,ik->i", J, res["cov"], J))
    return 10.0 ** (yb - sig_y), 10.0 ** (yb + sig_y)


def fit_broken_powerlaw_linear(logMeff, a1, a1_err, seed_fit=None):
    """4-parameter MINUIT fit of the broken power law in LINEAR A_IA.

    The model is identical to the log-space version -- a continuous broken
    power law A_IA = A0 * 10**(beta * (logM - logMb)) -- but the chi2 is formed
    on A_IA itself, where the measured errors are (approximately) Gaussian,
    instead of on log10(A_IA) with the linearised error
    sig_log = sig_A / (A ln10).

    That linearisation is a first-order expansion valid only for sig_A << A.
    Three of the fiducial bins have S/N of 0.1-0.4 (e.g. A_IA = 0.07 +/- 0.70),
    for which it returns sig_log up to 4.4 dex and a weight of essentially
    zero: in log space those bins carry 0.1% of the total weight, against 11.4%
    here. Fitting A_IA directly keeps them, at the cost of a non-linear solve.
    """
    from iminuit import Minuit

    x = np.asarray(logMeff, float)
    y = np.asarray(a1, float)
    ye = np.asarray(a1_err, float)

    def chi2(c0, beta1, beta2, logMb):
        m = 10.0 ** (c0 + np.where(x < logMb, beta1, beta2) * (x - logMb))
        return float((((y - m) / ye) ** 2).sum())

    seed = seed_fit or fit_broken_powerlaw(x, y, ye)
    m = Minuit(chi2, c0=np.log10(seed["A0"]), beta1=seed["beta1"],
               beta2=seed["beta2"], logMb=seed["logMb"])
    m.errordef = 1.0
    m.limits["logMb"] = (x.min() + 0.05, x.max() - 0.05)
    m.migrad()
    m.hesse()
    out = dict(A0=10.0 ** m.values["c0"], c0=m.values["c0"],
               beta1=m.values["beta1"], beta2=m.values["beta2"],
               logMb=m.values["logMb"],
               beta1_err=m.errors["beta1"], beta2_err=m.errors["beta2"],
               logMb_err=m.errors["logMb"],
               chi2=float(m.fval), dof=len(x) - 4,
               cov=np.array(m.covariance), valid=bool(m.valid))
    try:
        m.minos()
        out["minos"] = {k: (-m.merrors[k].lower, m.merrors[k].upper)
                        for k in ("c0", "beta1", "beta2", "logMb")}
    except Exception:
        out["minos"] = None

    # HESSE/MINOS are unreliable for the pivot -- chi2(logMb) has kinks where
    # the break crosses a data point, and on the BGS-only sample HESSE returns
    # sigma(logMb) = 0.00. Take the pivot error from an explicit delta-chi2 = 1
    # profile scan instead, which is smooth and well behaved on both samples.
    grid = np.linspace(x.min() + 0.05, x.max() - 0.05, 160)
    prof = []
    def _chi2_at(p_):
        def c2(c0, beta1, beta2):
            mm = 10.0 ** (c0 + np.where(x < p_, beta1, beta2) * (x - p_))
            return float((((y - mm) / ye) ** 2).sum())
        return c2

    for p_ in grid:
        mi = Minuit(_chi2_at(p_), c0=out["c0"], beta1=out["beta1"],
                    beta2=out["beta2"])
        mi.errordef = 1.0
        mi.migrad()
        prof.append(float(mi.fval))
    prof = np.array(prof)
    j = int(np.argmin(prof))
    ok = grid[prof - prof[j] < 1.0]
    out["logMb_prof"] = (grid[j] - ok.min(), ok.max() - grid[j])
    out["logMb_err"] = 0.5 * sum(out["logMb_prof"])
    return out


def fit_single_powerlaw_linear(logMeff, a1, a1_err, logM_pivot=13.0):
    """Single (unbroken) power law in LINEAR A_IA, for the chi2 comparison."""
    from iminuit import Minuit

    x = np.asarray(logMeff, float)
    y = np.asarray(a1, float)
    ye = np.asarray(a1_err, float)

    def chi2(c0, beta):
        return float((((y - 10.0 ** (c0 + beta * (x - logM_pivot))) / ye) ** 2).sum())

    m = Minuit(chi2, c0=np.log10(max(np.median(y), 0.1)), beta=0.4)
    m.errordef = 1.0
    m.migrad()
    m.hesse()
    A0 = 10.0 ** m.values["c0"]
    return dict(A0=A0, A0_err=m.errors["c0"] * np.log(10.0) * A0,
                beta=m.values["beta"], beta_err=m.errors["beta"],
                chi2=float(m.fval), dof=len(x) - 2, valid=bool(m.valid))


def report_powerlaw(logMeff, a1, a1_err, n_bgs, meff_min=12.0):
    """Print the four power-law chi2 comparisons quoted in Sect. 4.1.2.

    These were previously run interactively, so the numbers in the text could
    not be regenerated from the repository. The four cases are:

      1. single power law, all bins            -- the "poor over the full range"
      2. single power law, logMeff > meff_min  -- Fortuna+25's mass range
      3. single power law, BGS red bins only   -- where the SHMR is not extrapolated
      4. broken power law, BGS red bins only

    The broken-power-law fit over all bins is already printed by main(); it is
    repeated here so the whole comparison appears together.

    All fits use the LINEAR-A_IA chi2 and therefore keep every bin, including
    the negative ones (see fit_broken_powerlaw_linear for why).
    """
    x, y, ye = map(lambda v: np.asarray(v, float), (logMeff, a1, a1_err))

    def single(mask, tag):
        r = fit_single_powerlaw_linear(x[mask], y[mask], ye[mask])
        print(f"  [single PL] {tag:28s} beta={r['beta']:+.2f}+/-{r['beta_err']:.2f}"
              f"   chi2/dof = {r['chi2']:.1f}/{r['dof']}")
        return r

    def broken(mask, tag):
        pos = mask & (y > 0)
        seed = fit_broken_powerlaw(x[pos], y[pos], ye[pos])
        r = fit_broken_powerlaw_linear(x[mask], y[mask], ye[mask], seed_fit=seed)
        print(f"  [broken PL] {tag:28s} beta1={r['beta1']:+.2f}+/-{r['beta1_err']:.2f}"
              f"  beta2={r['beta2']:+.2f}+/-{r['beta2_err']:.2f}"
              f"  logMb={r['logMb']:.2f}+/-{r['logMb_err']:.2f}"
              f"   chi2/dof = {r['chi2']:.1f}/{r['dof']}")
        return r

    allm = np.ones(x.size, bool)
    bgsm = np.zeros(x.size, bool)
    bgsm[:n_bgs] = True

    print("\n[--report-powerlaw] chi2 comparisons quoted in Sect. 4.1.2")
    single(allm, "all bins")
    single(allm & (x > meff_min), f"logMeff > {meff_min}")
    single(bgsm, "BGS red only")
    broken(allm, "all bins")
    broken(bgsm, "BGS red only")
    print()


def band_from_cov(res, mgrid_log, linear=True):
    """1-sigma band on A_IA from the 4x4 covariance, pivot column included.

    m = 10**(c0 + beta*(x-xb))  ->  dm/dtheta = ln10 * m * [1, (x-xb) on the
    active side, -beta].  With linear=False the same Jacobian is applied in
    log10 space (the historic behaviour of minuit_band).
    """
    xb = np.asarray(mgrid_log, float) - res["logMb"]
    beta = np.where(xb < 0, res["beta1"], res["beta2"])
    u = res["c0"] + beta * xb
    J = np.column_stack([np.ones_like(xb),
                         np.where(xb < 0, xb, 0.0),
                         np.where(xb >= 0, xb, 0.0),
                         -beta])
    sig_u = np.sqrt(np.einsum("ij,jk,ik->i", J, res["cov"], J))
    if not linear:
        return 10.0 ** (u - sig_u), 10.0 ** (u + sig_u)
    m_ = 10.0 ** u
    sig_m = np.log(10.0) * m_ * sig_u
    return m_ - sig_m, m_ + sig_m


def mc_broken_powerlaw(logMeff, a1, a1_err, n_mc=2000, seed=42, mgrid_log=None):
    """Monte Carlo errors for the broken power law.

    The Hessian errors of fit_broken_powerlaw are conditional on the best-fit
    pivot and distorted by the log transform. Here the FULL estimator (log-space
    solve + pivot grid scan) is re-run on n_mc Gaussian realisations of the
    A_IA points, giving the sampling distribution of every parameter including
    the pivot. The log-space weights are FROZEN at the measured a1 (see
    fit_broken_powerlaw): recomputing them per realisation biased the band
    high by up to 0.07 dex, enough that the plotted best-fit curve fell outside
    its own 1-sigma band near the break. Realisation points with A_IA <= 0 are
    dropped (the log transform requires positivity); the fraction dropped is
    reported so the residual selection bias is visible. Returns percentile
    (16/50/84) summaries and, when mgrid_log is given, the 16/84 band of the
    model curves.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(logMeff, float)
    a1 = np.asarray(a1, float)
    a1_err = np.asarray(a1_err, float)
    sig_frozen = a1_err / (a1 * np.log(10.0))
    pars, curves = [], []
    n_dropped = 0
    for _ in range(n_mc):
        y = rng.normal(a1, a1_err)
        pos = y > 0
        n_dropped += (~pos).sum()
        if pos.sum() < 5:
            continue
        try:
            br = fit_broken_powerlaw(x[pos], y[pos], a1_err[pos],
                                     sigma_log=sig_frozen[pos])
        except (np.linalg.LinAlgError, ValueError):
            continue
        pars.append([br["A0"], br["beta1"], br["beta2"], br["logMb"]])
        if mgrid_log is not None:
            xb = mgrid_log - br["logMb"]
            curves.append(np.log10(br["A0"])
                          + np.where(xb < 0, br["beta1"], br["beta2"]) * xb)
    pars = np.array(pars)
    q = {name: np.percentile(pars[:, i], [16, 50, 84])
         for i, name in enumerate(("A0", "beta1", "beta2", "logMb"))}
    out = {name: dict(median=v[1], lo=v[1] - v[0], hi=v[2] - v[1])
           for name, v in q.items()}
    out["frac_dropped"] = n_dropped / (n_mc * len(x))
    out["n_ok"] = len(pars)
    if mgrid_log is not None and curves:
        band = np.percentile(np.array(curves), [16, 84], axis=0)
        out["band_lo"], out["band_hi"] = 10.0**band[0], 10.0**band[1]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits-dir",
                   default="results/fits/nautilus_split_rmin30-10_zerocross")
    p.add_argument("--suffix", default="multipoles_NLA_nautilus")
    p.add_argument("--relation-npz", default="plots/aia_vs_meff_snr_z5.npz",
                   help="saved table carrying rel_logmstar / rel_logmeff "
                        "(final z5 CSMF SHMR)")
    p.add_argument("--out", default="plot_for_paper/fig3_aia_vs_mass_red.png")
    p.add_argument("--lrg-by-z", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="draw the two LRG redshift bins in separate colours "
                        "(open squares for 0.4<z<0.75, filled circles for "
                        "0.75<z<1.1) instead of one combined 0.4<z<1.1 "
                        "series. On by default since 2026-09-01: the low-z "
                        "LRG bins sit low at fixed mass and the figure has to "
                        "show it. Pass --no-lrg-by-z for the pre-2026-09-01 "
                        "combined series. The fit is unchanged either way -- "
                        "it always uses all 22 bins -- this only changes what "
                        "the eye can see.")
    p.add_argument("--panel", choices=("both", "mass", "meff"), default="both",
                   help="which panel to save. 'both' (default) is the paper "
                        "two-panel figure; 'mass' saves only A_IA vs M* and "
                        "'meff' only A_IA vs M_eff, each as a standalone "
                        "square figure (for talks). Both panels are always "
                        "computed, so the printed fit numbers are identical.")
    p.add_argument("--mc", type=int, default=0,
                   help="Monte Carlo realisations for the broken-power-law "
                        "errors (re-runs the full estimator incl. the pivot "
                        "scan per realisation; 0 = Hessian errors as before)")
    p.add_argument("--mc-band", type=int, default=0,
                   help="Monte Carlo realisations for the shaded BAND and the "
                        "pivot error, keeping the Hessian (parabolic) errors on "
                        "the slopes. The analytic band propagates the "
                        "(c0,beta1,beta2) covariance at FIXED pivot, but logMb "
                        "is fitted (dof counts it), so that band is too narrow "
                        "-- ~1.6x on median and ~1.8x at the break. Use this "
                        "for the paper figure: quoted slope errors stay "
                        "parabolic/MINUIT, the band honestly includes the pivot "
                        "freedom, and logMb carries its MC percentile error.")
    p.add_argument("--report-powerlaw", action="store_true",
                   help="print the single/broken power-law chi2 comparisons "
                        "quoted in Sect. 4.1.2 and exit after plotting")
    p.add_argument("--fit-space", choices=("linear", "log"), default="linear",
                   help="metric for the broken-power-law chi2. 'linear' (default) "
                        "fits A_IA itself, where the measured errors are Gaussian. "
                        "'log' fits log10(A_IA) with the linearised error "
                        "sig_A/(A ln10), which is only valid for sig_A << A and "
                        "gives the three S/N~0.1-0.4 bins essentially zero weight.")
    args = p.parse_args()

    paper_style.apply_style()

    bgs = collect(args.fits_dir, f"BGS_RED_GMM_*massbin*_{args.suffix}.npz")
    lrg = collect(args.fits_dir, f"LRG_*massbin*_{args.suffix}.npz")
    lrg_lo = collect(args.fits_dir, f"LRG_zmin_0.40_*massbin*_{args.suffix}.npz")
    lrg_hi = collect(args.fits_dir, f"LRG_zmin_0.75_*massbin*_{args.suffix}.npz")
    if args.lrg_by_z:
        assert len(lrg_lo) + len(lrg_hi) == len(lrg), "LRG z-split lost bins"

    rel = np.load(args.relation_npz, allow_pickle=True)
    rel_x = np.asarray(rel["rel_logmstar"], float)
    rel_y = np.asarray(rel["rel_logmeff"], float)
    # 16-84 % band of the relation over the CSMF posterior (csmf_meff_relation.py
    # --n-samples): drawn as x error bars on the right panel, NOT propagated into the
    # broken power-law fit (which weights the points by their A_IA errors only).
    rel_band = ((np.asarray(rel["rel_logmeff_lo"], float), np.asarray(rel["rel_logmeff_hi"], float))
                if "rel_logmeff_lo" in rel else None)

    if args.panel == "both":
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(7, 3.5))
    else:
        figL, axL = plt.subplots(figsize=(5.0, 3.2))
        figR, axR = plt.subplots(figsize=(5.0, 3.2))
        fig = figL if args.panel == "mass" else figR

    # ---- left: A_IA vs logM* ----
    axL.errorbar(bgs[:, 0], bgs[:, 1], yerr=bgs[:, 2], fmt="o",
                 color="darkorange", ms=4, capsize=2,
                 label=r"BGS red, $0.1<z<0.5$")
    if args.lrg_by_z:
        axL.errorbar(lrg_lo[:, 0], lrg_lo[:, 1], yerr=lrg_lo[:, 2], fmt="s",
                     color="crimson", ms=4, capsize=2, mfc="white",
                     label=r"LRG, $0.4<z<0.75$")
        axL.errorbar(lrg_hi[:, 0], lrg_hi[:, 1], yerr=lrg_hi[:, 2], fmt="o",
                     color="crimson", ms=4, capsize=2,
                     label=r"LRG, $0.75<z<1.1$")
    else:
        axL.errorbar(lrg[:, 0], lrg[:, 1], yerr=lrg[:, 2], fmt="o",
                     color="crimson", ms=4, capsize=2,
                     label=r"LRG, $0.4<z<1.1$")
    axL.axhline(0.0, color="k", ls="--", lw=0.8)
    axL.set_xlabel(r"$\log_{10}(M_\ast\,/\,h^{-2}M_\odot)$")
    axL.set_ylabel(r"$A_{\rm IA}$")
    axL.legend(frameon=False, loc="upper left", fontsize=8 * paper_style.FONTSCALE)

    # ---- right: A_IA vs M_eff + broken power law ----
    allpts = np.vstack([bgs, lrg])
    logMeff = lin_extrap(allpts[:, 0], rel_x, rel_y)
    if rel_band is not None:
        xlo = lin_extrap(allpts[:, 0], rel_x, rel_band[0])
        xhi = lin_extrap(allpts[:, 0], rel_x, rel_band[1])
        xerr = np.vstack([np.clip(logMeff - xlo, 0, None), np.clip(xhi - logMeff, 0, None)])
        print(f"[M_eff band] median half-width {0.5 * (xhi - xlo).mean():.3f} dex over "
              f"{len(logMeff)} bins (posterior 16-84 %)")
    else:
        xerr = np.zeros((2, len(logMeff)))
    nb = len(bgs)
    axR.errorbar(logMeff[:nb], bgs[:, 1], yerr=bgs[:, 2], xerr=xerr[:, :nb], fmt="o",
                 color="darkorange", ms=4, capsize=2)
    if args.lrg_by_z:
        lo_rows = {tuple(r) for r in lrg_lo}
        is_lo = np.array([tuple(r) in lo_rows for r in lrg])
        mlrg = logMeff[nb:]
        xlrg = xerr[:, nb:]
        axR.errorbar(mlrg[is_lo], lrg[is_lo, 1], yerr=lrg[is_lo, 2], xerr=xlrg[:, is_lo], fmt="s",
                     color="crimson", ms=4, capsize=2, mfc="white")
        axR.errorbar(mlrg[~is_lo], lrg[~is_lo, 1], yerr=lrg[~is_lo, 2], xerr=xlrg[:, ~is_lo], fmt="o",
                     color="crimson", ms=4, capsize=2)
    else:
        axR.errorbar(logMeff[nb:], lrg[:, 1], yerr=lrg[:, 2], xerr=xerr[:, nb:], fmt="o",
                     color="crimson", ms=4, capsize=2)

    pos = allpts[:, 1] > 0
    br = fit_broken_powerlaw(logMeff[pos], allpts[pos, 1], allpts[pos, 2])
    print(f"[seed only -- log-space grid on positive bins, NOT a result; the fit "
          f"is the linear MINUIT below] A0={br['A0']:.2f} at logMb={br['logMb']:.2f}, "
          f"beta1={br['beta1']:.2f}+/-{br['beta1_err']:.2f}, "
          f"beta2={br['beta2']:.2f}+/-{br['beta2_err']:.2f}, "
          f"chi2/dof={br['chi2']:.1f}/{br['dof']}")
    if args.report_powerlaw:
        report_powerlaw(logMeff, allpts[:, 1], allpts[:, 2], len(bgs))

    loggrid = np.linspace(logMeff.min(), logMeff.max(), 200)
    xb = loggrid - br["logMb"]
    yb = np.log10(br["A0"]) + np.where(xb < 0, br["beta1"], br["beta2"]) * xb
    n_mc = args.mc or args.mc_band
    if n_mc:
        mc = mc_broken_powerlaw(logMeff[pos], allpts[pos, 1], allpts[pos, 2],
                                n_mc=n_mc, mgrid_log=loggrid)
        b1e = 0.5 * (mc["beta1"]["lo"] + mc["beta1"]["hi"])
        b2e = 0.5 * (mc["beta2"]["lo"] + mc["beta2"]["hi"])
        mbe = 0.5 * (mc["logMb"]["lo"] + mc["logMb"]["hi"])
        print(f"[MC x{n_mc}] A0={mc['A0']['median']:.2f} "
              f"-{mc['A0']['lo']:.2f}/+{mc['A0']['hi']:.2f}  "
              f"beta1={mc['beta1']['median']:.2f} "
              f"-{mc['beta1']['lo']:.2f}/+{mc['beta1']['hi']:.2f}  "
              f"beta2={mc['beta2']['median']:.2f} "
              f"-{mc['beta2']['lo']:.2f}/+{mc['beta2']['hi']:.2f}  "
              f"logMb={mc['logMb']['median']:.2f} "
              f"-{mc['logMb']['lo']:.2f}/+{mc['logMb']['hi']:.2f}  "
              f"(dropped A<=0 draws: {100 * mc['frac_dropped']:.1f}%)")
        band_lo, band_hi = mc["band_lo"], mc["band_hi"]
        label = (rf"$\beta_1={br['beta1']:.2f}\pm{b1e:.2f}$, "
                 rf"$\beta_2={br['beta2']:.2f}\pm{b2e:.2f}$" "\n"
                 rf"$\log M_b={br['logMb']:.2f}"
                 rf"^{{+{mc['logMb']['hi']:.2f}}}_{{-{mc['logMb']['lo']:.2f}}}$")
    else:
        # Default: 4-parameter MINUIT fit with the pivot free, band propagated
        # from the full 4x4 covariance -- the same construction as the appendix
        # SHMR band (draw/propagate the fitted covariance), and the only one of
        # the three that gives MARGINAL slope errors. The grid estimator's 3x3
        # is conditional on the pivot (too narrow, ~1.6x on median); the
        # bootstrap re-runs the grid scan per realisation and its logMb error
        # picks up a secondary chi2 minimum near logMb~12.3 that Delta-chi2 on
        # the real data disfavours by ~4.4, inflating the low side.
        if args.fit_space == "linear":
            # NO pos mask here. `pos` is a LOG-space necessity -- log10(A<=0) is
            # undefined -- not a linear-space one: this chi2 is formed on A_IA
            # itself, where a negative measurement is perfectly well defined
            # (positive model, negative datum, finite residual). Masking would
            # defeat the whole point of the linear metric, which exists to keep
            # the low-S/N bins that log space effectively deletes. So the linear
            # fit uses ALL bins and dof = n_bins - 4.
            mn = fit_broken_powerlaw_linear(logMeff, allpts[:, 1],
                                            allpts[:, 2], seed_fit=br)
        else:
            mn = minuit_broken_powerlaw(logMeff[pos], allpts[pos, 1],
                                        allpts[pos, 2], seed_fit=br)
        print(f"[MINUIT 4-par, pivot free, {args.fit_space} space] "
              f"valid={mn['valid']}  "
              f"A0={mn['A0']:.2f}  "
              f"beta1={mn['beta1']:.2f}+/-{mn['beta1_err']:.2f}  "
              f"beta2={mn['beta2']:.2f}+/-{mn['beta2_err']:.2f}  "
              f"logMb={mn['logMb']:.2f}+/-{mn['logMb_err']:.2f}  "
              f"chi2={mn['chi2']:.1f}/{mn['dof']}")
        if mn["minos"]:
            print("            MINOS: " + "  ".join(
                f"{k}=-{lo:.2f}/+{hi:.2f}" for k, (lo, hi) in mn["minos"].items()))
        br = mn
        xb = loggrid - br["logMb"]
        yb = br["c0"] + np.where(xb < 0, br["beta1"], br["beta2"]) * xb
        band_lo, band_hi = band_from_cov(br, loggrid,
                                         linear=(args.fit_space == "linear"))
        label = (rf"$\beta_1={br['beta1']:.2f}\pm{br['beta1_err']:.2f}$, "
                 rf"$\beta_2={br['beta2']:.2f}\pm{br['beta2_err']:.2f}$" "\n"
                 rf"$\log M_b={br['logMb']:.2f}\pm{br['logMb_err']:.2f}$")
    axR.plot(loggrid, 10**yb, "k-", lw=1.4, label=label)
    axR.fill_between(loggrid, band_lo, band_hi, color="k", alpha=0.15, lw=0)
    axR.axvline(br["logMb"], color="gray", ls=":", lw=0.8)
    # y stays LINEAR (Romain, 2026-09-03): a log y would drop the A_IA<=0 bin,
    # which the linear-space chi2 does fit.  Only x changes -- log10(M_eff) on a
    # linear axis, so the label reads like the left panel's.
    axR.axhline(0.0, color="k", ls="--", lw=0.8)
    axR.set_xlabel(r"$\log_{10}(M_{\rm eff}\,/\,h^{-1}M_\odot)$")
    axR.set_ylabel(r"$A_{\rm IA}$")
    axR.legend(frameon=False, loc="upper left", fontsize=8 * paper_style.FONTSCALE)

    fig.tight_layout()
    paper_style.savefig(fig, args.out, dpi=300)
    n_nonpos = int(np.count_nonzero(~pos))
    if args.fit_space == "linear":
        note = (f"all {len(allpts)} bins fitted"
                + (f" ({n_nonpos} non-positive, kept: linear metric)"
                   if n_nonpos else ""))
    else:
        note = f"{n_nonpos} non-positive A_IA excluded from the fit"
    print(f"({len(bgs)} BGS + {len(lrg)} LRG bins; {note})")


if __name__ == "__main__":
    main()
