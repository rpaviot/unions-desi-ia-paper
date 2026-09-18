#!/usr/bin/env python3
"""A_IA (=a1) versus effective halo mass M_eff, using the p99 CSMF SHMR.

Step 1 -- ESTIMATE the M*-M_eff relation from the p99 fiducial CSMF fit
(csmf_fit_vlim_ngalc_rmin05_bnl12_p99). The fit's 8 volume-limited BGS bins
(bin 7 capped at its 99th percentile, exactly as fitted) are re-built, the p99
best-fit HOD params are set, and the package's central-weighted effective halo
mass M_eff = <M_h>_cen = int N_c n(M) M dM / int N_c n(M) dM is evaluated per
bin. That gives 8 points (median log M*, log M_eff) = the relation.

Step 2 -- ASSUME THE SAME RELATION for the flux-limited IA samples: each IA mass
bin's measured A_IA (a1, fiducial NLA minuit fit) is paired with M_eff obtained
by interpolating its prop_logmstar_median through the p99 relation (log-log,
linear extrapolation at the ends). Applied identically to BGS_RED_GMM and LRG.

Step 3 -- plot A1 vs M_eff on a log mass axis.

Run from the NRV venv:
    /home/rpaviot/NRV_HOD/.venv_hod/bin/python scripts/plot_aia_meff.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu                       # noqa: E402
from fit_csmf import build_fitter            # noqa: E402

CSMF_KEYS = ["M0", "M1", "gamma1", "gamma2", "sigma_c", "alpha_s", "b0", "b1"]


def p99_relation(cfg):
    """Return (logM*_med[8], logMeff[8]) from the p99 fiducial fit.

    Rebuilds the VLIM fitter exactly as the p99 fit did (bin 7 -> p99 cap),
    sets the saved best-fit params, evaluates effective_halo_mass per bin.
    beta_nl is OFF here: M_eff = <M_h|N_c,HMF> does not depend on it, and
    skipping it avoids the slow emulator init."""
    dest = Path(cfg["dest"])
    ia = dest / "ia"
    stem = "BGS_RED_GMM_VLIM_zmin_0.10_zmax_0.50_shapes"

    # p99 of LOGMSTAR per bin (only bin 7 moves; bins 0-6 == their edge).
    p99 = {}
    for i in range(8):
        m = np.asarray(__import__("pandas").read_parquet(
            f"{ia}/{stem}_massbin{i}.parquet", columns=["LOGMSTAR"])["LOGMSTAR"])
        p99[i] = float(np.percentile(m, 99))

    args = Namespace(
        samples=["BGS_RED_GMM_VLIM"], no_beta_nl=True, beta_nl_log_m_min=12.0,
        fit_ngal=None, rp_min=0.5, rp_max=None, verbose=False,
        gamma1_flat=True, gamma1_mean=None, gamma1_std=None,
        drop_massbins=None, fix=None,
    )
    fitter = build_fitter(cfg, args)
    for mb in fitter.mass_bins:                      # apply the p99 cap (model side)
        mb.logmstar_max = p99[mb.massbin_id]
    fitter._initialize_halo_model()

    fit = np.load(dest / "ggl" / "csmf_fit"
                  / "csmf_fit_vlim_ngalc_rmin05_bnl12_p99_minuit.npz", allow_pickle=True)
    m = fit_method(fit, method)
    bf = {str(n): float(v) for n, v in zip(fit[f"{m}_param_names"], fit[f"{m}_best_fit"])}
    print(f"[{label}] point estimate: {m}" + (" (posterior median)" if m == "nautilus" else ""))
    fitter._halo_model.set_hod_params({k: bf[k] for k in CSMF_KEYS})
    fitter._halo_model.update_f(f_h=bf.get("f_h", bf.get("f_c", 1.0)), f_s=bf.get("f_s", 1.0))

    m_eff = np.atleast_1d(np.asarray(fitter._halo_model.effective_halo_mass(), float))
    logmed = np.array([mb.logmstar_median for mb in fitter.mass_bins])
    order = np.argsort(logmed)
    logmed, logMeff = logmed[order], np.log10(m_eff[order])
    print("\n[p99 relation]  bin   <logM*>   logM_eff[h^-1 Msun]")
    for x, y in zip(logmed, logMeff):
        print(f"            {x:10.3f} {y:10.3f}")
    return logmed, logMeff


def fit_method(fit, method=None):
    """Which point estimate a fit npz carries: 'minuit' / 'nautilus' / 'de' (fit_csmf.py
    keys <method>_param_names / <method>_best_fit); auto-detected in that order."""
    if method:
        if f"{method}_best_fit" not in fit:
            raise KeyError(f"fit has no {method}_best_fit")
        return method
    for m in ("minuit", "nautilus", "de"):
        if f"{m}_best_fit" in fit:
            return m
    raise KeyError("fit npz has no minuit_/nautilus_/de_ best fit")


def label_relation(cfg, label, samples, in_name=None, fit_npz=None, in_dir=None, method=None):
    """(logM*_med, logMeff) from any saved CSMF minuit fit (fit_csmf label, or an
    explicit ``fit_npz`` path with the csmf-input directory ``in_dir`` it was run on).

    Same machinery as p99_relation but with the fit's own mass bins as built
    (no p99 re-capping -- the S/N samples are already capped at logM*=11.40 by
    construction). beta_nl OFF (M_eff doesn't depend on it)."""
    dest = Path(cfg["dest"])
    if fit_npz is None:
        fit_npz = dest / "ggl" / "csmf_fit" / f"csmf_fit_{label}_minuit.npz"
    label = label or Path(fit_npz).stem
    fit = np.load(fit_npz, allow_pickle=True)
    # Halo mass definition the fit was run with (recorded by fit_csmf.py since
    # 2026-09-17; older fits = the package default MassDef200c). M_eff must be
    # evaluated with the same definition.
    mass_def = str(fit["mass_definition"]) if "mass_definition" in fit else "MassDef200c"
    print(f"[{label}] halo mass definition: {mass_def}")
    args = Namespace(
        samples=samples, in_name=in_name, in_dir=in_dir, no_beta_nl=True,
        beta_nl_log_m_min=12.0, fit_ngal=None, rp_min=0.5, rp_max=None,
        verbose=False, gamma1_flat=True, gamma1_mean=None, gamma1_std=None,
        drop_massbins=None, fix=None, mass_def=mass_def,
        halo_model_kwargs=(dict(fit["halo_model_kwargs"].item()) if "halo_model_kwargs" in fit else {}),
    )
    fitter = build_fitter(cfg, args)
    fitter._initialize_halo_model()

    bf = {str(n): float(v) for n, v in zip(fit["minuit_param_names"], fit["minuit_best_fit"])}
    fitter._halo_model.set_hod_params({k: bf[k] for k in CSMF_KEYS})
    fitter._halo_model.update_f(f_h=bf.get("f_h", bf.get("f_c", 1.0)), f_s=bf.get("f_s", 1.0))

    m_eff = np.atleast_1d(np.asarray(fitter._halo_model.effective_halo_mass(), float))
    logmed = np.array([mb.logmstar_median for mb in fitter.mass_bins])
    order = np.argsort(logmed)
    logmed, logMeff = logmed[order], np.log10(m_eff[order])
    print(f"\n[{label} relation]  <logM*>   logM_eff[h^-1 Msun]")
    for x, y in zip(logmed, logMeff):
        print(f"            {x:10.3f} {y:10.3f}")
    return logmed, logMeff


def map_meff(logmstar, rel_x, rel_y):
    """log M_eff at given log M* via the p99 relation (log-log, lin extrapolation)."""
    return np.interp(logmstar, rel_x, rel_y)  # np.interp clamps; we extrapolate below


def lin_extrap(x, xp, fp):
    """np.interp but linearly extrapolating beyond the node range."""
    y = np.interp(x, xp, fp)
    below, above = x < xp[0], x > xp[-1]
    if np.any(below):
        s = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y = np.where(below, fp[0] + s * (x - xp[0]), y)
    if np.any(above):
        s = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y = np.where(above, fp[-1] + s * (x - xp[-1]), y)
    return y


def collect_ia(fits_dir, pattern, rel_x, rel_y):
    """(logM*med, logMeff, a1, a1_err) for each IA fit matching pattern."""
    rows = []
    for f in sorted(glob.glob(os.path.join(fits_dir, pattern))):
        d = np.load(f, allow_pickle=True)
        names = list(d["param_names"].astype(str))
        j = names.index("a1")
        lm = float(d["prop_logmstar_median"])
        rows.append((lm, float(lin_extrap(np.array([lm]), rel_x, rel_y)[0]),
                     float(d["bestfit"][j]), float(d["errors"][j])))
    return np.array(rows).reshape(-1, 4)


def fit_powerlaw(logMeff, a1, a1_err, logM0=13.0):
    """Single power law A_IA = A0 (M/M0)^beta fitted in LINEAR A_IA (weighted
    chi2 on A_IA itself, MINUIT), pivot M0 = 10**logM0. Never fit in log space:
    the linearised log errors delete the low-S/N bins and cannot handle
    A_IA <= 0. Returns (A0, A0_err, beta, beta_err, chi2, dof)."""
    from plot_fig3_aia_mass_meff import fit_single_powerlaw_linear
    r = fit_single_powerlaw_linear(logMeff, a1, a1_err, logM_pivot=logM0)
    return r["A0"], r["A0_err"], r["beta"], r["beta_err"], r["chi2"], r["dof"]


def fit_broken_powerlaw(logMeff, a1, a1_err, pivot_grid=None):
    """Continuous broken power law fitted in LINEAR A_IA with the pivot free
    (4-parameter MINUIT, shared with the Fig. 3 script; ``pivot_grid`` is
    ignored, kept for call compatibility). Returns the dict of
    fit_broken_powerlaw_linear plus logMb_lo / logMb_hi (dchi2 = 1 profile)."""
    from plot_fig3_aia_mass_meff import (fit_broken_powerlaw_linear,
                                         fit_broken_powerlaw as _grid_seed)
    x, y, ye = (np.asarray(v, float) for v in (logMeff, a1, a1_err))
    pos = y > 0                       # seed only (starting point, not a result)
    seed = _grid_seed(x[pos], y[pos], ye[pos])
    br = fit_broken_powerlaw_linear(x, y, ye, seed_fit=seed)
    lo, hi = br["logMb_prof"]
    br["logMb_lo"], br["logMb_hi"] = br["logMb"] - lo, br["logMb"] + hi
    br["A0_err"] = br["A0"] * np.log(10.0) * float(np.sqrt(br["cov"][0, 0]))
    return br


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits-dir", default="results/fits_zerocross_rmin25",
                   help="directory holding the per-mass-bin IA fit npz files")
    p.add_argument("--suffix", default="multipoles_NLA_minuit",
                   help="fit-file suffix selecting stat/model/sampler")
    p.add_argument("--out", default="plots/aia_vs_meff_p99.png")
    p.add_argument("--meff-min", type=float, default=0.0,
                   help="restrict the power-law FIT to bins with M_eff > this "
                        "value [h^-1 Msun]; all bins are still plotted")
    p.add_argument("--broken", action="store_true",
                   help="also fit a continuous broken power law with a free "
                        "M_eff pivot (grid-scanned) and overlay it")
    p.add_argument("--linear-y", action="store_true",
                   help="force a linear A_IA axis (default: log if all A_IA > 0)")
    p.add_argument("--relation-label", default=None,
                   help="Source the M*-M_eff relation from this CSMF minuit fit "
                        "label instead of the p99 fiducial, e.g. "
                        "bgs_snr_dr6_rmin05_fhcap_g1prior_z5")
    p.add_argument("--relation-samples", nargs="+", default=["BGS_RED_GMM_VLIM_SNR"],
                   help="Samples of the --relation-label fit")
    p.add_argument("--relation-in-name", default=None,
                   help="csmf-input subdir the fit used (fit_csmf --in-name)")
    args = p.parse_args()

    cfg = gu.load_config(str(gu.CONFIG))
    if args.relation_label:
        rel_x, rel_y = label_relation(cfg, args.relation_label,
                                      args.relation_samples, args.relation_in_name)
        rel_tag = args.relation_label
    else:
        rel_x, rel_y = p99_relation(cfg)
        rel_tag = "p99 fiducial"

    fits = args.fits_dir
    sfx = args.suffix
    bgs = collect_ia(fits, f"BGS_RED_GMM_*massbin*_{sfx}.npz", rel_x, rel_y)
    lrg_lo = collect_ia(fits, f"LRG_zmin_0.40_*massbin*_{sfx}.npz", rel_x, rel_y)
    lrg_hi = collect_ia(fits, f"LRG_zmin_0.75_*massbin*_{sfx}.npz", rel_x, rel_y)

    all_a1 = np.concatenate([a[:, 2] for a in (bgs, lrg_lo, lrg_hi) if len(a)])
    logy = bool(np.all(all_a1 > 0)) and not args.linear_y

    # Power-law fit A_IA = A0 (M_eff/M0)^beta, pivot M0 = 1e13. --meff-min
    # restricts only the FIT; every measured bin is still drawn.
    allpts = np.vstack([a for a in (bgs, lrg_lo, lrg_hi) if len(a)])
    if args.meff_min > 0:
        n_before = len(allpts)
        allpts = allpts[allpts[:, 1] > np.log10(args.meff_min)]
        print(f"[cut] power-law fit restricted to M_eff > {args.meff_min:.2e}: "
              f"{len(allpts)}/{n_before} bins")
    logM0 = 13.0
    A0, A0e, beta, betae, chi2, dof = fit_powerlaw(
        allpts[:, 1], allpts[:, 2], allpts[:, 3], logM0=logM0)
    print(f"\n[power law]  A_IA = A0 (M_eff/1e{logM0:.0f})^beta")
    print(f"   A0   = {A0:.3f} +/- {A0e:.3f}   (at M0 = 1e{logM0:.0f} h^-1 Msun)")
    print(f"   beta = {beta:.3f} +/- {betae:.3f}")
    print(f"   chi2/dof = {chi2:.1f}/{dof} = {chi2/dof:.2f}")

    br = None
    if args.broken:
        br = fit_broken_powerlaw(allpts[:, 1], allpts[:, 2], allpts[:, 3])
        print(f"\n[broken power law]  A_IA = A0 (M/Mb)^beta1 [M<Mb], ^beta2 [M>=Mb]")
        print(f"   A0     = {br['A0']:.3f} +/- {br['A0_err']:.3f}   (at the pivot)")
        print(f"   beta1  = {br['beta1']:.3f} +/- {br['beta1_err']:.3f}")
        print(f"   beta2  = {br['beta2']:.3f} +/- {br['beta2_err']:.3f}")
        print(f"   logMb  = {br['logMb']:.3f}  [{br['logMb_lo']:.3f}, "
              f"{br['logMb_hi']:.3f}] (dchi2=1)")
        print(f"   chi2/dof = {br['chi2']:.1f}/{br['dof']} = "
              f"{br['chi2']/br['dof']:.2f}")

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    def draw(a, **kw):
        if len(a):
            ax.errorbar(10**a[:, 1], a[:, 2], yerr=a[:, 3], fmt="o", ms=5,
                        capsize=2, **kw)
    draw(bgs, color="darkorange", label=r"BGS RED (GMM), $0.1<z<0.5$")
    draw(lrg_lo, color="crimson", label=r"LRG, $0.40<z<0.75$")
    draw(lrg_hi, color="purple", label=r"LRG, $0.75<z<1.10$")

    # power-law overlay (+/-1sigma band on the slope/amplitude)
    mgrid = np.logspace(np.log10(10**allpts[:, 1].min()),
                        np.log10(10**allpts[:, 1].max()), 100)
    xg = np.log10(mgrid) - logM0
    c0, c0e = np.log10(A0), A0e / (A0 * np.log(10.0))
    ymid = c0 + beta * xg
    yerr = np.sqrt(c0e**2 + (xg * betae)**2)
    ax.plot(mgrid, 10**ymid, "k-", lw=1.8, zorder=1,
            label=(rf"$A_{{\rm IA}}=({A0:.2f}\pm{A0e:.2f})"
                   rf"(M_{{\rm eff}}/10^{{13}})^{{{beta:.2f}\pm{betae:.2f}}}$"))
    ax.fill_between(mgrid, 10**(ymid - yerr), 10**(ymid + yerr),
                    color="k", alpha=0.12, zorder=0)
    ax.axvline(10**logM0, color="gray", ls=":", lw=1, zorder=0)

    if br is not None:
        xb = np.log10(mgrid) - br["logMb"]
        yb = (np.log10(br["A0"])
              + np.where(xb < 0, br["beta1"], br["beta2"]) * xb)
        ax.plot(mgrid, 10**yb, "-", color="tab:blue", lw=1.8, zorder=2,
                label=(rf"broken: $\beta_1={br['beta1']:.2f}\pm{br['beta1_err']:.2f}$, "
                       rf"$\beta_2={br['beta2']:.2f}\pm{br['beta2_err']:.2f}$, "
                       rf"$\log M_b={br['logMb']:.2f}$"))
        ax.axvline(10**br["logMb"], color="tab:blue", ls="--", lw=1, alpha=0.6)

    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    else:
        ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel(rf"$M_{{\rm eff}}\;[h^{{-1}}M_\odot]$  ({rel_tag} CSMF SHMR)")
    ax.set_ylabel(r"$A_{\rm IA}$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    ax.set_title(rf"$A_{{\rm IA}}$ vs effective halo mass ({rel_tag} SHMR)",
                 fontsize=9)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"\nwrote {out}  (yscale={'log' if logy else 'linear'}; "
          f"{len(bgs)} BGS, {len(lrg_lo)} LRG-lo, {len(lrg_hi)} LRG-hi)")

    # dump the table for reuse / sanity
    np.savez(out.with_suffix(".npz"),
             rel_logmstar=rel_x, rel_logmeff=rel_y,
             bgs=bgs, lrg_lo=lrg_lo, lrg_hi=lrg_hi,
             columns=np.array(["logM*_med", "logM_eff", "a1", "a1_err"]))


if __name__ == "__main__":
    main()
