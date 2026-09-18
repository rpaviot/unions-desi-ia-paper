#!/usr/bin/env python3
"""Shared helpers for the galaxy-galaxy lensing (Delta Sigma) pipeline.

Used by ``compute_magnification.py``, ``estimate_nz.py`` and
``compute_deltasigma.py``. Ports the legacy logic from ``old_codes/``
(lens_magnification-2.py, get_IA_samples.py PSF leakage) without importing it:

  * flux <-> magnitude conversions and the lensed-magnitude perturbations
    (total mags, fiber mags with a De Vaucouleurs / exponential profile
    correction, stellar masses in log space);
  * the DESI BGS (bright+faint) and LRG (Zhou et al. 2022, N/S) photometric
    target-selection cuts re-applied under the perturbation;
  * the finite-difference alpha estimator (dN counts + linear fit);
  * source-catalogue preparation: veto mask, responsivity calibration
    (same formula as build_ia_samples.calibrate) and the two-pass per-object
    PSF leakage correction (Li et al. 2024), in that order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config" / "data_sources.yaml"


def load_config(path=CONFIG):
    with open(path) as fh:
        return yaml.safe_load(fh)


def lens_stem(sample, massbin):
    """Filename stem of one lens (sample, mass bin): the IA shapes parquet."""
    return (f"{sample['name']}_zmin_{sample['zmin']:.2f}_zmax_{sample['zmax']:.2f}"
            f"_shapes_massbin{massbin}")


# =============================================================================
# FLUX / MAGNITUDE CONVERSIONS AND LENSING PERTURBATION
# =============================================================================
def flux_to_mag(flux, transmission=1.0):
    """Nanomaggies -> AB magnitude, dereddened by MW_TRANSMISSION."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 22.5 - 2.5 * np.log10(np.asarray(flux, dtype=float) / transmission)


def _profile_intensity(r, r_e, exponential):
    if exponential:
        return np.exp(-1.678 * (r / r_e - 1.0))
    return np.exp(-7.669 * ((r / r_e) ** 0.25 - 1.0))   # De Vaucouleurs


def _fiber_flux(theta_e, fiber_radius, exponential, dr=0.01):
    r = np.arange(0.0, fiber_radius + dr, dr)
    return np.trapezoid(_profile_intensity(r, theta_e, exponential) * 2 * np.pi * r, x=r)


def fiber_correction_grid(exponential, fiber_radius=0.75):
    """(theta_e grid, dlnF/dlntheta at the fiber radius) for fast interpolation.

    The fiber flux of an extended source lensed by kappa scales as
    F * (1 + (2 - dlnF/dlntheta) * kappa) -- point sources recover the full 2k."""
    theta_grid = np.arange(0.05, 10.0, 0.05)
    fiber_sizes = np.arange(0.1, 3.0, 0.05)
    corr = np.empty_like(theta_grid)
    for i, te in enumerate(theta_grid):
        fluxes = np.array([_fiber_flux(te, fr, exponential) for fr in fiber_sizes])
        dlnf = np.gradient(np.log(fluxes), np.log(fiber_sizes))
        corr[i] = np.interp(fiber_radius, fiber_sizes, dlnf)
    return theta_grid, corr


def fiber_corrections(shape_r, theta_grid, corr_grid):
    """Per-galaxy fiber correction factor from SHAPE_R (arcsec); bad sizes
    (<=0.001, >20, non-finite) take the sample median, as in the legacy code."""
    theta_e = np.asarray(shape_r, dtype=float).copy()
    bad = (theta_e <= 0.001) | (theta_e > 20.0) | ~np.isfinite(theta_e)
    theta_e[bad] = np.median(theta_e[~bad]) if (~bad).any() else 1.0
    return np.interp(theta_e, theta_grid, corr_grid)


def lensed_total_mag(mag, kappa):
    """Total magnitude under convergence kappa: flux magnified by (1 + 2k)."""
    return mag - 2.5 * np.log10(1.0 + 2.0 * kappa)


def lensed_fiber_mag(mag_fiber, kappa, fiber_corr):
    """Fiber magnitude under kappa with the extended-profile aperture factor."""
    flux = 10.0 ** (-0.4 * (mag_fiber - 22.5))
    return 22.5 - 2.5 * np.log10(flux * (1.0 + (2.0 - fiber_corr) * kappa))


def lensed_logmass(logm, kappa):
    """Inferred log10 M* under kappa: luminosity (hence M* at fixed M/L)
    magnified by (1 + 2k)."""
    return logm + np.log10(1.0 + 2.0 * kappa)


# =============================================================================
# DESI TARGET-SELECTION CUTS (re-applied under the perturbation)
# =============================================================================
def bgs_selection(mag_g, mag_r, mag_z, mag_w1, mag_r_fiber, sample="BGS_ANY"):
    """DESI BGS bright/faint/any photometric selection (Hahn et al. 2023)."""
    fmc = (((mag_r < 17.8) & (mag_r_fiber < 22.9 + (mag_r - 17.8))) |
           ((mag_r >= 17.8) & (mag_r < 20.0) & (mag_r_fiber < 22.9)))
    bright = (mag_r < 19.5) & fmc
    colour = (mag_z - mag_w1) - 1.2 * (mag_g - mag_r) + 1.2
    fcc = (((colour < 0) & (mag_r_fiber < 20.75)) |
           ((colour >= 0) & (mag_r_fiber < 21.5)))
    faint = (mag_r >= 19.5) & (mag_r < 20.175) & fcc
    if sample == "BGS_BRIGHT":
        return bright
    if sample == "BGS_FAINT":
        return faint
    if sample == "BGS_ANY":
        return bright | faint
    raise ValueError(f"unknown BGS sample {sample!r}")


def _lrg_cuts(mag_g, mag_r, mag_z, mag_w1, mag_z_fiber, north):
    """Zhou et al. 2022 LRG cuts (Eq. 2 north / Eq. 1 south)."""
    if north:
        return ((mag_z_fiber < 21.61)
                & ((mag_z - mag_w1) > 0.8 * (mag_r - mag_z) - 0.6)
                & ((mag_g - mag_w1 > 2.97) | (mag_r - mag_w1 > 1.8))
                & (((mag_r - mag_w1 > 1.83 * (mag_w1 - 17.13))
                    & (mag_r - mag_w1 > mag_w1 - 16.31))
                   | (mag_r - mag_w1 > 3.4)))
    return ((mag_z_fiber < 21.60)
            & ((mag_z - mag_w1) > 0.8 * (mag_r - mag_z) - 0.6)
            & ((mag_g - mag_w1 > 2.9) | (mag_r - mag_w1 > 1.8))
            & (((mag_r - mag_w1 > 1.8 * (mag_w1 - 17.14))
                & (mag_r - mag_w1 > mag_w1 - 16.33))
               | (mag_r - mag_w1 > 3.3)))


def lrg_selection(mag_g, mag_r, mag_z, mag_w1, mag_z_fiber, photsys):
    """Zhou et al. 2022 LRG selection, N/S split by PHOTSYS."""
    photsys = np.char.strip(np.asarray(photsys, dtype=str))
    cond = np.zeros(len(mag_g), dtype=bool)
    for tag, north in (("N", True), ("S", False)):
        m = photsys == tag
        if m.any():
            cond[m] = _lrg_cuts(mag_g[m], mag_r[m], mag_z[m], mag_w1[m],
                                mag_z_fiber[m], north)
    return cond


def stellar_mass_selection(logm, m_lo, m_hi):
    logm = np.asarray(logm, dtype=float)
    return np.isfinite(logm) & (logm >= m_lo) & (logm <= m_hi)


# =============================================================================
# FINITE-DIFFERENCE ALPHA ESTIMATOR
# =============================================================================
def alpha_from_selections(sel_left, sel_right, kappas, weights):
    """Magnification alpha from selections at +/-kappa (port of get_dN + fit).

    sel_left/right are lists (one per kappa, kappas[0]==0) of boolean selections
    after applying +kappa / -kappa. Per kappa step we form the weighted count
    change dN, the per-step amplitudes A = ddN / (2 N0 dk), and fit a line
    A(kappa); the intercept extrapolates to kappa -> 0 and is the alpha.

    Returns a dict with the alpha (fit intercept), its error, the per-step
    diagnostics and the simple cumulative estimates."""
    from scipy.optimize import curve_fit

    assert kappas[0] == 0.0, "kappas must start at 0"
    w = np.asarray(weights, dtype=float)
    n0 = w.sum()
    n = len(kappas)
    dns = np.zeros(n)
    dns_err = np.zeros(n)
    for i, (sl, sr) in enumerate(zip(sel_left, sel_right)):
        dns[i] = w[sl].sum() - w[sr].sum()
        # weighted Poisson-like variance of the changed objects
        v_l = np.sum(w[sl] ** 2) - np.sum(w ** 2)
        v_r = np.sum(w[sr] ** 2) - np.sum(w ** 2)
        dns_err[i] = np.sqrt(np.abs(v_l) + np.abs(v_r))
    dk = kappas[1] - kappas[0]
    a_step = np.diff(dns) / (2.0 * n0 * dk)
    a_step_err = np.sqrt(np.abs(np.diff(dns_err ** 2))) / (2.0 * n0 * dk)
    a_step_err[a_step_err == 0] = a_step_err[a_step_err > 0].min() \
        if (a_step_err > 0).any() else 1.0

    func = lambda x, a, c: a * x + c                          # noqa: E731
    x = kappas[1:] - dk / 2.0
    res, cov = curve_fit(func, x, a_step, sigma=a_step_err, absolute_sigma=True)
    chi2 = float(np.sum((a_step - func(x, *res)) ** 2 / a_step_err ** 2))
    return {
        "alpha": float(res[1]), "alpha_err": float(np.sqrt(cov[1, 1])),
        "slope": float(res[0]), "slope_err": float(np.sqrt(cov[0, 0])),
        "chi2": chi2, "dof": len(x) - 2, "n0": float(n0),
        "kappas": kappas, "dns": dns, "a_step": a_step, "a_step_err": a_step_err,
        "alpha_simple": dns[1:] / (2.0 * n0 * kappas[1:]),
        "alpha_simple_err": dns_err[1:] / (2.0 * n0 * kappas[1:]),
    }


# =============================================================================
# SOURCE CATALOGUE PREPARATION (responsivity, then PSF leakage)
# =============================================================================
def calibrate_sources(df, weight_col="WEIGHT"):
    """e1/e2 = e_uncal / <w (R_g11 + R_g22)/2>; drops non-finite rows.
    Same formula as build_ia_samples.calibrate. Returns (df, r_mean)."""
    e1, e2 = df["e1_uncal"].to_numpy(float), df["e2_uncal"].to_numpy(float)
    r11, r22 = df["R_g11"].to_numpy(float), df["R_g22"].to_numpy(float)
    w = df[weight_col].to_numpy(float)
    good = (np.isfinite(e1) & np.isfinite(e2) & np.isfinite(r11)
            & np.isfinite(r22) & np.isfinite(w) & (w > 0))
    r_mean = np.sum(w[good] * (r11 + r22)[good] / 2.0) / np.sum(w[good])
    out = df[good].reset_index(drop=True).copy()
    out["e1"] = out["e1_uncal"].to_numpy(float) / r_mean
    out["e2"] = out["e2_uncal"].to_numpy(float) / r_mean
    return out, float(r_mean)


def _wls(x, y, w):
    """Weighted least squares y = a*x + c. Returns (slope, slope_err)."""
    sw = np.sum(w)
    mx, my = np.sum(w * x) / sw, np.sum(w * y) / sw
    sxx = np.sum(w * (x - mx) ** 2)
    a = np.sum(w * (x - mx) * (y - my)) / sxx
    c = my - a * mx
    # nonrobust OLS-style error: weighted residual variance / sxx, dof n-2
    resid = y - (a * x + c)
    s2 = np.sum(w * resid ** 2) / max(len(x) - 2, 1)
    return a, np.sqrt(s2 / sxx)


def psf_leakage_alphas(df, n_bins=20, weight_col="WEIGHT"):
    """Two-pass per-object PSF leakage coefficients (Li et al. 2024).

    Equipopulated bins in size_ratio = Tpsf/(T + Tpsf), then in SNR within each;
    per-bin WLS fit e_i = alpha * e_PSF_i + c; smooth polynomial trend
    alpha(SNR^-2, SNR^-3, size_ratio) fit to the bin alphas; subtract, refit the
    residual per bin, total alpha = trend + residual. Needs columns e1/e2 (after
    responsivity calibration), e1_PSF/e2_PSF, snr, NGMIX_T_NOSHEAR,
    NGMIX_Tpsf_NOSHEAR and the weight. Returns (alpha_1, alpha_2) per object."""
    snr = df["snr"].to_numpy(float)
    w = df[weight_col].to_numpy(float)
    t, tpsf = df["NGMIX_T_NOSHEAR"].to_numpy(float), df["NGMIX_Tpsf_NOSHEAR"].to_numpy(float)
    size_ratio = tpsf / (t + tpsf)
    e = {1: df["e1"].to_numpy(float), 2: df["e2"].to_numpy(float)}
    epsf = {1: df["e1_PSF"].to_numpy(float), 2: df["e2_PSF"].to_numpy(float)}

    bin_r = pd.qcut(size_ratio, n_bins, labels=False, duplicates="drop")
    bin_snr = np.full(len(df), -1)
    for ib in range(n_bins):
        m = bin_r == ib
        if m.sum():
            bin_snr[m] = pd.qcut(snr[m], n_bins, labels=False, duplicates="drop")
    group = bin_r * n_bins + bin_snr
    groups = np.unique(group)

    # first pass: per-bin WLS alphas + the bins' mean (size_ratio, SNR)
    stats = {k: np.zeros((len(groups), 2)) for k in (1, 2)}   # (alpha, err)
    gr_r = np.zeros(len(groups))
    gr_snr = np.zeros(len(groups))
    for gi, g in enumerate(groups):
        m = group == g
        gr_r[gi] = np.average(size_ratio[m], weights=w[m])
        gr_snr[gi] = np.average(snr[m], weights=w[m])
        for k in (1, 2):
            stats[k][gi] = _wls(epsf[k][m], e[k][m], w[m])

    # smooth polynomial trend in (SNR^-2, SNR^-3, R, R*SNR^-2), 1/err^2 weights
    def _design(snr_v, r_v):
        return np.vstack([np.ones_like(snr_v), snr_v ** -2.0, snr_v ** -3.0,
                          r_v, r_v * snr_v ** -2.0]).T

    alpha_obj = {}
    a_design = _design(gr_snr, gr_r)
    for k in (1, 2):
        fw = 1.0 / np.square(stats[k][:, 1])
        poly = np.linalg.lstsq(a_design * fw[:, None], fw * stats[k][:, 0],
                               rcond=None)[0]
        alpha_obj[k] = _design(snr, size_ratio) @ poly

    # second pass: residual per-bin alpha on the trend-corrected ellipticities
    for k in (1, 2):
        e_cor = e[k] - alpha_obj[k] * epsf[k]
        resid = np.zeros(len(df))
        for g in groups:
            m = group == g
            resid[m] = _wls(epsf[k][m], e_cor[m], w[m])[0]
        alpha_obj[k] = alpha_obj[k] + resid

    return alpha_obj[1], alpha_obj[2]


def read_source_bin(cfg, bin_name, columns, head=0):
    """Read the for_GGL parquet restricted to one tomographic bin: veto-mask
    (drop rows with ANY ShapePipe mask bit set), an optional photo-z width gate
    (ggl.source_zb_width_max on Z_B_MAX - Z_B_MIN), an optional ODDS reliability
    gate (ggl.source_odds_min on the BPZ ODDS column), then the Z_B cut. No shape
    calibration -- see load_source_bin for the full pipeline."""
    import pyarrow.parquet as pq

    ggl = cfg["ggl"]
    path = Path(cfg["dest"]) / ggl["source_catalogue"]
    zcut = ggl["source_bins"][bin_name]
    veto = list(ggl.get("source_veto_columns", []))
    width_max = ggl.get("source_zb_width_max")
    odds_min = ggl.get("source_odds_min")
    # Z_B_MIN/MAX (width gate) and ODDS (reliability gate) are only needed to
    # evaluate their gates; pull them transparently when a gate is on and the
    # caller didn't already ask.
    read_cols = list(columns)
    if width_max is not None:
        for c in ("Z_B_MIN", "Z_B_MAX"):
            if c not in read_cols:
                read_cols.append(c)
    if odds_min is not None and "ODDS" not in read_cols:
        read_cols.append("ODDS")
    gate = (f", width(Z_B_MAX-Z_B_MIN) <= {width_max}"
            if width_max is not None else "")
    gate += f", ODDS >= {odds_min}" if odds_min is not None else ""
    print(f"  loading sources {path.name} (bin {bin_name}: "
          f"{zcut['z_min']} <= Z_B <= {zcut['z_max']}{gate})", flush=True)
    pf = pq.ParquetFile(path)
    ngroups = pf.num_row_groups
    if head:
        ngroups = min(head // max(pf.metadata.row_group(0).num_rows, 1) + 1,
                      ngroups)

    # Stream row-group-wise, filtering each chunk before accumulating: holding
    # the full 200M-row table (then forking dsigma workers on top of it) OOMs
    # the 128 GB comp nodes; the filtered bin is only a few GB.
    parts = []
    n_read = n_veto = n_width = n_odds = n_bin = 0
    for ig in range(ngroups):
        t = pf.read_row_group(ig, columns=read_cols + veto).to_pandas()
        n_read += len(t)
        if veto:
            masked = np.zeros(len(t), dtype=bool)
            for c in veto:
                masked |= t[c].to_numpy(bool)
            t = t[~masked].drop(columns=veto)
        n_veto += len(t)
        if width_max is not None:
            width = (t["Z_B_MAX"].to_numpy(float)
                     - t["Z_B_MIN"].to_numpy(float))
            t = t[width <= width_max]
            t = t.drop(columns=[c for c in ("Z_B_MIN", "Z_B_MAX")
                                if c not in columns])
        n_width += len(t)
        if odds_min is not None:
            t = t[t["ODDS"].to_numpy(float) >= odds_min]
            if "ODDS" not in columns:
                t = t.drop(columns=["ODDS"])
        n_odds += len(t)
        zb = t["Z_B"].to_numpy(float)
        t = t[(zb >= zcut["z_min"]) & (zb <= zcut["z_max"])]
        n_bin += len(t)
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    del parts
    if head:
        df = df.iloc[:head]
    print(f"    {n_read:,} rows read", flush=True)
    if veto:
        print(f"    {n_veto:,} after veto mask ({', '.join(veto)})", flush=True)
    if width_max is not None:
        print(f"    {n_width:,} after photo-z width gate "
              f"(<= {width_max})", flush=True)
    if odds_min is not None:
        print(f"    {n_odds:,} after ODDS gate (>= {odds_min})", flush=True)
    print(f"    {n_bin:,} in tomographic bin", flush=True)
    return df


def load_source_bin(cfg, bin_name, head=0):
    """Load + prepare one tomographic source bin from the for_GGL parquet:
    veto-mask, Z_B cut, responsivity calibration, then the two-pass PSF leakage
    correction (e1/e2 are overwritten with the corrected ellipticities).
    Returns (df, r_mean)."""
    df = read_source_bin(cfg, bin_name,
                         ["RA", "DEC", "WEIGHT", "e1_uncal", "e2_uncal",
                          "R_g11", "R_g22", "e1_PSF", "e2_PSF", "snr",
                          "NGMIX_T_NOSHEAR", "NGMIX_Tpsf_NOSHEAR", "Z_B"],
                         head=head)
    # 1. responsivity calibration, 2. PSF leakage correction (locked order)
    df, r_mean = calibrate_sources(df, weight_col="WEIGHT")
    print(f"    responsivity R = {r_mean:.4f}", flush=True)
    n_leak = int(cfg["ggl"]["psf_leakage"]["n_bins"])
    a1, a2 = psf_leakage_alphas(df, n_bins=n_leak, weight_col="WEIGHT")
    df["e1"] = df["e1"].to_numpy(float) - a1 * df["e1_PSF"].to_numpy(float)
    df["e2"] = df["e2"].to_numpy(float) - a2 * df["e2_PSF"].to_numpy(float)
    print(f"    PSF leakage corrected: <alpha_1>={a1.mean():.4f}, "
          f"<alpha_2>={a2.mean():.4f}", flush=True)
    return df, r_mean
