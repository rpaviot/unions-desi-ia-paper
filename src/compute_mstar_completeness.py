#!/usr/bin/env python3
"""Stellar-mass completeness limits for the GGL/CSMF lens samples (Wright+17 turn-over).

Builds the limiting stellar mass ``M*_lim(z)`` of each flux-limited lens sample with
NO magnitude limit, via the Wright et al. 2017 (GAMA SMF; arXiv 1705.04074, MNRAS 470,
283; their ``MassFuncFitR``) automated *turn-over* method, and from it a per-mass-bin
upper-redshift cut ``z_lim`` so each CSMF mass bin can be made volume-complete (cf.
Dvornik et al. 2023, arXiv 2210.03110, Fig. 1). The CSMF ties the abundance Φ(M*) to
halo mass, so feeding it M*-INCOMPLETE bins biases the SHMR while ΔΣ (the mean halo
mass per bin) stays fine -- the "good ΔΣ / bad SHMR" symptom. Cutting each bin at its
z_lim removes the incomplete high-z wedge.

Method
------
Wright+17 (Sec. 3.2; MassFuncFitR) estimate "the turn-over point of the number density
distribution in bins of comoving distance and stellar mass independently": in each
comoving-distance bin, the STELLAR MASS at peak number density (the M* mode); in each
stellar-mass bin, "the largest comoving distance at median stellar mass density" (a
median-density threshold in chi). A polynomial is then fit to the combined turn-over
points to give the smooth M*_lim(z); Dvornik+23 (Sec. 3.1, Fig. 1) run exactly that
package on KiDS-BRIGHT. We follow the same turn-over PRINCIPLE but deviate deliberately
on all three ingredients (each deviation is motivated below):

  (b) COUNT turn-over  [PRIMARY -- adopted; replaces Wright's median-density threshold].
      In a thin stellar-mass slice the number of galaxies per comoving-distance shell
      ``dN/dchi`` rises with the (volume ~ chi^2) term while the slice is complete,
      peaks, then falls once the flux limit starts removing that slice's galaxies. The
      peak of ``dN/dchi`` is the completeness distance chi_lim(M*) -> z_lim(M*), which
      inverts to M*_lim(z). This is robust across the whole mass range, including the
      bright end where the SHMR bias lives. We use the count PEAK rather than Wright's
      in-mass-bin density-threshold criterion because for these red samples any
      threshold referenced to a low-/mid-chi density level inherits the noisy small-
      volume end and fires before the true flux-limit cliff -- the same failure mode as
      the density-threshold prototype below.

  (a) MODE turn-over  [high-z CROSS-CHECK only -- not enveloped into the limit].
      Wright's comoving-distance-bin estimator (here in thin z slices): the limiting
      mass is the stellar mass at peak number density (the mode of the M* histogram).
      For a *red* sample this only tracks incompleteness at high z; at low z the mode
      locks onto the intrinsic red-sequence SMF peak (~10.5), NOT incompleteness --
      fine for Wright's flux-limited full-population GAMA sample, wrong for a red
      subsample -- so (a) over-estimates M*_lim at low z and is used only as a
      consistency check.

  Smoothing: instead of Wright's high-degree polynomial through the combined (a)+(b)
      points we enforce monotone non-decreasing z_lim(M*) on the (b) curve alone
      (np.maximum.accumulate) -- a physical constraint (a brighter slice cannot go
      incomplete closer than a fainter one) with no polynomial edge oscillations.

Why not a density threshold (the earlier prototype's bug)
---------------------------------------------------------
An earlier prototype used the number *density* n(z)=N/Vshell and set z_lim where the
smoothed n dropped to 0.5x a "plateau" taken as ``max(n) over z<0.30``. For these bins
n(z) DECLINES monotonically (no flat plateau then cliff), and the max sat on the noisy,
tiny-volume lowest-z shell, so the half-max crossing preceded the true flux-limit cliff
-- worst at high M* (it gave z_lim ~ 0.30 for the most massive bin whose galaxies clearly
reach z ~ 0.43). The count turn-over fixes this: its peak coincides with the cliff
because the rising chi^2 volume term competes with the falling completeness, so the
turn-over is anchored to where completeness actually drops, not to a fragile reference
level.

Outputs (per sample)
--------------------
  * ``<dest>/ggl/completeness/<name>_mstar_completeness.npz`` -- the M*_lim(z) curve
    (fine slices), the per-mass-bin z_lim (at each bin's faint edge), the bin edges,
    and the estimator-(a) mode points + Pozzetti cross-check.
  * ``<plot_dir>/mstar_completeness_<name>.png`` -- Dvornik-Fig.1 analogue: M*-z hexbin,
    the adopted M*_lim(z) curve, estimator-(a) modes, the Pozzetti cross-check, and the
    volume-limited (z<z_lim) bin boxes.

Usage
-----
    python scripts/compute_mstar_completeness.py [--config CONFIG] [--samples NAME ...]
                                                 [--plot-dir DIR] [--no-plot]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402
import compute_deltasigma as cds  # noqa: E402  (the single real-h ΔΣ cosmology)

COSMO = cds.COSMO

# --- clean-sample window (drop FastSpecFit failures and the noisy extremes) ---
# Z_MIN is global; the UPPER edge is sample-aware (see z_window) so we never clip
# inside a lens sample's own redshift range -- otherwise the count turn-over would
# latch onto our own clip rather than the real completeness limit (bit LRG: lenses
# reach 0.75, a fixed 0.60 cap put the spurious turn-over right at the clip).
Z_MIN = 0.01
LOGM_MIN, LOGM_MAX = 8.5, 12.5


def z_window(lens):
    """Upper edge of the clean/turn-over redshift window for a sample: comfortably
    above the lens zmax so (i) the sample is never truncated inside its own range and
    (ii) the count turn-over is anchored to the data, not to our clip. The margin is
    wide (0.35) because the catalogues run well past the lens cut (LRG reaches z~1.1);
    BGS has no galaxies up there so its curve is unaffected by the larger window."""
    return max(0.62, ggl_zmax(lens) + 0.35)

# --- estimator (b): fine M* slices for the M*_lim(z) curve ---
SLICE_LO, SLICE_HI, SLICE_DM = 9.6, 11.6, 0.10   # thin stellar-mass slices
SLICE_MIN_N = 500                                # need enough galaxies to find a peak
COUNT_SMOOTH = 2.0                               # Gaussian smoothing of dN/dchi (bins)
N_CHI_BINS = 60                                  # comoving-distance shells over [zmin,zmax]

# --- estimator (a): redshift slices for the mode cross-check ---
A_DZ = 0.02
A_MIN_N = 200

# --- Pozzetti shift-to-limit cross-check (needs a nominal m_lim; cross-check only) ---
POZZETTI_MLIM = 20.0


def load_clean_catalogue(path: Path, z_hi: float) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=["Z", "LOGMSTAR", "MAG_R"])
    ok = (d["LOGMSTAR"].notna() & np.isfinite(d["MAG_R"])
          & (d["Z"] > Z_MIN) & (d["Z"] < z_hi)
          & (d["LOGMSTAR"] > LOGM_MIN) & (d["LOGMSTAR"] < LOGM_MAX))
    return d[ok].copy()


def bin_edges(dest: Path, in_subdir: str, stem: str):
    """Authoritative (logmstar_min, logmstar_max, z_eff) per mass bin from the ΔΣ npz."""
    fs = sorted(glob.glob(str(dest / in_subdir / f"{stem}_massbin*_deltasigma.npz")),
                key=lambda f: int(f.split("massbin")[1].split("_")[0]))
    edges = []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        edges.append((float(z["logmstar_min"]), float(z["logmstar_max"]), float(z["z_eff"])))
    return edges


def count_turnover_z(zvals, chi_edges, zc_of_chi):
    """Wright+17 count turn-over: peak of dN/dchi (galaxies per comoving-distance shell)
    for a fixed M* slice. Returns (z_turnover, N_per_shell, N_smoothed)."""
    chi = COSMO.comoving_distance(zvals.astype(float)).value
    N, _ = np.histogram(chi, bins=chi_edges)
    Ns = gaussian_filter1d(N.astype(float), COUNT_SMOOTH)
    return zc_of_chi[Ns.argmax()], N, Ns


def mstar_lim_curve(d: pd.DataFrame, chi_edges, zc_of_chi):
    """Estimator (b): M*_lim(z) from the count turn-over over thin M* slices.
    Returns (M*_centre, z_lim_raw, z_lim_monotone)."""
    lo = np.arange(SLICE_LO, SLICE_HI, SLICE_DM)
    cen, zlim = [], []
    for m0 in lo:
        s = d[(d["LOGMSTAR"] >= m0) & (d["LOGMSTAR"] < m0 + SLICE_DM)]
        if len(s) < SLICE_MIN_N:
            continue
        zt, _, _ = count_turnover_z(s["Z"].values, chi_edges, zc_of_chi)
        cen.append(m0 + 0.5 * SLICE_DM)
        zlim.append(zt)
    cen, zlim = np.array(cen), np.array(zlim)
    # completeness limit can only rise with z: enforce monotone non-decreasing z_lim(M*)
    return cen, zlim, np.maximum.accumulate(zlim)


def mode_turnover(d: pd.DataFrame, z_hi: float):
    """Estimator (a) cross-check: mode of the logM* histogram in each z slice."""
    mbins = np.arange(LOGM_MIN, LOGM_MAX, 0.02)
    mcen = 0.5 * (mbins[:-1] + mbins[1:])
    zc = np.arange(Z_MIN + A_DZ, z_hi, A_DZ)
    zt, mode = [], []
    for z0 in zc:
        s = d[(d["Z"] >= z0 - A_DZ / 2) & (d["Z"] < z0 + A_DZ / 2)]
        if len(s) < A_MIN_N:
            continue
        h, _ = np.histogram(s["LOGMSTAR"], bins=mbins)
        h = gaussian_filter1d(h.astype(float), 1.5)
        zt.append(z0)
        mode.append(mcen[h.argmax()])
    return np.array(zt), np.array(mode)


def pozzetti_curve(d: pd.DataFrame, z_hi: float, m_lim=POZZETTI_MLIM):
    """Pozzetti+10 shift-to-limit cross-check: M*_lim = logM* + 0.4(m_r - m_lim),
    95th percentile over the faintest 20% in each z slice. NEEDS m_lim (cross-check)."""
    Ml = d["LOGMSTAR"] + 0.4 * (d["MAG_R"] - m_lim)
    zz, ll = [], []
    for z0 in np.arange(0.06, z_hi - 0.06, A_DZ):
        sel = (d["Z"] >= z0 - A_DZ / 2) & (d["Z"] < z0 + A_DZ / 2)
        if sel.sum() < A_MIN_N:
            continue
        faint = d[sel].nlargest(max(20, int(0.2 * sel.sum())), "MAG_R")
        zz.append(z0)
        ll.append(np.percentile(Ml[faint.index], 95))
    return np.array(zz), np.array(ll)


def process_sample(sample, cfg, plot_dir: Path, make_plot: bool, out_dir=None):
    dest = Path(cfg["dest"])
    ggl = cfg["ggl"]
    csmf = cfg["csmf"]
    name = sample["name"]
    stem = sample["stem"]

    # catalogue: match the lens-sample entry, else fall back to the naming convention
    lens = next((s for s in ggl["lens_samples"] if s["name"] == name), None)
    cat_name = lens["catalogue"] if lens else f"{name}_clustering.dat.parquet"
    cat_path = dest / ggl["cat_subdir"] / cat_name

    edges = bin_edges(dest, csmf["in_subdir"], stem)
    if not edges:
        print(f"[{name}] no ΔΣ npz under {dest/csmf['in_subdir']}/{stem}_massbin* -- skipping")
        return
    nb = len(edges)

    # IA LENS sample = the UNIONS-matched RED shape sample for this (sample, z-bin),
    # equipopulated LOGMSTAR deciles -- THIS is what enters the GGL/CSMF lensing, not
    # the parent clustering catalogue. The cloud, the per-bin drop counts and the bin
    # boxes all use it, so they are self-consistent with the lenses (and start at the
    # lens z-floor). The M*_lim(z) CURVE is measured on the parent RED clustering
    # catalogue instead: same red selection, full-z leverage (needed where the lens
    # z-range is narrow, e.g. LRG), and verified to agree with the shape-sample curve
    # to <=0.02 in z_lim for BGS_RED_GMM.
    lens_path = dest / cfg["ia_samples"]["out_subdir"] / f"{stem}.parquet"

    z_hi = z_window(lens)
    zmin_lens, zmax_lens = ggl_zmin(lens), ggl_zmax(lens)
    print(f"\n=== {name} ===  curve_cat={cat_path.name}  lenses={lens_path.name}  "
          f"n_mass_bins={nb}  lens z=[{zmin_lens:.2f},{zmax_lens:.2f}]  turn-over window z<{z_hi:.2f}")
    d = load_clean_catalogue(cat_path, z_hi)            # parent red clustering (curve)
    L = pd.read_parquet(lens_path, columns=["Z", "LOGMSTAR"])  # IA red shape lenses (cloud/counts)
    L = L[L["LOGMSTAR"].notna() & (L["Z"] >= zmin_lens) & (L["Z"] < zmax_lens)].copy()
    print(f"  curve sample (red clustering): {len(d):,}   lens shapes: {len(L):,}")

    # comoving-distance shells over the (sample-aware) turn-over window
    chi_lo = COSMO.comoving_distance(Z_MIN).value
    chi_hi = COSMO.comoving_distance(z_hi).value
    chi_edges = np.linspace(chi_lo, chi_hi, N_CHI_BINS + 1)
    chi_cen = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    # invert each shell centre back to redshift (monotone interp on a fine grid)
    zgrid = np.linspace(Z_MIN, z_hi, 4000)
    chigrid = COSMO.comoving_distance(zgrid).value
    zc_of_chi = np.interp(chi_cen, chigrid, zgrid)

    # --- estimator (b): adopted M*_lim(z) curve ---
    m_cen, zlim_raw, zlim_mono = mstar_lim_curve(d, chi_edges, zc_of_chi)
    # --- estimator (a) + Pozzetti: cross-checks ---
    zt_a, mode_a = mode_turnover(d, z_hi)
    zp, lp = pozzetti_curve(d, z_hi)

    # --- per-mass-bin z_lim + how many LENS galaxies the volume-limit cut drops ---
    # counts taken on the IA shape LENS sample within the bin's M* window and the lens
    # z-range [zmin,zmax]; the volume limit keeps zmin <= z < z_lim.
    print(f"  {'mb':>3} {'M*_lo':>6} {'M*_hi':>6} {'z_eff':>6} {'z_lim':>6}  "
          f"{'complete?':>10}  {'N_tot':>9} {'N_keep':>9} {'N_drop':>9}  drop%")
    rows = []
    for i, (lo, hi, zeff) in enumerate(edges):
        zlim = float(np.interp(lo, m_cen, zlim_mono))
        b = L[(L["LOGMSTAR"] >= lo) & (L["LOGMSTAR"] < (hi if i < nb - 1 else hi + 1e-6))
              & (L["Z"] >= zmin_lens) & (L["Z"] < zmax_lens)]
        n_tot = int(len(b))
        n_keep = int(((b["Z"] >= zmin_lens) & (b["Z"] < zlim)).sum())
        n_drop = n_tot - n_keep
        keep = n_keep / n_tot if n_tot else np.nan
        flag = "yes" if zlim >= zeff else "NO (cut)"
        rows.append((i, lo, hi, zeff, zlim, keep, n_tot, n_keep, n_drop))
        print(f"  {i:>3} {lo:6.2f} {hi:6.2f} {zeff:6.3f} {zlim:6.3f}  {flag:>10}  "
              f"{n_tot:>9,} {n_keep:>9,} {n_drop:>9,}  {100*(1-keep):4.1f}%")

    # --- save ---
    out_dir = Path(out_dir) if out_dir else dest / "ggl" / "completeness"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{name}_mstar_completeness.npz"
    np.savez(
        out_npz,
        sample=name, stem=stem,
        mstar_lim_mcen=m_cen, mstar_lim_zlim_raw=zlim_raw, mstar_lim_zlim=zlim_mono,
        bin_index=np.array([r[0] for r in rows]),
        bin_logmstar_min=np.array([r[1] for r in rows]),
        bin_logmstar_max=np.array([r[2] for r in rows]),
        bin_z_eff=np.array([r[3] for r in rows]),
        bin_z_lim=np.array([r[4] for r in rows]),
        bin_keep_frac=np.array([r[5] for r in rows]),
        bin_n_total=np.array([r[6] for r in rows]),
        bin_n_keep=np.array([r[7] for r in rows]),
        bin_n_drop=np.array([r[8] for r in rows]),
        lens_zmin=zmin_lens, lens_zmax=zmax_lens, z_window=z_hi,
        mode_z=zt_a, mode_logmstar=mode_a,
        pozzetti_z=zp, pozzetti_logmstar=lp, pozzetti_mlim=POZZETTI_MLIM,
    )
    print(f"  saved -> {out_npz}")

    if make_plot:
        _plot(name, L, m_cen, zlim_mono, zt_a, mode_a, zp, lp, rows,
              zmin_lens, zmax_lens, plot_dir)


def ggl_zmin(lens):
    return float(lens["zmin"]) if lens else Z_MIN


def ggl_zmax(lens):
    return float(lens["zmax"]) if lens else Z_MAX


def _plot(name, L, m_cen, zlim_mono, zt_a, mode_a, zp, lp, rows,
          zmin_lens, zmax_lens, plot_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    # cloud = the IA shape LENS sample (red, within the lens z-range)
    ax.hexbin(L["Z"], L["LOGMSTAR"], gridsize=120, bins="log", cmap="Greys", mincnt=1)
    ax.plot(zlim_mono, m_cen, "b-o", lw=2, ms=4,
            label="count turn-over $M_*^{\\rm lim}(z)$ (Wright+17-style; adopted, no $m_{\\rm lim}$)")
    ax.scatter(zt_a, mode_a, c="orange", s=10, zorder=5,
               label="estimator (a) mode (low-$z$ SMF-peak biased, cross-check)")
    ax.plot(zp, lp, "r:", lw=1.5,
            label=f"Pozzetti shift-to-limit ($m_r<{POZZETTI_MLIM:.0f}$, cross-check)")
    # volume-limited bins span the lens range [zmin_lens, z_lim]
    for i, lo, hi, zeff, zlim, keep, *_ in rows:
        ax.add_patch(plt.Rectangle((zmin_lens, lo), zlim - zmin_lens, hi - lo,
                                   fill=False, ec="red", lw=0.9))
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$\log_{10} M_*$ (FastSpecFit)")
    # sample-aware x-axis: tight around the lens z-range (the cloud the boxes sit in)
    ax.set_xlim(max(0.0, zmin_lens - 0.04), zmax_lens + 0.04)
    ax.set_ylim(9.3, 12.0)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title(f"{name}: stellar-mass completeness + volume-limited mass bins")
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"mstar_completeness_{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  plot  -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(gu.CONFIG))
    ap.add_argument("--samples", nargs="*", default=None,
                    help="subset of csmf.samples names (default: all)")
    ap.add_argument("--plot-dir", default=str(gu.REPO / "plots"))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: <dest>/ggl/completeness)")
    args = ap.parse_args()

    cfg = gu.load_config(args.config)
    samples = cfg["csmf"]["samples"]
    if args.samples:
        samples = [s for s in samples if s["name"] in args.samples]
        if not samples:
            ap.error(f"no csmf.samples match {args.samples}")

    plot_dir = Path(args.plot_dir)
    for s in samples:
        process_sample(s, cfg, plot_dir, make_plot=not args.no_plot, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
