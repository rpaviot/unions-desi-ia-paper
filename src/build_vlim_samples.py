#!/usr/bin/env python3
"""Build stellar-mass-COMPLETE (volume-limited) lens samples for the SHMR/CSMF fit.

Re-bins the existing full RED shape sample (``<source>_..._shapes.parquet``, already
UNIONS-matched + responsivity-calibrated by ``build_ia_samples.py``) into ``n_mass_bins``
equal-N volume-limited "staircase" boxes, using the Wright+17-style count turn-over
completeness curve ``M*_lim(z)`` from ``compute_mstar_completeness.py`` (see that
script's docstring for the exact method + deviations from Wright+17). Each box is

    [M*_lo, M*_hi) x [zmin, z_max] ,   z_max = min(z_lim(M*_lo), zmax)

i.e. it extends in redshift only as far as its FAINT edge stays complete, so every
galaxy in the box is above the completeness limit -- volume-complete (cf. Dvornik+23
Fig. 1). Bin edges are chosen by a greedy sweep; two schemes (``bin_scheme``):

  equal_n   (default) boxes hold equal numbers of galaxies.
  equal_snr boxes hold equal FORECAST DeltaSigma S/N: signal ~ Mh(mean M*)^(2/3)
            with Mh(M*) inverted from a fitted SHMR (``snr_fit`` MINUIT npz), noise
            shape-noise ~ 1/sqrt(sum_lenses W(z)) with the per-lens weight
            W(z_l) = int n(z_s) Sigma_crit^-2(z_l,z_s) dz_s from the source n(z)
            (``snr_nz``). Bright bins get narrower (mass resolution moves to the
            bright end, where the SHMR question lives); a per-bin galaxy floor
            ``snr_n_min`` keeps the jackknife stable, and the top box is clipped
            at the ``snr_mass_pct_hi`` percentile (relaxed upper clip, as for the
            equal_snr IA samples). The SHMR/n(z) enter ONLY the edge placement,
            never the likelihood.

No UNIONS re-match is needed: the boxes are subsets of the full shape sample,
recalibrated per bin.

These samples calibrate the SHMR only (the CSMF is fit on their ΔΣ); the flux-limited IA
deciles are untouched, and halo masses are later painted onto them from the best-fit
model (the "model route"). Per bin we write, with the SAME naming + machinery as
``build_ia_samples`` so ``compute_deltasigma`` consumes them unchanged:

    <tag>_shapes_massbin{i}.parquet         calibrated shapes (per-bin responsivity)
    <tag>_shape_randoms_massbin{i}.parquet  matching shape randoms (DESI reconstruction)

where ``<tag> = <name>_zmin_<zmin>_zmax_<zmax>``.

Usage
-----
    python scripts/build_vlim_samples.py [--config CONFIG] [--samples NAME ...] [--no-plot]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402
import sysweights as sw  # noqa: E402
from build_ia_samples import (  # noqa: E402
    calibrate, write_shapes_parquet, build_shape_randoms, ztag,
)


def snr_lens_weight(nz_path):
    """Per-lens forecast weight W(z_l) = int n(z_s) Sigma_crit^-2(z_l, z_s) dz_s.

    Shape-noise forecast for the equal_snr binning: sigma(DeltaSigma) per pair
    ~ sigma_e * Sigma_crit, so the inverse-variance lens weight is the n(z)-average
    of Sigma_crit^-2 ~ (D_l D_ls / D_s)^2 (constants drop -- only used for ranking).
    Uses the same real-h COSMO as compute_deltasigma. Returns W(z) interpolator."""
    import compute_deltasigma as cds
    d = np.load(nz_path)
    zs = np.asarray(d["z"], float)
    nz = np.clip(np.asarray(d["nz"], float), 0.0, None)
    nz /= np.trapezoid(nz, zs)
    zl = np.linspace(0.01, 0.60, 120)
    Ds = cds.COSMO.angular_diameter_distance(zs).value
    W = np.empty_like(zl)
    for i, z0 in enumerate(zl):
        Dl = cds.COSMO.angular_diameter_distance(z0).value
        ok = zs > z0 + 1e-6
        w = np.zeros_like(zs)
        if ok.any():
            Dls = cds.COSMO.angular_diameter_distance_z1z2(z0, zs[ok]).value
            w[ok] = (Dl * Dls / Ds[ok]) ** 2
        W[i] = np.trapezoid(nz * w, zs)
    return lambda z: np.interp(z, zl, W)


def shmr_mh_of_logm(fit_path):
    """log10 Mh(log10 M*) by inverting the fitted central SHMR (Dvornik/NRV form
    Mstar_c = M0 (Mh/M1)^g1 / (1+Mh/M1)^(g1-g2); monotone, so a grid interp)."""
    z = np.load(fit_path, allow_pickle=True)
    p = dict(zip([str(n) for n in z["minuit_param_names"]],
                 np.asarray(z["minuit_best_fit"], float)))
    lg_mh = np.linspace(10.0, 16.0, 2000)
    x = 10.0 ** (lg_mh - p["M1"])
    lg_ms = p["M0"] + p["gamma1"] * np.log10(x) \
        - (p["gamma1"] - p["gamma2"]) * np.log10(1.0 + x)
    return lambda lm: np.interp(lm, lg_ms, lg_mh)


def vlim_boxes_snr(M, Z, z_lim_of, zmin, zmax, floor, n_bins, W_of, mh_of,
                   n_min, pct_hi, beta_cap=1.0, logm_break=None, mass_max=None,
                   beta_min=0.0, count_ratio=None):
    """K=n_bins S/N-weighted volume-limited staircase boxes (IA equal_snr structure).

    Same greedy sweep as ``vlim_boxes`` (z_max from the FAINT edge), but with the
    IA scheme's two-regime COUNT target: equal-N boxes (target C) below
    ``logm_break``, thinned above it as N_t = C * 10^(-2 int beta dm) with the
    lensing signal slope beta(m) = min(beta_cap, (2/3) dlogMh/dlogM*) from the
    fitted SHMR (DeltaSigma ~ Mh^(2/3); per-bin error ~ 1/sqrt(N)); C is bisected
    so the sweep yields exactly n_bins boxes, each floored at n_min galaxies.
    The cap matters: above the knee dlogMh/dlogM* ~ 1/gamma2 ~ 5-6, so the raw
    slope would floor every bright bin at n_min and dump the rest into one
    mega-bin at the knee -- losing exactly the mass resolution the SHMR fit needs
    -- while resting on an Mh(M*) extrapolation far beyond the constrained range.
    Equal-count below the break keeps the knee resolved even though the faint
    windows are volume-starved (a pure equal-S/N target degenerates there). The
    top box is [last edge, pct_hi percentile] -- galaxies above the clip are
    dropped, so the box window and its counts stay consistent for the abundance
    anchor. The source-n(z) weight W enters the REPORTED forecast S/N only."""
    keep = (M >= floor) & (Z >= zmin) & (Z < zmax)
    M, Z = M[keep], Z[keep]
    m_hi = float(np.percentile(M, pct_hi))
    if mass_max is not None:                         # hard bright-end cap
        m_hi = min(m_hi, float(mass_max))

    # thinning ratio R(m): the slope-capped lensing S/N slope, N_t ~ 1/S^2 with
    # dlog10 S/dm = min(beta_cap, 2/3 dlogMh/dm), active above logm_break (default:
    # from the floor, i.e. bin sizes decrease from the start -- below the SHMR knee
    # the slope is tiny anyway, so faint bins come out near-equal without a plateau)
    if logm_break is None:
        logm_break = floor
    def window(lo, zb):
        sel = (Z >= zmin) & (Z < zb) & (M >= lo) & (M <= m_hi)
        return np.sort(M[sel])

    if count_ratio is not None:
        # Dvornik-style SMOOTH thinning: per-bin target counts decrease
        # geometrically with N_first/N_last = count_ratio (on the scales driving
        # the fit the signal ~ linear bias x growth -- much flatter than the
        # 1-halo Mh^(2/3) proxy -- so the spread is pinned empirically). The
        # overall scale T is bisected so the LAST bin also lands on its target
        # (its faint edge slides accordingly) -- no capacity-limited remainder box.
        frac = float(count_ratio) ** (-np.arange(n_bins) / (n_bins - 1.0))

        def build_ratio(T):
            boxes, lo = [], float(floor)
            for i in range(n_bins - 1):
                zb = min(float(z_lim_of(lo)), zmax)
                ms = window(lo, zb)
                nt = int(np.clip(T * frac[i], n_min,
                                 max(len(ms) - n_min, n_min)))
                hi = float(ms[min(nt, len(ms) - 1)])
                boxes.append((lo, hi, zb))
                lo = hi
            boxes.append((lo, m_hi + 1e-6, min(float(z_lim_of(lo)), zmax)))
            return boxes

        def last_excess(T):
            lo, _, zb = build_ratio(T)[-1]
            n_rem = np.count_nonzero((M >= lo) & (M <= m_hi)
                                     & (Z >= zmin) & (Z < zb))
            return n_rem - T * frac[-1]

        t_lo, t_hi = float(n_min), float(len(M))
        for _ in range(60):
            mid = 0.5 * (t_lo + t_hi)
            if last_excess(mid) > 0:
                t_lo = mid
            else:
                t_hi = mid
        return build_ratio(0.5 * (t_lo + t_hi))

    # SHMR signal proxy. beta_min > 0 forces the bin sizes to DECREASE from
    # the first bin even where the SHMR is flat (below the knee its slope is
    # ~0, which with few bins over a wide range would push bin0 to its
    # window-capacity ceiling instead)
    mg = np.linspace(floor, max(m_hi + 0.1, 12.0), 500)
    beta = np.where(mg >= logm_break,
                    np.clip((2.0 / 3.0) * np.gradient(mh_of(mg), mg),
                            beta_min, beta_cap),
                    0.0)
    logR = -2.0 * np.concatenate(
        [[0.0], np.cumsum(0.5 * (beta[1:] + beta[:-1]) * np.diff(mg))])
    R_of = lambda m: 10.0 ** np.interp(m, mg, logR)  # noqa: E731

    def n_next(hi_try):
        """Galaxies available to a box whose FAINT edge is hi_try (its own window)."""
        zbn = min(float(z_lim_of(hi_try)), zmax)
        return int(np.count_nonzero((M >= hi_try) & (M <= m_hi)
                                    & (Z >= zmin) & (Z < zbn)))

    def build(C):
        boxes, lo = [], float(floor)
        while True:
            zb = min(float(z_lim_of(lo)), zmax)
            ms = window(lo, zb)
            if len(ms) < 2 * n_min:                  # window exhausted -> top box
                boxes.append((lo, m_hi + 1e-6, zb))
                break
            nt = max(int(C * R_of(lo)), n_min)
            for _ in range(2):                       # fixed point on the bin median
                nt = max(int(C * R_of(ms[min(nt // 2, len(ms) - 1)])), n_min)
            j = min(nt, len(ms) - n_min)             # never swallow the window
            hi = float(ms[j])
            if n_next(hi) < n_min:                   # no viable next box -> top box
                boxes.append((lo, m_hi + 1e-6, zb))
                break
            boxes.append((lo, hi, zb))
            lo = hi
        return boxes

    # boxes(C) is non-increasing in C: bisect to the SMALLEST C that yields
    # <= n_bins boxes -- bins stay as small/balanced as the thinning allows.
    # (The IA "largest C" convention would push the faint end into its window
    # clamp here, recreating the mega-bin across the SHMR knee.)
    lo_c, hi_c = n_min, len(M)
    while lo_c < hi_c:
        mid = (lo_c + hi_c) // 2
        if len(build(mid)) <= n_bins:
            hi_c = mid
        else:
            lo_c = mid + 1
    boxes = build(lo_c)
    if len(boxes) != n_bins:
        print(f"      WARNING equal_snr: {len(boxes)} boxes built, {n_bins} "
              f"requested (snr_n_min too tight for this sample?)")
    return boxes


def vlim_boxes(M, Z, z_lim_of, zmin, zmax, floor, n_bins):
    """K=n_bins equal-N volume-limited staircase boxes. Returns [(lo, hi, z_max), ...].

    Greedy from the faint floor: fix M*_lo, set z_max = min(z_lim(M*_lo), zmax), then take
    the Ntarget-th smallest M* among galaxies inside [M*_lo, .) x [zmin, z_max] as M*_hi.
    Ntarget is binary-searched so the sweep yields exactly n_bins boxes."""
    keep = (M >= floor) & (Z >= zmin) & (Z < zmax)
    M, Z = M[keep], Z[keep]

    def build(ntarget):
        boxes, lo = [], float(floor)
        while True:
            zb = min(float(z_lim_of(lo)), zmax)
            in_slice = M[(Z >= zmin) & (Z < zb) & (M >= lo)]
            if len(in_slice) <= ntarget:                 # remainder -> top box
                boxes.append((lo, float(M[M >= lo].max()) + 1e-6, zb))
                break
            hi = float(np.sort(in_slice)[ntarget])
            boxes.append((lo, hi, zb))
            lo = hi
        return boxes

    lo, hi = 100, len(M)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(build(mid)) > n_bins:
            lo = mid + 1
        else:
            hi = mid
    return build(lo)


def process(sample, cfg, vc, rng, make_plot, plot_dir):
    dest = Path(cfg["dest"])
    out_dir = dest / vc["out_subdir"]
    name = sample["name"]
    zmin, zmax = float(sample["source_zmin"]), float(sample["source_zmax"])
    floor, n_bins = float(sample["floor"]), int(sample["n_mass_bins"])

    scheme = sample.get("bin_scheme", "equal_n")

    src_tag = ztag(sample["source"], zmin, zmax)
    tr_tag = ztag(sample["tracer"], zmin, zmax)
    out_tag = ztag(name, zmin, zmax)

    # completeness curve z_lim(M*)
    c = np.load(dest / vc["completeness_subdir"] / sample["completeness"], allow_pickle=True)
    mcen, zlim = c["mstar_lim_mcen"], c["mstar_lim_zlim"]
    z_lim_of = lambda m: np.interp(m, mcen, zlim, left=zlim[0], right=zlim[-1])  # noqa: E731

    # full RED shape sample (already matched/calibrated) + footprint randoms
    shapes = pd.read_parquet(out_dir / f"{src_tag}_shapes.parquet")
    tr = pd.read_parquet(out_dir / f"{tr_tag}_randoms.parquet")
    print(f"\n=== {name} ===  source={src_tag}_shapes ({len(shapes):,})  "
          f"randoms={tr_tag}_randoms ({len(tr):,})  floor={floor}  n_bins={n_bins}  "
          f"scheme={scheme}")

    M = shapes["LOGMSTAR"].to_numpy()
    Z = shapes["Z"].to_numpy()
    if scheme == "equal_snr":
        ggl_dir = dest / cfg["ggl"]["out_subdir"]
        W_of = snr_lens_weight(ggl_dir / "nz" / sample["snr_nz"])
        mh_of = shmr_mh_of_logm(ggl_dir / "csmf_fit" / sample["snr_fit"])
        boxes = vlim_boxes_snr(M, Z, z_lim_of, zmin, zmax, floor, n_bins,
                               W_of, mh_of, int(sample.get("snr_n_min", 5000)),
                               float(sample.get("snr_mass_pct_hi", 99.9)),
                               beta_cap=float(sample.get("snr_beta_cap", 1.0)),
                               logm_break=(float(sample["snr_logm_break"])
                                           if "snr_logm_break" in sample else None),
                               mass_max=(float(sample["snr_mass_max"])
                                         if "snr_mass_max" in sample else None),
                               beta_min=float(sample.get("snr_beta_min", 0.0)),
                               count_ratio=(float(sample["snr_count_ratio"])
                                            if "snr_count_ratio" in sample else None))
    else:
        boxes = vlim_boxes(M, Z, z_lim_of, zmin, zmax, floor, n_bins)

    srfac, p0 = int(vc["shape_randoms_factor"]), float(vc["p0"])
    print(f"  {'mb':>3} {'M*_lo':>6} {'M*_hi':>6} {'z_max':>6} {'N':>8} {'R':>7}  n_rand")
    rows = []
    for i, (lo, hi, zb) in enumerate(boxes):
        sel = (shapes["LOGMSTAR"] >= lo) & (shapes["LOGMSTAR"] < hi) & (shapes["Z"] < zb)
        bdf = shapes[sel].reset_index(drop=True)
        bcal, br = calibrate(bdf, weight_col="WEIGHT")
        write_shapes_parquet(bcal, out_dir / f"{out_tag}_shapes_massbin{i}.parquet", br)
        sr = build_shape_randoms(bcal, tr, srfac, p0, sw.create_clustering_randoms_desi, rng)
        sr.to_parquet(out_dir / f"{out_tag}_shape_randoms_massbin{i}.parquet", index=False)
        zeff = float(bcal["Z"].mean())
        rows.append((i, lo, hi, zb, len(bcal), br, len(sr), zeff))
        hi_s = f"{hi:6.2f}" if i < len(boxes) - 1 else "  top "
        print(f"  {i:>3} {lo:6.2f} {hi_s} {zb:6.3f} {len(bcal):>8,} {br:7.4f}  {len(sr):,}")
    print(f"  total kept: {sum(r[4] for r in rows):,}")

    if make_plot:
        _plot(name, shapes, mcen, zlim, rows, zmin, zmax, floor, plot_dir, scheme)


def _plot(name, shapes, mcen, zlim, rows, zmin, zmax, floor, plot_dir,
          scheme="equal_n"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hexbin(shapes["Z"], shapes["LOGMSTAR"], gridsize=120, bins="log", cmap="Greys", mincnt=1)
    mm = np.linspace(9.6, 11.7, 200)
    ax.plot(np.interp(mm, mcen, zlim, left=zlim[0], right=zlim[-1]), mm, "b-", lw=2,
            label="$M_*^{\\rm lim}(z)$ (Wright+17-style count turn-over)")
    for i, lo, hi, zb, n, br, nr, zeff in rows:
        ax.add_patch(plt.Rectangle((zmin, lo), zb - zmin, hi - lo, fill=False, ec="red", lw=1.3))
        ax.text(zmin + 0.004, 0.5 * (lo + hi), f"{i}", fontsize=7, color="red", va="center")
    ax.axhline(floor, color="green", ls=":", lw=1, label=f"M* floor = {floor}")
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$\log_{10} M_*$ (FastSpecFit)")
    ax.set_xlim(max(0.0, zmin - 0.03), zmax + 0.03)
    ax.set_ylim(9.6, 11.8)
    ax.legend(loc="lower right", fontsize=8)
    lab = "equal-N" if scheme == "equal_n" else "S/N-weighted (equal_snr)"
    ax.set_title(f"{name}: {len(rows)} {lab} volume-limited bins (built)")
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"vlim_bins_{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  plot  -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(gu.CONFIG))
    ap.add_argument("--samples", nargs="*", default=None,
                    help="subset of vlim_samples names (default: all)")
    ap.add_argument("--plot-dir", default=str(gu.REPO / "plots"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = gu.load_config(args.config)
    vc = cfg["vlim_samples"]
    samples = vc["samples"]
    if args.samples:
        samples = [s for s in samples if s["name"] in args.samples]
        if not samples:
            ap.error(f"no vlim_samples match {args.samples}")

    rng = np.random.default_rng(int(vc.get("seed", 42)))
    plot_dir = Path(args.plot_dir)
    for s in samples:
        process(s, cfg, vc, rng, make_plot=not args.no_plot, plot_dir=plot_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
