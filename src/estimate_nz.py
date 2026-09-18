#!/usr/bin/env python3
"""Clustering-redshift n(z) of the GGL source tomographic bins.

Clean-pipeline port of ``old_codes/photoz_estimation.ipynb``. Each source bin
(``ggl.source_bins``) is first cut to the DESI overlap (healpix pixels occupied
by the reference randoms, ``ia_samples.footprint_nside`` -- same strategy as
build_ia_samples.py), then cross-correlated against DESI spectroscopic reference
tracers (``ggl.nz.reference_tracers``: the footprint-matched IA clustering +
randoms parquets) sliced in dz bins:

  * w_ru(z_i): treecorr NNCorrelation between the reference slice and the
    (unknown-z) source bin, estimator DD/RD - 1 with the reference randoms.
    Sources enter with unit weights (positions only -- the lensing WEIGHT is
    a shape weight and does not belong in a clustering cross-correlation);
  * the angular signal is compressed to a scalar by a normalised W(theta) =
    theta^alpha scale weighting. The integration range is either fixed
    [theta_min, theta_max] arcmin (--scale-mode theta) or a fixed comoving
    [rp_min, rp_max] Mpc converted to theta at each slice centre
    (--scale-mode rp, the old-notebook convention);
  * the reference bias drops out via the slice auto-correlation w_xx and a CCL
    dark-matter prediction:  b(z_i) = sqrt(w_xx * dz / w_dm), so
    n(z_i) propto w_ru / (b * w_dm). If pyccl is unavailable the raw
    w_ru / sqrt(w_xx * dz) is saved instead (flagged in the output).

All slices of all tracers share the jackknife patches built from the source
bin, so the cross-slice covariance is assembled from the per-slice jackknife
design vectors. LRG (0.4-1.1) and ELG (0.8-1.6) overlap on z=[0.8,1.1]: the
per-tracer estimates are kept separately in the output and combined by inverse
variance on the joint grid. The combined n(z) is normalised to integrate to 1.

Outputs ``<dest>/ggl/nz/nz_<bin>_<scale_mode>.npz`` with (z, nz, nz_err,
nz_cov) plus the per-tracer raw measurements -- consumed by
``compute_deltasigma.py`` (dsigma ``table_n``).

Usage
-----
    python scripts/estimate_nz.py [--config CONFIG] [--bins bgs lrg]
                                  [--scale-mode theta|rp]
                                  [--head-sources N] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402


# =============================================================================
# CCL DARK-MATTER PREDICTION (theta-space scale-weighted w_dm)
# =============================================================================
def make_wdm_theta(nzc, ells=np.arange(1, 4000)):
    """Return (w_dm_theta(z, theta_min_deg, theta_max_deg, alpha), chi_mpc(z))
    using pyccl, or (None, None) if pyccl is not importable (the caller then
    skips the bias correction, flags the output, and cannot use rp mode)."""
    try:
        import camb  # noqa: F401  (pyccl's default linear P(k) backend)
        import pyccl as ccl
    except ImportError:
        return None, None
    from scipy.integrate import quad

    c = nzc["cosmology"]
    h = float(c["h"])
    cosmo = ccl.Cosmology(Omega_c=float(c["omega_c_h2"]) / h**2,
                          Omega_b=float(c["omega_b_h2"]) / h**2,
                          h=h, A_s=float(c["A_s"]), n_s=float(c["n_s"]))
    c_kms = 299792.458

    def wdm(z, theta_min, theta_max, alpha):
        a = 1.0 / (1.0 + z)
        chi = ccl.comoving_radial_distance(cosmo, a)
        hz = 100.0 * h * ccl.background.h_over_h0(cosmo, a)
        p_ell = np.array([ccl.nonlin_matter_power(cosmo, k=(ell + 0.5) / chi, a=a)
                          for ell in ells])

        def xi(theta):
            return (hz / c_kms / chi**2 *
                    ccl.correlations.correlation(cosmo, ell=ells, C_ell=p_ell,
                                                 theta=theta, type="NN",
                                                 method="Legendre"))

        norm = quad(lambda t: t**alpha, theta_min, theta_max, limit=10000)[0]
        return quad(lambda t: t**alpha / norm * xi(t), theta_min, theta_max,
                    epsrel=1e-6, epsabs=1e-6, limit=10000)[0]

    def chi_mpc(z):
        return float(ccl.comoving_angular_distance(cosmo, 1.0 / (1.0 + z)))

    return wdm, chi_mpc


# =============================================================================
# REFERENCE CATALOGUES AND CORRELATION MACHINERY
# =============================================================================
def load_reference(cfg, tracer):
    """Concatenated clustering data + randoms of one reference tracer (the
    footprint-matched parquets). Data weight = WEIGHT_CORR when present
    (BGS) else WEIGHT; randoms = WEIGHT -- the project weight schema.

    Directory: a tracer may set `subdir` to read from somewhere other than the
    default `ggl.ia_subdir`. The LRG/ELG references live in ia/ (built and
    restored by build_ia_samples), but the n(z)-only combined LRGpELG reference
    lives under ggl/reference/ so build_ia_samples --clean cannot wipe it."""
    ref_dir = Path(cfg["dest"]) / tracer.get("subdir", cfg["ggl"]["ia_subdir"])
    data, rand = [], []
    for stem in tracer["stems"]:
        data.append(pd.read_parquet(ref_dir / f"{stem}_clustering.parquet",
                                    columns=None))
        rand.append(pd.read_parquet(ref_dir / f"{stem}_randoms.parquet",
                                    columns=["RA", "DEC", "Z", "WEIGHT"]))
    data = pd.concat(data, ignore_index=True)
    rand = pd.concat(rand, ignore_index=True)
    wcol = "WEIGHT_CORR" if "WEIGHT_CORR" in data.columns else "WEIGHT"
    print(f"  {tracer['name']}: {len(data):,} ref galaxies (weight={wcol}), "
          f"{len(rand):,} randoms", flush=True)
    return (data["RA"].to_numpy(), data["DEC"].to_numpy(), data["Z"].to_numpy(),
            data[wcol].to_numpy(float),
            rand["RA"].to_numpy(), rand["DEC"].to_numpy(), rand["Z"].to_numpy(),
            rand["WEIGHT"].to_numpy(float))


def scale_weighted(xi, rnom, widths, alpha):
    """Compress xi(theta) to a scalar with W(theta) = theta^alpha weighting."""
    w = rnom**alpha * widths
    return np.sum(xi * w) / np.sum(w)


def slice_correlations(ref, src_cat, patch_centers, zlo, zhi, nzc, rng,
                       theta_min, theta_max):
    """w_ru and w_xx (scale-weighted scalars) for one reference z slice,
    over [theta_min, theta_max] degrees.

    Returns (w_ru, w_ru jackknife design vector, w_xx, w_xx_err)."""
    import treecorr

    ra, dec, z, w, rra, rdec, rz, rw = ref
    alpha = float(nzc["alpha_weight"])
    nn_kw = dict(min_sep=theta_min, max_sep=theta_max, nbins=int(nzc["n_theta"]),
                 sep_units="degree", bin_slop=float(nzc["bin_slop"]),
                 var_method="jackknife", cross_patch_weight="match")

    sel = (z >= zlo) & (z < zhi)
    rsel = (rz >= zlo) & (rz < zhi)
    n_rand = int(nzc["eta_rand"]) * int(sel.sum())
    idx = rng.choice(rsel.sum(), size=n_rand, replace=True)
    cat_ref = treecorr.Catalog(ra=ra[sel], dec=dec[sel], w=w[sel],
                               ra_units="degrees", dec_units="degrees",
                               patch_centers=patch_centers)
    cat_rand = treecorr.Catalog(ra=rra[rsel][idx], dec=rdec[rsel][idx],
                                w=rw[rsel][idx], ra_units="degrees",
                                dec_units="degrees", patch_centers=patch_centers)

    # cross: DD/RD - 1 with the reference randoms (no source randoms needed)
    w_ru = treecorr.NNCorrelation(**nn_kw)
    rd = treecorr.NNCorrelation(**nn_kw)
    w_ru.process(cat_ref, src_cat)
    rd.process(cat_rand, src_cat)
    w_ru.calculateXi(rr=rd)
    widths = w_ru.right_edges - w_ru.left_edges
    func = lambda corrs: np.atleast_1d(                       # noqa: E731
        scale_weighted(corrs[0].xi, corrs[0].rnom, widths, alpha))
    wru = float(func([w_ru])[0])
    design, _ = treecorr.build_multi_cov_design_matrix(
        [w_ru], "jackknife", func=func)
    design = design[:, 0]

    # auto: Landy-Szalay within the slice (for the reference bias)
    w_xx = treecorr.NNCorrelation(**nn_kw)
    rr = treecorr.NNCorrelation(**nn_kw)
    dr = treecorr.NNCorrelation(**nn_kw)
    w_xx.process(cat_ref, cat_ref)
    rr.process(cat_rand, cat_rand)
    dr.process(cat_ref, cat_rand)
    w_xx.calculateXi(rr=rr, dr=dr)
    wxx = float(scale_weighted(w_xx.xi, w_xx.rnom, widths, alpha))
    cov_xx = treecorr.estimate_multi_cov([w_xx], "jackknife", func=func)
    return wru, design, wxx, float(np.sqrt(cov_xx[0, 0]))


def footprint_cut(src, refs, nside):
    """Cut the source bin to the DESI overlap: keep sources whose healpix pixel
    (nested, footprint nside) is occupied by ANY reference tracer's randoms --
    the same footprint-matching strategy as build_ia_samples.py. Without it the
    UNIONS-only sky leaves empty jackknife patches (the old-code limitation)."""
    import healpy as hp

    from build_ia_samples import ang2pix

    occ = np.zeros(hp.nside2npix(nside), dtype=bool)
    for ref in refs.values():
        rra, rdec = ref[4], ref[5]
        occ[ang2pix(rra, rdec, nside)] = True
    keep = occ[ang2pix(src["RA"].to_numpy(), src["DEC"].to_numpy(), nside)]
    src = src[keep].reset_index(drop=True)
    print(f"    {len(src):,} sources in the DESI footprint "
          f"(nside={nside}, {keep.mean():.1%} kept)", flush=True)
    return src


def process_bin(cfg, bin_name, out_dir, scale_mode, head=0, tag=""):
    """n(z) of one source tomographic bin against every reference tracer.

    scale_mode 'theta': fixed [theta_min, theta_max] arcmin for every slice;
    scale_mode 'rp': fixed comoving [rp_min, rp_max] Mpc, converted to theta
    at each slice centre (requires pyccl). `tag` is appended to the output
    filename (e.g. the photo-z width cut) so sweep runs don't clobber."""
    import treecorr

    nzc = cfg["ggl"]["nz"]
    dz = float(nzc["delta_z"])
    alpha = float(nzc["alpha_weight"])
    rng = np.random.default_rng(42)

    # positions only: the lensing WEIGHT is a shape weight, not a density one
    src = gu.read_source_bin(cfg, bin_name, ["RA", "DEC", "Z_B"], head=head)
    refs = {t["name"]: load_reference(cfg, t) for t in nzc["reference_tracers"]}
    src = footprint_cut(src, refs, int(cfg["ia_samples"]["footprint_nside"]))
    print(f"  building {nzc['n_patches']} jackknife patches from the source bin",
          flush=True)
    src_cat = treecorr.Catalog(ra=src["RA"].to_numpy(), dec=src["DEC"].to_numpy(),
                               ra_units="degrees", dec_units="degrees",
                               npatch=int(nzc["n_patches"]),
                               kmeans_init="kmeans++")
    patch_centers = src_cat.patch_centers

    wdm_func, chi_mpc = make_wdm_theta(nzc)
    if wdm_func is None:
        if scale_mode == "rp":
            raise RuntimeError("--scale-mode rp needs pyccl for chi(z)")
        print("  WARNING: pyccl not importable -- skipping the w_dm bias "
              "correction (n(z) saved with w_dm = 1, flagged)", flush=True)

    def theta_range(zc):
        """Scale-weighting bounds in degrees for the slice centred on zc."""
        if scale_mode == "rp":
            deg_per_mpc = np.degrees(1.0 / chi_mpc(zc))
            return (float(nzc["rp_min_mpc"]) * deg_per_mpc,
                    float(nzc["rp_max_mpc"]) * deg_per_mpc)
        return (float(nzc["theta_min_arcmin"]) / 60.0,
                float(nzc["theta_max_arcmin"]) / 60.0)

    out = {"delta_z": dz, "bias_corrected": wdm_func is not None,
           "alpha_weight": alpha, "n_patches": int(nzc["n_patches"]),
           "scale_mode": scale_mode}
    per_tracer = []
    for tracer in nzc["reference_tracers"]:
        name = tracer["name"]
        zlo_t, zhi_t = (float(v) for v in tracer["z_range"])
        n_slices = int(round((zhi_t - zlo_t) / dz))
        centers = zlo_t + (np.arange(n_slices) + 0.5) * dz
        ref = refs[name]

        wru = np.zeros(n_slices)
        wxx = np.zeros(n_slices)
        wxx_err = np.zeros(n_slices)
        designs = np.zeros((int(nzc["n_patches"]), n_slices))
        wdm = np.ones(n_slices)
        t0 = time.time()
        for i, zc in enumerate(centers):
            tmin, tmax = theta_range(zc)
            wru[i], designs[:, i], wxx[i], wxx_err[i] = slice_correlations(
                ref, src_cat, patch_centers, zc - dz / 2, zc + dz / 2, nzc, rng,
                tmin, tmax)
            if wdm_func is not None:
                wdm[i] = wdm_func(zc, tmin, tmax, alpha)
            print(f"    {name} z={zc:.3f} theta=[{tmin*60:.2f},{tmax*60:.2f}]': "
                  f"w_ru={wru[i]:+.5f}  w_xx={wxx[i]:.5f}"
                  f"  w_dm={wdm[i]:.5f}  [{time.time()-t0:.0f}s]", flush=True)

        # n(z) propto w_ru / (b w_dm) with b = sqrt(w_xx dz / w_dm) (Limber,
        # dz top-hat slice); without pyccl: w_dm = 1 -> w_ru / sqrt(w_xx dz).
        denom = np.sqrt(np.clip(wxx, 0, None) * wdm * dz)
        good = denom > 0
        nz = np.where(good, wru / np.where(good, denom, 1.0), 0.0)
        # cross-slice jackknife covariance of w_ru from the shared patches,
        # scaled by the (deterministic, w_xx-error-ignored) 1/denom factor
        npat = designs.shape[0]
        dev = designs - designs.mean(axis=0)
        cov_wru = (npat - 1) / npat * dev.T @ dev
        scale = np.where(good, 1.0 / np.where(good, denom, 1.0), 0.0)
        cov_nz = cov_wru * np.outer(scale, scale)
        per_tracer.append({"name": name, "z": centers, "nz": nz, "cov": cov_nz,
                           "wru": wru, "wxx": wxx, "wxx_err": wxx_err, "wdm": wdm})
        out.update({f"{name}_z": centers, f"{name}_nz": nz, f"{name}_cov": cov_nz,
                    f"{name}_wru": wru, f"{name}_wru_err": np.sqrt(np.diag(cov_wru)),
                    f"{name}_wxx": wxx, f"{name}_wxx_err": wxx_err,
                    f"{name}_wdm": wdm})

    # ---- combine tracers on the joint grid (inverse variance in the overlap)
    z_all = np.unique(np.round(np.concatenate([t["z"] for t in per_tracer]), 6))
    nz_w = np.zeros(len(z_all))       # sum of weights
    nz_acc = np.zeros(len(z_all))     # sum of w * nz
    maps = []                         # (tracer index -> grid index, ivw weights)
    for t in per_tracer:
        gi = np.searchsorted(z_all, np.round(t["z"], 6))
        var = np.diag(t["cov"]).copy()
        var[var <= 0] = np.inf
        wt = 1.0 / var
        nz_acc[gi] += wt * t["nz"]
        nz_w[gi] += wt
        maps.append((gi, wt))
    nz = nz_acc / np.where(nz_w > 0, nz_w, 1.0)
    cov = np.zeros((len(z_all), len(z_all)))
    for t, (gi, wt) in zip(per_tracer, maps):
        frac = wt / nz_w[gi]          # this tracer's combination weight
        cov[np.ix_(gi, gi)] += t["cov"] * np.outer(frac, frac)

    # normalise the combined n(z) to integrate to 1 (clip negatives to 0 for
    # the normalisation only; the raw combined values are kept in nz_raw)
    norm = np.trapezoid(np.clip(nz, 0, None), z_all)
    out.update({"z": z_all, "nz_raw": nz, "nz": nz / norm, "nz_cov": cov / norm**2,
                "nz_err": np.sqrt(np.diag(cov)) / norm})

    out_file = out_dir / f"nz_{bin_name}_{scale_mode}{tag}.npz"
    np.savez(out_file, **out)
    print(f"  saved {out_file}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(gu.CONFIG))
    ap.add_argument("--bins", nargs="+", help="Restrict to these source bins (bgs lrg).")
    ap.add_argument("--scale-mode", choices=["theta", "rp"], default="theta",
                    help="Scale-weighting range: fixed arcmin (theta) or fixed "
                         "comoving Mpc converted per slice (rp).")
    ap.add_argument("--tracers", nargs="+",
                    help="Restrict to these reference tracers (LRG ELG); with "
                         "a single tracer no inverse-variance combination "
                         "happens.")
    ap.add_argument("--head-sources", type=int, default=0,
                    help="Read only the first N source rows (smoke test).")
    ap.add_argument("--zb-width-max", type=float, default=None,
                    help="Override ggl.source_zb_width_max: drop sources whose "
                         "BPZ interval Z_B_MAX-Z_B_MIN exceeds this. Tags the "
                         "output filename (nz_<bin>_<mode>_w<width>.npz) so a "
                         "sweep doesn't clobber the no-cut baseline.")
    ap.add_argument("--odds-min", type=float, default=None,
                    help="Override ggl.source_odds_min: keep only sources with "
                         "BPZ ODDS >= this (photo-z reliability). Tags the output "
                         "filename (nz_<bin>_<mode>_odds<val>.npz) so it doesn't "
                         "clobber the no-cut baseline.")
    ap.add_argument("--delta-z", type=float, default=None,
                    help="Override ggl.nz.delta_z (the reference-slice width). Tags "
                         "the output filename (nz_<bin>_<mode>_dz<val>.npz) so a dz "
                         "sweep doesn't clobber the fiducial dz=0.05 n(z) that ΔΣ "
                         "depends on.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = gu.load_config(args.config)
    ggl = cfg["ggl"]
    tag = ""
    if args.zb_width_max is not None:
        ggl["source_zb_width_max"] = args.zb_width_max
        tag += f"_w{args.zb_width_max:g}"
    if args.odds_min is not None:
        ggl["source_odds_min"] = args.odds_min
        tag += f"_odds{args.odds_min:g}"
    if args.delta_z is not None:
        ggl["nz"]["delta_z"] = args.delta_z
        tag += f"_dz{args.delta_z:g}"
    if args.tracers:
        nzc = ggl["nz"]
        unknown = set(args.tracers) - {t["name"] for t in nzc["reference_tracers"]}
        if unknown:
            ap.error(f"unknown tracers: {sorted(unknown)}")
        nzc["reference_tracers"] = [t for t in nzc["reference_tracers"]
                                    if t["name"] in args.tracers]
    out_dir = Path(cfg["dest"]) / ggl["out_subdir"] / "nz"
    bins = [b for b in ggl["source_bins"] if not args.bins or b in args.bins]

    print(f"source bins : {', '.join(bins)}")
    print(f"references  : {', '.join(t['name'] for t in ggl['nz']['reference_tracers'])}")
    print(f"scale mode  : {args.scale_mode}")
    print(f"zb width cut: {ggl.get('source_zb_width_max')}")
    print(f"odds cut    : {ggl.get('source_odds_min')}")
    print(f"out         : {out_dir}\n")
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for b in bins:
        print(f"\n{'='*70}\nsource bin: {b}\n{'='*70}", flush=True)
        process_bin(cfg, b, out_dir, args.scale_mode, head=args.head_sources,
                    tag=tag)
    print(f"\n[{time.time()-t0:.0f}s] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
