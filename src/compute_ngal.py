#!/usr/bin/env python3
"""Comoving number density n_gal per (lens sample, mass bin) -- the CSMF abundance anchor.

For each (lens sample, mass bin) the model fit needs ONE measured number density to
pair with the model's ``halo_model.ngal()`` (the abundance term that breaks the
ΔΣ-alone SHMR degeneracy -- see the csmf modelling notes). This script measures it
on the SHAPE / LENS catalogue itself (the same parquets used as GGL lenses):

    n_gal = (sum of WEIGHT_COMP*WEIGHT_ZFAIL*WEIGHT_SYS over the lenses) / V_eff  [h^3/Mpc^3]
    V_eff = Omega_eff * (D_c(z_hi)^3 - D_c(z_lo)^3) / 3                           [(Mpc/h)^3]

with the angular area ``Omega_eff`` the FIDUCIAL density-ratio footprint area of the
lens tracer (compute_footprint_area.py: N_kept/(n_files x 2500 deg^-2) on the FULL
randoms -- the same DESI-style, nside-free area the Gaussian covariance uses). The
old healpix occupied-pixel area of the downsampled shape randoms is resolution-
dependent (nside=256 overestimates by ~37% from partially covered edge pixels; see
the compute_footprint_area docstring) and is recorded as a diagnostic only.
Comoving distances are in Mpc/h (real-h Planck18 = the ΔΣ COSMO), so n_gal is in
h^3/Mpc^3 to match the HOD model run with ``units_per_h=True``.

COMPLETENESS: the CSMF model n_gal (halo_model.ngal) is the INTRINSIC comoving density,
so the measurement must be completeness-corrected. Two completenesses are in play and
must NOT be confused:
  * UNIONS footprint match -- >99% complete (build_ia_samples masks DESI at nside=8192
    and sky-matches within 1"), encoded by the shape randoms; no correction needed.
  * DESI fiber assignment -- LARGE (<WEIGHT_COMP> ~ 2.1, i.e. ~47% complete), corrected
    per-object by WEIGHT_COMP*WEIGHT_ZFAIL (and WEIGHT_SYS for imaging).
So we sum the COMPLETENESS-CORRECTED weight ``w_corr = WEIGHT_COMP*WEIGHT_ZFAIL*WEIGHT_SYS``.
NB ``WEIGHT_CLUSTERING`` is NOT used for the density: it has the per-NTILE <WEIGHT_COMP>
divided out (a relative clustering weight, mean ~1, so Sum(W_CLUSTERING) ~ the RAW
observed count) -- using it under-counts the intrinsic density by ~<WEIGHT_COMP> ~= 1.9x.
The raw (un-corrected) count is also recorded -- it is the density to use for SHOT NOISE.

Errors come from a leave-one-out jackknife over the SAME regions dsigma builds for
the ΔΣ covariance: ``compute_jackknife_fields`` (KMeans on the lens positions,
``ggl.dsigma.n_jackknife`` regions), applied to the randoms with the same centers.
Each resample drops a patch from BOTH the lens count and the random-traced area, so
the variance includes cosmic variance and is self-consistent with the ΔΣ jackknife.

CAVEAT: only the VOLUME-LIMITED sample (BGS_RED_GMM_VLIM) gives a clean intrinsic
density. The non-volume-limited BGS_RED_GMM bins are flux-limited inside [zmin,zmax]
so their N/V is a naive density (biased low at the high-z faint edge) -- computed
here only for comparison.

Outputs ``<dest>/ggl/csmf_input/ngal_<sample>.npz`` (per-bin arrays + provenance).

Run with the project .venv (dsigma 1.2 + healpy + astropy + pyarrow):
    .venv/bin/python scripts/compute_ngal.py [--samples NAME ...] [--nside 256]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402
from compute_deltasigma import COSMO  # the SAME real-h Planck18 cosmology as ΔΣ  # noqa: E402

H = COSMO.h
DEG2_PER_SR = (180.0 / np.pi) ** 2


def shell_volume_per_sr(zlo, zhi):
    """Comoving volume per steradian between zlo and zhi, in (Mpc/h)^3."""
    dlo = COSMO.comoving_distance(zlo).value * H   # Mpc -> Mpc/h
    dhi = COSMO.comoving_distance(zhi).value * H
    return (dhi ** 3 - dlo ** 3) / 3.0


def healpix_area_sr(ra, dec, nside):
    """Effective angular area (steradians) = occupied healpix pixels x pixel area."""
    import healpy as hp
    pix = np.unique(hp.ang2pix(nside, ra, dec, lonlat=True))
    return len(pix) * hp.nside2pixarea(nside), len(pix)


def radec_table(ra, dec):
    """Minimal dsigma-style table with ra/dec as degree Quantities (for jackknife)."""
    t = Table()
    t["ra"] = np.asarray(ra, float) * u.deg
    t["dec"] = np.asarray(dec, float) * u.deg
    return t


def ngal_one(shapes_file, randoms_file, zlo, n_jk, nside, omega_fid,
             zlo_percentile=None):
    """n_gal + jackknife error for one (sample, mass bin).

    omega_fid: fiducial footprint area in steradians (density-ratio estimator,
    footprint_area_<tracer>.json). The randoms still set the JACKKNIFE area
    fractions (patch area = omega_fid * N_rand_kept/N_rand_tot)."""
    from dsigma.jackknife import compute_jackknife_fields

    cols = ["RA", "DEC", "Z", "LOGMSTAR",
            "WEIGHT_COMP", "WEIGHT_ZFAIL", "WEIGHT_SYS", "WEIGHT_CLUSTERING"]
    lenses = pd.read_parquet(shapes_file, columns=cols)
    randoms = pd.read_parquet(randoms_file, columns=["RA", "DEC"])

    ra_l = lenses["RA"].to_numpy(float)
    dec_l = lenses["DEC"].to_numpy(float)
    # Completeness-CORRECTED weight = the intrinsic-density weight (see header).
    # WEIGHT_CLUSTERING has the per-NTILE <WEIGHT_COMP> divided out, so it is NOT
    # the right column for an absolute number density -- recorded only for comparison.
    w_l = (lenses["WEIGHT_COMP"] * lenses["WEIGHT_ZFAIL"]
           * lenses["WEIGHT_SYS"]).to_numpy(float)
    w_clust = lenses["WEIGHT_CLUSTERING"].to_numpy(float)
    z_l = lenses["Z"].to_numpy(float)
    lm = lenses["LOGMSTAR"].to_numpy(float)
    lm = lm[np.isfinite(lm)]
    ra_r = randoms["RA"].to_numpy(float)
    dec_r = randoms["DEC"].to_numpy(float)

    # Optional per-bin lower edge: the sample's own z percentile instead of the
    # fixed config zmin. Trims the depleted low-z end of the window (BGS bright
    # limit pushes the 5th percentile from 0.104 to 0.138 with bin mass) so the
    # volume matches where the sample actually lives. Galaxies below the new
    # edge are dropped from the count (and jackknife) for consistency.
    if zlo_percentile is not None:
        zlo = max(float(np.percentile(z_l, zlo_percentile)), zlo)
        keep = z_l >= zlo
        ra_l, dec_l, z_l = ra_l[keep], dec_l[keep], z_l[keep]
        w_l, w_clust = w_l[keep], w_clust[keep]

    zhi = float(z_l.max())
    shell = shell_volume_per_sr(zlo, zhi)          # (Mpc/h)^3 / sr
    omega_hp, n_occ = healpix_area_sr(ra_r, dec_r, nside)   # diagnostic only
    omega = omega_fid                              # fiducial density-ratio area
    v_eff = omega * shell                          # (Mpc/h)^3
    n_w = float(w_l.sum())                         # completeness-corrected count
    n_gal = n_w / v_eff
    n_gal_clustering = float(w_clust.sum()) / v_eff   # old (under-counted) value
    n_gal_raw = len(w_l) / v_eff                      # raw count -> for shot noise

    # Jackknife: SAME patch construction as the ΔΣ covariance (KMeans on lenses,
    # centers reused on randoms). Drop patch k from both count and area.
    tl, tr = radec_table(ra_l, dec_l), radec_table(ra_r, dec_r)
    centers = compute_jackknife_fields(tl, n_jk)
    compute_jackknife_fields(tr, centers)
    fjl = np.asarray(tl["field_jk"])
    fjr = np.asarray(tr["field_jk"])
    n_r_tot = len(fjr)
    n_k = []
    for k in range(n_jk):
        keep_r = int(np.count_nonzero(fjr != k))
        v_k = omega * (keep_r / n_r_tot) * shell
        if v_k <= 0:
            continue
        n_k.append(float(w_l[fjl != k].sum()) / v_k)
    n_k = np.asarray(n_k)
    m = len(n_k)
    var = (m - 1) / m * np.sum((n_k - n_k.mean()) ** 2)

    return dict(
        n_gal=n_gal, n_gal_err=float(np.sqrt(var)),
        n_gal_clustering=n_gal_clustering, n_gal_raw=n_gal_raw,
        comp_factor=n_gal / n_gal_clustering,     # <WEIGHT_COMP>-style ratio (~1.9)
        V_eff=v_eff, Omega_sr=omega, area_deg2=omega * DEG2_PER_SR,
        area_deg2_healpix=omega_hp * DEG2_PER_SR, n_occ=n_occ,
        N_weighted=n_w, N_raw=len(w_l), zlo=float(zlo), zhi=zhi,
        logmstar_min=float(lm.min()), logmstar_max=float(lm.max()), n_jk_used=m,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(gu.CONFIG))
    ap.add_argument("--samples", nargs="+",
                    default=["BGS_RED_GMM_VLIM", "BGS_RED_GMM"],
                    help="Lens samples to process (default: VLIM + standard BGS red).")
    ap.add_argument("--nside", type=int, default=256,
                    help="healpix nside for the random-traced area (default 256).")
    ap.add_argument("--zlo-percentile", type=float, default=None,
                    help="Per-bin lower z edge = this percentile of the bin's own "
                         "n(z) (floored at the config zmin), instead of the fixed "
                         "config zmin. Galaxies below it are dropped from the count.")
    ap.add_argument("--suffix", default="",
                    help="Appended to the output stem, e.g. '_z5' -> "
                         "ngal_<sample>_z5.npz (keeps the fiducial npz intact)")
    ap.add_argument("--area-mode", choices=["geometric", "effective"], default="geometric",
                    help="'geometric' (default): the density-ratio footprint area "
                         "N_ran/(n_files x 2500 deg^-2). 'effective': the same area "
                         "weighted by FRAC_TLOBS_TILES over the tracer randoms -- the "
                         "DESI DR1 convention (the '#effective area' of the released "
                         "*_nz.txt tables; Vol_bin there = shell x effective area). "
                         "Measured 2026-09-17: BGS overlap <FRAC_TLOBS_TILES> = 0.988.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = gu.load_config(args.config)
    dest = Path(cfg["dest"])
    ggl = cfg["ggl"]
    ia_dir = dest / ggl["ia_subdir"]
    out_dir = dest / ggl["out_subdir"] / "csmf_input"
    area_dir = dest / cfg["ia_samples"]["out_subdir"] / "sample_properties"
    n_jk = int(ggl["dsigma"]["n_jackknife"])
    lens_samples = {s["name"]: s for s in ggl["lens_samples"]}

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"cosmology: H0={COSMO.H0.value:.2f}, Om0={COSMO.Om0:.4f}, h={H:.4f}")
    print(f"area: FIDUCIAL density-ratio footprint (compute_footprint_area.py), "
          f"healpix nside={args.nside} recorded as diagnostic; jackknife: {n_jk} "
          f"regions; weight: WEIGHT_COMP*WEIGHT_ZFAIL*WEIGHT_SYS (no FKP); "
          f"units: h^3/Mpc^3\n")

    for name in args.samples:
        if name not in lens_samples:
            print(f"!! {name} not in ggl.lens_samples; skipping")
            continue
        s = lens_samples[name]
        n_mb = int(s["n_mass_bins"])
        zlo = float(s["zmin"])
        vlim = "VLIM" in name
        # fiducial area: the lens tracer's density-ratio footprint (DESI way,
        # same JSON the Gaussian covariance uses). Lens sample -> tracer by prefix.
        tracer = name.split("_")[0]
        area_file = area_dir / f"footprint_area_{tracer}.json"
        if not area_file.exists():
            raise SystemExit(f"missing {area_file} -- run compute_footprint_area.py "
                             f"--tracers {tracer}")
        import json
        area = json.load(open(area_file))
        omega_fid = float(area["area_sr"])
        frac_tlobs = 1.0
        if args.area_mode == "effective":
            # DESI effective area = geometric area x <FRAC_TLOBS_TILES> over the
            # randoms (uniform on the sky, so any random subsample gives the mean).
            tr_file = ia_dir / f"{tracer}_zmin_{float(s['zmin']):.2f}_zmax_{float(s['zmax']):.2f}_randoms.parquet"
            frac_tlobs = float(pd.read_parquet(tr_file, columns=["FRAC_TLOBS_TILES"])
                               ["FRAC_TLOBS_TILES"].mean())
            omega_fid *= frac_tlobs
        print(f"=== {name}  ({n_mb} bins, zmin={zlo:.2f}, "
              f"{'volume-limited' if vlim else 'FLUX-LIMITED -> naive density, comparison only'}) ===")
        print(f"    {args.area_mode} area: {omega_fid * DEG2_PER_SR:.2f} deg^2 "
              f"({area_file.name}; geometric {area['area_deg2']:.2f} deg^2"
              f"{f', <FRAC_TLOBS_TILES> = {frac_tlobs:.4f}' if args.area_mode == 'effective' else ''})")
        print(f"{'bin':>3} {'logM*lo':>8} {'logM*hi':>8} {'zhi':>6} {'N_raw':>8} "
              f"{'area/deg2':>9} {'n_gal':>11} {'err':>10} {'rel':>6} "
              f"{'n_clust':>10} {'fcomp':>6}")
        rows = []
        for i in range(n_mb):
            stem = gu.lens_stem(s, i)
            sf = ia_dir / f"{stem}.parquet"
            rf = ia_dir / f"{stem.replace('_shapes_', '_shape_randoms_')}.parquet"
            if not sf.exists() or not rf.exists():
                print(f"{i:>3}  MISSING {sf.name if not sf.exists() else rf.name}")
                continue
            r = ngal_one(sf, rf, zlo, n_jk, args.nside, omega_fid,
                         zlo_percentile=args.zlo_percentile)
            rows.append((i, r))
            print(f"{i:>3} {r['logmstar_min']:>8.3f} {r['logmstar_max']:>8.3f} "
                  f"{r['zhi']:>6.3f} {r['N_raw']:>8} {r['area_deg2']:>9.0f} "
                  f"{r['n_gal']:>11.4e} {r['n_gal_err']:>10.3e} "
                  f"{r['n_gal_err'] / r['n_gal']:>6.1%} "
                  f"{r['n_gal_clustering']:>10.3e} {r['comp_factor']:>6.3f}")

        if rows and not args.dry_run:
            idx = np.array([i for i, _ in rows])
            col = lambda k: np.array([r[k] for _, r in rows])  # noqa: E731
            out = out_dir / f"ngal_{name}{args.suffix}.npz"
            np.savez(
                out, massbin=idx, n_gal=col("n_gal"), n_gal_err=col("n_gal_err"),
                n_gal_clustering=col("n_gal_clustering"), n_gal_raw=col("n_gal_raw"),
                comp_factor=col("comp_factor"),
                V_eff=col("V_eff"), area_deg2=col("area_deg2"),
                area_deg2_healpix=col("area_deg2_healpix"),
                area_source=str(area_file),
                N_weighted=col("N_weighted"), N_raw=col("N_raw"),
                zlo=col("zlo"), zhi=col("zhi"),
                logmstar_min=col("logmstar_min"), logmstar_max=col("logmstar_max"),
                nside=args.nside, n_jackknife=n_jk, sample=name,
                area_mode=args.area_mode, frac_tlobs_tiles=frac_tlobs,
                zlo_percentile=(args.zlo_percentile
                                if args.zlo_percentile is not None else -1.0),
                volume_limited=vlim)
            print(f"  -> {out}\n")
        else:
            print()


if __name__ == "__main__":
    main()
