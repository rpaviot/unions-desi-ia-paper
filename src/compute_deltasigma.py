#!/usr/bin/env python3
"""Galaxy-galaxy lensing Delta Sigma for the GGL lens samples.

Clean-pipeline replacement for ``old_codes/compute_deltasigma_andwgg_LATEST.py``
(Delta Sigma part only -- the clustering wp lives in compute_correlations.py).
For each (lens sample, mass bin) from ``ggl.lens_samples``:

  * LENSES   = the IA mass-bin shapes parquet (RA/DEC/Z + WEIGHT_CLUSTERING,
               the matched lens density weight: WEIGHT_CORR for BGS, WEIGHT
               for LRG); RANDOMS = the matching shape_randoms parquet (WEIGHT).
  * SOURCES  = the for_GGL tomographic bin of the lens family, prepared once
               per bin (veto, Z_B cut, responsivity calibration, then two-pass
               PSF leakage correction -- ggl_utils.load_source_bin).
  * Source redshifts enter ONLY through the clustering-z n(z) of the bin
    (estimate_nz.py output), via dsigma's ``table_n`` machinery: every source
    carries z_bin=0 and sigma_crit is the n(z)-effective one.
  * dsigma precompute (comoving rp bins in h^-1 Mpc, sigma_crit^-2 weighting),
    excess_surface_density with random subtraction, and a jackknife covariance
    over ``ggl.dsigma.n_jackknife`` regions (built from the lenses, shared with
    the randoms).

Outputs ``<dest>/ggl/deltasigma/<stem>_deltasigma.npz`` per mass bin with
rp, ds, ds_err, the covariance and the sample metadata.

Usage
-----
    python scripts/compute_deltasigma.py [--config CONFIG] [--samples NAME ...]
                                         [--massbins I ...] [--auto-resume]
                                         [--head-sources N] [--dry-run]
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402

# Planck18 -- the SINGLE cosmology for the whole ΔΣ measurement. Use the REAL h
# (H0=67.66), NOT H0=100. dsigma 1.2 handles little-h natively via astropy units
# (a plain rp_bins array defaults to Mpc/h, ΔΣ is returned in h Msun/pc^2). The
# correct 1.2 usage is therefore a real-h cosmology -- and crucially the n(z)
# effective critical surface density (effective_critical_surface_density, the
# path we use by passing table_n with z_bin=0 sources) is only little-h-correct
# with a real cosmology: an h=1 (H0=100) cosmology silently INFLATES ΔΣ by 1/h
# (~1.478). [Verified against a hand n(z)-average of critical_surface_density.]
# rp binning stays in Mpc/h (h-robust), consistent with the IA/clustering rp.
# These densities match HOD_NRV DEFAULT_COSMO_PARAMS, fit_ia.DICT_COSMO and
# old_codes.setup_camb (omch2=0.11933, ombh2=0.02242, sum m_nu=0.06 eV).
_h = 0.6766
COSMO = FlatLambdaCDM(
    H0=100 * _h,
    Om0=(0.11933 + 0.02242) / _h**2,   # (omch2 + ombh2)/h^2  (cdm + baryon)
    Ob0=0.02242 / _h**2,               # ombh2/h^2
    Tcmb0=2.7255,                      # K, Planck18 (also lets CAMB run for the mag template)
    m_nu=[0.0, 0.0, 0.06] * u.eV,      # sum m_nu = 0.06 eV
)


def read_r_mean(path):
    """Responsivity recorded in the shapes parquet schema metadata (b"r_mean")."""
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(path).schema_arrow.metadata or {}
    return float(meta[b"r_mean"]) if b"r_mean" in meta else np.nan


def source_table(src):
    """dsigma source table: positions, weights, corrected shapes, z_bin=0.
    No `z` column -- the redshift information is carried by table_n."""
    t = Table()
    t["ra"] = src["RA"].to_numpy(float)
    t["dec"] = src["DEC"].to_numpy(float)
    t["w"] = src["WEIGHT"].to_numpy(float)
    t["e_1"] = src["e1"].to_numpy(float)
    t["e_2"] = src["e2"].to_numpy(float)
    t["w_sys"] = np.ones(len(src))
    t["z_bin"] = np.zeros(len(src), dtype=int)
    return t


def lens_table(df, wcol):
    t = Table()
    t["ra"] = df["RA"].to_numpy(float)
    t["dec"] = df["DEC"].to_numpy(float)
    t["z"] = df["Z"].to_numpy(float)
    t["w_sys"] = df[wcol].to_numpy(float)
    return t


def load_nz_table(nz_dir, bin_name, mode, smooth_sigma):
    """dsigma `table_n`: columns z (grid) and n with shape (n_z, n_bins=1).

    Reads nz_<bin>_<mode>.npz (mode = the baseline scale convention, e.g. rp).
    The n(z) is lightly Gaussian-smoothed (smooth_sigma grid cells; 0 = off),
    THEN negatives are clipped to zero, THEN it is renormalised to integrate
    to 1 -- so dsigma sees a smooth, non-negative source redshift kernel."""
    f = nz_dir / f"nz_{bin_name}_{mode}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} -- run estimate_nz.py (--scale-mode {mode}) first")
    d = np.load(f)
    z = np.asarray(d["z"], float)
    nz = np.asarray(d["nz"], float)
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        nz = gaussian_filter1d(nz, float(smooth_sigma), mode="nearest")
    nz = np.clip(nz, 0.0, None)               # n(z) -> 0 wherever negative
    norm = np.trapezoid(nz, z)
    if norm > 0:
        nz = nz / norm
    t = Table()
    t["z"] = z
    t["n"] = nz[:, None]                       # (n_z, 1)
    if not bool(d["bias_corrected"]):
        print(f"  WARNING: {f.name} was built WITHOUT the w_dm bias correction "
              "(pyccl missing at estimate_nz time)", flush=True)
    print(f"  n(z): {f.name}  (smooth_sigma={smooth_sigma}, clipped>=0, renormalised)",
          flush=True)
    return t, f


def process(job, table_s, table_n, nz_file, dsc, out_dir, n_jobs_override=None,
            jk_seed=None):
    """One (sample, mass bin): precompute, stack, jackknife, save."""
    rp_bins = np.geomspace(float(dsc["rp_min"]), float(dsc["rp_max"]),
                           int(dsc["n_rp_bins"]) + 1)
    n_jk = int(dsc["n_jackknife"])
    # Resolve n_jobs per source bin. precompute forks n_jobs long-lived workers;
    # under the nodes' strict no-overcommit policy (vm.overcommit_memory=2) each
    # fork must reserve commit ~ the parent's VM, which is dominated by the
    # (duplicated, float64) source engine arrays. The big bgs source bin (~41M)
    # OOMs at n_jobs=20 (the LRG bin, ~13.6M, barely fits) -- so allow a smaller
    # per-source override via dsigma.n_jobs_by_source. A CLI/SLURM --n-jobs wins
    # over both (lets the slurm pass $SLURM_CPUS_PER_TASK straight to precompute).
    if n_jobs_override is not None:
        n_jobs = int(n_jobs_override)
    else:
        n_jobs = int(dsc.get("n_jobs_by_source", {}).get(job["source_bin"],
                                                         dsc["n_jobs"]))

    from dsigma.precompute import precompute
    from dsigma.stacking import excess_surface_density
    from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling

    lenses = pd.read_parquet(job["shapes_file"])
    randoms = pd.read_parquet(job["randoms_file"],
                              columns=["RA", "DEC", "Z", "WEIGHT"])
    print(f"  lenses {len(lenses):,} (weight={job['wcol']}), "
          f"randoms {len(randoms):,}", flush=True)
    table_l = lens_table(lenses, job["wcol"])
    table_r = lens_table(randoms, "WEIGHT")

    t0 = time.time()
    print(f"  precompute lenses ({n_jobs} jobs)...", flush=True)
    precompute(table_l, table_s, rp_bins, table_n=table_n, cosmology=COSMO,
               comoving=True, weighting=-2, n_jobs=n_jobs)
    print(f"  precompute randoms...", flush=True)
    precompute(table_r, table_s, rp_bins, table_n=table_n, cosmology=COSMO,
               comoving=True, weighting=-2, n_jobs=n_jobs)
    print(f"    done in {time.time()-t0:.0f}s", flush=True)

    kwargs = dict(random_subtraction=True, table_r=table_r)
    result = excess_surface_density(table_l, return_table=True, **kwargs)

    # Lens-magnification bias TEMPLATE (NOT subtracted here). dsigma scales it as
    # 2*(alpha_l-1); we evaluate at alpha_l_template (=2.0 -> unit prefactor) so the
    # real correction is later (alpha_real-1)*ds_mag_unit -- fully rescalable.
    alpha_l_tpl = float(dsc.get("alpha_l_template", 2.0))
    ds_mag_unit = None
    try:
        # Single-cosmology stock call: meta['cosmology'] is the real-h Planck18
        # COSMO, so CAMB gets correct physical densities and Tcmb0.
        from dsigma.stacking import lens_magnification_bias
        ds_mag_unit = np.asarray(lens_magnification_bias(
            table_l, alpha_l_tpl,
            sigma_8=float(dsc.get("mag_sigma_8", 0.8102)),
            n_s=float(dsc.get("mag_n_s", 0.9665)),
            photo_z_dilution_correction=False, shear=False))
        print(f"  magnification template @ alpha_l={alpha_l_tpl} "
              f"(unit, rescale by (alpha-1)): ds_mag[0]={ds_mag_unit[0]:.3e}", flush=True)
    except Exception as exc:
        print(f"  WARNING: magnification template failed ({exc}); saving NaNs", flush=True)
        ds_mag_unit = np.full(int(dsc["n_rp_bins"]), np.nan)

    print(f"  jackknife covariance ({n_jk} regions)...", flush=True)
    centers = compute_jackknife_fields(table_l, n_jk, seed=jk_seed)
    compute_jackknife_fields(table_r, centers)
    cov = jackknife_resampling(excess_surface_density, table_l, **kwargs)

    w_l = table_l["w_sys"].data
    logm = lenses["LOGMSTAR"].to_numpy(float)
    logm = logm[np.isfinite(logm)]
    out = {
        "rp_bins": rp_bins,
        "rp": np.sqrt(result["rp_min"].data * result["rp_max"].data),
        "ds": np.asarray(result["ds"]),
        "ds_err": np.sqrt(np.diag(cov)),
        "cov_ds": cov,
        "ds_raw": np.asarray(result["ds_raw"]),
        "ds_r": np.asarray(result["ds_r"]),
        # lens-magnification bias template at alpha_l_template (unit prefactor):
        # real correction = (alpha_real - 1) * ds_mag_unit  (NOT applied here)
        "ds_mag_unit": ds_mag_unit,
        "alpha_l_template": alpha_l_tpl,
        "n_pairs": np.asarray(result["n_pairs"]),
        "z_l_mean": np.asarray(result["z_l"]),
        "z_s_mean": np.asarray(result["z_s"]),
        "z_eff": float(np.sum(table_l["z"].data * w_l) / np.sum(w_l)),
        "n_lenses": len(lenses), "n_randoms": len(randoms),
        "n_sources": len(table_s),
        "r_mean_lens": read_r_mean(job["shapes_file"]),
        "logmstar_mean": logm.mean(), "logmstar_median": np.median(logm),
        "logmstar_min": logm.min(), "logmstar_max": logm.max(),
        "logmstar_std": logm.std(),
        "sample": job["sample"], "massbin": job["massbin"],
        "zmin": job["zmin"], "zmax": job["zmax"],
        "source_bin": job["source_bin"], "nz_file": str(nz_file),
        "n_jackknife": n_jk,
    }
    np.savez(job["out_file"], **out)
    print(f"  saved {job['out_file']}  (z_eff={out['z_eff']:.3f})", flush=True)
    del lenses, randoms, table_l, table_r
    gc.collect()


def build_jobs(cfg, out_dir, samples=None, massbins=None):
    ggl = cfg["ggl"]
    ia_dir = Path(cfg["dest"]) / ggl["ia_subdir"]
    jobs = []
    for s in ggl["lens_samples"]:
        if samples and s["name"] not in samples:
            continue
        for i in range(int(s["n_mass_bins"])):
            if massbins and i not in massbins:
                continue
            stem = gu.lens_stem(s, i)
            jobs.append({
                "sample": s["name"], "massbin": i, "stem": stem,
                "zmin": s["zmin"], "zmax": s["zmax"],
                "source_bin": s["source_bin"],
                "wcol": "WEIGHT_CLUSTERING",
                "shapes_file": ia_dir / f"{stem}.parquet",
                "randoms_file": ia_dir / f"{stem.replace('_shapes_', '_shape_randoms_')}.parquet",
                "out_file": out_dir / f"{stem}_deltasigma.npz",
            })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(gu.CONFIG))
    ap.add_argument("--samples", nargs="+",
                    help="Restrict to these lens samples (e.g. BGS_RED_GMM LRG).")
    ap.add_argument("--massbins", nargs="+", type=int,
                    help="Restrict to these mass bin indices.")
    ap.add_argument("--shard", type=int, default=None,
                    help="With --n-shards, run only this contiguous shard of the "
                         "job list (for SLURM job arrays).")
    ap.add_argument("--n-shards", type=int, default=None,
                    help="Split the job list into this many contiguous shards.")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="Override precompute n_jobs for ALL source bins (wins "
                         "over config dsigma.n_jobs and n_jobs_by_source). The "
                         "slurm passes $SLURM_CPUS_PER_TASK here.")
    ap.add_argument("--head-sources", type=int, default=0,
                    help="Read only the first N source rows (smoke test).")
    ap.add_argument("--auto-resume", action="store_true",
                    help="Skip mass bins whose output .npz already exists.")
    ap.add_argument("--out-name", default="deltasigma",
                    help="Subdirectory under <dest>/<ggl.out_subdir> for the output "
                         ".npz (default 'deltasigma'). Point at a separate name to "
                         "recompute (e.g. on a new n(z)) without overwriting an "
                         "existing set.")
    ap.add_argument("--out-dir", default=None,
                    help="Explicit output directory (wins over --out-name).")
    ap.add_argument("--nz-dir", default=None,
                    help="Directory of the clustering-z nz_<bin>_<mode>.npz files "
                         "(default <dest>/<ggl.out_subdir>/nz).")
    ap.add_argument("--nz-mode", choices=["rp", "theta"], default=None,
                    help="Scale convention of the n(z) to read (default: config "
                         "ggl.dsigma.nz_mode, rp).")
    ap.add_argument("--jk-seed", type=int, default=None,
                    help="Seed of the dsigma jackknife-region KMeans (default None = unseeded, as in the "
                         "production runs: the released jackknife errors are one random draw of "
                         "the tessellation; set an int for a bit-reproducible covariance).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = gu.load_config(args.config)
    ggl = cfg["ggl"]
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(cfg["dest"]) / ggl["out_subdir"] / args.out_name)
    nz_dir = Path(args.nz_dir) if args.nz_dir else Path(cfg["dest"]) / ggl["out_subdir"] / "nz"
    nz_mode = args.nz_mode or ggl["dsigma"].get("nz_mode", "rp")
    jobs = build_jobs(cfg, out_dir, args.samples, args.massbins)
    if args.auto_resume:
        n0 = len(jobs)
        jobs = [j for j in jobs if not j["out_file"].exists()]
        print(f"[auto-resume] {n0 - len(jobs)}/{n0} outputs already present")

    # Contiguous shard selection for SLURM job arrays (keeps a source bin's mass
    # bins together so the prepared source bin is reused within a shard).
    if args.n_shards:
        if args.shard is None or not (0 <= args.shard < args.n_shards):
            ap.error("--shard must be in [0, --n-shards)")
        n = len(jobs)
        sizes = [n // args.n_shards + (1 if i < n % args.n_shards else 0)
                 for i in range(args.n_shards)]
        starts = np.cumsum([0] + sizes)
        jobs = jobs[starts[args.shard]:starts[args.shard + 1]]
        print(f"[shard {args.shard}/{args.n_shards}] {len(jobs)} of {n} jobs")

    print(f"out  : {out_dir}")
    print(f"jobs : {len(jobs)}")
    for j in jobs:
        print(f"  - {j['stem']}  (sources: {j['source_bin']})")
    if args.dry_run or not jobs:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # prepare each tomographic source bin once, shared across its lens samples
    src_cache = {}
    for j in jobs:
        b = j["source_bin"]
        if b not in src_cache:
            print(f"\npreparing source bin '{b}'...", flush=True)
            src, r_mean = gu.load_source_bin(cfg, b, head=args.head_sources)
            table_n, nz_file = load_nz_table(
                nz_dir, b, nz_mode,
                ggl["dsigma"].get("nz_smooth_sigma", 0.0))
            src_cache[b] = (source_table(src), table_n, nz_file, r_mean)
            del src
            gc.collect()
        table_s, table_n, nz_file, _ = src_cache[b]
        print(f"\n{'-'*70}\n{j['stem']}\n{'-'*70}", flush=True)
        try:
            process(j, table_s, table_n, nz_file, ggl["dsigma"], out_dir,
                    n_jobs_override=args.n_jobs, jk_seed=args.jk_seed)
        except Exception as exc:  # keep going across mass bins
            print(f"ERROR on {j['stem']}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
