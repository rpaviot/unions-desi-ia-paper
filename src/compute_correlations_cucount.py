#!/usr/bin/env python3
"""IA multipoles xi_0 / xi~22 with cucount's NATIVE jackknife (no scripts/djk).

The release under data/correlations_* was produced by compute_correlations.py, whose
delete-one jackknife lives in the vendored ``djk/`` package: per-patch pair-count blocks
NN[i][j] assembled into realisations

    realisation_k = total - auto(k) - alpha * cross(k),

with alpha = 1 (standard) or alpha = Ns / (2 + sqrt(2) (Ns - 1)) (Mohammad & Percival
2022) applied to the counts AND the weight-sum normalisations. cucount >= 0.2.7 does the
same natively: ``Particles(..., splits=labels)`` + ``SplitAttrs(mode='jackknife',
nsplits=Ns)`` return the per-region ii / ij / ji blocks as an
``lsstypes.Count2Jackknife`` whose ``realization(k, correction='mohammad21' | None)`` is
the same algebra. This script is the measurement stage on that path, kept deliberately
small: one cucount call per pair-count term, the estimators and the (s, mu) -> multipole
projection written out in full.

Same inputs, tessellation and grids as compute_correlations.py (the patch centres are
loaded from <dest>/ia/patch_centers, the labels assigned by treecorr exactly as djk does),
same output layout (<out>/<tracer>/<stem>_GPU_multipoles.npz with the keys fit_ia.py
reads), so the two scripts are interchangeable upstream of the fits. Only the multipoles
are produced here; the projected wp / wg+ stay with compute_correlations.py.

Estimators (each term normalised by its weight-sum norm):
    xi_gg(s, mu) = (DD - 2 DR + RR) / RR                      Landy-Szalay
    xi_g+(s, mu) = (D S_+ - R S_+) / (R_D S)                  shape randoms unused
    xi_0  = 1/2 * sum_mu xi_gg L_00 dmu,  L_00 = 1
    xi~22 = 5/48 * sum_mu xi_g+ L_22 dmu, L_22 = 3 (1 - mu^2)     (mu at bin midpoints)
    Cov   = (Ns - 1)/Ns * sum_k (x_k - <x>)(x_k - <x>)^T ; the quoted xi is <x>_k.

Validation against the release (cross-check job 870968, LRG 0.40 < z < 0.75, fine grid):
xi_0 agrees with djk to 1e-13 (alpha = 1 and Mohammad & Percival); xi~22 agrees to 6e-6,
which is the D x S normalisation: the shape sample is a subsample of the density sample
and djk drops the self-pairs from that norm, cucount does not (an offset of order
1 / N_shapes, not a covariance effect). Needs a CUDA GPU.

    ${IA_PYTHON} src/compute_correlations_cucount.py --config config/data_sources.yaml \
        --tracer LRG --zmin 0.4 --zmax 0.75 --sample LRG \
        --s-nbins 19 --s-min 6 --s-max 100 --n-patches 70 --kmeans-init random \
        --fkp fkp --sysweights recomputed --out results/correlations_cucount
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_correlations as cc  # noqa: E402  (config, jobs, weights, n(z), grids)

ENGINE = "cucount-jackknife"


# =============================================================================
# geometry + particles
# =============================================================================
def cartesian(ra_rad, dec_rad, dc):
    """RA/Dec (rad) + comoving distance -> Cartesian positions, as djk.compute_pairs."""
    x = dc * np.cos(dec_rad) * np.cos(ra_rad)
    y = dc * np.cos(dec_rad) * np.sin(ra_rad)
    z = dc * np.sin(dec_rad)
    return np.column_stack([x, y, z])


def treecorr_catalog(df, w, patch_centers, n_patches, g1=None, g2=None, is_rand=False):
    """The treecorr Catalog djk builds (ra/dec in degrees, r = D_c(z) in h^-1 Mpc):
    it assigns every object to the nearest cached patch centre -- the tessellation of
    the release -- and converts ra/dec to radians."""
    import treecorr
    dc = cc.COSMO.comoving_distance(np.asarray(df["Z"], float)).value
    kw = dict(ra=df["RA"].values, dec=df["DEC"].values, r=dc, w=np.asarray(w, float),
              ra_units="degree", dec_units="degree", npatch=n_patches,
              patch_centers=patch_centers)
    if g1 is not None:
        kw.update(g1=np.asarray(g1, float), g2=np.asarray(g2, float))
    if is_rand:
        kw["is_rand"] = 1
    return treecorr.Catalog(**kw)


def particles(cat, spin=False):
    """cucount Particles from a treecorr Catalog, with the patch label as the split."""
    from cucount.numpy import Particles
    pos = cartesian(cat.ra, cat.dec, cat.r)
    kw = {"splits": np.asarray(cat.patch, dtype=np.int64)}
    if spin:
        kw["spin_values"] = np.column_stack([cat.g1, cat.g2])
    return Particles(pos, np.asarray(cat.w, dtype=np.float64), **kw)


# =============================================================================
# jackknife realisations, projection, covariance
# =============================================================================
def realisations(counts, estimator, correction):
    """(Ns, ns, nmu) stack of xi(s, mu) over the delete-one realisations.

    counts    : dict name -> lsstypes.Count2Jackknife
    estimator : callable(dict name -> normalised counts array) -> xi(s, mu)
    correction: None (alpha = 1) or 'mohammad21'"""
    first = next(iter(counts.values()))
    out = []
    for lab in first.realizations:
        nc = {name: np.asarray(c.realization(lab, correction=correction).value())
              for name, c in counts.items()}
        out.append(estimator(nc))
    return np.array(out)


def project(xi2d, mu_edges, ell, sab):
    """(s, mu) -> multipole with the Legendre / associated-Legendre kernel evaluated at
    the mu-bin midpoints and a midpoint sum over mu (djk.compute_correlation_multipoles)."""
    mu = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = np.diff(mu_edges)
    if (ell, sab) == (0, 0):
        kernel, factor = np.ones_like(mu), 0.5
    elif (ell, sab) == (2, 2):
        kernel = 3.0 * (1.0 - mu**2)
        factor = (2 * ell + 1) / 2.0 * math.factorial(ell - sab) / math.factorial(ell + sab)
    else:
        raise ValueError(f"unsupported (ell, sab) = ({ell}, {sab})")
    return factor * np.sum(xi2d * kernel * dmu, axis=-1)


def jackknife_mean_cov(real):
    """Delete-one jackknife: mean over realisations and (Ns-1)/Ns * scatter."""
    ns = real.shape[0]
    mean = real.mean(axis=0)
    xc = real - mean
    return mean, (ns - 1) / ns * (xc.T @ xc)


def s_mid(edges):
    return np.sqrt(edges[1:] * edges[:-1])


LS = lambda c: (c["DD"] - 2.0 * c["DR"] + c["RR"]) / c["RR"]        # noqa: E731
IA = lambda c: (c["DS"] - c["RS"]) / c["RDS"]                        # noqa: E731


# =============================================================================
# per-tracer driver
# =============================================================================
def process(job, out_dir, auto_resume=False):
    from cucount.numpy import BinAttrs, SplitAttrs, WeightAttrs
    from cucount.types import count2

    tracer, (zlo, zhi) = job["tracer"], job["z_range"]
    shapes_todo = job["shapes"]
    if auto_resume:
        shapes_todo = [sh for sh in job["shapes"]
                       if not (Path(out_dir) / sh["out_tracer"]
                               / f"{sh['stem']}_GPU_multipoles.npz").exists()]
        if not shapes_todo:
            print(f"[auto-resume] {tracer} z=[{zlo:.2f}, {zhi:.2f}]: nothing left", flush=True)
            return

    print("\n" + "#" * 70 + f"\n# {tracer}  z=[{zlo:.2f}, {zhi:.2f}]   shapes={len(shapes_todo)}"
          f"   engine={ENGINE}\n" + "#" * 70, flush=True)

    ns = cc.N_PATCHES
    patch_centers = cc.get_patch_centers(job)
    clustering = cc.load_parquet(job["clustering_file"])
    randoms = cc.load_parquet(job["randoms_file"])
    w_clust, w_rand = cc.density_weights(clustering, randoms)
    z_eff_clustering = cc.effective_redshift(clustering["Z"], w_clust)
    print(f"  z_eff (clustering) = {z_eff_clustering:.4f}", flush=True)

    battrs = BinAttrs(s=cc.S_BINS, mu=(cc.MU_BINS, "midpoint"))
    spattrs = SplitAttrs(mode="jackknife", nsplits=ns)
    wspin = WeightAttrs(spin=(0, 2))

    # --- density auto: DD, DR, RR once per tracer -------------------------------
    print("\n[xi0] clustering monopole (ell=0)...", flush=True)
    t0 = time.time()
    cat_d = treecorr_catalog(clustering, w_clust, patch_centers, ns)
    cat_r = treecorr_catalog(randoms, w_rand, patch_centers, ns, is_rand=True)
    p_d, p_r = particles(cat_d), particles(cat_r)
    DD = count2(p_d, battrs=battrs, spattrs=spattrs)["weight"]
    DR = count2(p_d, p_r, battrs=battrs, spattrs=spattrs)["weight"]
    RR = count2(p_r, battrs=battrs, spattrs=spattrs)["weight"]
    gg = dict(DD=DD, DR=DR, RR=RR)
    xi0_real = {corr: project(realisations(gg, LS, corr), cc.MU_BINS, 0, 0)
                for corr in (None, "mohammad21")}
    xi0, cov_xi0 = jackknife_mean_cov(xi0_real[None])
    _, cov_xi0_corr = jackknife_mean_cov(xi0_real["mohammad21"])
    print(f"    done in {time.time() - t0:.1f}s", flush=True)
    d0 = {"xi0": xi0, "cov_xi0": cov_xi0, "cov_xi0_corrected": cov_xi0_corr}
    del cat_d, p_d
    gc.collect()

    # --- per shape sample: D S_+, R S_+, R_D S ------------------------------------
    for sh in shapes_todo:
        print("\n" + "-" * 70 + f"\n  shape sample: {sh['stem']}\n" + "-" * 70, flush=True)
        shapes = cc.load_parquet(sh["shape_file"])
        shape_randoms = cc.load_parquet(sh["shape_randoms_file"])
        good = np.isfinite(shapes["e1"]) & np.isfinite(shapes["e2"])
        if not good.all():
            print(f"  dropping {(~good).sum()} shapes with non-finite e1/e2", flush=True)
            shapes = shapes[good].reset_index(drop=True)
        w_shape, _ = cc.shape_weights(shapes, shape_randoms)   # shape randoms: R_D S norm only

        meta = {
            "z_eff_clustering": z_eff_clustering,
            "z_eff_IA": cc.effective_redshift(shapes["Z"], shapes["WEIGHT"]),
            "n_clustering": len(clustering), "n_shapes": len(shapes),
            "n_randoms": len(randoms), "n_shape_randoms": len(shape_randoms),
        }
        meta.update(cc.compute_nz(clustering, shapes, w_clust, shapes["WEIGHT"].values))
        meta.update(use_fkp=cc.USE_FKP, n_patches=ns, n_deleted=1,
                    kmeans_init=cc.KMEANS_INIT, no_sysweights=cc.NO_SYSWEIGHTS,
                    engine=ENGINE)
        meta.update(cc.mass_stats(shapes))
        print(f"  z_eff (IA) = {meta['z_eff_IA']:.4f}", flush=True)

        print("[xi2] IA quadrupole (ell=2)...", flush=True)
        t0 = time.time()
        cat_s = treecorr_catalog(shapes, w_shape, patch_centers, ns,
                                 g1=shapes["e1"].values, g2=shapes["e2"].values)
        p_s, p_s_pos = particles(cat_s, spin=True), particles(cat_s)
        p_d = particles(treecorr_catalog(clustering, w_clust, patch_centers, ns))
        DS = count2(p_d, p_s, battrs=battrs, wattrs=wspin, spattrs=spattrs)["weight_plus"]
        RS = count2(p_r, p_s, battrs=battrs, wattrs=wspin, spattrs=spattrs)["weight_plus"]
        RDS = count2(p_r, p_s_pos, battrs=battrs, spattrs=spattrs)["weight"]
        gp = dict(DS=DS, RS=RS, RDS=RDS)
        xi2_real = {corr: project(realisations(gp, IA, corr), cc.MU_BINS, 2, 2)
                    for corr in (None, "mohammad21")}
        xi2, cov_xi2 = jackknife_mean_cov(xi2_real[None])
        _, cov_xi2_corr = jackknife_mean_cov(xi2_real["mohammad21"])
        # joint [xi0, xi2] covariance from the stacked realisations (same patch order)
        _, cov_c = jackknife_mean_cov(np.column_stack([xi0_real[None], xi2_real[None]]))
        _, cov_cc = jackknife_mean_cov(np.column_stack([xi0_real["mohammad21"],
                                                        xi2_real["mohammad21"]]))
        print(f"    done in {time.time() - t0:.1f}s", flush=True)

        sh_out_dir = Path(out_dir) / sh["out_tracer"]
        sh_out_dir.mkdir(parents=True, exist_ok=True)
        out = sh_out_dir / f"{sh['stem']}_GPU_multipoles.npz"
        np.savez(out, s_bins=cc.S_BINS, s_mid=s_mid(cc.S_BINS),
                 xi2=xi2, cov_xi2=cov_xi2, cov_xi2_corrected=cov_xi2_corr, **d0,
                 cov_combined=cov_c, cov_combined_corrected=cov_cc, **meta)
        print(f"  saved {out}", flush=True)
        del shapes, shape_randoms, cat_s, p_s, p_s_pos, p_d, DS, RS, RDS
        gc.collect()

    del clustering, randoms, cat_r, p_r, DD, DR, RR
    gc.collect()


# =============================================================================
# CLI (the compute_correlations.py subset that applies to the multipoles)
# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(cc.CONFIG))
    p.add_argument("--out", default=None, help="output root (default <dest>/correlations_cucount)")
    p.add_argument("--tracer", default=None, help="only this tracer (BGS/LRG/ELG)")
    p.add_argument("--zmin", type=float, default=None)
    p.add_argument("--zmax", type=float, default=None)
    p.add_argument("--sample", default=None, help="substring match on the shape-sample stem")
    p.add_argument("--no-massbins", action="store_true")
    p.add_argument("--no-sysweights", action="store_true")
    p.add_argument("--sysweights", choices=["recomputed", "desi"], default=None)
    p.add_argument("--use-fkp", action="store_true")
    p.add_argument("--fkp", choices=["fkp", "none"], default=None)
    p.add_argument("--n-patches", type=int, default=None)
    p.add_argument("--kmeans-init", default=None, choices=["kmeans++", "kmeanspp", "random"])
    p.add_argument("--s-nbins", type=int, default=None)
    p.add_argument("--s-min", type=float, default=None)
    p.add_argument("--s-max", type=float, default=None)
    p.add_argument("--auto-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # same globals as compute_correlations.py, so its helpers see the same settings
    cc.USE_FKP = args.use_fkp if args.fkp is None else (args.fkp == "fkp")
    cc.NO_SYSWEIGHTS = (args.no_sysweights if args.sysweights is None
                        else args.sysweights == "desi")
    if args.n_patches is not None:
        cc.N_PATCHES = args.n_patches
    if args.kmeans_init is not None:
        cc.KMEANS_INIT = "kmeans++" if args.kmeans_init == "kmeanspp" else args.kmeans_init
    if args.s_nbins is not None:
        cc.S_BINS = np.geomspace(args.s_min if args.s_min else 2.0,
                                 args.s_max if args.s_max else 90.0, args.s_nbins + 1)
    elif args.s_min is not None or args.s_max is not None:
        raise SystemExit("--s-min/--s-max require --s-nbins")

    cfg = cc.load_config(args.config)
    out_root = Path(args.out) if args.out else Path(cfg["dest"]) / "correlations_cucount"
    jobs = cc.tracer_jobs(cfg, include_massbins=not args.no_massbins)
    if args.tracer:
        jobs = [j for j in jobs if j["tracer"] == args.tracer]
    if args.zmin is not None:
        jobs = [j for j in jobs if abs(j["z_range"][0] - args.zmin) < 1e-6]
    if args.zmax is not None:
        jobs = [j for j in jobs if abs(j["z_range"][1] - args.zmax) < 1e-6]
    if args.sample:
        for j in jobs:
            j["shapes"] = [sh for sh in j["shapes"] if args.sample in sh["stem"]]
        jobs = [j for j in jobs if j["shapes"]]

    print(f"engine: {ENGINE};  jackknife: N_PATCHES={cc.N_PATCHES}, kmeans_init={cc.KMEANS_INIT}; "
          f"s-bins={len(cc.S_BINS) - 1} over [{cc.S_BINS[0]:g},{cc.S_BINS[-1]:g}]; "
          f"FKP={cc.USE_FKP}; no_sysweights={cc.NO_SYSWEIGHTS}", flush=True)
    print(f"output root: {out_root}\ntracer jobs to process: {len(jobs)}", flush=True)
    for j in jobs:
        print(f"  - {j['base_stem']}  ({len(j['shapes'])} shape samples)", flush=True)
    if args.dry_run:
        return 0
    out_root.mkdir(parents=True, exist_ok=True)
    for j in jobs:
        process(j, out_root, auto_resume=args.auto_resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
