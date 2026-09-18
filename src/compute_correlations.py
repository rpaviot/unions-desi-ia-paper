#!/usr/bin/env python3
"""
Correlation-function estimation for the UNIONS x DESI IA samples.

This is the new-pipeline replacement for ``old_codes/compute_correlations.py``.
It consumes the analysis-ready IA samples written by ``build_ia_samples.py`` under
``<dest>/ia/``. Samples are organised per TRACER: one clustering + one randoms
catalogue define the density field, and the colour / mass shape samples are
cross-correlated against it. Per (tracer, z-bin) we compute the density-only
clustering ONCE; per shape sample we compute the density-shape cross:

  - GPU multipoles (cucount):  xi0  (clustering monopole, density-density) -- per tracer
                               xi2  (IA quadrupole,        density-shape)   -- per shape
  - CPU projected  (pycorr):   wp   (projected clustering) -- per tracer, optional
                               wgp  (projected IA, w_g+)    -- per shape,  optional

with deleted-jackknife covariances (standard + Mohammad+21 corrected) from the
vendored ``scripts/djk`` (corrd / jackknife / compute_pairs).

Layout + conventions (see ``build_ia_samples.py``):

  * Tracers + shape samples are enumerated from ``config/data_sources.yaml``
    (``ia_samples.tracers``). File names are ``<name>_zmin_{a:.2f}_zmax_{b:.2f}_{kind}.parquet``,
    with ``<name>`` the tracer for clustering/randoms and the shape sample for
    shapes/shape_randoms.
  * Density (clustering) data weight is ``WEIGHT_CORR`` when present -- BGS, where
    WEIGHT_CORR = WEIGHT * recomputed WEIGHT_SYS folds in the imaging systematics; LRG/ELG
    have no WEIGHT_CORR and keep DESI's official ``WEIGHT``. The randoms use ``WEIGHT``
    (for BGS already rebuilt with the recomputed WEIGHT_SYS, so data and randoms share the
    same systematics). With ``--use-fkp`` the density data and randoms are additionally
    multiplied by ``WEIGHT_FKP``; the shape side is never FKP-weighted. See density_weights().
  * The density field (cat1/rand1) is the tracer's single clustering + randoms; the
    shape field (cat2/rand2) is the shape sample's shapes + its own shape randoms.
  * Shape tracer weight  W2  = shapes.WEIGHT (lensing) * shapes.WEIGHT_CLUSTERING
    (the matched lens density weight); shape ellipticities g1/g2 = e1/e2.
    Shape-random weight W_r2 = shape_randoms.WEIGHT.
  * Jackknife patch centres are computed once per tracer from the RANDOMS catalogue
    (uniform footprint sampling -> equal-area patches) and cached to
    ``<dest>/ia/patch_centers/``; all of a tracer's shape samples share them.

Outputs: numpy ``.npz`` under ``<out>/<tracer>/`` plus n(z) and effective redshifts.
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.cosmology import FlatLambdaCDM

# Vendored deleted-jackknife correlation code (corrd / jackknife / compute_pairs).
sys.path.insert(0, str(Path(__file__).resolve().parent / "djk"))
from corrd import deleted_jackknife  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================
REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config" / "data_sources.yaml"

COSMO = FlatLambdaCDM(H0=100, Om0=0.3110)

# Jackknife
N_PATCHES = 70
N_DELETED = 1
# k-means init for the jackknife patch tessellation (treecorr Catalog kmeans_init).
# 'random' (plain k-means seeding) is the analysis default since 2026-08-20:
#   * 'tree' (treecorr's own default) is BROKEN on the BGS randoms -- it fails to fill
#     all 70 centre rows and leaves uninitialised memory behind, non-deterministically,
#     in ~89% of runs (16/18 measured). Never use it.
#   * 'kmeans++' (the previous default here) never failed, but tessellates BGS badly:
#     patch-count spread (max/min) 16.6 and 38.3 on two independent draws vs 4.6 for
#     'random', and its sparsest BGS_RED_GMM massbin10 patch held 26 galaxies vs 139.
# Overridable via --kmeans-init; pass 'kmeans++' to reproduce the pre-2026-08-20
# fiducial tessellation (which is the one cached under the bare filename, see
# get_patch_centers).
KMEANS_INIT = "random"

# GPU multipole binning (s, mu). 17 log s-bins over [2, 90] h^-1 Mpc.
S_BINS = np.geomspace(2, 90, 18)
MU_BINS = np.linspace(-1, 1, 201)
OVERSAMPLE_FACTOR = 1
RP_CUT = None

# Density-side FKP weighting (WEIGHT*WEIGHT_FKP on clustering data + randoms). Set from
# --use-fkp in main(); shapes are never FKP-weighted.
USE_FKP = False

# Null test for the home-made BGS imaging-systematics weights (--no-sysweights).
# BGS is the only tracer whose WEIGHT_SYS is recomputed here (scripts/sysweights.py);
# LRG/ELG keep DESI's official WEIGHT and are unaffected by this switch.
NO_SYSWEIGHTS = False

# Projected binning (rp, pi)
RP_BINS = np.geomspace(6, 130, 21)
PI_BINS = np.linspace(-100, 100, 21)

# Machine for the projected wp/wgp (djk supports both: pycorr/treecorr on CPU,
# cucount rp-pi on GPU). Set from --gpu-projected in main(); also names the
# output file (<stem>_{CPU,GPU}_projected.npz).
PROJ_MACHINE = "CPU"


# =============================================================================
# CONFIG / PATH HELPERS
# =============================================================================
def load_config(path=CONFIG):
    with open(path) as fh:
        return yaml.safe_load(fh)


def tracer_jobs(cfg, include_massbins=True):
    """One job per (tracer, z-bin). Each job carries the tracer's single clustering +
    randoms catalogue and jackknife patch-centre identity (shared across all its shape
    samples) plus the list of shape samples to cross-correlate against it -- each shape
    sample's full catalogue and, when requested, its per-mass-bin catalogues. Shape
    samples reuse the tracer density field; only the shape + shape-randoms files differ."""
    ia = cfg["ia_samples"]
    dest = Path(cfg["dest"])
    ia_dir = dest / ia["out_subdir"]
    jobs = []
    for t in ia["tracers"]:
        tracer = t["name"]
        # One jackknife tessellation per TRACER, pinned to its first z-bin's randoms and
        # shared across every z-bin of that tracer (so e.g. both LRG z-bins use the
        # 0.40-0.75 patch centres). The per-z-bin randoms still drive the correlations.
        zlo0, zhi0 = t["zbins"][0]
        pc_base = f"{tracer}_zmin_{zlo0:.2f}_zmax_{zhi0:.2f}"
        for zlo, zhi in t["zbins"]:
            ztag = f"zmin_{zlo:.2f}_zmax_{zhi:.2f}"
            base = f"{tracer}_{ztag}"             # density / clustering identity
            shapes = []
            for s in t["shape_samples"]:
                sbase = f"{s['name']}_{ztag}"
                shapes.append({
                    "stem": sbase, "out_tracer": s["name"].split("_")[0],
                    "shape_file": ia_dir / f"{sbase}_shapes.parquet",
                    "shape_randoms_file": ia_dir / f"{sbase}_shape_randoms.parquet",
                })
                if include_massbins and s.get("n_mass_bins"):
                    for i in range(int(s["n_mass_bins"])):
                        shapes.append({
                            "stem": f"{sbase}_massbin{i}", "out_tracer": s["name"].split("_")[0],
                            "shape_file": ia_dir / f"{sbase}_shapes_massbin{i}.parquet",
                            "shape_randoms_file": ia_dir / f"{sbase}_shape_randoms_massbin{i}.parquet",
                        })
            jobs.append({
                "tracer": tracer, "z_range": (zlo, zhi), "ia_dir": ia_dir, "base_stem": base,
                "clustering_file": ia_dir / f"{base}_clustering.parquet",
                "randoms_file": ia_dir / f"{base}_randoms.parquet",
                # jackknife identity: shared per tracer, computed from the first z-bin's randoms
                "pc_stem": pc_base,
                "pc_randoms_file": ia_dir / f"{pc_base}_randoms.parquet",
                "shapes": shapes,
            })
    return jobs


def shape_outputs(sh, out_dir, with_projected, projected_only=False):
    """Output .npz paths a shape sample produces -- used by --auto-resume to decide
    whether the (tracer, z-bin, shape) cell is already done."""
    sh_out_dir = Path(out_dir) / sh["out_tracer"]
    outs = []
    if not projected_only:
        outs.append(sh_out_dir / f"{sh['stem']}_GPU_multipoles.npz")
    if with_projected or projected_only:
        outs.append(sh_out_dir / f"{sh['stem']}_{PROJ_MACHINE}_projected.npz")
    return outs


def load_parquet(path):
    print(f"  loading {Path(path).name}", flush=True)
    df = pd.read_parquet(path)
    print(f"    {len(df):,} rows", flush=True)
    return df


def comoving(z):
    return COSMO.comoving_distance(np.asarray(z)).value


# =============================================================================
# JACKKNIFE PATCH CENTRES (computed once per sample, cached)
# =============================================================================
def get_patch_centers(job, n_patches=None):
    """Load cached patch centres or compute them from the tracer RANDOMS catalogue.

    One tessellation per TRACER, pinned to the tracer's first z-bin randoms (``pc_stem`` /
    ``pc_randoms_file``) and shared across all of that tracer's z-bins, colour and mass
    shape samples -- so e.g. both LRG z-bins reuse the 0.40-0.75 regions. Centres are
    k-means'd on the *randoms*, not the data: the randoms sample the footprint uniformly
    (no clustering, no shot-noise holes), so they tessellate the survey AREA evenly --
    the right basis for a deleted-jackknife -- and avoid the tiny / dead patches that
    the clustered data produces on the fragmented DESI footprint. ``kmeans_init``
    defaults to ``'random'`` (see KMEANS_INIT): treecorr's 'tree' init leaves centre
    rows uninitialised on the BGS randoms, and 'kmeans++' produces badly unbalanced
    patches. The full randoms are used (k-means is ~10-45 s even at ~30M)."""
    import treecorr

    if n_patches is None:
        n_patches = N_PATCHES
    pc_dir = job["ia_dir"] / "patch_centers"
    pc_dir.mkdir(parents=True, exist_ok=True)
    # The bare filename is reserved for the pre-2026-08-20 fiducial tessellation
    # (70, kmeans++) so those cached centres stay reachable via --kmeans-init kmeans++
    # even though 'random' is now the default; every other (n_patches, init) -- the
    # current default included -- gets its own cache and can never clobber it.
    suffix = "" if (n_patches == 70 and KMEANS_INIT == "kmeans++") \
        else f"_np{n_patches}_{KMEANS_INIT}"
    pc_file = pc_dir / f"{job['pc_stem']}{suffix}_patch_centers.txt"

    if pc_file.exists():
        centers = np.loadtxt(pc_file)
        print(f"  patch centres: loaded {len(centers)} from {pc_file.name}", flush=True)
        return centers

    print(f"  patch centres: computing {n_patches} ({KMEANS_INIT}) from {Path(job['pc_randoms_file']).name}", flush=True)
    df = pd.read_parquet(job["pc_randoms_file"], columns=["RA", "DEC"])
    cat = treecorr.Catalog(ra=df["RA"].values, dec=df["DEC"].values, ra_units="degree",
                           dec_units="degree", npatch=n_patches, kmeans_init=KMEANS_INIT)
    centers = cat.patch_centers
    # sanity: every patch must own randoms (i.e. real footprint area)
    _, counts = np.unique(cat.patch, return_counts=True)
    if len(counts) != n_patches or counts.min() == 0:
        raise RuntimeError(f"empty jackknife patch for {job['pc_stem']} "
                           f"({len(counts)}/{n_patches} populated)")
    np.savetxt(pc_file, centers)
    print(f"    saved -> {pc_file}", flush=True)
    return centers


# =============================================================================
# n(z) AND EFFECTIVE REDSHIFT
# =============================================================================
def effective_redshift(z, w):
    return float(np.sum(np.asarray(z) * np.asarray(w)) / np.sum(w))


def compute_nz(clustering, shapes, w_clust, w_shape, delta_z=0.005):
    """Normalised n(z) for the density and shape tracers (integral = 1)."""
    zmin = min(clustering["Z"].min(), shapes["Z"].min())
    zmax = max(clustering["Z"].max(), shapes["Z"].max())
    edges = np.arange(zmin, zmax + delta_z, delta_z)
    centers = 0.5 * (edges[:-1] + edges[1:])

    nz_c, _ = np.histogram(clustering["Z"], bins=edges, weights=w_clust)
    nz_c = nz_c.astype(float)
    nz_c /= (nz_c.sum() * delta_z)

    nz_s, _ = np.histogram(shapes["Z"], bins=edges, weights=w_shape)
    nz_s = nz_s.astype(float)
    nz_s /= (nz_s.sum() * delta_z)

    return {"z_bins": edges, "z_nz": centers, "nz_clustering": nz_c,
            "nz_shape": nz_s, "delta_z": delta_z}


def mass_stats(shapes, mass_column="LOGMSTAR"):
    """Mean / spread of LOGMSTAR for the shape sample (metadata only)."""
    if mass_column not in shapes.columns:
        return {}
    v = np.asarray(shapes[mass_column], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    return {"mass_column": mass_column, "mass_mean": v.mean(), "mass_min": v.min(),
            "mass_max": v.max(), "mass_std": v.std(), "mass_median": np.median(v)}


# =============================================================================
# COMMON INPUT DICT FOR corrd
# =============================================================================
def density_weights(clustering, randoms):
    """Density-field weights for cat1/rand1 (the clustering tracer).

    Data weight is ``WEIGHT_CORR`` when present -- BGS, where WEIGHT_CORR = WEIGHT *
    recomputed WEIGHT_SYS folds in the imaging-systematics correction (LRG/ELG have no
    WEIGHT_CORR and keep DESI's official WEIGHT). The randoms keep ``WEIGHT`` (for BGS it
    is already rebuilt with the recomputed WEIGHT_SYS, so data WEIGHT_CORR and random
    WEIGHT carry the same systematics). With USE_FKP both data and randoms are additionally
    multiplied by ``WEIGHT_FKP`` (FKP weighting); the shape side is never FKP-weighted."""
    wcol = "WEIGHT_CORR" if "WEIGHT_CORR" in clustering.columns else "WEIGHT"
    if NO_SYSWEIGHTS:
        wcol = "WEIGHT"          # drop the recomputed WEIGHT_SYS on the data side
    w_clust = clustering[wcol].to_numpy().astype(float)
    w_rand = randoms["WEIGHT"].to_numpy().astype(float)
    rand_note = ""
    if NO_SYSWEIGHTS and "WEIGHT_SYS" in randoms.columns:
        # the randoms' WEIGHT was rebuilt as COMP*ZFAIL*SYS/<COMP>_NTILE, so divide the
        # shuffled SYS factor back out to keep the data/random weight sums matched
        w_rand = w_rand / randoms["WEIGHT_SYS"].to_numpy()
        rand_note = " / WEIGHT_SYS"
    if USE_FKP:
        for name, df in (("clustering", clustering), ("randoms", randoms)):
            if "WEIGHT_FKP" not in df.columns:
                raise KeyError(f"--use-fkp set but {name} has no WEIGHT_FKP column; "
                               "rebuild the IA samples with the FKP-propagation build")
        w_clust = w_clust * clustering["WEIGHT_FKP"].to_numpy()
        w_rand = w_rand * randoms["WEIGHT_FKP"].to_numpy()
    print(f"  density weights: data={wcol}{' * WEIGHT_FKP' if USE_FKP else ''}, "
          f"randoms=WEIGHT{rand_note}{' * WEIGHT_FKP' if USE_FKP else ''}"
          f"{'   [NO_SYSWEIGHTS null test]' if NO_SYSWEIGHTS else ''}", flush=True)
    return w_clust, w_rand


def _base_inputs(clustering, randoms, w_clust, w_rand, computation, binsfile,
                 patch_centers, thetamax):
    """Density-density auto inputs (cat1 == cat2 == clustering)."""
    n = len(clustering)
    return {
        "cosmology": COSMO, "Ns": N_PATCHES, "Nd": N_DELETED,
        "computation": computation, "binsfile": binsfile,
        "RA": clustering["RA"].values, "DEC": clustering["DEC"].values,
        "Z": clustering["Z"].values, "W": w_clust,
        "RA2": clustering["RA"].values, "DEC2": clustering["DEC"].values,
        "Z2": clustering["Z"].values, "W2": w_clust,
        "g12": np.zeros(n), "g22": np.zeros(n),
        "RA_r": randoms["RA"].values, "DEC_r": randoms["DEC"].values,
        "Z_r": randoms["Z"].values, "W_r": w_rand,
        "RA_r2": randoms["RA"].values, "DEC_r2": randoms["DEC"].values,
        "Z_r2": randoms["Z"].values, "W_r2": w_rand,
        "compute_RR": True, "patch_centers": patch_centers, "thetamax": thetamax,
    }


def _cross_inputs(clustering, randoms, shapes, shape_randoms,
                  w_clust, w_rand, w_shape, w_shape_rand,
                  computation, binsfile, patch_centers, thetamax):
    """Density-shape cross inputs: cat1/rand1 = the tracer's clustering + clustering
    randoms (density), cat2/rand2 = the shape sample's shapes + shape randoms."""
    return {
        "cosmology": COSMO, "Ns": N_PATCHES, "Nd": N_DELETED,
        "computation": computation, "binsfile": binsfile,
        "RA": clustering["RA"].values, "DEC": clustering["DEC"].values,
        "Z": clustering["Z"].values, "W": w_clust,
        "RA2": shapes["RA"].values, "DEC2": shapes["DEC"].values,
        "Z2": shapes["Z"].values, "W2": w_shape,
        "g12": shapes["e1"].values, "g22": shapes["e2"].values,
        "RA_r": randoms["RA"].values, "DEC_r": randoms["DEC"].values,
        "Z_r": randoms["Z"].values, "W_r": w_rand,
        "RA_r2": shape_randoms["RA"].values, "DEC_r2": shape_randoms["DEC"].values,
        "Z_r2": shape_randoms["Z"].values, "W_r2": w_shape_rand,
        "compute_RR": False, "patch_centers": patch_centers, "thetamax": thetamax,
    }


# =============================================================================
# GPU MULTIPOLES (xi0 clustering, xi2 IA)
# =============================================================================
# xi0 (clustering monopole, density-density auto) depends only on the tracer's density
# field, so it is computed ONCE per tracer job and reused across all of its shape
# samples. xi2 (IA quadrupole, density-shape cross) is computed per shape sample.
def compute_xi0(clustering, randoms, w_clust, w_rand, patch_centers, thetamax):
    """Clustering monopole (density auto). Returns (corr0, results-dict)."""
    print("\n[xi0] clustering monopole (ell=0)...", flush=True)
    t0 = time.time()
    ic = _base_inputs(clustering, randoms, w_clust, w_rand, "xil", [S_BINS, MU_BINS],
                      patch_centers, thetamax)
    ic.update(machine="GPU", oversample_factor=OVERSAMPLE_FACTOR, correction_cov=False)
    corr0 = deleted_jackknife(**ic)
    s0, xi0, cov_xi0 = corr0.get_measurements(rp_cut=RP_CUT, correction_cov=False)
    _, _, cov_xi0_corr = corr0.get_measurements(rp_cut=RP_CUT, correction_cov=True)
    print(f"    done in {time.time() - t0:.1f}s", flush=True)
    return corr0, {"s_mid": s0, "xi0": xi0, "cov_xi0": cov_xi0,
                   "cov_xi0_corrected": cov_xi0_corr}


def shape_weights(shapes, shape_randoms):
    """Shape-field weights (cat2/rand2): W2 = WEIGHT (UNIONS lensing) x WEIGHT_CLUSTERING
    (the matched lens density weight), W_r2 = shape_randoms.WEIGHT. Never FKP.

    With NO_SYSWEIGHTS the recomputed imaging-systematics factor is divided back out of
    WEIGHT_CLUSTERING. That is exact: WEIGHT_CLUSTERING = C(NTILE) x WEIGHT_COMP x
    WEIGHT_ZFAIL x WEIGHT_SYS, with C constant to ~1e-3 within each NTILE group
    (verified on BGS 2026-08-22), so the division removes WEIGHT_SYS and nothing else.

    The shape RANDOMS are left untouched, deliberately: create_clustering_randoms_desi
    shuffles WEIGHT_SYS from the data onto random positions (sysweights.py,
    ``out['WEIGHT_SYS'] = random_z['WEIGHT_SYS'].values``), so their weights carry the
    1-D distribution of WEIGHT_SYS but NONE of its spatial pattern -- there is no
    systematics correction on that side to undo, only a ~0.3% constant (mean
    WEIGHT_SYS = 0.997) that the estimator's weight-sum normalisation absorbs."""
    w_shape = (shapes["WEIGHT"] * shapes["WEIGHT_CLUSTERING"]).values
    if NO_SYSWEIGHTS:
        if "WEIGHT_SYS" not in shapes.columns:
            raise KeyError("--no-sysweights: shapes have no WEIGHT_SYS to divide out")
        w_shape = w_shape / shapes["WEIGHT_SYS"].to_numpy()
    return w_shape, shape_randoms["WEIGHT"].values


def compute_xi2(clustering, randoms, shapes, shape_randoms, w_clust, w_rand,
                patch_centers, thetamax):
    """IA quadrupole (density-shape cross). Returns (corr2, results-dict)."""
    print("[xi2] IA quadrupole (ell=2)...", flush=True)
    t0 = time.time()
    w_shape, w_shape_rand = shape_weights(shapes, shape_randoms)
    iq = _cross_inputs(clustering, randoms, shapes, shape_randoms, w_clust, w_rand,
                       w_shape, w_shape_rand, "xi2p", [S_BINS, MU_BINS], patch_centers, thetamax)
    iq.update(machine="GPU", oversample_factor=OVERSAMPLE_FACTOR, correction_cov=False)
    corr2 = deleted_jackknife(**iq)
    s2, xi2, cov_xi2 = corr2.get_measurements(rp_cut=RP_CUT, correction_cov=False)
    _, _, cov_xi2_corr = corr2.get_measurements(rp_cut=RP_CUT, correction_cov=True)
    print(f"    done in {time.time() - t0:.1f}s", flush=True)
    return corr2, {"s_mid": s2, "xi2": xi2, "cov_xi2": cov_xi2,
                   "cov_xi2_corrected": cov_xi2_corr}


def combine_cov(corr_a, corr_b):
    """Joint covariance of two jackknife measurements (standard + Mohammad+21)."""
    corr_a.get_measurements(rp_cut=RP_CUT, correction_cov=False)
    corr_b.get_measurements(rp_cut=RP_CUT, correction_cov=False)
    cov = corr_a.combine_measurements(corr_b)
    corr_a.get_measurements(rp_cut=RP_CUT, correction_cov=True)
    corr_b.get_measurements(rp_cut=RP_CUT, correction_cov=True)
    cov_corr = corr_a.combine_measurements(corr_b)
    return cov, cov_corr


# =============================================================================
# CPU PROJECTED (wp clustering, wgp IA)
# =============================================================================
def compute_wp(clustering, randoms, w_clust, w_rand, patch_centers, thetamax):
    """Projected clustering wp (density auto). Computed once per tracer job."""
    print("\n[wp] projected clustering (WGG)...", flush=True)
    t0 = time.time()
    ig = _base_inputs(clustering, randoms, w_clust, w_rand, "WGG", [RP_BINS, PI_BINS],
                      patch_centers, thetamax)
    ig.update(machine=PROJ_MACHINE, correction_cov=False)
    corr_g = deleted_jackknife(**ig)
    rp_g, wp, cov_wp = corr_g.get_measurements(correction_cov=False)
    _, _, cov_wp_corr = corr_g.get_measurements(correction_cov=True)
    print(f"    done in {time.time() - t0:.1f}s", flush=True)
    return corr_g, {"rp_mid": rp_g, "wp": wp, "cov_wp": cov_wp,
                    "cov_wp_corrected": cov_wp_corr}


def compute_wgp(clustering, randoms, shapes, shape_randoms, w_clust, w_rand,
                patch_centers, thetamax):
    """Projected IA wg+ (density-shape cross). Computed per shape sample."""
    print("[wgp] projected IA (WGP)...", flush=True)
    t0 = time.time()
    w_shape, w_shape_rand = shape_weights(shapes, shape_randoms)
    ip = _cross_inputs(clustering, randoms, shapes, shape_randoms, w_clust, w_rand,
                       w_shape, w_shape_rand, "WGP", [RP_BINS, PI_BINS], patch_centers, thetamax)
    ip.update(machine=PROJ_MACHINE, correction_cov=False)
    corr_p = deleted_jackknife(**ip)
    rp_p, wgp, cov_wgp = corr_p.get_measurements(correction_cov=False)
    _, _, cov_wgp_corr = corr_p.get_measurements(correction_cov=True)
    print(f"    done in {time.time() - t0:.1f}s", flush=True)
    return corr_p, {"rp_mid": rp_p, "wgp": wgp, "cov_wgp": cov_wgp,
                    "cov_wgp_corrected": cov_wgp_corr}


# =============================================================================
# PER-TRACER DRIVER
# =============================================================================
def process(job, out_dir, with_projected=False, ia_only=False, auto_resume=False,
            projected_only=False):
    """Process one tracer job: load the density field once (clustering + randoms +
    patch centres), compute the clustering xi0/wp once, then cross-correlate every
    shape sample (and mass bin) against it for xi2/wgp.

    With ``auto_resume`` the shape samples whose output .npz already exist are skipped;
    if a tracer job has nothing left to do, it returns before loading any catalogue."""
    tracer, (zlo, zhi) = job["tracer"], job["z_range"]

    # projected-only is a CPU path (wp/wgp via pycorr); it never touches the GPU
    # multipoles, so it can run on a plain CPU node. It implies with_projected.
    if projected_only:
        with_projected = True

    shapes_todo = job["shapes"]
    if auto_resume:
        shapes_todo = [sh for sh in job["shapes"]
                       if not all(o.exists()
                                  for o in shape_outputs(sh, out_dir, with_projected,
                                                          projected_only))]
        n_done = len(job["shapes"]) - len(shapes_todo)
        if n_done:
            print(f"\n[auto-resume] {tracer} z=[{zlo:.2f}, {zhi:.2f}]: "
                  f"{n_done}/{len(job['shapes'])} shape outputs already present", flush=True)
        if not shapes_todo:
            print(f"[auto-resume] {tracer} z=[{zlo:.2f}, {zhi:.2f}]: nothing left, skipping",
                  flush=True)
            return

    print("\n" + "#" * 70, flush=True)
    print(f"# {tracer}  z=[{zlo:.2f}, {zhi:.2f}]   shapes={len(shapes_todo)}  "
          f"projected={with_projected}", flush=True)
    print("#" * 70, flush=True)

    for kind in ("clustering_file", "randoms_file"):
        if not Path(job[kind]).exists():
            raise FileNotFoundError(job[kind])

    # Patch centres first: building the treecorr catalogue from the ~30M randoms is the
    # most memory-hungry step, so compute (or load) it before the clustering/randoms
    # catalogues are resident -- otherwise peak RSS = clustering + randoms + treecorr and
    # the node OOM-kills the k-means.
    patch_centers = get_patch_centers(job)

    clustering = load_parquet(job["clustering_file"])
    randoms = load_parquet(job["randoms_file"])

    thetamax = float(S_BINS[-1] / comoving(clustering["Z"].min()))
    print(f"\n  thetamax = {thetamax:.6f} rad", flush=True)

    # density-field weights (WEIGHT_CORR for BGS, optional WEIGHT_FKP), shared by the
    # clustering auto and every shape cross; also drive z_eff and n(z) for consistency.
    w_clust, w_rand = density_weights(clustering, randoms)

    z_eff_clustering = effective_redshift(clustering["Z"], w_clust)
    print(f"  z_eff (clustering) = {z_eff_clustering:.4f}", flush=True)

    # --- density-only correlations, computed ONCE per tracer ---
    corr0 = d0 = corr_g = dwp = None
    if not ia_only and not projected_only:
        corr0, d0 = compute_xi0(clustering, randoms, w_clust, w_rand, patch_centers, thetamax)
    if with_projected and not ia_only:
        corr_g, dwp = compute_wp(clustering, randoms, w_clust, w_rand, patch_centers, thetamax)

    # --- per shape sample: cross-correlate against the shared density field ---
    for sh in shapes_todo:
        for f in ("shape_file", "shape_randoms_file"):
            if not Path(sh[f]).exists():
                raise FileNotFoundError(sh[f])
        print("\n" + "-" * 70 + f"\n  shape sample: {sh['stem']}\n" + "-" * 70, flush=True)
        shapes = load_parquet(sh["shape_file"])
        shape_randoms = load_parquet(sh["shape_randoms_file"])

        # drop shapes with non-finite ellipticities (build cleaned these; belt+braces)
        good = np.isfinite(shapes["e1"]) & np.isfinite(shapes["e2"])
        if not good.all():
            print(f"  dropping {(~good).sum()} shapes with non-finite e1/e2", flush=True)
            shapes = shapes[good].reset_index(drop=True)

        meta = {
            "z_eff_clustering": z_eff_clustering,
            "z_eff_IA": effective_redshift(shapes["Z"], shapes["WEIGHT"]),
            "n_clustering": len(clustering), "n_shapes": len(shapes),
            "n_randoms": len(randoms), "n_shape_randoms": len(shape_randoms),
        }
        meta.update(compute_nz(clustering, shapes, w_clust, shapes["WEIGHT"].values))
        meta["use_fkp"] = USE_FKP
        # Jackknife tessellation identity. Re-drawing the 70 patches is NOT a null
        # operation -- it moved the ELG IA amplitude from -3.7 to -1.0 sigma on
        # identical data -- so the (n_patches, kmeans_init) that produced this
        # covariance is recorded in the file itself, not only in the run log.
        meta["n_patches"] = N_PATCHES
        meta["n_deleted"] = N_DELETED
        meta["kmeans_init"] = KMEANS_INIT
        meta["no_sysweights"] = NO_SYSWEIGHTS
        meta.update(mass_stats(shapes))
        print(f"  z_eff (IA) = {meta['z_eff_IA']:.4f}", flush=True)

        sh_out_dir = Path(out_dir) / sh["out_tracer"]
        sh_out_dir.mkdir(parents=True, exist_ok=True)

        # GPU multipoles: xi2 (+ xi0 reused from the tracer). Skipped entirely for
        # --projected-only so the run needs no GPU and fits on a plain CPU node.
        corr2 = None
        if not projected_only:
            corr2, d2 = compute_xi2(clustering, randoms, shapes, shape_randoms,
                                    w_clust, w_rand, patch_centers, thetamax)
            gpu = {"s_bins": S_BINS, **d2}
            if not ia_only:
                gpu.update(d0)
                cov_c, cov_cc = combine_cov(corr0, corr2)
                gpu.update(cov_combined=cov_c, cov_combined_corrected=cov_cc)
            gpu_out = sh_out_dir / f"{sh['stem']}_GPU_multipoles.npz"
            np.savez(gpu_out, **gpu, **meta)
            print(f"  saved {gpu_out}", flush=True)

        # CPU projected: wgp (+ wp reused from the tracer)
        corr_p = None
        if with_projected:
            corr_p, dwgp = compute_wgp(clustering, randoms, shapes, shape_randoms,
                                       w_clust, w_rand, patch_centers, thetamax)
            cpu = {"rp_bins": RP_BINS, **dwgp}
            if not ia_only:
                cpu.update(dwp)
                cov_c, cov_cc = combine_cov(corr_g, corr_p)
                cpu.update(cov_combined=cov_c, cov_combined_corrected=cov_cc)
            cpu_out = sh_out_dir / f"{sh['stem']}_{PROJ_MACHINE}_projected.npz"
            np.savez(cpu_out, **cpu, **meta)
            print(f"  saved {cpu_out}", flush=True)

        # free this shape sample's parquets + correlation objects before the next one
        del shapes, shape_randoms, corr2, corr_p
        gc.collect()

    # free the tracer density field before moving to the next job
    del clustering, randoms, corr0, corr_g
    gc.collect()


# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Compute UNIONS x DESI IA correlations")
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--out", default=None,
                   help="output root (default: <dest>/correlations)")
    p.add_argument("--tracer", default=None, help="only this tracer (e.g. BGS, LRG, ELG)")
    p.add_argument("--zmin", type=float, default=None)
    p.add_argument("--zmax", type=float, default=None)
    p.add_argument("--with-projected", action="store_true",
                   help="also compute projected wp/wgp")
    p.add_argument("--gpu-projected", action="store_true",
                   help="compute the projected wp/wgp on the GPU (cucount rp-pi) "
                        "instead of CPU pycorr/treecorr")
    p.add_argument("--sample", default=None,
                   help="substring match on the shape-sample stem (e.g. BLUE, "
                        "RED_GMM); the tracer's xi0/wp is still computed")
    p.add_argument("--ia-only", action="store_true",
                   help="compute only the IA quadrupole xi2 (skip clustering xi0)")
    p.add_argument("--projected-only", action="store_true",
                   help="compute only the CPU projected wp/wgp (skip the GPU "
                        "multipoles xi0/xi2); runs on a plain CPU node. Implies "
                        "--with-projected.")
    p.add_argument("--no-sysweights", action="store_true",
                   help="NULL TEST: strip the recomputed imaging-systematics weight. "
                        "Density data uses WEIGHT instead of WEIGHT_CORR, density "
                        "randoms are divided by WEIGHT_SYS, and the shape weight uses "
                        "WEIGHT_CLUSTERING/WEIGHT_SYS. Only BGS is affected (LRG/ELG "
                        "have no recomputed WEIGHT_SYS). Write to a SEPARATE --out tree")
    p.add_argument("--use-fkp", action="store_true",
                   help="apply FKP weights (WEIGHT*WEIGHT_FKP) to the density data + "
                        "randoms; the shape side is never FKP-weighted")
    p.add_argument("--no-massbins", action="store_true",
                   help="full shape samples only (skip per-mass-bin shape samples)")
    p.add_argument("--n-patches", type=int, default=None,
                   help="override the jackknife patch count (default 70); also drives "
                        "the Mohammad+21 cross-pair down-weighting (Ns)")
    p.add_argument("--kmeans-init", default=None, choices=["tree", "kmeans++", "random"],
                   help="treecorr k-means init for the patch tessellation (default "
                        "random); 'kmeans++' reproduces the pre-2026-08-20 fiducial "
                        "tessellation, 'tree' is treecorr's own default and is broken "
                        "on the BGS randoms -- do not use it")
    p.add_argument("--s-nbins", type=int, default=None,
                   help="override the number of log s-bins for the multipoles over "
                        "[--s-min, --s-max] (default 17 over [2,90]); finer binning "
                        "-> more fit points")
    p.add_argument("--s-min", type=float, default=None,
                   help="lower edge of the multipole s-grid [h^-1 Mpc] (default 2). "
                        "Only takes effect together with --s-nbins.")
    p.add_argument("--s-max", type=float, default=None,
                   help="upper edge of the multipole s-grid [h^-1 Mpc] (default 90). "
                        "Only takes effect together with --s-nbins. NOTE: the "
                        "jackknife patches are ~5.9 deg across (70 patches over "
                        "2397 deg^2), which at the BGS z_eff=0.35 corresponds to "
                        "~99 h^-1 Mpc -- beyond that the JK covariance is being "
                        "extrapolated past the patch scale even with the "
                        "Mohammad+21 cross-pair correction. LRG/ELG are safe to "
                        "~150 h^-1 Mpc.")
    p.add_argument("--rp-match-s", action="store_true",
                   help="use the multipole s-grid as the projected rp-grid (RP_BINS = "
                        "S_BINS, i.e. geomspace(2,90,18)) instead of the default "
                        "geomspace(6,130,21); pi-binning is unchanged")
    p.add_argument("--rp-cut", type=float, default=None,
                   help="transverse-separation cut [h^-1 Mpc] applied to the multipole "
                        "measurement (default none); the fit must mirror it with "
                        "fit_ia --rp-min-wedge")
    p.add_argument("--oversample", type=int, default=None,
                   help="cucount GPU oversample factor for the multipole pair counts "
                        "(default 1); higher = more accurate, slower")
    p.add_argument("--auto-resume", action="store_true",
                   help="skip shape samples whose output .npz already exist, so a "
                        "re-run picks up where a killed/partial run left off")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    global USE_FKP, PROJ_MACHINE, N_PATCHES, KMEANS_INIT, S_BINS, RP_BINS, RP_CUT, OVERSAMPLE_FACTOR
    global NO_SYSWEIGHTS
    USE_FKP = args.use_fkp
    NO_SYSWEIGHTS = args.no_sysweights
    if args.gpu_projected:
        PROJ_MACHINE = "GPU"
    if args.n_patches is not None:
        N_PATCHES = args.n_patches
    if args.kmeans_init is not None:
        KMEANS_INIT = args.kmeans_init
    if args.s_nbins is not None:
        S_BINS = np.geomspace(args.s_min if args.s_min else 2.0,
                              args.s_max if args.s_max else 90.0,
                              args.s_nbins + 1)
    elif args.s_min is not None or args.s_max is not None:
        raise SystemExit("--s-min/--s-max require --s-nbins")
    if args.rp_match_s:
        RP_BINS = S_BINS  # projected rp-grid = the multipole s-grid (resolved after --s-nbins)
    if args.rp_cut is not None:
        RP_CUT = args.rp_cut
    if args.oversample is not None:
        OVERSAMPLE_FACTOR = args.oversample
    print(f"jackknife: N_PATCHES={N_PATCHES}, kmeans_init={KMEANS_INIT}; "
          f"s-bins={len(S_BINS) - 1} over [{S_BINS[0]:.0f},{S_BINS[-1]:.0f}]; "
          f"rp_cut={RP_CUT}, oversample={OVERSAMPLE_FACTOR}", flush=True)
    print(f"projected: rp-bins={len(RP_BINS) - 1} over [{RP_BINS[0]:.0f},{RP_BINS[-1]:.0f}], "
          f"pi-bins={len(PI_BINS) - 1} over [{PI_BINS[0]:.0f},{PI_BINS[-1]:.0f}] "
          f"(dpi={(PI_BINS[1] - PI_BINS[0]):.0f}), machine={PROJ_MACHINE}", flush=True)

    cfg = load_config(args.config)
    out_root = Path(args.out) if args.out else Path(cfg["dest"]) / "correlations"
    jobs = tracer_jobs(cfg, include_massbins=not args.no_massbins)

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

    print(f"output root: {out_root}", flush=True)
    print(f"tracer jobs to process: {len(jobs)}", flush=True)
    for j in jobs:
        print(f"  - {j['base_stem']}  ({len(j['shapes'])} shape samples)", flush=True)
    if args.dry_run or not jobs:
        return

    t0 = time.time()
    for j in jobs:
        try:
            process(j, out_root, with_projected=args.with_projected, ia_only=args.ia_only,
                    auto_resume=args.auto_resume, projected_only=args.projected_only)
        except Exception as exc:  # keep going across tracers
            print(f"\nERROR on {j['tracer']} z={j['z_range']}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
