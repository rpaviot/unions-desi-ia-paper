#!/usr/bin/env python3
"""Gaussian covariance for the IA multipoles (xi0 clustering, xi2 IA), following
Grieb et al. 2016 (arXiv:1509.04293), generalized to the spin-2 IA quadrupole.

The survey-independent core (bin-averaged Bessel kernels, density integrals,
per-mode variance and the covariance blocks) lives in ``ia2pt.gausscov``; this
script is the UNIONS x DESI driver: sample enumeration from the config, the
catalogue reads for the n(z) / weights / shape noise, the b1 fit, the outputs.

For every shape sample measured by compute_correlations.py this predicts the
Gaussian covariance of the data vector [xi0, xi~_{2,2}] on the SAME s bins, for
cross-checking / replacing the deleted-jackknife covariance.

Formalism
---------
Grieb et al. eq. (16)+(18): the 2PCF multipole covariance is

    C_{l1,l2}(s_i, s_j) = i^{l1+l2}/(2 pi^2) Int dk k^2 sigma2_{l1,l2}(k)
                          jbar_{l1}(k s_i) jbar_{l2}(k s_j),

with jbar_l the VOLUME-averaged spherical Bessel function over the s bin
(eq. 19: jbar_l = 4 pi / V_bin * Int s^2 j_l(ks) ds; closed forms below) and
sigma2 the per-mode power spectrum variance. We write sigma2 uniformly as

    sigma2_{l1,l2}(k) = 1/2 Int_{-1}^{1} dmu c_{l1}(mu) c_{l2}(mu) Q(k, mu),

where c_l(mu) is the per-side estimator kernel and Q = (per-mode covariance)/V:

  * clustering monopole (scalar Legendre, ell=0):  c_0(mu) = 1
        per-mode cov = 2 [P_gg(k,mu) + 1/nbar_g]^2          (Grieb eq. 15)
  * IA quadrupole. compute_correlations measures the ASSOCIATED-Legendre spin-2
    multipole xi~_{2,2} = (5/2)(1/24) Int dmu 3(1-mu^2) xi_g+(s,mu). Expanding
    the plane wave against the cos(2 phi) shear projection gives the per-side
    kernel  c_2(mu) = (5/24) P_2^2(mu) = (5/8)(1-mu^2)  with jbar_2, and
        per-mode cov = [P_gg + 1/nbar_g][P_EE + sigma_gamma^2/nbar_s] + P_gE^2
    (cross-spectrum variance; sigma_gamma = per-component ellipticity dispersion
    weighted like the estimator). In the scalar limit this machinery reproduces
    Grieb eq. 15 exactly.
  * xi0 x xi2 cross block: per-mode cov = 2 [P_gg + 1/nbar_g] P_gE, kernels
    c_0 x c_2, prefactor i^2 = -1. (The fits zero this block by convention;
    it is stored for completeness and its SIGN is validated against the
    jackknife cross block.)

Varying number density: instead of picking a constant nbar and an effective
volume, the FKP-style local approximation is integrated over the survey
(reduces exactly to Grieb's [P + 1/nbar]^2 / V_s for constant nbar).
Density weights MIRROR THE ESTIMATOR: WEIGHT_CORR/WEIGHT, additionally
multiplied by WEIGHT_FKP when the measurement npz records use_fkp=True
(read per sample -- the shape side is never FKP-weighted, as in
compute_correlations.density_weights):

    Q_gg(k,mu) = Int dV [ nbar_w^2(z) (P_gg + 1/nbar_eff(z)) ]^2 / [Int dV nbar_w^2]^2

with nbar_w = weighted density (Sum w / V_shell) and the SHOT-NOISE density
nbar_eff = (Sum w)^2 / Sum w^2 / V_shell (weighted counts raise the shot noise).
dV = A_sr chi^2(z) dchi uses the converged footprint area measured by
compute_footprint_area.py (density-ratio estimator on the full randoms).
V_eff is only reported as a diagnostic. Pure white-noise (P-independent) terms
are integrated ANALYTICALLY (Int k^2 jbar_l jbar_l dk = 2 pi^2 delta_ij / V_bin)
so they carry no k-truncation error.

P(k,mu) model: pyccl nonlinear P_dd (camb/mead2020, the ia_model.py setup) at the
block's effective redshift, Kaiser RSD (b1 + f mu^2)^2 on the density side. b1 is
fit internally to the measured xi0 over --bfit-range (jackknife-diag weighted;
the Kaiser monopole is a single template so b has a closed form) unless --b1 is
given. NLA P_gE = (b1 + f mu^2) F (1-mu^2) P_dd, P_EE = F^2 (1-mu^2)^2 P_dd with
F = -a1 * 0.0134 * Omega_m / D(z); a1 defaults to 0 (these signal terms are
negligible next to the shape noise -- pass --a1 to include them).

Outputs <corr-dir>/<tracer>/<stem>_gaussian_cov.npz (+ optional comparison plot
vs the stored jackknife under plots/gaussian_cov/).

Run with the project .venv (pyccl + healpy + pyarrow):
    .venv/bin/python scripts/compute_gaussian_cov.py --tracers BGS \
        [--samples 'BGS_ANY*'] [--plot]
"""
import argparse
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from astropy.cosmology import FlatLambdaCDM

from ia2pt.gausscov import (GaussianCov, density_profile, gg_coeffs,  # noqa: F401
                            gp_coeffs, cross_coeffs, fit_b1, C1RHOC)

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config" / "data_sources.yaml"

# Distances: SAME cosmology as compute_correlations.py (H0=100 -> Mpc/h).
COSMO = FlatLambdaCDM(H0=100, Om0=0.3110)

# P(k) cosmology: fit_ia.DICT_COSMO (Planck-18-like, old_codes/lestgobaby.py).
# Replicated here so importing this script does not pull the sampler stack.
DICT_COSMO = {
    "h": 0.6766,
    "Omc": 0.11933 / 0.6766**2,
    "Omb": 0.02242 / 0.6766**2,
    "A_s": 2.105209331337507e-09,
    "n_s": 0.9665,
}
DZ_SHELL = 0.01  # n(z) shell width for the dV integrals


def chi_of_z(z):
    """Comoving distance [Mpc/h] on the H0=100 distance cosmology."""
    return COSMO.comoving_distance(z).value


# =============================================================================
# SAMPLE ENUMERATION + DRIVER
# =============================================================================
def ztag(name, zlo, zhi):
    return f"{name}_zmin_{zlo:.2f}_zmax_{zhi:.2f}"


def shape_jobs(cfg, only_tracers=None):
    """(tracer, zlo, zhi, shape_stem, shape_name) per measured sample, incl. mass bins."""
    jobs = []
    for t in cfg["ia_samples"]["tracers"]:
        if only_tracers and t["name"] not in only_tracers:
            continue
        for zlo, zhi in t["zbins"]:
            for s in t["shape_samples"]:
                stems = [ztag(s["name"], zlo, zhi)]
                if s.get("n_mass_bins"):
                    stems += [f"{ztag(s['name'], zlo, zhi)}_massbin{i}"
                              for i in range(int(s["n_mass_bins"]))]
                jobs += [(t["name"], zlo, zhi, st, s["name"]) for st in stems]
    return jobs


def load_area(dest, ia, tracer):
    f = dest / ia["out_subdir"] / "sample_properties" / f"footprint_area_{tracer}.json"
    if not f.exists():
        raise SystemExit(f"missing {f} -- run compute_footprint_area.py --tracers {tracer}")
    return json.load(open(f))


def weighted_sigma_gamma2(shapes):
    """Per-component ellipticity dispersion, weighted like the xi2 estimator."""
    w = (shapes["WEIGHT"] * shapes["WEIGHT_CLUSTERING"]).to_numpy()
    e1 = shapes["e1"].to_numpy()
    e2 = shapes["e2"].to_numpy()
    return float(np.sum(w**2 * (e1**2 + e2**2)) / (2.0 * np.sum(w**2)))


def process_sample(gc, npz_path, clu, w_clu, shapes_path, cgg_cache, cosmo_ccl,
                   z_edges, args, use_fkp=False):
    """Covariance for one shape-sample npz. Returns (out dict, meta dict)."""
    import pandas as pd
    d = np.load(npz_path)
    z_gg = float(d["z_eff_clustering"])
    z_ia = float(d["z_eff_IA"])

    # density-side coefficients (shared per tracer z-bin -> cached)
    if "cgg" not in cgg_cache:
        ng_w, ng_eff, v = density_profile(clu["Z"].to_numpy(), w_clu, z_edges,
                                          gc.area_sr, chi_of_z)
        cgg_cache.update(cgg=gg_coeffs(ng_w, ng_eff, v),
                         ng=(ng_w, ng_eff, v))
    cgg = cgg_cache["cgg"]
    ng_w, ng_eff, v = cgg_cache["ng"]

    # shape-side profile + shape noise
    shp = pd.read_parquet(shapes_path,
                          columns=["Z", "WEIGHT", "WEIGHT_CLUSTERING", "e1", "e2"])
    w_s = (shp["WEIGHT"] * shp["WEIGHT_CLUSTERING"]).to_numpy()
    ns_w, ns_eff, _ = density_profile(shp["Z"].to_numpy(), w_s, z_edges, gc.area_sr,
                                      chi_of_z)
    sig2 = weighted_sigma_gamma2(shp)
    cgp = gp_coeffs(ng_w, ng_eff, ns_w, ns_eff, v, sig2)
    cx = cross_coeffs(ng_w, ng_eff, ns_w, v)

    # b1: CLI override or closed-form fit on the measured xi0
    if "b1" not in cgg_cache:
        tpl, f_gg = gc.xi0_template(cosmo_ccl, z_gg)
        if args.b1 is not None:
            b1 = float(args.b1)
        else:
            b1 = fit_b1(d["xi0"], np.diag(d["cov_xi0"]), tpl, f_gg,
                        d["s_mid"], args.bfit_range)
        cgg_cache["b1"] = b1
    b1 = cgg_cache["b1"]

    pk_gg = gc.pk_model(cosmo_ccl, z_gg, b1, args.a1)
    pk_ia = gc.pk_model(cosmo_ccl, z_ia, b1, args.a1)
    cov00, cov22, cov02, comb = gc.blocks(pk_gg, pk_ia, cgg, cgp, cx)

    v_geo = float(np.sum(v))
    veff = float(np.sum((ng_w * args.p0 / (1.0 + ng_w * args.p0)) ** 2 * v))
    meta = dict(b1=b1, f_gg=pk_gg["f"], f_ia=pk_ia["f"], a1=args.a1,
                use_fkp=use_fkp,
                sigma_gamma2=sig2, z_eff_clustering=z_gg, z_eff_IA=z_ia,
                area_deg2=gc.area_sr * (180 / np.pi) ** 2, v_geo=v_geo,
                v_eff_p0=veff, p0=args.p0, kmax=args.kmax,
                n_gal_weighted=float(np.sum(ng_w * v)),
                n_shape_weighted=float(np.sum(ns_w * v)))
    out = dict(s_mid=d["s_mid"], s_bins=d["s_bins"],
               cov_xi0_gauss=cov00, cov_xi2_gauss=cov22,
               cov_cross_gauss=cov02, cov_combined_gauss=comb, **meta)
    return out, d, meta


def make_plot(stem, d, out, plot_dir):
    """Jackknife vs Gaussian: diagonal errors + correlation matrices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = out["s_mid"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # compare against the Mohammad+21-corrected jackknife -- the one the fits use
    for row, (xi, jk_key, g_key) in enumerate(
            [("xi0", "cov_xi0_corrected", "cov_xi0_gauss"),
             ("xi2", "cov_xi2_corrected", "cov_xi2_gauss")]):
        jk, ga = d[jk_key], out[g_key]
        ax = axes[row, 0]
        ax.semilogx(s, s * np.sqrt(np.diag(jk)), "o-", label="jackknife (corrected)")
        ax.semilogx(s, s * np.sqrt(np.diag(ga)), "s-", label="Gaussian")
        ax.set_xlabel(r"$s$ [$h^{-1}$Mpc]")
        ax.set_ylabel(rf"$s\,\sigma(\{xi[:2]}_{xi[2]})$ [$h^{{-1}}$Mpc]")
        ax.legend()
        ax = axes[row, 1]
        ax.semilogx(s, np.sqrt(np.diag(ga) / np.diag(jk)), "k.-")
        ax.axhline(1, color="r", ls=":")
        ax.set_xlabel(r"$s$ [$h^{-1}$Mpc]")
        ax.set_ylabel("Gaussian / jackknife")
        ax.set_ylim(0, 2)
        ax = axes[row, 2]
        dj = np.sqrt(np.diag(jk))
        dg = np.sqrt(np.diag(ga))
        n = len(s)
        both = np.zeros((n, n))
        iu = np.triu_indices(n, 1)
        il = np.tril_indices(n, -1)
        both[iu] = (jk / np.outer(dj, dj))[iu]
        both[il] = (ga / np.outer(dg, dg))[il]
        im = ax.imshow(both, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
        ax.set_title(f"{xi}: corr (upper=JK, lower=Gauss)")
        fig.colorbar(im, ax=ax)
    fig.suptitle(stem)
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    f = plot_dir / f"{stem}_gaussian_vs_jk.png"
    fig.savefig(f, dpi=120)
    plt.close(fig)
    print(f"    plot -> {f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--corr-dir", default=None,
                    help="compute_correlations output dir (default <dest>/correlations)")
    ap.add_argument("--tracers", nargs="*", default=None)
    ap.add_argument("--samples", nargs="*", default=None,
                    help="fnmatch patterns on the shape-sample stem")
    ap.add_argument("--b1", type=float, default=None,
                    help="linear bias (default: closed-form fit to the measured xi0)")
    ap.add_argument("--a1", type=float, default=0.0,
                    help="NLA amplitude for the (negligible) P_gE/P_EE signal terms")
    ap.add_argument("--p0", type=float, default=7000.0,
                    help="FKP P0 for the V_eff diagnostic [Mpc/h]^3")
    ap.add_argument("--kmax", type=float, default=10.0)
    ap.add_argument("--bfit-range", type=float, nargs=2, default=(10.0, 50.0))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    import pyccl as ccl

    cfg = yaml.safe_load(open(args.config))
    dest = Path(cfg["dest"])
    ia = cfg["ia_samples"]
    ia_dir = dest / ia["out_subdir"]
    corr_dir = Path(args.corr_dir) if args.corr_dir else dest / "correlations"
    plot_dir = REPO / "plots" / "gaussian_cov"

    cosmo_ccl = ccl.Cosmology(
        Omega_c=DICT_COSMO["Omc"], Omega_b=DICT_COSMO["Omb"], h=DICT_COSMO["h"],
        A_s=DICT_COSMO["A_s"], n_s=DICT_COSMO["n_s"],
        transfer_function="boltzmann_camb", matter_power_spectrum="camb",
        extra_parameters={"camb": {"halofit_version": "mead2020"}})

    jobs = shape_jobs(cfg, args.tracers)
    if args.samples:
        jobs = [j for j in jobs if any(fnmatch.fnmatch(j[3], p) for p in args.samples)]
    if not jobs:
        raise SystemExit("no samples matched")

    # group by (tracer, z-bin): density side + b1 shared
    state = {}
    for tracer, zlo, zhi, stem, sname in jobs:
        npz = corr_dir / tracer / f"{stem}_GPU_multipoles.npz"
        out_npz = corr_dir / tracer / f"{stem}_gaussian_cov.npz"
        if not npz.exists():
            print(f"[skip] {npz.name}: not measured")
            continue
        if out_npz.exists() and not args.overwrite:
            print(f"[skip] {out_npz.name}: exists (--overwrite to redo)")
            continue

        key = (tracer, zlo, zhi)
        if key not in state:
            area = load_area(dest, ia, tracer)
            clu = pd.read_parquet(
                ia_dir / f"{ztag(tracer, zlo, zhi)}_clustering.parquet")
            wcol = "WEIGHT_CORR" if "WEIGHT_CORR" in clu.columns else "WEIGHT"
            d0 = np.load(npz, allow_pickle=True)
            use_fkp = bool(d0["use_fkp"]) if "use_fkp" in d0.files else False
            w_clu = clu[wcol].to_numpy().astype(float)
            if use_fkp:
                if "WEIGHT_FKP" not in clu.columns:
                    raise SystemExit(f"{stem}: measurement has use_fkp=True but "
                                     "the clustering parquet lacks WEIGHT_FKP")
                w_clu = w_clu * clu["WEIGHT_FKP"].to_numpy()
            gc = GaussianCov(np.asarray(d0["s_bins"]), area["area_sr"],
                             kmax=args.kmax)
            z_edges = np.arange(zlo, zhi + DZ_SHELL / 2, DZ_SHELL)
            state[key] = dict(gc=gc, clu=clu, w=w_clu, use_fkp=use_fkp,
                              z_edges=z_edges, cache={},
                              area_deg2=area["area_deg2"])
            print(f"\n[{tracer} z={zlo}-{zhi}] area = {area['area_deg2']:.2f} deg^2 "
                  f"(density weight: {wcol}"
                  f"{' * WEIGHT_FKP' if use_fkp else ''})")
        st = state[key]

        print(f"  {stem}")
        # shape parquet naming: <sample>_zmin_..zmax_.._shapes[_massbinN].parquet
        if "massbin" in stem:
            base, mb = stem.rsplit("_massbin", 1)
            shapes_path = ia_dir / f"{base}_shapes_massbin{mb}.parquet"
        else:
            shapes_path = ia_dir / f"{stem}_shapes.parquet"
        out, d, meta = process_sample(st["gc"], npz, st["clu"], st["w"],
                                      shapes_path, st["cache"], cosmo_ccl,
                                      st["z_edges"], args,
                                      use_fkp=st["use_fkp"])
        np.savez(out_npz, **out)
        rat0 = np.sqrt(np.diag(out["cov_xi0_gauss"]) / np.diag(d["cov_xi0"]))
        rat2 = np.sqrt(np.diag(out["cov_xi2_gauss"]) / np.diag(d["cov_xi2"]))
        print(f"    b1={meta['b1']:.3f}  f={meta['f_gg']:.3f}  "
              f"sigma_gamma={np.sqrt(meta['sigma_gamma2']):.4f}  "
              f"V_geo={meta['v_geo']:.3e}  V_eff(P0={args.p0:.0f})={meta['v_eff_p0']:.3e}")
        print(f"    sigma ratio gauss/JK  xi0: median {np.median(rat0):.3f} "
              f"(range {rat0.min():.2f}-{rat0.max():.2f})  "
              f"xi2: median {np.median(rat2):.3f} ({rat2.min():.2f}-{rat2.max():.2f})")
        print(f"    -> {out_npz.name}")
        if args.plot:
            make_plot(stem, d, out, plot_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
