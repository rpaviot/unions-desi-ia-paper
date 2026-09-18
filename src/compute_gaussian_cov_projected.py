#!/usr/bin/env python3
"""Gaussian covariance for the PROJECTED statistics (wp clustering, wg+ IA),
the exact analogue of compute_gaussian_cov.py for the multipoles. The core is
``ia2pt.gausscov.GaussianCovProjected``; this script is the UNIONS x DESI driver.

Modernizes old_codes/gaussiancov_.py: same per-mode Wick variance and
FKP-style varying-nbar dV integrals as the multipole script, but with the
projected estimator's exact kernels instead of the old Limber-style shortcut.

Formalism
---------
The estimator is w_A(rp_i) = Int_{-Pi}^{Pi} dpi xi_A(rp, pi), annulus-averaged
over the rp bin (pair counts weight the annulus uniformly in area). Writing the
estimator in Fourier space factorizes the covariance exactly:

    C_AB(rp_i, rp_j) = sign_AB Int_0^inf dkp kp/(2 pi) Jbar_a(kp; i) Jbar_b(kp; j)
                       x Int dkz/(2 pi) W(kz)^2 Q_AB(k, mu)

with kp/kz the transverse/LOS wavenumbers, k = sqrt(kp^2 + kz^2), mu = kz/k,
W(kz) = 2 sin(kz Pi) / kz the exact finite-pimax LOS window, and Jbar_m the
AREA-averaged Bessel kernel over the rp annulus (closed forms):

    Jbar_0 = 2 [ x J1(x) ]_{lo}^{hi} / (x_hi^2 - x_lo^2)
    Jbar_2 = 2 [ -2 J0(x) - x J1(x) ]_{lo}^{hi} / (x_hi^2 - x_lo^2),  x = kp rp.

Per side: scalar density -> J0; spin-2 shear w.r.t. the projected separation ->
the azimuthal integral (1/2pi) Int dphi cos 2(phi_k - phi_r) e^{i kp.rp}
= -J2(kp rp), so wgp x wgp gets (+)Jbar_2 Jbar_2 and the wp x wgp cross block
gets sign -1 (one -J2 side), mirroring the multipoles' i^{0+2} = -1.

Q_AB is the SAME per-mode covariance / dV machinery as the multipole script
(gg_coeffs / gp_coeffs / cross_coeffs, FKP weights mirrored from use_fkp):

    Q_gg = 2 [c4 P_gg^2 + 2 c3 P_gg + c2] / N^2
    Q_g+ = [dPP (P_gg P_EE + P_gE^2) + dgn P_gg + dsn P_EE + dwhite] / M^2
    Q_x  = 2 [x1 P_gg P_gE + x0 P_gE] / (N M)

with the full mu dependence: P_gg = (b1 + f mu^2)^2 P_dd, P_gE = (b1 + f mu^2)
F (1 - mu^2) P_dd, P_EE = F^2 (1 - mu^2)^2 P_dd (a1 = 0 default -> signal terms
off next to shape noise, as in the multipole script).

Pure white (P-independent) terms are analytic: Int dkz W^2 / 2pi = 2 Pi and the
transverse Hankel closure Int kp dkp Jbar_m(i) Jbar_m(j) / 2pi = delta_ij / A_i
(A_i = pi (rp_hi^2 - rp_lo^2)), so e.g. the wgp shape-noise diagonal is exactly
sigma_gamma^2-weighted dwhite/M^2 x 2 Pi / A_i -- no k-truncation error.

What the old code missed: the finite-LOS-depth factor (the kz integral; its
P-terms lacked the 2 Pi / L_depth normalization), the Kaiser + spin-2 mu
dependence (it used P at mu = 0), the varying-nbar dV weighting, FKP weights,
and the cross-block sign.

b1 is read from the multipole Gaussian-cov npz of the same stem (fit there to
the measured xi0) unless --b1 is given.

Outputs <corr-dir>/<tracer>/<stem>_gaussian_cov_projected.npz with blocks
cov_wp_gauss, cov_wgp_gauss, cov_cross_gauss, cov_combined_gauss ([wp, wgp]
ordering, matching the measurement's cov_combined).

Run with the project .venv (pyccl + pyarrow):
    .venv/bin/python scripts/compute_gaussian_cov_projected.py --tracers LRG [--plot]
"""
import argparse
import fnmatch
import sys
from pathlib import Path

import numpy as np
import yaml

from ia2pt.gausscov import (GaussianCovProjected, density_profile, gg_coeffs,
                            gp_coeffs, cross_coeffs)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_gaussian_cov import (CONFIG, DICT_COSMO, DZ_SHELL, chi_of_z,  # noqa: E402
                                  shape_jobs, ztag, load_area, weighted_sigma_gamma2)

PIMAX = 100.0  # compute_correlations PI_BINS = linspace(-100, 100, 21)


def make_plot(stem, d, out, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rp = out["rp_mid"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # compare against the Mohammad+21-corrected jackknife -- the one the fits use
    for row, (w, jk_key, g_key) in enumerate(
            [("wp", "cov_wp_corrected", "cov_wp_gauss"),
             ("wgp", "cov_wgp_corrected", "cov_wgp_gauss")]):
        jk, ga = d[jk_key], out[g_key]
        ax = axes[row, 0]
        ax.loglog(rp, np.sqrt(np.diag(jk)), "o-", label="jackknife (corrected)")
        ax.loglog(rp, np.sqrt(np.diag(ga)), "s-", label="Gaussian")
        ax.set_xlabel(r"$r_p$ [$h^{-1}$Mpc]")
        ax.set_ylabel(rf"$\sigma({w})$")
        ax.legend()
        ax = axes[row, 1]
        ax.semilogx(rp, np.sqrt(np.diag(ga) / np.diag(jk)), "k.-")
        ax.axhline(1, color="r", ls=":")
        ax.set_xlabel(r"$r_p$ [$h^{-1}$Mpc]")
        ax.set_ylabel("Gaussian / jackknife")
        ax.set_ylim(0, 2)
        ax = axes[row, 2]
        dj, dg = np.sqrt(np.diag(jk)), np.sqrt(np.diag(ga))
        n = len(rp)
        both = np.zeros((n, n))
        iu, il = np.triu_indices(n, 1), np.tril_indices(n, -1)
        both[iu] = (jk / np.outer(dj, dj))[iu]
        both[il] = (ga / np.outer(dg, dg))[il]
        im = ax.imshow(both, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
        ax.set_title(f"{w}: corr (upper=JK, lower=Gauss)")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"{stem} (projected)")
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    f = plot_dir / f"{stem}_projected_gaussian_vs_jk.png"
    fig.savefig(f, dpi=120)
    plt.close(fig)
    print(f"    plot -> {f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--corr-dir", default=None)
    ap.add_argument("--tracers", nargs="*", default=None)
    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--b1", type=float, default=None,
                    help="linear bias (default: read from the multipole "
                         "<stem>_gaussian_cov.npz)")
    ap.add_argument("--a1", type=float, default=0.0)
    ap.add_argument("--kmax", type=float, default=10.0)
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

    state = {}
    for tracer, zlo, zhi, stem, sname in jobs:
        npz = corr_dir / tracer / f"{stem}_CPU_projected.npz"
        out_npz = corr_dir / tracer / f"{stem}_gaussian_cov_projected.npz"
        if not npz.exists():
            print(f"[skip] {npz.name}: not measured")
            continue
        if out_npz.exists() and not args.overwrite:
            print(f"[skip] {out_npz.name}: exists (--overwrite to redo)")
            continue

        d = np.load(npz, allow_pickle=True)
        key = (tracer, zlo, zhi)
        if key not in state:
            area = load_area(dest, ia, tracer)
            clu = pd.read_parquet(
                ia_dir / f"{ztag(tracer, zlo, zhi)}_clustering.parquet")
            wcol = "WEIGHT_CORR" if "WEIGHT_CORR" in clu.columns else "WEIGHT"
            use_fkp = bool(d["use_fkp"]) if "use_fkp" in d.files else False
            w_clu = clu[wcol].to_numpy().astype(float)
            if use_fkp:
                w_clu = w_clu * clu["WEIGHT_FKP"].to_numpy()
            gc = GaussianCovProjected(np.asarray(d["rp_bins"]), area["area_sr"],
                                      kmax=args.kmax)
            z_edges = np.arange(zlo, zhi + DZ_SHELL / 2, DZ_SHELL)
            ng_w, ng_eff, v = density_profile(clu["Z"].to_numpy(), w_clu,
                                              z_edges, area["area_sr"], chi_of_z)
            state[key] = dict(gc=gc, use_fkp=use_fkp, z_edges=z_edges,
                              cgg=gg_coeffs(ng_w, ng_eff, v),
                              ng=(ng_w, ng_eff, v))
            print(f"\n[{tracer} z={zlo}-{zhi}] area = {area['area_deg2']:.2f} deg^2 "
                  f"(density weight: {wcol}{' * WEIGHT_FKP' if use_fkp else ''}, "
                  f"pimax = {PIMAX:.0f})")
        st = state[key]
        gc = st["gc"]
        ng_w, ng_eff, v = st["ng"]

        print(f"  {stem}")
        if "massbin" in stem:
            base, mb = stem.rsplit("_massbin", 1)
            shapes_path = ia_dir / f"{base}_shapes_massbin{mb}.parquet"
        else:
            shapes_path = ia_dir / f"{stem}_shapes.parquet"
        shp = pd.read_parquet(shapes_path,
                              columns=["Z", "WEIGHT", "WEIGHT_CLUSTERING", "e1", "e2"])
        w_s = (shp["WEIGHT"] * shp["WEIGHT_CLUSTERING"]).to_numpy()
        ns_w, ns_eff, _ = density_profile(shp["Z"].to_numpy(), w_s,
                                          st["z_edges"], gc.area_sr, chi_of_z)
        sig2 = weighted_sigma_gamma2(shp)
        cgp = gp_coeffs(ng_w, ng_eff, ns_w, ns_eff, v, sig2)
        cx = cross_coeffs(ng_w, ng_eff, ns_w, v)

        if args.b1 is not None:
            b1 = float(args.b1)
        else:
            gfile = corr_dir / tracer / f"{stem}_gaussian_cov.npz"
            if not gfile.exists():
                raise SystemExit(f"{stem}: no --b1 and no multipole Gaussian cov "
                                 f"({gfile.name}) to read b1 from")
            b1 = float(np.load(gfile)["b1"])

        z_gg = float(d["z_eff_clustering"])
        z_ia = float(d["z_eff_IA"])
        pk_gg = gc.pk_model(cosmo_ccl, z_gg, b1, args.a1)
        pk_ia = gc.pk_model(cosmo_ccl, z_ia, b1, args.a1)
        cov_gg, cov_gp, cov_x, comb = gc.blocks(pk_gg, pk_ia, st["cgg"], cgp, cx)

        out = dict(rp_mid=d["rp_mid"], rp_bins=d["rp_bins"],
                   cov_wp_gauss=cov_gg, cov_wgp_gauss=cov_gp,
                   cov_cross_gauss=cov_x, cov_combined_gauss=comb,
                   b1=b1, f_gg=pk_gg["f"], f_ia=pk_ia["f"], a1=args.a1,
                   use_fkp=st["use_fkp"], sigma_gamma2=sig2, pimax=PIMAX,
                   z_eff_clustering=z_gg, z_eff_IA=z_ia, kmax=args.kmax)
        np.savez(out_npz, **out)
        rat_p = np.sqrt(np.diag(cov_gg) / np.diag(d["cov_wp"]))
        rat_gp = np.sqrt(np.diag(cov_gp) / np.diag(d["cov_wgp"]))
        print(f"    b1={b1:.3f}  f={pk_gg['f']:.3f}  "
              f"sigma_gamma={np.sqrt(sig2):.4f}")
        print(f"    sigma ratio gauss/JK  wp: median {np.median(rat_p):.3f} "
              f"(range {rat_p.min():.2f}-{rat_p.max():.2f})  "
              f"wgp: median {np.median(rat_gp):.3f} "
              f"({rat_gp.min():.2f}-{rat_gp.max():.2f})")
        print(f"    -> {out_npz.name}")
        if args.plot:
            make_plot(stem, d, out, plot_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
