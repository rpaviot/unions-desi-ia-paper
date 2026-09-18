#!/usr/bin/env python3
"""Plot the best-fit CSMF ΔΣ model against the measured profiles, bin by bin.

Reads a saved CSMF fit (``<dest>/ggl/csmf_fit/csmf_fit_<label>_<method>.npz``,
written by fit_csmf.py), rebuilds the fitter exactly as fit_csmf.py does (same
config-driven samples / rp range / β^NL / cosmology), re-injects the saved
best-fit parameters, and evaluates the analytical halo model via
``CSMFFitter.compute_model_prediction``. No re-fitting -- the saved free
parameters are merged with the fixed priors (``_build_full_params``) so the
curve shown is the one the fit actually landed on.

One panel per fitted mass bin: measured ΔΣ(r_p) with the jackknife errors, the
best-fit model curve, and the per-bin χ² over the fitted scale range. That χ² is
the FULL-covariance one, r^T C^-1 r with the bin's jackknife covariance -- i.e.
the quantity the likelihood actually minimises, so the panels sum to the ΔΣ part
of the reported total (see scripts/decompose_csmf_chi2.py).

Run from the NRV venv (needs jax / dark_emulator via the package import):
    /home/rpaviot/NRV_HOD/.venv_hod/bin/python scripts/plot_csmf_bestfit.py \
        --label bgs_vlim --samples BGS_RED_GMM_VLIM
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402
import paper_style  # noqa: E402
from fit_csmf import build_fitter  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(gu.CONFIG))
    p.add_argument("--label", default="bgs_vlim", help="fit label (default: bgs_vlim)")
    p.add_argument("--method", default="minuit", choices=["minuit", "de"])
    p.add_argument("--samples", nargs="+", default=None,
                   help="Samples to rebuild (must match the fit; default: all in config)")
    p.add_argument("--rp-min", type=float, default=None, help="[Mpc/h]; default from config")
    p.add_argument("--rp-max", type=float, default=None, help="[Mpc/h]; default from config")
    p.add_argument("--no-beta-nl", action="store_true")
    p.add_argument("--fix", nargs="*", default=None, metavar="PARAM=VALUE")
    p.add_argument("--drop-massbins", nargs="*", default=None, metavar="NAME:i[,j]")
    p.add_argument("--ncols", type=int, default=4, help="panel columns")
    p.add_argument("--in-name", default=None,
                   help="csmf-input subdir the fit used (fit_csmf --in-name)")
    p.add_argument("--in-dir", default=None, metavar="DIR",
                   help="csmf-input directory the fit used (fit_csmf --in-dir); wins over --in-name")
    p.add_argument("--fit-npz", default=None,
                   help="explicit path to the fit npz (default: "
                        "<dest>/ggl/csmf_fit/csmf_fit_<label>_<method>.npz)")
    p.add_argument("--paper", action="store_true",
                   help="paper styling: PLOT13NRV rcParams, no suptitle, "
                        "compact panels, dpi 300 + bbox tight")
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", dest="verbose", action="store_false")
    args = p.parse_args()

    if args.fix:
        args.fix = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.fix}
    if args.drop_massbins:
        drop = {}
        for ent in args.drop_massbins:
            name, idxs = ent.split(":")
            drop[name] = {int(i) for i in idxs.split(",") if i != ""}
        args.drop_massbins = drop

    cfg = gu.load_config(args.config)
    dest = Path(cfg["dest"])
    fit_npz = (Path(args.fit_npz) if args.fit_npz
               else dest / "ggl" / "csmf_fit" / f"csmf_fit_{args.label}_{args.method}.npz")
    if not fit_npz.exists():
        sys.exit(f"fit file not found: {fit_npz}")

    d = np.load(fit_npz, allow_pickle=True)
    names = [str(s) for s in d[f"{args.method}_param_names"]]
    best = np.asarray(d[f"{args.method}_best_fit"], float)
    chi2, ndof = float(d[f"{args.method}_chi2"]), int(d[f"{args.method}_ndof"])
    free = dict(zip(names, best))

    # Rebuild the fitter exactly like fit_csmf.py (build_fitter takes a namespace).
    bf_args = SimpleNamespace(
        config=args.config, samples=args.samples, in_name=args.in_name, in_dir=args.in_dir,
        rp_min=args.rp_min, rp_max=args.rp_max, no_beta_nl=args.no_beta_nl,
        fix=args.fix, drop_massbins=args.drop_massbins, verbose=args.verbose,
        # n_gal anchor + gamma1 prior only touch the likelihood, not the dSigma
        # model curve drawn here -- mirror the fit flags so build_fitter is happy.
        fit_ngal=True, gamma1_flat=True, gamma1_mean=None, gamma1_std=None,
        # None -> take csmf.beta_nl.log_M_min from the config, i.e. the value the
        # fit itself ran with; hardcoding 12.0 here drew a slightly different model.
        beta_nl_log_m_min=None,
        # halo mass definition recorded in the fit npz (older fits: 200c default)
        mass_def=str(d["mass_definition"]) if "mass_definition" in d else None,
        halo_model_kwargs=(dict(d["halo_model_kwargs"].item()) if "halo_model_kwargs" in d else {}),
    )
    fitter = build_fitter(cfg, bf_args)
    fitter._initialize_halo_model()

    # Merge the saved free params with the fixed priors, then evaluate the model.
    full = fitter._build_full_params(free)
    preds = fitter.compute_model_prediction(full)

    # Per-bin chi2 with the FULL jackknife covariance (what the likelihood uses),
    # via the fitter's own scale cut so the mask matches the fit exactly.
    cov_chi2 = {}
    for mb in fitter.mass_bins:
        uid = (mb.sample_type.value, mb.massbin_id)
        _, ds_data, cov, mask = fitter._apply_scale_cuts(mb)
        res = ds_data - preds[uid]["delta_sigma_model"][mask]
        try:
            c2 = float(res @ np.linalg.inv(cov) @ res)
        except np.linalg.LinAlgError:  # fall back to the diagonal
            c2 = float(np.sum((res / mb.delta_sigma_err[mask]) ** 2))
        cov_chi2[uid] = (c2, int(mask.sum()))

    rp_min = args.rp_min if args.rp_min is not None else float(cfg["csmf"]["rp_min"])
    rp_max = args.rp_max if args.rp_max is not None else float(cfg["csmf"]["rp_max"])

    # Order panels by (sample, massbin) so bins read left-to-right, low M* -> high.
    keys = sorted(preds.keys(), key=lambda k: (k[0], k[1]))
    n = len(keys)
    ncols = min(args.ncols, n)
    nrows = int(np.ceil(n / ncols))
    if args.paper:
        paper_style.apply_style()
        fig, axes = plt.subplots(nrows, ncols, figsize=(7, 2.1 * nrows),
                                 sharex=True, sharey=True, squeeze=False)
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows),
                                 sharex=True, sharey=True, squeeze=False)
    flat = axes.flat

    for ax, key in zip(flat, keys):
        pr = preds[key]
        rp, model = pr["rp"], pr["delta_sigma_model"]
        data, err = pr["delta_sigma_data"], pr["delta_sigma_err"]
        sel = (rp >= rp_min) & (rp <= rp_max)
        c2, npts = cov_chi2[key]
        ax.errorbar(rp, data, yerr=err, fmt="o", ms=3 if args.paper else 4,
                    color="k", capsize=2, label="data")
        ax.plot(rp, model, color="crimson", lw=1.4 if args.paper else 2,
                label="best-fit model")
        ax.axvspan(rp.min() / 2, rp_min, color="gray", alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if args.paper:
            ax.text(0.95, 0.92, f"bin {key[1]}\n"
                    rf"$\chi^2$={c2:.1f}/{npts}",
                    transform=ax.transAxes, fontsize=7, va="top", ha="right")
        else:
            ax.set_title(f"{key[0]} bin {key[1]}  z={pr['z_eff']:.2f}\n"
                         rf"$\chi^2$={c2:.1f}/{npts} pts", fontsize=9)
            ax.legend(fontsize=7)

    for ax in flat[n:]:
        ax.set_visible(False)
    for c in range(ncols):
        col = [axes[rr][c] for rr in range(nrows) if axes[rr][c].get_visible()]
        if col:
            col[-1].set_xlabel(r"$r_{\rm p}$ [Mpc/$h$]")
            col[-1].xaxis.set_tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta\Sigma$ [$M_\odot h$/pc$^2$]")

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "plots" / f"csmf_bestfit_{args.label}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.paper:
        fig.tight_layout()
        paper_style.savefig(fig, out, dpi=300)
    else:
        fig.suptitle(f"CSMF {args.method} fit — {args.label}:  "
                     rf"$\chi^2$/ndof = {chi2:.1f}/{ndof} = {chi2/ndof:.2f}",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out, dpi=140)
    print(f"saved {out}")
    print(f"  chi2/ndof = {chi2:.1f}/{ndof} = {chi2/ndof:.2f}")
    print(f"  sum of panel chi2 (dSigma, full cov) = "
          f"{sum(v[0] for v in cov_chi2.values()):.2f} over "
          f"{sum(v[1] for v in cov_chi2.values())} points")


if __name__ == "__main__":
    main()
