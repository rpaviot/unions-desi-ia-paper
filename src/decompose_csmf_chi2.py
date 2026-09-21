#!/usr/bin/env python3
"""Decompose a saved CSMF minuit fit's total chi2 into its contributions.

The reported minuit chi2 is 2*m.fval where the minuit cost is
    -log L + prior_penalty
and  -log L = 0.5 * [ sum_bin (ΔΣ residual^T Cov^-1 residual)  +  sum_bin n_gal anchor ].
So  chi2_total = sum_bin chi2_DS(bin)  +  sum_bin chi2_ngal(bin)  +  prior_penalty.

This rebuilds the fitter exactly as fit_csmf.py did (same samples / rp cut /
β^NL / gamma1 Gaussian / n_gal anchor), re-injects the saved best fit, and
recomputes each term via the fitter's own code paths (_apply_scale_cuts, full
per-bin covariance, _halo_model.ngal, _compute_prior_penalty) so the pieces sum
to the saved total.

    /home/rpaviot/NRV_HOD/.venv_hod/bin/python scripts/decompose_csmf_chi2.py \
        --label bgs_snr_dr6_rmin01_fh_g1prior --samples BGS_RED_GMM_VLIM_SNR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402
from fit_csmf import build_fitter  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(gu.CONFIG))
    p.add_argument("--label", default=None,
                   help="fit label (locates <dest>/ggl/csmf_fit/csmf_fit_<label>_<method>.npz "
                        "unless --fit-npz is given)")
    p.add_argument("--method", default="minuit", choices=["minuit", "de", "nautilus"],
                   help="which point estimate of the fit npz to decompose (nautilus: the "
                        "posterior median, fit_csmf's nautilus_* keys)")
    p.add_argument("--samples", nargs="+", default=["BGS_RED_GMM_VLIM_SNR"])
    p.add_argument("--rp-min", type=float, default=None,
                   help="[Mpc/h] scale cut the fit was run with (default: config)")
    p.add_argument("--in-name", default=None,
                   help="csmf-input subdir override the fit was run with (fit_csmf --in-name)")
    p.add_argument("--in-dir", default=None, metavar="DIR",
                   help="csmf-input directory the fit was run with (fit_csmf --in-dir); "
                        "wins over --in-name")
    p.add_argument("--fit-npz", default=None,
                   help="explicit path to the saved fit npz (default: "
                        "<dest>/ggl/csmf_fit/csmf_fit_<label>_<method>.npz). Needed for "
                        "arms whose fits live outside the production tree, e.g. the "
                        "PIP/IIP arm in pip_snr/ggl/csmf_fit/.")
    p.add_argument("--gamma1-flat", action="store_true",
                   help="set if the fit used --gamma1-flat (no Gaussian penalty)")
    p.add_argument("--gamma1-prior", choices=["gaussian", "flat"], default=None,
                   help="decision-style form of the same switch (fit_csmf --gamma1-prior)")
    args = p.parse_args()
    if args.fit_npz is None and args.label is None:
        p.error("give --fit-npz or --label")

    cfg = gu.load_config(args.config)
    dest = Path(cfg["dest"])
    fit_npz = (Path(args.fit_npz) if args.fit_npz
               else dest / "ggl" / "csmf_fit" / f"csmf_fit_{args.label}_{args.method}.npz")
    if not fit_npz.exists():
        sys.exit(f"fit file not found: {fit_npz}")

    d = np.load(fit_npz, allow_pickle=True)
    names = [str(s) for s in d[f"{args.method}_param_names"]]
    best = np.asarray(d[f"{args.method}_best_fit"], float)
    tot_saved = float(d[f"{args.method}_chi2"])
    ndof = int(d[f"{args.method}_ndof"])
    free = dict(zip(names, best))

    bf_args = SimpleNamespace(
        config=args.config, samples=args.samples,
        rp_min=args.rp_min, rp_max=None, in_name=args.in_name, in_dir=args.in_dir,
        no_beta_nl=False, beta_nl_log_m_min=None,
        fix=None, drop_massbins=None, verbose=False,
        fit_ngal=True, gamma1_flat=args.gamma1_flat, gamma1_prior=args.gamma1_prior,
        gamma1_mean=None, gamma1_std=None,
        # halo mass definition recorded in the fit npz (older fits: 200c default)
        mass_def=str(d["mass_definition"]) if "mass_definition" in d else None,
        halo_model_kwargs=(dict(d["halo_model_kwargs"].item()) if "halo_model_kwargs" in d else {}),
    )
    fitter = build_fitter(cfg, bf_args)
    fitter._initialize_halo_model()

    full = fitter._build_full_params(free)
    ds_dict, _ = fitter._compute_model_observables(full)   # also sets HOD on halo model
    ng_model = np.asarray(fitter._halo_model.ngal()).ravel()

    print(f"\n{'='*72}")
    print(f"chi2 decomposition — {args.label or fit_npz.name} ({args.method})")
    print(f"{'='*72}")
    hdr = f"{'sample/bin':22s} {'z_eff':>6s} {'npts':>4s} {'chi2_DS':>9s} {'DS/npts':>8s} {'chi2_ngal':>10s}"
    print(hdr)
    print("-" * len(hdr))

    ds_total = 0.0
    ng_total = 0.0
    npts_total = 0
    rows = []
    for i, mb in enumerate(fitter.mass_bins):
        uid = (mb.sample_type.value, mb.massbin_id)
        rp_cut, ds_data, cov, mask = fitter._apply_scale_cuts(mb)
        ds_model = ds_dict[uid][mask]
        residual = ds_data - ds_model
        try:
            c2 = float(residual @ np.linalg.inv(cov) @ residual)
        except np.linalg.LinAlgError:
            err = mb.delta_sigma_err[mask]
            c2 = float(np.sum((residual / err) ** 2))
        npts = int(mask.sum())
        # n_gal anchor term for this bin
        if mb.n_gal is not None and mb.n_gal_err is not None and np.isfinite(ng_model[i]):
            r = (ng_model[i] - mb.n_gal) / mb.n_gal_err
            c2_ng = float(r * r)
        else:
            c2_ng = 0.0
        ds_total += c2
        ng_total += c2_ng
        npts_total += npts
        zeff = float(getattr(mb, "z_eff", np.nan))
        print(f"{uid[0]+' bin'+str(uid[1]):22s} {zeff:6.3f} {npts:4d} "
              f"{c2:9.2f} {c2/max(npts,1):8.2f} {c2_ng:10.2f}")
        rows.append((uid, c2, npts, c2_ng))

    # _compute_prior_penalty returns the -log(prior) (= 0.5 * chi2_prior); it
    # enters the reported chi2 = 2*fval, so its chi2 contribution is 2x.
    prior_pen = 2.0 * float(fitter._compute_prior_penalty(free))

    print("-" * len(hdr))
    print(f"{'ΔΣ total':22s} {'':6s} {npts_total:4d} {ds_total:9.2f}")
    print(f"{'n_gal anchor total':22s} {'':6s} {len(fitter.mass_bins):4d} {'':9s} {'':8s} {ng_total:10.2f}")
    print(f"{'gamma1 prior (chi2)':22s} {'':6s} {'':4s} {prior_pen:9.2f}")
    recomputed = ds_total + ng_total + prior_pen
    print("=" * len(hdr))
    print(f"  recomputed total chi2 = {ds_total:.2f} (DS) + {ng_total:.2f} (ngal) "
          f"+ {prior_pen:.2f} (prior) = {recomputed:.2f}")
    print(f"  saved {args.method} chi2 =  {tot_saved:.2f}   (ndof={ndof}, "
          f"chi2/ndof={tot_saved/ndof:.3f})")
    print(f"  match: {'OK' if abs(recomputed - tot_saved) < 0.5 else 'MISMATCH'} "
          f"(Δ={recomputed - tot_saved:+.3f})")
    # DS-only reduced chi2 (ndof convention = n_DS_points - n_free)
    n_free = len(names)
    print(f"\n  ΔΣ-only: chi2_DS/{npts_total - n_free} = "
          f"{ds_total/(npts_total - n_free):.3f}  (npts {npts_total} - {n_free} free)")


if __name__ == "__main__":
    main()
