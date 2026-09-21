#!/usr/bin/env python3
"""Fit the analytical CSMF HOD to the measured ΔΣ profiles.

Uses ``HOD_NRV.HOD_analytical.sampler.CSMFFitter`` (develop branch, NRV venv) to
jointly constrain the stellar-to-halo mass relation of the red lenses from their
galaxy-galaxy lensing signal. Each (sample, mass bin) contributes one ΔΣ(r_p)
profile at its own effective redshift and stellar-mass window; the per-bin
inputs are built by ``build_csmf_input.py`` (magnification already subtracted on
the data side).

Samples, rp range and β^NL options come from the ``csmf:`` block of
config/data_sources.yaml. BGS_RED_GMM loads via the BGS (GMM) path, LRG via the
LRG path -- but since build_csmf_input.py writes no ``mag_contribution`` key, the
package's LRG magnification subtraction is inert (already done upstream).

Cosmology: HOD_NRV DEFAULT_COSMO_PARAMS (Planck18). Units: units_per_h=True
(rp in Mpc/h, ΔΣ in Msun h/pc^2), matching the measurement.

Run from the NRV venv:
    /home/rpaviot/NRV_HOD/.venv_hod/bin/python scripts/fit_csmf.py --method smoke
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402

from HOD_NRV.HOD_analytical.sampler import (  # noqa: E402
    CSMFFitter, DEFAULT_CSMF_PRIORS, DEFAULT_COSMO_PARAMS, ParameterPrior,
)


def ccl_name(name):
    """CCL halo-model class names are CamelCase (Tinker10, Duffy08); accept the
    lower-case decision ids of the ASTRA spec (tinker10 -> Tinker10)."""
    name = str(name)
    return name if name[:1].isupper() else name[:1].upper() + name[1:]


def build_fitter(cfg, args):
    csmf = cfg["csmf"]
    dest = Path(cfg["dest"])
    in_name = getattr(args, "in_name", None)
    in_dir = (dest / "ggl" / in_name) if in_name else (dest / csmf["out_subdir"])
    if getattr(args, "in_dir", None):          # explicit directory wins (ASTRA recipes)
        in_dir = Path(args.in_dir)

    beta_nl = bool(csmf.get("include_beta_nl", False)) and not args.no_beta_nl
    beta_nl_opts = dict(csmf.get("beta_nl", {}))
    beta_nl_opts.setdefault("verbose", args.verbose)
    # Optional override of the emulator β^NL interpolation mass floor [log10 M].
    # The package's own default is 12.0 (dark_emulator's validity floor); the
    # config lowers it to 11.5 (extrapolating below 1e12). --beta-nl-log-m-min
    # restores/sets it per run without touching the global config.
    if args.beta_nl_log_m_min is not None:
        beta_nl_opts["log_M_min"] = float(args.beta_nl_log_m_min)
        print(f"[beta_nl] log_M_min = {float(args.beta_nl_log_m_min)} "
              f"(M_min = {10**float(args.beta_nl_log_m_min):.2e} M_sun/h)", flush=True)

    # n_gal abundance anchor: CLI --fit-ngal/--no-fit-ngal wins over config csmf.fit_ngal.
    fit_ngal = csmf.get("fit_ngal", False) if args.fit_ngal is None else args.fit_ngal
    anchor = getattr(args, "ngal_anchor", None)
    if anchor is not None:                      # --ngal-anchor ngal|none wins over both
        fit_ngal = (anchor == "ngal")
    observables = ["DeltaSigma"] + (["ngal"] if fit_ngal else [])
    if fit_ngal:
        print("[observables] DeltaSigma + n_gal (abundance anchor ON)", flush=True)

    # Halo-model ingredients (mass_definition / concentration / halo_bias /
    # mass_function), config `csmf.halo_model` with a CLI override for the
    # mass definition. Package defaults when absent: Tinker08 HMF, Tinker10
    # bias, Duffy08 c(M), MassDef200c. NB dark_emulator's beta^NL is defined
    # for M200b (= 200m) haloes, so MassDef200m is the self-consistent choice.
    halo_model_kwargs = dict(csmf.get("halo_model", {}) or {})
    # a downstream script re-evaluating a saved fit passes that fit's recorded
    # halo_model_kwargs wholesale (mass definition, HMF, bias, concentration)
    halo_model_kwargs.update(dict(getattr(args, "halo_model_kwargs", None) or {}))
    mass_def = getattr(args, "mass_def", None)
    if mass_def:
        halo_model_kwargs["mass_definition"] = mass_def
    for key, arg in (("mass_function", "mass_function"), ("halo_bias", "halo_bias"),
                     ("concentration", "concentration")):
        val = getattr(args, arg, None)
        if val:
            halo_model_kwargs[key] = ccl_name(val)
    if halo_model_kwargs:
        print(f"[halo model] overrides: {halo_model_kwargs}", flush=True)

    fitter = CSMFFitter(
        cosmo_params=dict(DEFAULT_COSMO_PARAMS),
        observables=observables,
        rp_min=args.rp_min if args.rp_min is not None else float(csmf["rp_min"]),
        rp_max=args.rp_max if args.rp_max is not None else float(csmf["rp_max"]),
        include_beta_nl=beta_nl,
        beta_nl_kwargs=beta_nl_opts if beta_nl else None,
        k_array=np.geomspace(1e-5, 300, 1024),
        units_per_h=True,
        verbose=args.verbose,
        halo_model_kwargs=halo_model_kwargs or None,
    )

    want = set(args.samples) if args.samples else None
    drop_all = args.drop_massbins or {}
    for s in csmf["samples"]:
        if want is not None and s["name"] not in want:
            continue
        drop = drop_all.get(s["name"], set())
        mass_bins = [b for b in range(int(s["n_mass_bins"])) if b not in drop]
        if drop:
            print(f"[{s['name']}] dropping mass bins {sorted(drop)} -> "
                  f"using {mass_bins}", flush=True)
        sel = s["selection"].upper()
        pattern = f"csmf_{s['name']}_massbin{{}}.npz"
        if sel == "LRG":
            fitter.load_lrg_data(str(in_dir), mass_bins=mass_bins, file_pattern=pattern)
        else:  # GMM / SFR -> BGS path
            fitter.load_bgs_data(str(in_dir), mass_bins=mass_bins,
                                 selection=sel, file_pattern=pattern)

    # Optional Gaussian prior on gamma1 (the low-mass SHMR slope). Our lens bins
    # sit at/above the SHMR knee, so DeltaSigma alone barely constrains gamma1 --
    # Dvornik+23 pin it via galaxy abundance, which we do not fit. Adopt a Gaussian
    # prior centred on their best fit instead of the flat [2.5,15] default. The
    # package honours this in de/minuit/nautilus (minuit adds it as a 0.5*z^2
    # penalty). Precedence: --gamma1-flat > CLI mean/std > config csmf.gamma1_prior.
    priors = dict(DEFAULT_CSMF_PRIORS)
    gamma1_flat = args.gamma1_flat or getattr(args, "gamma1_prior", None) == "flat"
    g1 = {} if gamma1_flat else dict(csmf.get("gamma1_prior") or {})
    if args.gamma1_mean is not None:
        g1["mean"] = args.gamma1_mean
    if args.gamma1_std is not None:
        g1["std"] = args.gamma1_std
    if g1:
        if "mean" not in g1 or "std" not in g1:
            raise SystemExit("gamma1 Gaussian prior needs both 'mean' and 'std' "
                             "(config csmf.gamma1_prior or --gamma1-mean/--gamma1-std)")
        priors["gamma1"] = ParameterPrior(name="gamma1", prior_type="gaussian",
                                          mean=float(g1["mean"]), std=float(g1["std"]))
        print(f"[priors] gamma1 ~ Gaussian(mean={float(g1['mean'])}, "
              f"std={float(g1['std'])})", flush=True)
    else:
        print("[priors] gamma1 ~ Uniform (package default [2.5, 15])", flush=True)

    fitter.set_priors(priors, fixed_params=args.fix or None)
    return fitter


def nautilus_summary(fitter):
    """Point-estimate keys for a nautilus run, in the layout of the minuit ones
    (``nautilus_param_names / _best_fit / _errors / _covariance / _chi2 / _ndof``) so the
    downstream scripts (--method nautilus) can re-evaluate the model at one point.

    best_fit = the weighted posterior MEDIAN of each parameter (what the IA fits quote);
    errors = half the 16-84 % interval; covariance = the weighted sample covariance;
    chi2 = -2 log L + 2 x prior penalty AT THE MEDIAN (the minuit convention, so
    decompose_csmf_chi2 reproduces it); ndof = Delta Sigma points after the scale cut -
    free parameters (the fitter's minuit convention). Two more point estimates with
    their own chi2: nautilus_map / nautilus_map_chi2 = the posterior sample with the
    highest log L - prior penalty (nautilus's log_l is the likelihood alone, the Gaussian
    gamma1 prior sits in its Prior object, so the MAP is taken on the same objective
    MINUIT minimises and nautilus_map_chi2 is its minimum over the chain) and
    nautilus_mean / nautilus_mean_chi2 = the weighted posterior mean."""
    res = fitter.results
    names = [str(n) for n in res["param_names"]]
    pts = np.asarray(res["points"], float)
    logw = np.asarray(res["log_w"], float)
    w = np.exp(logw - logw.max())
    w /= w.sum()

    def wq(x, q):
        i = np.argsort(x)
        c = np.cumsum(w[i])
        return float(np.interp(q, c, x[i]))

    med = np.array([wq(pts[:, j], 0.5) for j in range(len(names))])
    lo = np.array([wq(pts[:, j], 0.16) for j in range(len(names))])
    hi = np.array([wq(pts[:, j], 0.84) for j in range(len(names))])
    cov = np.cov(pts, rowvar=False, aweights=w)
    mean = np.average(pts, weights=w, axis=0)

    def objective(vec):
        free = {n: float(v) for n, v in zip(names, vec)}
        return (-2.0 * float(fitter.log_likelihood(free))
                + 2.0 * float(fitter._compute_prior_penalty(free)))

    # MAP = the chain sample maximising log L - prior penalty (MINUIT's objective).
    log_l = np.asarray(res["log_l"], float)
    penalty = np.array([fitter._compute_prior_penalty({n: float(v) for n, v in zip(names, pt)})
                        for pt in pts])
    imap = int(np.argmax(log_l - penalty))
    chi2 = objective(med)
    chi2_mean = objective(mean)
    chi2_map = -2.0 * log_l[imap] + 2.0 * penalty[imap]
    npts = sum(int(fitter._apply_scale_cuts(mb)[3].sum()) for mb in fitter.mass_bins)
    ndof = npts - len(names)
    print(f"\n[nautilus] posterior medians: {dict(zip(names, med))}\n"
          f"[nautilus] posterior means:   {dict(zip(names, mean))}\n"
          f"[nautilus] MAP sample:        {dict(zip(names, pts[imap]))}\n"
          f"[nautilus] chi2 at the median = {chi2:.2f}, mean = {chi2_mean:.2f}, "
          f"MAP = {chi2_map:.2f} for ndof = {ndof}  (log Z = {float(res['log_z']):.2f})",
          flush=True)
    return {"nautilus_param_names": np.array(names, dtype=str), "nautilus_best_fit": med,
            "nautilus_errors": 0.5 * (hi - lo), "nautilus_lo16": lo, "nautilus_hi84": hi,
            "nautilus_covariance": cov, "nautilus_chi2": chi2, "nautilus_ndof": ndof,
            "nautilus_mean": mean, "nautilus_mean_chi2": chi2_mean,
            "nautilus_map": pts[imap], "nautilus_map_chi2": chi2_map,
            "nautilus_map_log_l": float(log_l[imap])}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(gu.CONFIG))
    p.add_argument("--samples", nargs="+", default=None,
                   help="Restrict to these sample names (default: all in csmf config)")
    p.add_argument("--in-name", default=None,
                   help="Override the csmf-input subdir under <dest>/ggl (default: "
                        "csmf.out_subdir). Use e.g. 'csmf_input_z5' for the "
                        "percentile-zmin n_gal build.")
    p.add_argument("--in-dir", default=None, metavar="DIR",
                   help="Directory holding the csmf_<sample>_massbin{i}.npz inputs; "
                        "overrides --in-name / config (used by the ASTRA recipes, e.g. "
                        "data/ggl/csmf_input_z5_effective)")
    p.add_argument("--rp-min", type=float, default=None, help="[Mpc/h]; default from config")
    p.add_argument("--rp-max", type=float, default=None, help="[Mpc/h]; default from config")
    p.add_argument("--no-beta-nl", action="store_true", help="Disable β^NL even if config enables it")
    p.add_argument("--mass-function", default=None, metavar="NAME",
                   help="CCL halo mass function (e.g. Tinker08 [package default], Tinker10)")
    p.add_argument("--halo-bias", default=None, metavar="NAME",
                   help="CCL halo bias model (default Tinker10)")
    p.add_argument("--concentration", default=None, metavar="NAME",
                   help="CCL concentration-mass relation (default Duffy08)")
    p.add_argument("--mass-def", default=None, metavar="MassDefXXX",
                   help="CCL halo mass definition for the halo model (e.g. MassDef200m, "
                        "MassDef200c); overrides config csmf.halo_model.mass_definition. "
                        "Default: package default MassDef200c unless the config sets it.")
    p.add_argument("--beta-nl-log-m-min", type=float, default=None,
                   help="Override the emulator β^NL interpolation mass floor "
                        "[log10 M_sun/h] (config default 11.5; package default 12.0)")
    p.add_argument("--fix", nargs="*", default=None,
                   metavar="PARAM=VALUE", help="Fix parameters, e.g. --fix f_h=1.0 f_s=1.0")
    p.add_argument("--gamma1-mean", type=float, default=None,
                   help="Mean of a Gaussian prior on gamma1 (overrides config csmf.gamma1_prior)")
    p.add_argument("--gamma1-std", type=float, default=None,
                   help="Std of the Gaussian prior on gamma1 (overrides config)")
    p.add_argument("--gamma1-flat", action="store_true",
                   help="Force the flat [2.5,15] gamma1 prior, ignoring any config Gaussian")
    p.add_argument("--gamma1-prior", choices=["gaussian", "flat"], default=None,
                   help="Decision-style switch: 'gaussian' = the config Gaussian "
                        "(Dvornik+23, 7.10 +- 2.0), 'flat' = the package flat prior "
                        "(same as --gamma1-flat)")
    p.add_argument("--fit-ngal", dest="fit_ngal", action="store_true", default=None,
                   help="Add the galaxy number-density (abundance) anchor to the "
                        "likelihood (needs n_gal in the csmf_input npz). Default: "
                        "config csmf.fit_ngal.")
    p.add_argument("--no-fit-ngal", dest="fit_ngal", action="store_false",
                   help="Disable the n_gal anchor even if config enables it.")
    p.add_argument("--ngal-anchor", choices=["ngal", "none"], default=None,
                   help="Decision-style switch for the abundance anchor: 'ngal' = "
                        "Delta Sigma + n_gal, 'none' = Delta Sigma only (wins over "
                        "--fit-ngal / config)")
    p.add_argument("--drop-massbins", nargs="*", default=None, metavar="NAME:i[,j]",
                   help="Exclude mass bins per sample (highest = last index), e.g. "
                        "--drop-massbins BGS_RED_GMM:9 LRG:3")
    p.add_argument("--method", default="de", choices=["smoke", "de", "minuit", "nautilus"])
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--n-live", type=int, default=2000)
    p.add_argument("--n-eff", type=int, default=10000)
    p.add_argument("--vectorized", action="store_true",
                   help="nautilus only: use the jax.vmap-batched ΔΣ likelihood "
                        "(vectorized=True). Equivalent posterior, much faster.")
    p.add_argument("--resume-from", default=None, metavar="CKPT.h5",
                   help="nautilus only: resume an interrupted run from a copy of its "
                        "checkpoint. The file is copied to <out>.h5 (unless that already "
                        "exists) before the sampler starts, and nautilus resumes from it "
                        "(same seed / n_live / --vectorized required). Needed under "
                        "Snakemake, which wipes the output directory -- and with it the "
                        "checkpoint -- before re-running an interrupted rule.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-from", default=None,
                   help="minuit only: seed the minimiser from the best-fit params of an "
                        "existing csmf_fit_*.npz (reads minuit_best_fit/minuit_param_names). "
                        "Cold-start minuit rails to a bad low-M1 basin on the VLIM-S/N samples; "
                        "seeding from a good SHMR recovers χ²/dof≈1. See csmf-snr-fit-coldstart.")
    p.add_argument("--passes", type=int, default=1,
                   help="minuit only: number of seeded Migrad passes. Pass i > 1 starts from "
                        "pass i-1's best fit (the paper's two-pass recipe); the intermediate "
                        "results are saved as <out stem>_pass{i}.npz. Ignored by de / nautilus.")
    p.add_argument("--label", default=None,
                   help="Tag inserted into the default output filename, e.g. 'bgs' or "
                        "'bgs_lrg', to keep multi-sample runs from clobbering each other")
    p.add_argument("--out", default=None, help="Results npz path (default under <dest>/ggl/csmf_fit)")
    p.add_argument("--quiet", dest="verbose", action="store_false")
    args = p.parse_args()

    # Parse --fix PARAM=VALUE entries into a dict.
    if args.fix:
        args.fix = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.fix}

    # Parse --drop-massbins NAME:i,j entries into {name: {indices}}.
    if args.drop_massbins:
        drop = {}
        for ent in args.drop_massbins:
            name, idxs = ent.split(":")
            drop[name] = {int(i) for i in idxs.split(",") if i != ""}
        args.drop_massbins = drop

    cfg = gu.load_config(args.config)
    fitter = build_fitter(cfg, args)

    if args.out is None:
        out_dir = Path(cfg["dest"]) / "ggl" / "csmf_fit"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{args.label}_" if args.label else ""
        args.out = str(out_dir / f"csmf_fit_{tag}{args.method}.npz")

    if args.method == "smoke":
        fitter._initialize_halo_model()
        init = fitter._get_initial_values()  # fiducial free-param start values
        ll = fitter.log_likelihood(init)
        print(f"\n[smoke] halo model OK; {len(fitter.mass_bins)} mass bins; "
              f"{len(init)} free params; log L at fiducial = {ll:.2f}")
        return

    if args.method == "de":
        res = fitter.minimize_de(maxiter=args.maxiter, workers=args.workers, seed=args.seed)
        res.print_summary()
    elif args.method == "minuit":
        start_params = None
        if args.start_from:
            d = np.load(args.start_from, allow_pickle=True)
            start_params = {str(n): float(v) for n, v in
                            zip(d["minuit_param_names"], d["minuit_best_fit"])}
            print(f"[seed] minuit start_params from {args.start_from}:\n"
                  f"       {start_params}", flush=True)
        for i in range(1, max(1, args.passes) + 1):
            res = fitter.minimize(run_hesse=True, start_params=start_params)
            res.print_summary()
            if i < args.passes:
                out_i = Path(args.out).with_name(Path(args.out).stem + f"_pass{i}.npz")
                fitter.save_results(str(out_i))
                start_params = {k: float(res.best_fit[k]) for k in res.param_names}
                print(f"[pass {i}/{args.passes}] saved {out_i}; re-seeding pass {i + 1} "
                      f"from its best fit", flush=True)
    elif args.method == "nautilus":
        ckpt = args.out + ".h5"
        if args.resume_from and not Path(ckpt).exists():
            Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.resume_from, ckpt)
            print(f"[resume] nautilus checkpoint copied from {args.resume_from} -> {ckpt}",
                  flush=True)
        fitter.run(n_live=args.n_live, n_eff=args.n_eff, seed=args.seed,
                   filepath=ckpt, vectorized=args.vectorized)

    fitter.save_results(args.out)
    # Record the halo-model ingredients actually used, so downstream scripts
    # (plot_aia_meff, plot_csmf_bestfit, decompose_csmf_chi2) can rebuild the
    # same model when they re-evaluate this fit.
    hm = fitter._halo_model
    extra = {"mass_definition": getattr(hm, "_mass_def_name", "MassDef200c"),
             "halo_model_kwargs": np.array(fitter.halo_model_kwargs, dtype=object)}
    if args.method == "nautilus":
        extra.update(nautilus_summary(fitter))
    d = dict(np.load(args.out, allow_pickle=True))
    d.update(extra)
    np.savez(args.out, **d)
    print(f"\nSaved fit results to {args.out}  (mass definition: {extra['mass_definition']})")


if __name__ == "__main__":
    main()
