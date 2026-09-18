#!/usr/bin/env python3
"""Fit the NLA / TATT IA model to the UNIONS x DESI correlation measurements.

Single entry point replacing the old collect_fits / nautilus driver scripts
(old_codes/lestgobaby.py, nautilus_sampler-2.py). Samples are enumerated from
``config/data_sources.yaml`` exactly like compute_correlations.tracer_jobs();
their measurement .npz files live under ``<corr-dir>/<tracer>/<stem>_GPU_multipoles.npz``
(xi0 + xi2 wedge multipoles) or ``..._CPU_projected.npz`` (wp + wgp). The fit uses
the Mohammad+21-corrected joint jackknife covariance (``cov_combined_corrected``)
plus the Hartlap-style inverse-covariance correction with n_realisations = the
jackknife patch count (read from compute_correlations.py / N_PATCHES).

SPLIT GRIDS (``--corr-dir-gp``)
------------------------------
The two statistics need not share a separation grid. ``--corr-dir-gp`` points the
IA block (xi2, or wgp) at a SECOND measurement tree, so the density block (xi0,
or wp) can keep a fine grid while the IA block uses a coarse one -- the
split-binning convention: xi0 on geomspace(6,100,20)=19 bins for b1 in the linear
regime, xi2 on geomspace(6,100,11)=10 bins because finer xi2 binning buys no IA
precision and only spends the fixed jackknife covariance budget.

The joint covariance is then BLOCK-DIAGONAL, built from each tree's own
per-statistic covariance. The cross-block is zero by construction, not by choice:
the per-patch jackknife realisations are not stored in the measurement npz, so no
cross-covariance between two different grids can be reconstructed. This is exactly
the standing ``--zero-cross-cov`` convention, under which the two statistics are
already fitted as independent blocks. The two trees are checked for the same
sample and the same jackknife tessellation before they are combined
(``--no-split-checks`` to override).

Examples
--------
    # Minuit NLA fit of the multipoles for every BGS shape sample
    python scripts/fit_ia.py --stat multipoles --model NLA --sampler minuit --tracer BGS

    # nautilus TATT fit of the projected statistics for the red GMM samples
    python scripts/fit_ia.py --stat projected --model TATT --sampler nautilus \\
        --sample RED_GMM --nlive 3000 --neff 50000 --pool 30

    # emcee, one LRG mass bin
    python scripts/fit_ia.py --stat multipoles --sampler emcee --tracer LRG --massbin 2

    # split grids: xi0 from the 19-bin tree, xi2 from the 10-bin coarse tree
    python scripts/fit_ia.py --stat multipoles --model NLA --sampler minuit \\
        --corr-dir    /n09data/rpaviot/DESIxUnions/correlations_highscales \\
        --corr-dir-gp /n09data/rpaviot/DESIxUnions/correlations_xi2coarse \\
        --rmin 30 10 --rmax 100 --no-hartlap --zero-cross-cov
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ia2pt import TwoPointModel, IALikelihood, PARAM_NAMES  # noqa: E402  (github.com/rpaviot/IA2pt)

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config" / "data_sources.yaml"
CORR_SCRIPT = Path(__file__).resolve().parent / "compute_correlations.py"

# Fixed fiducial cosmology (old_codes/lestgobaby.py, Planck-18-like).
DICT_COSMO = {
    "h": 0.6766,
    "Omc": 0.11933 / 0.6766**2,
    "Omb": 0.02242 / 0.6766**2,
    "s8": 0.8102,
    "A_s": 2.105209331337507e-09,
    "n_s": 0.9665,
    "Omnu": 0.0014034,
}

# Free params and priors per model (old_codes/lestgobaby.py / nautilus_sampler-2.py).
# scalar = fixed, (lo, hi) tuple = free with a flat prior. Edit here.
PRIORS = {
    "NLA": {
        "b1": (0.1, 4.0),
        "b2": 0.0,
        "a1": (-10.0, 10.0),
        "a2": 0.0,
        "bTA": 0.0,
    },
    "TATT": {
        "b1": (0.1, 4.0),
        "b2": (-2.0, 2.0),
        "a1": (-10.0, 10.0),
        "a2": (-3.0, 3.0),
        "bTA": (-3.0, 3.0),
    },
}

# Per-statistic default radial cuts [h^-1 Mpc] (lestgobaby defaults for the
# multipoles; the projected ones mirror them on wp/wgp).
RMIN_DEFAULTS = {"multipoles": (10.0, 6.0), "projected": (10.0, 6.0)}
RMAX_DEFAULT = 100.0  # fiducial: fit out to 100 h^-1 Mpc

# npz key names per statistic. "gg" is the density block (fitted first), "gp" the
# IA block (fitted second); the split-grid mode reads the gp block from a second
# tree, so every key that can differ between the two grids is named here.
STAT_KEYS = {
    "multipoles": {"r": "s_mid", "edges": "s_bins",
                   "gg": "xi0", "gp": "xi2",
                   "cov_gg": "cov_xi0", "cov_gp": "cov_xi2",
                   "gauss_gg": "cov_xi0_gauss", "gauss_gp": "cov_xi2_gauss",
                   "gauss_file": "gaussian_cov"},
    "projected": {"r": "rp_mid", "edges": "rp_bins",
                  "gg": "wp", "gp": "wgp",
                  "cov_gg": "cov_wp", "cov_gp": "cov_wgp",
                  "gauss_gg": "cov_wp_gauss", "gauss_gp": "cov_wgp_gauss",
                  "gauss_file": "gaussian_cov_projected"},
}

# Metadata that must agree between the two trees of a split-grid fit: same lens
# sample, same shape sample, same jackknife tessellation. Tolerance 0 -> exact.
SPLIT_CHECKS = (("n_patches", 0), ("kmeans_init", 0), ("use_fkp", 0),
                ("no_sysweights", 0), ("n_clustering", 0), ("n_shapes", 0),
                ("z_eff_clustering", 1e-6), ("z_eff_IA", 1e-6))
# Of those, the ones that pin the jackknife tessellation -- the covariance depends
# on it, and re-drawing the patches on identical data has moved an amplitude by
# >2 sigma. Trees written before these keys existed (e.g. correlations_highscales)
# cannot be checked from the npz; say so loudly rather than pass in silence.
SPLIT_CRITICAL = {"n_patches", "kmeans_init"}


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def read_n_patches(script=CORR_SCRIPT):
    """Jackknife patch count from compute_correlations.py (N_PATCHES = ...)."""
    m = re.search(r"^N_PATCHES\s*=\s*(\d+)", script.read_text(), re.M)
    if not m:
        raise RuntimeError(f"N_PATCHES not found in {script}")
    return int(m.group(1))


def enumerate_samples(cfg):
    """Replicate compute_correlations.tracer_jobs() shape enumeration: one entry per
    (tracer, z-bin, shape sample [, mass bin]) with the output identity."""
    ia = cfg["ia_samples"]
    out = []
    for t in ia["tracers"]:
        for zlo, zhi in t["zbins"]:
            ztag = f"zmin_{zlo:.2f}_zmax_{zhi:.2f}"
            for s in t["shape_samples"]:
                sbase = f"{s['name']}_{ztag}"
                out_tracer = s["name"].split("_")[0]
                out.append({"tracer": t["name"], "z_range": (zlo, zhi),
                            "stem": sbase, "out_tracer": out_tracer, "massbin": None})
                if s.get("n_mass_bins"):
                    for i in range(int(s["n_mass_bins"])):
                        out.append({"tracer": t["name"], "z_range": (zlo, zhi),
                                    "stem": f"{sbase}_massbin{i}",
                                    "out_tracer": out_tracer, "massbin": i})
    return out


def npz_path(corr_dir, sample, stat):
    suffix = "GPU_multipoles" if stat == "multipoles" else "CPU_projected"
    return Path(corr_dir) / sample["out_tracer"] / f"{sample['stem']}_{suffix}.npz"


def load_sample_properties(cfg, stem):
    """sigma_e / n_eff / r_mean etc. from the sample_properties JSON, if present."""
    props_json = (Path(cfg["dest"]) / cfg["ia_samples"]["out_subdir"]
                  / "sample_properties" / f"{stem}.json")
    if props_json.exists():
        with open(props_json) as fh:
            return json.load(fh)
    return {}


def block_diag_cov(cov_gg, cov_gp):
    """Joint covariance of two statistics measured on DIFFERENT separation grids.

    The cross-block is zero by construction, not by choice: the per-patch
    jackknife realisations are not stored in the measurement npz, so no
    cross-covariance between two grids can be reconstructed. Under the standing
    --zero-cross-cov convention the fit zeroes that block anyway."""
    n_gg, n_gp = len(cov_gg), len(cov_gp)
    cov = np.zeros((n_gg + n_gp, n_gg + n_gp))
    cov[:n_gg, :n_gg] = cov_gg
    cov[n_gg:, n_gg:] = cov_gp
    return cov


def check_split_consistency(data_gg, data_gp, f_gg, f_gp, strict=True):
    """The two trees of a split-grid fit must describe the same sample measured on
    the same jackknife tessellation; otherwise stacking them into one data vector
    with a block-diagonal covariance is meaningless. Re-drawing the 70 patches on
    identical data has moved an amplitude by >2 sigma, so this is checked, not
    assumed."""
    problems, unchecked = [], []
    for key, tol in SPLIT_CHECKS:
        if key not in data_gg or key not in data_gp:
            unchecked.append(key)
            continue
        a, b = data_gg[key], data_gp[key]
        same = (abs(float(a) - float(b)) <= tol) if tol else np.array_equal(a, b)
        if not same:
            problems.append(f"{key}: {a} [{f_gg.parent.parent.name}] "
                            f"!= {b} [{f_gp.parent.parent.name}]")
    if unchecked:
        print(f"  NOTE: absent from one or both npz, so NOT compared: "
              f"{', '.join(unchecked)}", flush=True)
        missed = SPLIT_CRITICAL & set(unchecked)
        if missed:
            print(f"  NOTE: the jackknife tessellation ({', '.join(sorted(missed))}) "
                  f"could NOT be verified from the measurement npz -- older trees "
                  f"predate these keys. Confirm from the run logs that both "
                  f"measurements loaded the same cached patch centres.", flush=True)
    if not problems:
        return
    msg = ("split-grid sources disagree, so a block-diagonal covariance is not "
           "valid:\n    " + "\n    ".join(problems))
    if strict:
        raise RuntimeError(msg + "\n  (override with --no-split-checks)")
    print(f"  WARNING: {msg}", flush=True)


def load_blocks(data_gg, data_gp, stat, raw_cov):
    """Per-statistic separations, bin edges, data vectors and the joint covariance.

    ``data_gp is data_gg`` is the usual single-tree fit and keeps the measured
    joint covariance (cross-blocks included). Two distinct trees means split
    grids, and the covariance is assembled block-diagonally."""
    K = STAT_KEYS[stat]
    suffix = "" if raw_cov else "_corrected"
    r_list = [np.asarray(data_gg[K["r"]]), np.asarray(data_gp[K["r"]])]
    edges_list = [np.asarray(data_gg[K["edges"]]), np.asarray(data_gp[K["edges"]])]
    vectors = [np.asarray(data_gg[K["gg"]]), np.asarray(data_gp[K["gp"]])]
    if data_gp is data_gg:
        cov = np.asarray(data_gg["cov_combined" + suffix])
    else:
        cov = block_diag_cov(np.asarray(data_gg[K["cov_gg"] + suffix]),
                             np.asarray(data_gp[K["cov_gp"] + suffix]))
    for r, e, v in zip(r_list, edges_list, vectors):
        if not (len(r) == len(v) == len(e) - 1):
            raise RuntimeError(f"grid/data length mismatch: r={len(r)}, "
                               f"data={len(v)}, edges={len(e)}")
    return r_list, edges_list, vectors, cov


def load_gaussian_cov(stem, f_gg, f_gp, data_gg, data_gp, stat):
    """Analytic Gaussian covariance, from each block's own tree when grids split."""
    K = STAT_KEYS[stat]

    def _read(npz_file, data):
        gfile = npz_file.with_name(f"{stem}_{K['gauss_file']}.npz")
        g = np.load(gfile, allow_pickle=True)
        if not np.allclose(g[K["edges"]], data[K["edges"]]):
            raise RuntimeError(f"{gfile.name}: {K['edges']} do not match "
                               f"{npz_file.name}")
        return g

    g_gg = _read(f_gg, data_gg)
    if data_gp is data_gg:
        return np.asarray(g_gg["cov_combined_gauss"])
    g_gp = _read(f_gp, data_gp)
    return block_diag_cov(np.asarray(g_gg[K["gauss_gg"]]),
                          np.asarray(g_gp[K["gauss_gp"]]))


def pair_effective_redshift(data):
    """Pair-weighted effective redshift of the density x shape cross-correlation,
    z_pair = int z W(z) dz with W ∝ n_d(z) n_s(z) / (chi^2 dchi/dz) -- the same
    window TwoPointModel.set_nz builds for the n(z)-integrated prediction, so the
    single-z model evaluated at z_pair is its effective-redshift approximation.
    n_d is the (FKP-weighted, when used) density n(z), n_s the lensing-weighted
    shape n(z), both as stored by compute_correlations.py."""
    import pyccl as ccl
    h = DICT_COSMO["h"]
    mnu = DICT_COSMO["Omnu"] * 93.14 * h**2
    cosmo = ccl.Cosmology(Omega_c=DICT_COSMO["Omc"], Omega_b=DICT_COSMO["Omb"],
                          m_nu=mnu, h=h, A_s=DICT_COSMO["A_s"], n_s=DICT_COSMO["n_s"])
    z = np.asarray(data["z_nz"], float)
    nd = np.asarray(data["nz_clustering"], float)
    ns = np.asarray(data["nz_shape"], float)
    a = 1.0 / (1.0 + z)
    chi = ccl.comoving_radial_distance(cosmo, a)
    dchi_dz = (2997.92458 / h) / ccl.h_over_h0(cosmo, a)   # c/H(z) [Mpc]
    w = nd * ns / np.maximum(chi**2 * dchi_dz, 1e-30)
    return float(np.sum(z * w) / np.sum(w))


def build_likelihood(data, args, model_config, cov=None, data_gp=None):
    """Instantiate TwoPointModel + IALikelihood for one sample's npz.

    ``data_gp`` is the second measurement npz when the IA block lives on its own
    separation grid (--corr-dir-gp); it defaults to ``data``, the single-tree case.
    ``cov`` overrides the covariance read from the measurement npz (used for the
    analytic Gaussian covariance)."""
    if data_gp is None:
        data_gp = data
    h = DICT_COSMO["h"]
    mnu = DICT_COSMO["Omnu"] * 93.14 * h**2  # Omega_nu h^2 -> sum m_nu [eV]
    zeff = float(data["z_eff_clustering"])
    cosmology = [DICT_COSMO["Omc"], DICT_COSMO["Omb"], mnu,
                 DICT_COSMO["A_s"], DICT_COSMO["n_s"], h, zeff]

    def _make_model(z):
        return TwoPointModel(cosmology[:-1] + [z], model_config, computation=[],
                             do_rsd=True, pimax=100, include_B=True,
                             bin_avg=args.bin_avg,
                             evolve_bias=False, rp_min_wedge=args.rp_min_wedge,
                             n_mu_wedge=101)

    # Clustering block (density auto-correlation): always at the density z_eff.
    model = _make_model(zeff)
    if args.use_nz:
        model.set_nz(data["z_bins"], data["z_nz"],
                     data["nz_clustering"], data["nz_shape"])

    # IA block (density x shape cross-correlation). Default 'clustering' keeps
    # the historical behaviour (one model, everything at the density z_eff --
    # wrong for a shape sub-sample whose n(z) differs from the density's, as the
    # NLA/TATT amplitude scales with D(z_pair)). 'pair' evaluates the IA
    # statistic at the pair-weighted redshift, 'shape' at the shape z_eff. With
    # --use-nz the n(z) window does the job and one model serves both blocks.
    zeff_ia_mode = getattr(args, "zeff_ia", "clustering")
    if args.use_nz or zeff_ia_mode == "clustering":
        model_ia, zeff_ia = model, zeff
    else:
        zeff_ia = (pair_effective_redshift(data_gp) if zeff_ia_mode == "pair"
                   else float(data_gp["z_eff_IA"]))
        model_ia = _make_model(zeff_ia)
    model.zeff_ia_used = zeff_ia          # recorded in the output npz
    print(f"  model z_eff: clustering {zeff:.4f}  IA {zeff_ia:.4f} "
          f"({'n(z)-integrated' if args.use_nz else zeff_ia_mode})", flush=True)

    if args.stat == "multipoles":
        funcs = {"xi0e": model.compute_xi_gg_wedge_monopole,
                 "xi2p": model_ia.compute_xi_gi_wedge_quadrupole}
    else:
        funcs = {"WGG": model.compute_wgg_v2, "WGP": model_ia.compute_wgp_v2}

    r_list, edges_list, vectors, cov_meas = load_blocks(data, data_gp, args.stat,
                                                        args.raw_cov)
    if cov is None:
        cov = cov_meas
    n_real = args.n_patches if args.n_patches is not None else read_n_patches()
    lik = IALikelihood(funcs, r_list, vectors, cov,
                       n_realisations=n_real, jack=not args.no_hartlap,
                       edges_list=edges_list if args.bin_avg else None,
                       bin_avg=args.bin_avg, taper_scale=args.cov_taper,
                       zero_cross_cov=args.zero_cross_cov)
    lik.set_cut([args.rmin_gg, args.rmin_gp], [args.rmax, args.rmax])
    prior = dict(PRIORS[model_config])
    for name, lo, hi in (getattr(args, "prior", None) or []):
        if name not in prior:
            raise SystemExit(f"--prior: unknown parameter {name!r}; "
                             f"expected one of {sorted(prior)}")
        prior[name] = (float(lo), float(hi))
    if getattr(args, "fix_bta", None) is not None:
        prior["bTA"] = float(args.fix_bta)  # fix the TA density weighting (0 = off)
    if getattr(args, "fix_b2", False):
        prior["b2"] = 0.0   # drop the nonlinear-bias term (linear bias only)
    lik.set_prior(prior)
    return model, lik, r_list, vectors, edges_list


def weighted_quantile(values, weights, q):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(q) * c[-1])]]


def run_minuit(lik):
    """Migrad over the free params; returns the result dict."""
    from iminuit import Minuit
    # start values from the old driver (lestgobaby): b1=1, a1=1, rest 0
    init = {"b1": 1.0, "b2": 0.0, "a1": 1.0, "a2": 0.0, "bTA": 0.0}
    start = {k: (init[k] if isinstance(v, tuple) else v)
             for k, v in lik.prior.items()}
    m = Minuit(lik, **start)
    for name, value in lik.prior.items():
        if isinstance(value, tuple):
            m.limits[name] = value
        else:
            m.fixed[name] = True
    m.migrad()
    bestfit = {k: m.values[k] for k in PARAM_NAMES}
    errors = {k: (m.errors[k] if k in lik.free_params else 0.0) for k in PARAM_NAMES}
    return {"bestfit": bestfit, "errors": errors, "chi2": float(m.fval),
            "valid": bool(m.valid)}


def run_nautilus(lik, n_live, n_eff, pool):
    points, log_w, log_l = lik.call_sampler(n_eff=n_eff, n_live=n_live, pool=pool)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    bestfit, errors = {}, {}
    free = list(lik.free_params)
    for name in PARAM_NAMES:
        if name in free:
            j = free.index(name)  # points columns follow free_params order
            bestfit[name] = float(weighted_quantile(points[:, j], w, 0.5))
            errors[name] = float(0.5 * (weighted_quantile(points[:, j], w, 0.84)
                                        - weighted_quantile(points[:, j], w, 0.16)))
        else:
            bestfit[name] = float(lik.fixed_params[name])
            errors[name] = 0.0
    chi2 = float(-2.0 * log_l.max())
    return {"bestfit": bestfit, "errors": errors, "chi2": chi2,
            "chain": points, "log_w": log_w, "log_l": log_l}


def run_emcee(lik, nwalkers, nsteps):
    import emcee
    ndim = lik.ndim
    rng = np.random.default_rng(42)
    p0 = np.array([[rng.uniform(*lik.prior[name]) for name in lik.free_params]
                   for _ in range(nwalkers)])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lik.log_prob_emcee)
    sampler.run_mcmc(p0, nsteps, progress=False)
    burn = nsteps // 2
    flat = sampler.get_chain(discard=burn, flat=True)
    logp = sampler.get_log_prob(discard=burn, flat=True)
    bestfit, errors = {}, {}
    for name in PARAM_NAMES:
        if name in lik.free_params:
            j = lik.free_params.index(name)
            bestfit[name] = float(np.median(flat[:, j]))
            errors[name] = float(np.std(flat[:, j]))
        else:
            bestfit[name] = float(lik.fixed_params[name])
            errors[name] = 0.0
    return {"bestfit": bestfit, "errors": errors, "chi2": float(-2.0 * logp.max()),
            "chain": flat, "log_l": logp}


def fit_sample(sample, npz_file, cfg, args, npz_file_gp=None):
    """Run one fit and save its result npz; returns the output path.

    ``npz_file_gp`` is the IA block's own measurement npz when the two statistics
    live on different separation grids (--corr-dir-gp)."""
    K = STAT_KEYS[args.stat]
    data = dict(np.load(npz_file, allow_pickle=True))
    split = npz_file_gp is not None and Path(npz_file_gp) != Path(npz_file)
    if split:
        data_gp = dict(np.load(npz_file_gp, allow_pickle=True))
        check_split_consistency(data, data_gp, Path(npz_file), Path(npz_file_gp),
                                strict=not args.no_split_checks)
    else:
        data_gp = data

    cov = None
    if args.gaussian_cov:
        cov = load_gaussian_cov(sample["stem"], Path(npz_file),
                                Path(npz_file_gp) if split else Path(npz_file),
                                data, data_gp, args.stat)
    model, lik, r_full, vectors, edges_full = build_likelihood(data, args, args.model,
                                                               cov=cov, data_gp=data_gp)
    cov_used = cov if cov is not None else load_blocks(data, data_gp, args.stat,
                                                       args.raw_cov)[3]
    if split:
        print(f"  split grids: {K['gg']} {len(r_full[0])} bins "
              f"[{r_full[0][0]:.2f}, {r_full[0][-1]:.2f}] from {Path(npz_file).parent.parent.name}"
              f"  |  {K['gp']} {len(r_full[1])} bins "
              f"[{r_full[1][0]:.2f}, {r_full[1][-1]:.2f}] from "
              f"{Path(npz_file_gp).parent.parent.name}; block-diagonal covariance",
              flush=True)

    if args.sampler == "minuit":
        res = run_minuit(lik)
    elif args.sampler == "nautilus":
        res = run_nautilus(lik, args.nlive, args.neff, args.pool)
    else:
        res = run_emcee(lik, args.nwalkers, args.nsteps)

    bf = [res["bestfit"][k] for k in PARAM_NAMES]
    dof = lik.ndata - lik.ndim
    pred_cut = lik.get_bestfit(*bf)
    if args.bin_avg:
        pred_full = lik.get_bestfit(*bf, edges_list=edges_full)
    else:
        pred_full = lik.get_bestfit(*bf, r_list=r_full)

    props = load_sample_properties(cfg, sample["stem"])

    out = {
        "stat": args.stat, "model": args.model, "sampler": args.sampler,
        "sample": sample["stem"], "tracer": sample["tracer"],
        "z_range": np.array(sample["z_range"]),
        "param_names": np.array(PARAM_NAMES),
        "bestfit": np.array([res["bestfit"][k] for k in PARAM_NAMES]),
        "errors": np.array([res["errors"][k] for k in PARAM_NAMES]),
        "free_params": np.array(lik.free_params),
        "chi2": res["chi2"], "dof": dof,
        "rmin_gg": args.rmin_gg, "rmin_gp": args.rmin_gp, "rmax": args.rmax,
        "bin_avg": args.bin_avg, "cov_taper": args.cov_taper or 0.0,
        "zero_cross_cov": args.zero_cross_cov, "raw_cov": args.raw_cov,
        "n_realisations": lik.n_realisations, "hartlap": not args.no_hartlap,
        # per-block separations. r_mid (both blocks share one grid) is written
        # ONLY when they really do -- a split-grid fit omits it on purpose, so
        # readers that slice model_full at len(r_mid) fail loudly instead of
        # silently mis-slicing two blocks of different length.
        "r_mid_gg": r_full[0], "r_mid_gp": r_full[1],
        "edges_gg": edges_full[0], "edges_gp": edges_full[1],
        "n_gg": len(r_full[0]), "n_gp": len(r_full[1]),
        "split_grid": split,
        "corr_dir_gg": str(Path(npz_file).parent.parent),
        "corr_dir_gp": str(Path(npz_file_gp if split else npz_file).parent.parent),
        "data_gg": vectors[0], "data_gp": vectors[1],
        "cov": cov_used,
        "cov_source": "gaussian" if args.gaussian_cov else "jackknife",
        "r_fit_gg": lik.r_fit[0], "r_fit_gp": lik.r_fit[1],
        "model_cut": pred_cut, "model_full": pred_full,
        "cosmology": json.dumps(DICT_COSMO),
        "z_eff_model_gg": float(data["z_eff_clustering"]),
        "z_eff_model_gp": float(getattr(model, "zeff_ia_used",
                                        data["z_eff_clustering"])),
        "zeff_ia_mode": "nz" if args.use_nz else getattr(args, "zeff_ia", "clustering"),
        "priors": json.dumps({k: list(v) if isinstance(v, tuple) else v
                              for k, v in lik.prior.items()}),
    }
    if not split:
        out["r_mid"] = r_full[0]
    # posterior chains, when produced
    for key in ("chain", "log_w", "log_l"):
        if key in res:
            out[key] = res[key]
    # carry every input-npz metadata key not already stored (z_eff, n(z), mass stats..)
    for k, v in data.items():
        if k not in out:
            out[k] = v
    if split:
        # the gg tree carries same-named keys for the gp block on the WRONG grid;
        # the gp block's measurement and covariance must come from its own tree.
        for k in (K["gp"], K["cov_gp"], K["cov_gp"] + "_corrected"):
            if k in data_gp:
                out[k] = data_gp[k]
        out[K["r"] + "_gp"] = data_gp[K["r"]]
        out[K["edges"] + "_gp"] = data_gp[K["edges"]]
        # the gg tree's joint covariance spans a grid pair this fit never used
        for k in ("cov_combined", "cov_combined_corrected"):
            out.pop(k, None)
    # sample properties (sigma_e / n_eff / r_mean ...) when available
    for k, v in props.items():
        if k not in ("sample", "parquet"):
            out[f"prop_{k}"] = v

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (f"{sample['stem']}_{args.stat}_{args.model}_"
                          f"{args.sampler}.npz")
    np.savez(out_file, **out)
    print(f"  bestfit: " + ", ".join(f"{k}={res['bestfit'][k]:.3f}"
                                     f"+-{res['errors'][k]:.3f}"
                                     for k in lik.free_params), flush=True)
    print(f"  chi2/dof = {res['chi2']:.2f}/{dof}   -> {out_file}", flush=True)
    return out_file


def main():
    p = argparse.ArgumentParser(description="Fit the IA model to UNIONS x DESI correlations")
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--stat", choices=["multipoles", "projected"], default="multipoles")
    p.add_argument("--model", choices=["NLA", "TATT"], default="NLA")
    p.add_argument("--fix-bta", nargs="?", type=float, const=0.0, default=None,
                   metavar="VALUE",
                   help="fix the TATT tidal-alignment bias bTA (bare flag: 0, i.e. "
                        "TA+TT with no density weighting; or give a value, e.g. 1)")
    p.add_argument("--fix-b2", action="store_true",
                   help="fix the nonlinear galaxy bias b2 to 0 (linear bias only)")
    p.add_argument("--prior", action="append", nargs=3, default=None,
                   metavar=("PARAM", "LO", "HI"),
                   help="widen/narrow the flat prior on PARAM, e.g. --prior a2 -5 5. "
                        "Repeatable; overrides the model default. Applied before "
                        "--fix-bta/--fix-b2, so fixing a parameter still wins.")
    p.add_argument("--sampler", choices=["minuit", "nautilus", "emcee"], default="minuit")
    p.add_argument("--tracer", default=None, help="only this tracer (BGS/LRG/ELG)")
    p.add_argument("--sample", "--colour", dest="sample", action="append",
                   default=None, metavar="PATTERN",
                   help="substring match on the shape-sample stem "
                        "(e.g. RED_GMM, BLUE_SFR, ANY); repeatable, a sample is "
                        "kept if it matches ANY pattern")
    p.add_argument("--exclude", action="append", default=None, metavar="PATTERN",
                   help="drop samples whose stem contains PATTERN; repeatable. "
                        "Applied AFTER --sample/--tracer (e.g. --exclude SFR)")
    p.add_argument("--shard", nargs=2, type=int, default=None,
                   metavar=("INDEX", "NTASKS"),
                   help="process only samples with (position %% NTASKS) == INDEX, "
                        "for SLURM job arrays: --shard $SLURM_ARRAY_TASK_ID $N. "
                        "Sharding is applied last, after every other filter, so "
                        "the shards of one array partition the corpus exactly")
    p.add_argument("--zbin", default=None,
                   help="z-bin filter as 'zlo,zhi' (e.g. 0.40,0.75)")
    p.add_argument("--massbin", default=None,
                   help="'full' = no mass bins, an int = only that mass bin; "
                        "default: everything")
    p.add_argument("--rmin", type=float, nargs=2, default=None,
                   metavar=("RMIN_GG", "RMIN_GP"),
                   help="per-statistic rmin (defaults: multipoles 10 6; projected 10 6)")
    p.add_argument("--rmax", type=float, default=RMAX_DEFAULT)
    p.add_argument("--rp-min-wedge", type=float, default=None,
                   help="transverse wedge cut for the multipole model "
                        "(default: none, matching the norpcut measurements)")
    p.add_argument("--use-nz", action="store_true",
                   help="n(z)-weight the model (default: single z_eff, as the old fits)")
    p.add_argument("--zeff-ia", default="clustering",
                   choices=["clustering", "pair", "shape"],
                   help="redshift at which the IA block (xi2 / wgp) is modelled: "
                        "'clustering' = the density z_eff (historical default, one "
                        "model for both blocks); 'pair' = the pair-weighted z of the "
                        "density x shape cross (n_d n_s / chi^2 dchi window, "
                        "recommended for shape sub-samples); 'shape' = z_eff_IA. "
                        "The clustering block always stays at the density z_eff. "
                        "Ignored with --use-nz.")
    p.add_argument("--bin-avg", action="store_true",
                   help="volume-average the model over each radial bin (uses the "
                        "stored bin edges) instead of evaluating at the bin centre")
    p.add_argument("--cov-taper", type=float, default=None, metavar="R_TAP",
                   help="taper the covariance with a Wendland kernel of lag |s_i-s_j|, "
                        "scale R_TAP [h^-1 Mpc] (Paz & Sanchez 2015); suppresses noisy "
                        "long-lag off-diagonals")
    p.add_argument("--zero-cross-cov", action="store_true",
                   help="zero the cross-covariance between statistics (e.g. monopole vs "
                        "quadrupole), fitting them as independent; old_codes diagnostic")
    p.add_argument("--no-hartlap", action="store_true",
                   help="skip the Hartlap-style inverse-covariance correction")
    p.add_argument("--gaussian-cov", action="store_true",
                   help="use the analytic Gaussian covariance "
                        "(<stem>_gaussian_cov.npz from compute_gaussian_cov.py, "
                        "cov_combined_gauss block) instead of the jackknife; "
                        "for --stat projected reads <stem>_gaussian_cov_projected.npz "
                        "(compute_gaussian_cov_projected.py); requires --no-hartlap")
    p.add_argument("--raw-cov", action="store_true",
                   help="use the un-corrected jackknife covariance (cov_combined) "
                        "instead of the Mohammad+21-corrected one (cov_combined_corrected)")
    p.add_argument("--n-patches", type=int, default=None,
                   help="override the jackknife patch count used for the Hartlap "
                        "correction (default: read N_PATCHES from compute_correlations.py)")
    p.add_argument("--corr-dir", default=None,
                   help="measurement npz root (default: <dest>/correlations); with "
                        "--corr-dir-gp this supplies only the density block "
                        "(xi0 / wp)")
    p.add_argument("--corr-dir-gp", default=None,
                   help="separate measurement root for the IA block (xi2 / wgp), so "
                        "the two statistics are fitted on DIFFERENT separation grids "
                        "(split binning). The joint covariance is then block-diagonal: "
                        "each block's own cov from its own tree, zero cross-block "
                        "(the per-patch realisations needed to build one are not "
                        "stored). Both trees must be the same sample on the same "
                        "jackknife tessellation.")
    p.add_argument("--no-split-checks", action="store_true",
                   help="downgrade the --corr-dir-gp consistency check (same sample, "
                        "same n_patches / tessellation / weighting) from an error to "
                        "a warning")
    p.add_argument("--out-dir", default=str(REPO / "results" / "fits"))
    p.add_argument("--nlive", type=int, default=3000, help="nautilus live points")
    p.add_argument("--neff", type=int, default=50000, help="nautilus effective samples")
    p.add_argument("--nwalkers", type=int, default=32, help="emcee walkers")
    p.add_argument("--nsteps", type=int, default=5000, help="emcee steps")
    p.add_argument("--pool", type=int, default=1, help="nautilus pool size")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.rmin_gg, args.rmin_gp = args.rmin if args.rmin else RMIN_DEFAULTS[args.stat]

    if args.gaussian_cov:
        if not args.no_hartlap:
            p.error("--gaussian-cov requires --no-hartlap (the Gaussian covariance "
                    "is analytic; a Hartlap correction would be meaningless)")
        if args.raw_cov or args.cov_taper:
            p.error("--gaussian-cov is incompatible with --raw-cov / --cov-taper")

    if args.no_split_checks and not args.corr_dir_gp:
        p.error("--no-split-checks only means anything with --corr-dir-gp")

    cfg = load_config(args.config)
    corr_dir = Path(args.corr_dir) if args.corr_dir else Path(cfg["dest"]) / "correlations"
    corr_dir_gp = Path(args.corr_dir_gp) if args.corr_dir_gp else None

    samples = enumerate_samples(cfg)
    if args.tracer:
        samples = [s for s in samples if s["tracer"] == args.tracer]
    if args.sample:
        samples = [s for s in samples
                   if any(pat in s["stem"] for pat in args.sample)]
    if args.exclude:
        samples = [s for s in samples
                   if not any(pat in s["stem"] for pat in args.exclude)]
    if args.zbin:
        zlo, zhi = (float(x) for x in args.zbin.split(","))
        samples = [s for s in samples
                   if abs(s["z_range"][0] - zlo) < 1e-6 and abs(s["z_range"][1] - zhi) < 1e-6]
    if args.massbin is not None:
        if args.massbin == "full":
            samples = [s for s in samples if s["massbin"] is None]
        else:
            samples = [s for s in samples if s["massbin"] == int(args.massbin)]

    # Shard LAST: every other filter has run, so the sample list is identical
    # across array tasks and the shards partition it exactly (no gaps/overlaps).
    n_total = len(samples)
    if args.shard is not None:
        idx, ntasks = args.shard
        if ntasks < 1 or not (0 <= idx < ntasks):
            p.error(f"--shard INDEX NTASKS: need 1<=NTASKS and 0<=INDEX<NTASKS, "
                    f"got {idx} {ntasks}")
        samples = [s for i, s in enumerate(samples) if i % ntasks == idx]

    print(f"stat={args.stat} model={args.model} sampler={args.sampler} "
          f"rmin=({args.rmin_gg}, {args.rmin_gp}) rmax={args.rmax}", flush=True)
    if args.shard is not None:
        print(f"shard {args.shard[0]}/{args.shard[1]}: "
              f"{len(samples)} of {n_total} samples", flush=True)
    print(f"{len(samples)} samples selected; corr dir = {corr_dir}", flush=True)
    if corr_dir_gp:
        K = STAT_KEYS[args.stat]
        print(f"SPLIT GRIDS: {K['gg']} from {corr_dir}\n"
              f"             {K['gp']} from {corr_dir_gp}\n"
              f"             -> block-diagonal covariance, zero cross-block",
              flush=True)

    gsuffix = STAT_KEYS[args.stat]["gauss_file"]
    for sample in samples:
        f = npz_path(corr_dir, sample, args.stat)
        f_gp = npz_path(corr_dir_gp, sample, args.stat) if corr_dir_gp else None
        out_file = (Path(args.out_dir) / (f"{sample['stem']}_{args.stat}_"
                                          f"{args.model}_{args.sampler}.npz"))
        gfiles = [f.with_name(f"{sample['stem']}_{gsuffix}.npz")]
        if f_gp is not None:
            gfiles.append(f_gp.with_name(f"{sample['stem']}_{gsuffix}.npz"))
        status = ("MISSING npz" if not f.exists()
                  else "MISSING gp npz" if f_gp is not None and not f_gp.exists()
                  else "MISSING gaussian cov"
                       if args.gaussian_cov and not all(g.exists() for g in gfiles)
                  else "done (skip)" if out_file.exists() and not args.overwrite
                  else "fit")
        print(f"\n[{sample['stem']}] {f.name}: {status}", flush=True)
        if args.dry_run or status != "fit":
            continue
        try:
            fit_sample(sample, f, cfg, args, npz_file_gp=f_gp)
        except Exception as exc:  # keep going across samples
            print(f"  ERROR: {exc}", flush=True)
            import traceback
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
