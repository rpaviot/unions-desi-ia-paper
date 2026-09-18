#!/usr/bin/env python3
"""Magnification-bias alphas for the GGL lens samples.

Clean-pipeline port of ``old_codes/lens_magnification-2.py``: alpha =
dlnN/dkappa / 2 by finite differences, perturbing the photometry at +/-kappa
and re-applying the full lens selection. Per (lens sample, mass bin):

  1. the bin's galaxies come from the IA shapes parquet (TARGETID membership);
     their fluxes/morphology are pulled from the analysis-ready per-tracer
     catalogue (``ggl.lens_samples[].catalogue``) by TARGETID;
  2. mags are rebuilt from FLUX_*/MW_TRANSMISSION_* (and FIBERFLUX_*); per
     kappa step the total mags get -2.5 log10(1+2k), the fiber fluxes a
     profile-dependent (2 - dlnF/dlntheta(SHAPE_R)) k factor, and LOGMSTAR a
     +log10(1+2k) shift;
  3. the selection re-applied at each +/-kappa is the photometric target cut
     (BGS_ANY or Zhou+22 LRG with PHOTSYS N/S split) AND the bin's stellar-mass
     window, recovered as [min, max] LOGMSTAR of the bin parquet;
  4. alpha is the kappa->0 intercept of a linear fit to the per-step amplitudes
     (weights = the bin's WEIGHT_CLUSTERING lens density weight).

Outputs one CSV + one JSON (full fit diagnostics) with the alphas + errors per
(sample, massbin) under ``<dest>/ggl/``.

Usage
-----
    python scripts/compute_magnification.py [--config CONFIG] [--samples NAME ...]
                                            [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402


def lens_alpha(cat, weights, m_lo, m_hi, selection, kappas, fiber_grid,
               use_exp_profile=False):
    """Alpha for one mass bin: selections at every +/-kappa, then the fit."""
    mag = {b: gu.flux_to_mag(cat[f"FLUX_{b}"], cat[f"MW_TRANSMISSION_{b}"])
           for b in ("G", "R", "Z", "W1")}
    fib = {b: gu.flux_to_mag(cat[f"FIBERFLUX_{b}"], cat[f"MW_TRANSMISSION_{b}"])
           for b in ("G", "R", "Z")}
    fc = gu.fiber_corrections(cat["SHAPE_R"].to_numpy(), *fiber_grid)
    logm = cat["LOGMSTAR"].to_numpy(float)
    photsys = cat["PHOTSYS"].to_numpy() if selection == "LRG" else None

    def select(kappa):
        m = {b: gu.lensed_total_mag(mag[b], kappa) for b in mag}
        lm = gu.lensed_logmass(logm, kappa)
        if selection == "LRG":
            fz = gu.lensed_fiber_mag(fib["Z"], kappa, fc)
            phot = gu.lrg_selection(m["G"], m["R"], m["Z"], m["W1"], fz, photsys)
        else:
            fr = gu.lensed_fiber_mag(fib["R"], kappa, fc)
            phot = gu.bgs_selection(m["G"], m["R"], m["Z"], m["W1"], fr,
                                    sample=selection)
        return phot & gu.stellar_mass_selection(lm, m_lo, m_hi)

    sel_left = [select(+k) for k in kappas]
    sel_right = [select(-k) for k in kappas]
    return gu.alpha_from_selections(sel_left, sel_right, kappas, weights)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(gu.CONFIG))
    ap.add_argument("--samples", nargs="+",
                    help="Restrict to these lens sample names (e.g. BGS_RED_GMM LRG).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = gu.load_config(args.config)
    ggl = cfg["ggl"]
    mc = ggl["magnification"]
    dest = Path(cfg["dest"])
    ia_dir = dest / ggl["ia_subdir"]
    cat_dir = dest / ggl["cat_subdir"]
    out_dir = dest / ggl["out_subdir"]
    wcol = mc.get("weight_column", "WEIGHT_CLUSTERING")
    kappas = float(mc["kappa_step"]) * np.arange(int(mc["n_kappa"]))
    samples = [s for s in ggl["lens_samples"]
               if not args.samples or s["name"] in args.samples]

    print(f"IA lens samples : {ia_dir}")
    print(f"catalogues      : {cat_dir}")
    print(f"out             : {out_dir}")
    print(f"kappas          : {kappas[1]:.4f} step, {len(kappas)} values; weight={wcol}\n")
    for s in samples:
        print(f"  {s['name']} z=[{s['zmin']:.2f},{s['zmax']:.2f}] "
              f"{s['n_mass_bins']} mass bins  selection={s['selection']}")
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    detail = {}
    for s in samples:
        use_exp = bool(mc.get("bgs_use_exp_profile", False)) and s["selection"] != "LRG"
        fiber_grid = gu.fiber_correction_grid(
            use_exp, fiber_radius=float(mc.get("fiber_radius_arcsec", 0.75)))
        cat_full = pd.read_parquet(cat_dir / s["catalogue"])
        cat_full = cat_full.set_index("TARGETID", drop=False)
        for i in range(int(s["n_mass_bins"])):
            stem = gu.lens_stem(s, i)
            shapes = pd.read_parquet(ia_dir / f"{stem}.parquet",
                                     columns=["TARGETID", "LOGMSTAR", wcol])
            logm = shapes["LOGMSTAR"].to_numpy(float)
            m_lo, m_hi = float(np.nanmin(logm)), float(np.nanmax(logm))
            cat = cat_full.loc[shapes["TARGETID"].to_numpy()]
            weights = shapes[wcol].to_numpy(float)

            res = lens_alpha(cat, weights, m_lo, m_hi, s["selection"], kappas,
                             fiber_grid, use_exp_profile=use_exp)
            print(f"  {stem}: n={len(cat):,}  logM*=[{m_lo:.2f},{m_hi:.2f}]  "
                  f"alpha = {res['alpha']:.3f} +/- {res['alpha_err']:.3f}  "
                  f"(chi2/dof = {res['chi2']:.1f}/{res['dof']})", flush=True)
            rows.append({"sample": s["name"], "massbin": i, "n_gal": len(cat),
                         "zmin": s["zmin"], "zmax": s["zmax"],
                         "logm_lo": m_lo, "logm_hi": m_hi,
                         "alpha": res["alpha"], "alpha_err": res["alpha_err"],
                         "slope": res["slope"], "slope_err": res["slope_err"],
                         "chi2": res["chi2"], "dof": res["dof"]})
            detail[f"{s['name']}_massbin{i}"] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in res.items()}
        del cat_full

    csv_out = out_dir / "magnification_alphas.csv"
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    json_out = out_dir / "magnification_alphas.json"
    with open(json_out, "w") as fh:
        json.dump({"kappa_step": float(mc["kappa_step"]),
                   "n_kappa": int(mc["n_kappa"]), "weight_column": wcol,
                   "results": detail}, fh, indent=2)
    print(f"\n[{time.time()-t0:.0f}s] saved {csv_out} + {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
