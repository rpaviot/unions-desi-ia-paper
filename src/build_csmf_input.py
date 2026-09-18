#!/usr/bin/env python3
"""Bridge the measured ΔΣ profiles to CSMF-fitter input files.

Maps the dsigma stacking outputs
``<dest>/ggl/deltasigma/<stem>_massbin{i}_deltasigma.npz`` (compute_deltasigma.py)
onto the per-bin npz format expected by
``HOD_NRV.HOD_analytical.sampler.CSMFFitter.{load_bgs_data,load_lrg_data}``
(via ``MassBinData.from_npz``), one file per (sample, mass bin).

Lens-magnification correction is applied here on the DATA side, the same way the
old code did, so the fitter sees an already-corrected ΔΣ:

    ΔΣ_corr(rp) = ΔΣ(rp) - (alpha - 1) * ds_mag_unit(rp)

with `alpha` per (sample, massbin) read from ``ggl/magnification_alphas.csv`` and
`ds_mag_unit` the unit (alpha_l=2.0) magnification template stored alongside ΔΣ.
Both red samples (BGS_RED_GMM, LRG) are corrected identically. The corrected ΔΣ
is written as the fitter's ``delta_sigma`` and NO ``mag_contribution`` key is
emitted, so the package's own LRG magnification path stays inert (no double
subtraction). If ``ds_mag_unit`` is non-finite (e.g. the template failed in the
measurement -- see the "Parameter values not set" warning), the correction is
skipped for that bin and a flag is recorded.

Key mapping (measured npz -> fitter npz)
  rp           -> rp, rp_delta_sigma
  rp_bins      -> rp_bins
  ds           -> delta_sigma            (after magnification subtraction)
  ds_err       -> delta_sigma_err
  cov_ds       -> cov_delta_sigma
  z_eff        -> z_eff
  logmstar_{min,max,median} -> same
Provenance kept for the record: ds_uncorrected, ds_mag_unit, alpha,
mag_correction, mag_applied, alpha_l_template.

Run from the NRV venv (numpy only; no NRV import needed here), e.g.:
    /home/rpaviot/NRV_HOD/.venv_hod/bin/python scripts/build_csmf_input.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402

OUT_PATTERN = "csmf_{sample}_massbin{massbin}.npz"


def load_alphas(csv_path: Path) -> dict:
    """(sample, massbin) -> alpha from the magnification fit table."""
    df = pd.read_csv(csv_path)
    return {
        (str(r["sample"]), int(r["massbin"])): float(r["alpha"])
        for _, r in df.iterrows()
    }


def load_ngal(npz_path: Path) -> dict:
    """massbin -> (n_gal, n_gal_err) from compute_ngal.py's ngal_<sample>.npz.

    The abundance anchor (h^3/Mpc^3). Returns {} if the file is absent so the
    pipeline still runs without it (n_gal keys simply omitted from the per-bin npz).
    """
    if not npz_path.exists():
        return {}
    d = np.load(npz_path)
    return {int(mb): (float(n), float(e))
            for mb, n, e in zip(d["massbin"], d["n_gal"], d["n_gal_err"])}


def build_one(in_npz: Path, alpha: float, subtract_mag: bool) -> dict:
    """Map one measured ΔΣ npz onto the fitter npz dict (+ provenance)."""
    d = np.load(in_npz, allow_pickle=True)

    ds = np.asarray(d["ds"], dtype=float)
    ds_mag_unit = np.asarray(d["ds_mag_unit"], dtype=float)
    alpha_l_tpl = float(d["alpha_l_template"])

    # Magnification subtraction: ΔΣ_corr = ΔΣ - (alpha - 1) * ds_mag_unit.
    # ds_mag_unit already carries the 2*(alpha_l-1)=2 prefactor at alpha_l=2.0,
    # so multiplying by (alpha-1) gives the full 2*(alpha-1)*template correction.
    mag_correction = np.zeros_like(ds)
    mag_applied = False
    if subtract_mag:
        if np.all(np.isfinite(ds_mag_unit)):
            mag_correction = (alpha - 1.0) * ds_mag_unit
            ds = ds - mag_correction
            mag_applied = True
        else:
            print(f"    WARNING: ds_mag_unit non-finite in {in_npz.name}; "
                  f"magnification NOT subtracted", flush=True)

    return dict(
        # --- keys read by MassBinData.from_npz ---
        rp=np.asarray(d["rp"], dtype=float),
        rp_delta_sigma=np.asarray(d["rp"], dtype=float),
        rp_bins=np.asarray(d["rp_bins"], dtype=float),
        delta_sigma=ds,
        delta_sigma_err=np.asarray(d["ds_err"], dtype=float),
        cov_delta_sigma=np.asarray(d["cov_ds"], dtype=float),
        z_eff=float(d["z_eff"]),
        logmstar_min=float(d["logmstar_min"]),
        logmstar_max=float(d["logmstar_max"]),
        logmstar_median=float(d["logmstar_median"]),
        # --- provenance (ignored by the fitter) ---
        ds_uncorrected=np.asarray(d["ds"], dtype=float),
        ds_mag_unit=ds_mag_unit,
        alpha=float(alpha),
        alpha_l_template=alpha_l_tpl,
        mag_correction=mag_correction,
        mag_applied=mag_applied,
        sample=str(d["sample"]),
        massbin=int(d["massbin"]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(gu.CONFIG))
    p.add_argument("--samples", nargs="+", default=None,
                   help="Restrict to these sample names (default: all in config)")
    p.add_argument("--no-magnification", dest="subtract_mag", action="store_false",
                   help="Do not subtract the lens-magnification contribution")
    p.add_argument("--in-name", default=None,
                   help="Override the ΔΣ input subdir under <dest>/ggl (default: "
                        "csmf.in_subdir, i.e. 'deltasigma'). Use e.g. 'deltasigma_dr6' "
                        "to build the CSMF input off the DR6-photo-z ΔΣ set.")
    p.add_argument("--out-name", default=None,
                   help="Override the output subdir under <dest>/ggl (default: "
                        "csmf.out_subdir). Use e.g. 'csmf_input_z5' to keep an "
                        "alternative n_gal build next to the fiducial one.")
    p.add_argument("--ngal-suffix", default="",
                   help="Suffix of the ngal npz to read (compute_ngal --suffix), "
                        "e.g. '_z5' -> ngal_<sample>_z5.npz. Always read from the "
                        "fiducial csmf.out_subdir.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = gu.load_config(args.config)
    dest = Path(cfg["dest"])
    csmf = cfg["csmf"]
    in_dir = (dest / "ggl" / args.in_name) if args.in_name else (dest / csmf["in_subdir"])
    out_dir = (dest / "ggl" / args.out_name) if args.out_name else (dest / csmf["out_subdir"])
    ngal_dir = dest / csmf["out_subdir"]   # ngal npz always lives with the fiducial build
    subtract_mag = args.subtract_mag and bool(csmf.get("subtract_magnification", True))

    alphas = load_alphas(dest / csmf["alphas_csv"])

    samples = csmf["samples"]
    if args.samples:
        samples = [s for s in samples if s["name"] in args.samples]

    print(f"in   : {in_dir}")
    print(f"out  : {out_dir}")
    print(f"alpha: {dest / csmf['alphas_csv']}")
    print(f"magnification subtraction (global default): {'ON' if subtract_mag else 'OFF'}\n")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for s in samples:
        name, stem, n_mb = s["name"], s["stem"], int(s["n_mass_bins"])
        # Per-sample magnification override (BGS negligible -> off; LRG -> on).
        sample_mag = subtract_mag and bool(s.get("subtract_magnification", True))
        ngal_map = load_ngal(ngal_dir / f"ngal_{name}{args.ngal_suffix}.npz")
        print(f"{name}  ({n_mb} mass bins, stem={stem}, "
              f"magnification={'ON' if sample_mag else 'OFF'}, "
              f"n_gal={'YES' if ngal_map else 'no'})")
        for i in range(n_mb):
            in_npz = in_dir / f"{stem}_massbin{i}_deltasigma.npz"
            if not in_npz.exists():
                print(f"  bin {i}: MISSING {in_npz.name} (skipped -- still running?)")
                continue
            alpha = alphas.get((name, i))
            if alpha is None:
                print(f"  bin {i}: no alpha for ({name},{i}) in csv; magnification skipped")
            rec = build_one(in_npz, alpha if alpha is not None else 1.0,
                            sample_mag and alpha is not None)
            if i in ngal_map:
                rec["n_gal"], rec["n_gal_err"] = ngal_map[i]
            out_npz = out_dir / OUT_PATTERN.format(sample=name, massbin=i)
            snr = float(np.sqrt(rec["delta_sigma"]
                                @ np.linalg.solve(rec["cov_delta_sigma"],
                                                  rec["delta_sigma"])))
            tag = (f"alpha={alpha:.2f} mag={'Y' if rec['mag_applied'] else 'N'}"
                   if alpha is not None else "mag=N")
            ng_tag = (f"  n_gal={rec['n_gal']:.3e}+/-{rec['n_gal_err']:.1e}"
                      if "n_gal" in rec else "")
            print(f"  bin {i}: z_eff={rec['z_eff']:.3f}  "
                  f"logM*=[{rec['logmstar_min']:.2f},{rec['logmstar_max']:.2f}]  "
                  f"{tag}  S/N={snr:.1f}{ng_tag}  -> {out_npz.name}")
            if not args.dry_run:
                np.savez(out_npz, **rec)
                n_written += 1

    print(f"\nWrote {n_written} CSMF input files to {out_dir}"
          + (" (dry run -- nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
