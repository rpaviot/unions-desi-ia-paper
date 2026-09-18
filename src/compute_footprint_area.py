#!/usr/bin/env python3
"""Converged UNIONS-overlap footprint area from the FULL (non-downsampled) randoms.

The Gaussian covariance (compute_gaussian_cov.py) needs the effective survey area
to turn the weighted redshift histograms into comoving densities n(z) and to build
V_eff. The existing area estimates (compute_ngal.py) count healpix pixels occupied
by the DOWNSAMPLED shape randoms at a fixed nside, which is resolution-dependent
(overestimates at low nside from partially-covered edge pixels, underestimates at
high nside once pixels outrun the random density). Here we instead use the full
random density, before any randoms_factor cap:

  * The assembled randoms parquet (<dest>/catalogues/<tracer>_clustering.ran.parquet,
    build_catalogues.assemble_randoms) concatenates ALL configured DESI random files
    (random_indices x caps) at the DESI parent density of 2500/deg^2 PER FILE. The
    surface density over the combined footprint is therefore exactly
    n_files_per_cap x 2500 deg^-2 (caps are disjoint sky), independent of any pixels.

  * DENSITY-RATIO area (the fiducial, nside-free estimator):
        A = N_kept / (n_files_per_cap * 2500)  [deg^2]
    where N_kept are the randoms that survive the SAME UNIONS-overlap footprint cut
    as build_ia_samples.py: keep iff the random lands in a healpix (footprint_nside,
    NESTED) pixel occupied by >= 1 UNIONS for_IA shape. Verified per cap so a
    lopsided file count cannot bias the total.

  * HEALPIX CONVERGENCE SCAN (cross-check): occupied-pixel area of the kept randoms
    at nside 256..8192. This should converge towards the density-ratio value from
    above as nside grows, until pixel size approaches the random inter-particle
    separation; the scan is diagnostic output only.

The footprint cut itself is part of the SAMPLE DEFINITION (fixed at
ia_samples.footprint_nside = 8192, matching build_ia_samples), so it is NOT varied:
only the area *measurement* must be resolution-free, which the density ratio is.

The randoms' Z/weights are irrelevant here -- the footprint cut is purely angular --
so one area serves every z-bin and shape sample of a tracer (shape randoms are drawn
from the tracer randoms footprint by construction).

Outputs one JSON per tracer:
    <dest>/ia/sample_properties/footprint_area_<tracer>.json

Run with the project .venv (healpy + pyarrow):
    .venv/bin/python scripts/compute_footprint_area.py [--tracers BGS_ANY ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config" / "data_sources.yaml"

DESI_RANDOM_DENSITY = 2500.0  # deg^-2 per random file (DESI DR1 LSS parent density)
SCAN_NSIDES = [256, 512, 1024, 2048, 4096, 8192]


def ang2pix(ra, dec, nside):
    """Same NESTED ang2pix convention as build_ia_samples.py."""
    import healpy as hp
    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))
    return hp.ang2pix(nside, theta, phi, nest=True)


def unions_occupancy(unions_parquet, nside):
    """Boolean healpix map (NESTED) of pixels occupied by UNIONS for_IA shapes."""
    import healpy as hp
    import pyarrow.parquet as pq
    t0 = time.time()
    pf = pq.ParquetFile(unions_parquet)
    occ = np.zeros(hp.nside2npix(nside), dtype=bool)
    n = 0
    # stream row groups: only RA/DEC are read, but the full column is ~200M rows
    for i in range(pf.num_row_groups):
        tb = pf.read_row_group(i, columns=["RA", "DEC"])
        occ[ang2pix(tb["RA"].to_numpy(), tb["DEC"].to_numpy(), nside)] = True
        n += tb.num_rows
    print(f"  [{time.time()-t0:.0f}s] UNIONS for_IA: {n:,} shapes, "
          f"{occ.sum():,} occupied nside={nside} pixels", flush=True)
    return occ


def tracer_area(randoms_parquet, occ, nside):
    """Density-ratio + healpix-scan areas for one tracer's full randoms."""
    import healpy as hp
    import pyarrow.parquet as pq
    t0 = time.time()
    pf = pq.ParquetFile(randoms_parquet)
    cols = [c for c in ["RA", "DEC", "CAP", "RAN_IDX"] if c in pf.schema_arrow.names]
    if "CAP" not in cols or "RAN_IDX" not in cols:
        raise SystemExit(f"{randoms_parquet} lacks CAP/RAN_IDX provenance columns -- "
                         "cannot establish the per-file random density; rebuild with "
                         "build_catalogues.py")
    ra_k, dec_k, cap_k = [], [], []
    n_total = 0
    files_per_cap = {}
    for i in range(pf.num_row_groups):
        tb = pf.read_row_group(i, columns=cols)
        ra = tb["RA"].to_numpy()
        dec = tb["DEC"].to_numpy()
        cap = tb["CAP"].to_numpy(zero_copy_only=False)
        idx = tb["RAN_IDX"].to_numpy()
        for c in np.unique(cap):
            files_per_cap.setdefault(str(c), set()).update(
                np.unique(idx[cap == c]).tolist())
        keep = occ[ang2pix(ra, dec, nside)]
        ra_k.append(ra[keep])
        dec_k.append(dec[keep])
        cap_k.append(cap[keep])
        n_total += tb.num_rows
    ra = np.concatenate(ra_k)
    dec = np.concatenate(dec_k)
    cap = np.concatenate(cap_k)
    del ra_k, dec_k, cap_k
    print(f"  [{time.time()-t0:.0f}s] randoms: {n_total:,} total, "
          f"{len(ra):,} kept in UNIONS overlap "
          f"({100*len(ra)/n_total:.2f}%)", flush=True)

    # density-ratio area, per cap (each cap is n_files x 2500 deg^-2)
    per_cap = {}
    area = 0.0
    for c, files in sorted(files_per_cap.items()):
        n_c = int((cap == c).sum())
        dens = len(files) * DESI_RANDOM_DENSITY
        a_c = n_c / dens
        per_cap[c] = {"n_kept": n_c, "n_files": len(files), "area_deg2": a_c}
        area += a_c
        print(f"    cap {c}: {len(files)} files, {n_c:,} kept -> {a_c:.2f} deg^2")

    # healpix occupied-pixel convergence scan on the kept randoms
    scan = {}
    for ns in SCAN_NSIDES:
        pix = ang2pix(ra, dec, ns)
        n_occ = len(np.unique(pix))
        a = n_occ * hp.nside2pixarea(ns, degrees=True)
        scan[ns] = a
        print(f"    nside {ns:5d}: {n_occ:,} pixels -> {a:.2f} deg^2 "
              f"({100*(a/area-1):+.2f}% vs density ratio)")
    return {
        "area_deg2": area,
        "area_sr": area * (np.pi / 180.0) ** 2,
        "per_cap": per_cap,
        "n_randoms_total": int(n_total),
        "n_randoms_kept": int(len(ra)),
        "random_density_per_file_deg2": DESI_RANDOM_DENSITY,
        "healpix_scan_deg2": {str(k): v for k, v in scan.items()},
        "footprint_nside": nside,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--tracers", nargs="*", default=None,
                    help="tracer names (default: all in ia_samples.tracers)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dest = Path(cfg["dest"])
    ia = cfg["ia_samples"]
    nside = int(ia["footprint_nside"])
    cat_dir = dest / ia["cat_subdir"]
    out_dir = dest / ia["out_subdir"] / "sample_properties"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracers = [t for t in ia["tracers"] if not args.tracers or t["name"] in args.tracers]
    if not tracers:
        raise SystemExit(f"no tracers matched {args.tracers}")

    print(f"footprint mask: UNIONS for_IA occupancy at nside={nside} (nested)")
    occ = unions_occupancy(dest / ia["unions_for_ia"], nside)

    for t in tracers:
        print(f"\n{t['name']}  ({t['randoms']})")
        res = tracer_area(cat_dir / t["randoms"], occ, nside)
        res["tracer"] = t["name"]
        res["randoms_parquet"] = str(cat_dir / t["randoms"])
        res["unions_for_ia"] = str(dest / ia["unions_for_ia"])
        out = out_dir / f"footprint_area_{t['name']}.json"
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
