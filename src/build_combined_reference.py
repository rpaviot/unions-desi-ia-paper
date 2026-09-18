#!/usr/bin/env python3
"""Build the combined LRG+ELG clustering-redshift REFERENCE from the DESI DR1
combined LSS catalogue.

The combined ``LRG+ELG_LOPnotqso`` clustering catalogue (data + randoms) is
clustering-only -- no full.dat / fastspec -- and its randoms encode the UNION
(LRG OR ELG) selection function, which cannot be reconstructed by concatenating
the separate LRG and ELG randoms. We use it as a single dense clustering-z
reference in the LRG/ELG overlap (z ~ 0.8-1.1), replacing the inverse-variance
combination of the two noisier separate references there.

This bypasses build_catalogues.py (which assumes full.dat + fastspec) and
build_ia_samples.py (which builds shape samples); it does only what an n(z)
reference needs:

  1. read + stack the combined data (RA, DEC, Z, WEIGHT, WEIGHT_FKP) over caps;
  2. read + stack the configured random files (RA, DEC, Z, WEIGHT);
  3. z-cut both to the reference z_range;
  4. footprint-cut both to the UNIONS overlap (healpix pixels occupied by the
     for_IA shape sample -- identical mask to build_ia_samples);
  5. cap the randoms at randoms_factor x data;
  6. write ia/<stem>_clustering.parquet + ia/<stem>_randoms.parquet, the exact
     files estimate_nz.load_reference reads.

Driven by config/data_sources.yaml (`combined_reference:` block).

Usage
-----
    python scripts/build_combined_reference.py [--config CONFIG] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent

DATA_COLS = ["RA", "DEC", "Z", "WEIGHT", "WEIGHT_FKP"]
RAND_COLS = ["RA", "DEC", "Z", "WEIGHT"]


def ang2pix(ra, dec, nside):
    """RA/Dec (deg) -> nested healpix pixel index (matches build_ia_samples)."""
    import healpy as hp
    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))
    return hp.ang2pix(nside, theta, phi, nest=True)


def read_fits(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a subset of 1-D columns from a FITS binary table, native byte order."""
    import fitsio
    with fitsio.FITS(str(path)) as f:
        hdu = f[1]
        present = set(hdu.get_colnames())
        want = [c for c in columns if c in present]
        missing = [c for c in columns if c not in present]
        if missing:
            print(f"    [warn] {path.name}: missing {missing}")
        data = hdu.read(columns=want)
    return pd.DataFrame(
        {c: np.asarray(data[c]).astype(data[c].dtype.newbyteorder("="), copy=False)
         for c in want})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config" / "data_sources.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None, help="output directory (default: the config location under <dest>)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dest = Path(cfg["dest"])
    cr = cfg["combined_reference"]
    lss_dir = dest / cr["lss_subdir"]
    tracer = cr["tracer"]
    caps = cr["caps"]
    indices = cr["random_indices"]
    out_dir = Path(args.out_dir) if args.out_dir else dest / cr["out_subdir"]
    zlo, zhi = (float(v) for v in cr["z_range"])
    stem = f"{cr['stem']}_zmin_{zlo:.2f}_zmax_{zhi:.2f}"
    nside = int(cr["footprint_nside"])
    rfac = int(cr["randoms_factor"])
    upath = dest / cr["unions_for_ia"]

    data_files = [lss_dir / f"{tracer}_{cap}_clustering.dat.fits" for cap in caps]
    rand_files = [lss_dir / f"{tracer}_{cap}_{i}_clustering.ran.fits"
                  for cap in caps for i in indices]

    print(f"tracer    : {tracer}")
    print(f"z range   : [{zlo}, {zhi}]  -> stem {stem}")
    print(f"data files: {len(data_files)}  random files: {len(rand_files)}")
    print(f"footprint : for_IA occupied pixels, nside={nside}")
    print(f"out       : {out_dir}/{stem}_{{clustering,randoms}}.parquet\n")
    if args.dry_run:
        for p in data_files + rand_files:
            print(f"  {'OK' if p.exists() else '!!':>2}  {p.name}")
        return 0

    import pyarrow.parquet as pq
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- 1. UNIONS footprint: occupied healpix pixels from for_IA ---------------
    import healpy as hp
    u = pq.read_table(upath, columns=["RA", "DEC"]).to_pandas()
    u_pix = ang2pix(u["RA"].to_numpy(np.float64), u["DEC"].to_numpy(np.float64), nside)
    occ = np.zeros(hp.nside2npix(nside), dtype=bool)
    occ[u_pix] = True
    print(f"[{time.time()-t0:.0f}s] UNIONS for_IA: {len(u):,} shapes, "
          f"{occ.sum():,} occupied pixels", flush=True)
    del u, u_pix

    def stack_zcut_footprint(files, cols, label):
        parts = []
        for p in files:
            df = read_fits(p, cols)
            df = df[(df["Z"] >= zlo) & (df["Z"] <= zhi)]
            pix = ang2pix(df["RA"], df["DEC"], nside)
            df = df[occ[pix]].reset_index(drop=True)
            parts.append(df)
            print(f"    {p.name}: {len(df):,} in z-bin & footprint", flush=True)
        out = pd.concat(parts, ignore_index=True)
        print(f"  {label} total: {len(out):,}", flush=True)
        return out

    # ---- 2. data ----------------------------------------------------------------
    data = stack_zcut_footprint(data_files, DATA_COLS, "data")
    data.to_parquet(out_dir / f"{stem}_clustering.parquet", index=False)

    # ---- 3. randoms (capped at rfac x data) -------------------------------------
    rand = stack_zcut_footprint(rand_files, RAND_COLS, "randoms")
    cap = rfac * len(data)
    if len(rand) > cap:
        rand = rand.sample(n=cap, replace=False, random_state=42).reset_index(drop=True)
        print(f"  randoms capped to {rfac}x data = {len(rand):,}", flush=True)
    rand.to_parquet(out_dir / f"{stem}_randoms.parquet", index=False)

    print(f"\n[{time.time()-t0:.0f}s] done: {stem}  "
          f"clustering {len(data):,}, randoms {len(rand):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
