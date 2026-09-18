#!/usr/bin/env python3
"""Downsample the UNIONS ShapePipe weak-lensing catalogue into reduced parquet samples.

The source is a single ~215 GB hdf5 with one compound dataset ('data', 644M rows).
In one streaming pass we write two column-reduced parquet samples (config-driven):

  * for_IA  -- columns for shape calibration only (RA, DEC, WEIGHT, e1/e2_uncal,
               R_g11, R_g22); DEC>=22 cut; NO quality filter (matched to DESI later).
  * for_GGL -- shear-quality selection (size ratio / MAG_WIN / snr = `cond1`) plus the
               extra columns needed for response recomputation, DES binned weights and
               PSF-leakage correction.

Metacalibration responsivity is recomputed per object from the metacal ladder:
    R_g11 = (NGMIX_ELL_1P_0 - NGMIX_ELL_1M_0) / (2 * dg)
    R_g22 = (NGMIX_ELL_2P_1 - NGMIX_ELL_2M_1) / (2 * dg)
with e1_uncal/e2_uncal == NGMIX_ELL_NOSHEAR_{0,1}. Calibration itself (global weighted
response, PSF leakage) is applied downstream AFTER selection -- see CosmoStat/sp_validation.

Reads only the source fields it needs, in chunks, and streams row groups straight to
parquet, so peak memory is ~one chunk regardless of the 644M total.

Usage
-----
    python scripts/build_unions_shape.py [--config CONFIG] [--sample IA GGL]
                                         [--max-chunks N] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parent.parent

# Source fields needed to compute the derived responsivity columns.
METACAL_FIELDS = ["NGMIX_ELL_1P_0", "NGMIX_ELL_1M_0", "NGMIX_ELL_2P_1", "NGMIX_ELL_2M_1"]


def ext_fields(spec: dict) -> list[str]:
    """data_ext columns to carry: veto bit flags + any non-veto extras (npoint3)."""
    return list(spec.get("ext_columns", [])) + list(spec.get("ext_extra", []))


def derived_responsivity(chunk, dg: float) -> dict[str, np.ndarray]:
    """R_g11, R_g22 from the metacal ladder (native byte order, float32)."""
    r11 = (chunk["NGMIX_ELL_1P_0"] - chunk["NGMIX_ELL_1M_0"]) / (2 * dg)
    r22 = (chunk["NGMIX_ELL_2P_1"] - chunk["NGMIX_ELL_2M_1"]) / (2 * dg)
    return {"R_g11": r11.astype("f4"), "R_g22": r22.astype("f4")}


def native(a: np.ndarray) -> np.ndarray:
    """Normalise byte order (the hdf5 mixes >f8 and <f4)."""
    return a.astype(a.dtype.newbyteorder("="), copy=False)


def sample_mask(spec: dict, src: dict) -> np.ndarray:
    """Boolean selection for a sample from its config (DEC cut and/or cond1)."""
    n = len(next(iter(src.values())))
    m = np.ones(n, dtype=bool)
    if "dec_min" in spec:                       # IA: footprint cut only
        m &= src["Dec"] >= spec["dec_min"]
    if "size_ratio_min" in spec:                # GGL: cond1 shear-quality cut
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = src["NGMIX_T_NOSHEAR"] / src["NGMIX_Tpsf_NOSHEAR"]
        m &= (ratio > spec["size_ratio_min"]) & (ratio < spec["size_ratio_max"])
        m &= src["MAG_WIN"] > spec["mag_win_min"]
        m &= (src["snr"] > spec["snr_min"]) & (src["snr"] < spec["snr_max"])
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config" / "data_sources.yaml"))
    ap.add_argument("--sample", nargs="+", help="Restrict to these samples (IA/GGL).")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="Process only the first N chunks (for sizing/testing).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None, help="output directory (default: the config location under <dest>)")
    args = ap.parse_args()

    import h5py

    cfg = yaml.safe_load(open(args.config))
    us = cfg["unions_shape"]
    dest = Path(args.out_dir) if args.out_dir else Path(cfg["dest"]) / us["out_subdir"]
    dg = float(us["metacal_dg"])
    chunk_rows = int(us["chunk_rows"])
    samples = {k: v for k, v in us["samples"].items()
               if not args.sample or k in args.sample}

    # Source fields to read: every mapped source column + metacal ladder + the fields
    # the filters touch (Dec, ratio inputs, MAG_WIN, snr).
    need = set(METACAL_FIELDS)
    need_ext: set[str] = set()
    for spec in samples.values():
        need.update(spec["columns"].values())
        need.update(["Dec", "NGMIX_T_NOSHEAR", "NGMIX_Tpsf_NOSHEAR", "MAG_WIN", "snr"])
        need_ext.update(ext_fields(spec))
    need = sorted(need)
    need_ext = sorted(need_ext)

    print(f"hdf5    : {us['hdf5']}")
    print(f"samples : {', '.join(samples)}")
    print(f"out     : {dest}")
    print(f"reading {len(need)} source fields in chunks of {chunk_rows:,} rows\n")
    if args.dry_run:
        for name, spec in samples.items():
            cols = list(spec["columns"]) + spec.get("derived", []) + ext_fields(spec)
            print(f"  {name}: {len(cols)} cols -> {spec['out']}  ({', '.join(cols)})")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    f = h5py.File(us["hdf5"], "r")
    d = f[us["dataset"]]
    d_ext = f[us["dataset_ext"]] if need_ext else None
    ntot = d.shape[0]
    nchunks = (ntot + chunk_rows - 1) // chunk_rows
    if args.max_chunks:
        nchunks = min(nchunks, args.max_chunks)

    writers: dict[str, pq.ParquetWriter] = {}
    kept = {k: 0 for k in samples}
    t0 = time.time()
    seen = 0
    for ci in range(nchunks):
        i0 = ci * chunk_rows
        i1 = min(i0 + chunk_rows, ntot)
        chunk = d.fields(need)[i0:i1]                     # only needed fields
        src = {k: native(chunk[k]) for k in need}
        deriv = derived_responsivity(chunk, dg)
        if d_ext is not None:
            ext_chunk = d_ext.fields(need_ext)[i0:i1]
            src_ext = {k: native(ext_chunk[k]) for k in need_ext}
        seen += i1 - i0

        for name, spec in samples.items():
            m = sample_mask(spec, src)
            out_cols = {}
            for out_name, src_name in spec["columns"].items():
                out_cols[out_name] = src[src_name][m]
            for dname in spec.get("derived", []):
                out_cols[dname] = deriv[dname][m]
            for ename in ext_fields(spec):
                out_cols[ename] = src_ext[ename][m]
            table = pa.table({k: native(v) for k, v in out_cols.items()})
            if name not in writers:
                writers[name] = pq.ParquetWriter(dest / spec["out"], table.schema,
                                                 compression="snappy")
            writers[name].write_table(table)
            kept[name] += int(m.sum())

        if (ci + 1) % 5 == 0 or ci == nchunks - 1:
            rate = seen / (time.time() - t0) / 1e6
            print(f"  chunk {ci+1}/{nchunks}  ({seen:,} rows, {rate:.1f} M/s)  "
                  + "  ".join(f"{k}:{kept[k]:,}" for k in samples), flush=True)

    for w in writers.values():
        w.close()

    dt = time.time() - t0
    print(f"\nProcessed {seen:,} rows in {dt/60:.1f} min")
    for name, spec in samples.items():
        out = dest / spec["out"]
        sz = out.stat().st_size / 1e9 if out.exists() else 0
        print(f"  {name}: {kept[name]:,} kept "
              f"({100*kept[name]/max(seen,1):.1f}%) -> {out.name}  ({sz:.1f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
