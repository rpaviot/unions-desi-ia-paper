#!/usr/bin/env python3
"""Per-sample shape-catalogue properties for the UNIONS x DESI IA samples.

For every shape sample parquet under ``<dest>/ia`` (enumerated from
``config/data_sources.yaml`` exactly like build_ia_samples / compute_correlations,
including mass bins) compute, from the e1/e2/WEIGHT columns:

  * sigma_e -- per-component weighted shape noise (lestgobaby.compute_shape_noise),
  * n_eff   -- Heymans-style effective count (sum w)^2 / sum w^2, plus the raw
               count and sum of weights,
  * z_eff   -- weighted mean redshift,
  * LOGMSTAR median / std,
  * r_mean  -- the calibration responsivity, read from the parquet schema metadata
               (key b"r_mean", written by build_ia_samples). Samples built before
               the metadata patch fall back to the ``R=`` lines harvested from the
               build_ia_samples slurm logs (--build-logs, default logs/build_ia_samples_*.out,
               newest log wins); NaN with a warning if found nowhere.

One JSON per sample is written to ``<dest>/ia/sample_properties/<sample_stem>.json``
and a combined summary table is printed. Idempotent: existing JSONs are skipped
unless --overwrite.

Usage
-----
    python scripts/sample_properties.py [--config CONFIG] [--only PATTERN]
                                        [--overwrite]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config" / "data_sources.yaml"


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def shape_sample_stems(cfg):
    """Enumerate every shape-sample stem (incl. mass bins), mirroring
    compute_correlations.tracer_jobs(). Returns (stem, parquet_path) pairs."""
    ia = cfg["ia_samples"]
    ia_dir = Path(cfg["dest"]) / ia["out_subdir"]
    out = []
    for t in ia["tracers"]:
        for zlo, zhi in t["zbins"]:
            ztag = f"zmin_{zlo:.2f}_zmax_{zhi:.2f}"
            for s in t["shape_samples"]:
                sbase = f"{s['name']}_{ztag}"
                out.append((sbase, ia_dir / f"{sbase}_shapes.parquet"))
                if s.get("n_mass_bins"):
                    for i in range(int(s["n_mass_bins"])):
                        out.append((f"{sbase}_massbin{i}",
                                    ia_dir / f"{sbase}_shapes_massbin{i}.parquet"))
    return out


def compute_shape_noise(e1, e2, w):
    """Per-component weighted shape noise (lestgobaby.compute_shape_noise):
    sigma_e^2 = 1/2 [ sum(w e1)^2 + sum(w e2)^2 ] / (sum w)^2 * (sum w)^2/sum(w^2)."""
    mask = np.isfinite(w) & np.isfinite(e1) & np.isfinite(e2) & (w > 0)
    if not np.any(mask):
        return np.nan
    w, e1, e2 = w[mask], e1[mask], e2[mask]
    sum_w = np.sum(w)
    sum_w2 = np.sum(w**2)
    term1 = np.sum((w * e1)**2) / sum_w**2
    term2 = np.sum((w * e2)**2) / sum_w**2
    n_eff_factor = sum_w**2 / sum_w2
    return float(np.sqrt(0.5 * (term1 + term2) * n_eff_factor))


_LOG_SAMPLE = re.compile(r"^\s+(\S+_zmin_[\d.]+_zmax_[\d.]+): matched .*R=([\d.]+)")
_LOG_MASSBIN = re.compile(r"^\s+(massbin\d+) \[.*\]: .*R=([\d.]+)")


def harvest_r_from_logs(log_files):
    """Parse build_ia_samples logs into {sample_stem: r_mean}. Logs are read in
    the given order, so later (newer) logs override earlier ones. Massbin lines
    are attached to the most recent sample line above them."""
    r_map, stag = {}, None
    for lf in log_files:
        with open(lf) as fh:
            for line in fh:
                m = _LOG_SAMPLE.match(line)
                if m:
                    stag = m.group(1)
                    r_map[stag] = float(m.group(2))
                    continue
                m = _LOG_MASSBIN.match(line)
                if m and stag:
                    r_map[f"{stag}_{m.group(1)}"] = float(m.group(2))
    return r_map


def sample_properties(parquet_path, stem=None, r_fallback=None):
    """Compute the property dict for one shape-sample parquet."""
    pf = pq.ParquetFile(parquet_path)
    meta = pf.schema_arrow.metadata or {}
    cols = [c for c in ("e1", "e2", "WEIGHT", "Z", "LOGMSTAR")
            if c in pf.schema_arrow.names]
    tab = pf.read(columns=cols)

    w = tab["WEIGHT"].to_numpy().astype(np.float64)
    e1 = tab["e1"].to_numpy().astype(np.float64)
    e2 = tab["e2"].to_numpy().astype(np.float64)

    props = {"n_raw": int(len(w)), "sum_w": float(np.sum(w))}
    props["n_eff"] = float(np.sum(w)**2 / np.sum(w**2)) if len(w) else np.nan
    props["sigma_e"] = compute_shape_noise(e1, e2, w)

    if "Z" in cols:
        z = tab["Z"].to_numpy().astype(np.float64)
        props["z_eff"] = float(np.sum(w * z) / np.sum(w))
    else:
        props["z_eff"] = np.nan

    if "LOGMSTAR" in cols:
        m = tab["LOGMSTAR"].to_numpy().astype(np.float64)
        m = m[np.isfinite(m)]
        props["logmstar_median"] = float(np.median(m)) if m.size else np.nan
        props["logmstar_std"] = float(np.std(m)) if m.size else np.nan
    else:
        props["logmstar_median"] = props["logmstar_std"] = np.nan

    if b"r_mean" in meta:
        props["r_mean"] = float(meta[b"r_mean"].decode())
        props["r_mean_source"] = "parquet"
    elif r_fallback and stem in r_fallback:
        props["r_mean"] = r_fallback[stem]
        props["r_mean_source"] = "build_log"
    else:
        print(f"  WARNING: no r_mean for {Path(parquet_path).name} "
              "(not in schema metadata nor in the build logs)", flush=True)
        props["r_mean"] = np.nan
        props["r_mean_source"] = None
    return props


def main():
    p = argparse.ArgumentParser(description="Compute IA shape-sample properties")
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--only", default=None, help="substring filter on the sample stem")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: <dest>/<ia_samples.out_subdir>/sample_properties)")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute samples whose JSON already exists")
    p.add_argument("--build-logs", nargs="*", default=None,
                   help="build_ia_samples logs to harvest r_mean from when absent "
                        "from the parquet metadata (default: logs/build_ia_samples_*.out)")
    args = p.parse_args()

    if args.build_logs is None:
        log_files = sorted((REPO / "logs").glob("build_ia_samples_*.out"),
                           key=lambda f: f.stat().st_mtime)
    else:
        log_files = [Path(f) for f in args.build_logs]
    r_fallback = harvest_r_from_logs(log_files) if log_files else {}

    cfg = load_config(args.config)
    ia_dir = Path(cfg["dest"]) / cfg["ia_samples"]["out_subdir"]
    out_dir = Path(args.out_dir) if args.out_dir else ia_dir / "sample_properties"
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = shape_sample_stems(cfg)
    if args.only:
        stems = [(s, f) for s, f in stems if args.only in s]
    print(f"{len(stems)} shape samples to process -> {out_dir}", flush=True)

    rows = []
    for stem, fpath in stems:
        out_json = out_dir / f"{stem}.json"
        if not fpath.exists():
            print(f"[skip] {stem}: missing {fpath.name}", flush=True)
            continue
        if out_json.exists() and not args.overwrite:
            with open(out_json) as fh:
                props = json.load(fh)
            print(f"[cached] {stem}", flush=True)
        else:
            print(f"[compute] {stem}", flush=True)
            props = sample_properties(fpath, stem=stem, r_fallback=r_fallback)
            props["sample"] = stem
            props["parquet"] = str(fpath)
            with open(out_json, "w") as fh:
                json.dump(props, fh, indent=2)
        rows.append((stem, props))

    # combined summary table
    hdr = ["sample", "n_raw", "n_eff", "sum_w", "sigma_e", "z_eff",
           "logM_med", "logM_std", "r_mean"]
    print("\n" + "  ".join(f"{h:>12}" for h in hdr[1:]) + "    sample")
    for stem, pr in rows:
        print(f"{pr['n_raw']:>12,}  {pr['n_eff']:>12,.1f}  {pr['sum_w']:>12,.1f}  "
              f"{pr['sigma_e']:>12.4f}  {pr['z_eff']:>12.4f}  "
              f"{pr['logmstar_median']:>12.3f}  {pr['logmstar_std']:>12.3f}  "
              f"{pr['r_mean']:>12.4f}    {stem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
