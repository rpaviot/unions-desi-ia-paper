#!/usr/bin/env python3
"""Bundle all Nautilus IA fit npz files into a single downloadable archive.

Walks the ``results/fits/nautilus_*`` directories (or any dirs passed on the
command line) and packs every ``*_NLA_nautilus.npz`` into one compressed npz.
Each input file is stored verbatim (all 71 keys, including the posterior
``chain``/``log_w``/``log_l``) as a 0-d object array under the key

    "<run-dir-name>/<sample-stem>"        e.g. "nautilus_fid_rmin25-10/LRG_..._massbin0"

Plus a ``manifest`` (sorted list of those keys) and ``runs`` (the run-dir names).

Read it back with::

    d = np.load("nautilus_all_fits.npz", allow_pickle=True)
    print(d["manifest"])
    fit = d["nautilus_fid_rmin25-10/LRG_zmin_0.40_zmax_0.75_massbin0"].item()
    a1 = fit["bestfit"][list(fit["param_names"].astype(str)).index("a1")]

Usage:
    python scripts/bundle_nautilus_fits.py [--out results/fits/nautilus_all_fits.npz] [DIR ...]
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="*", help="run dirs (default: results/fits/nautilus_*)")
    p.add_argument("--out", default="results/fits/nautilus_all_fits.npz")
    p.add_argument("--stat", default="multipoles", choices=["multipoles", "projected"])
    p.add_argument("--model", default="NLA", choices=["NLA", "TATT"])
    p.add_argument("--sampler", default="nautilus", choices=["nautilus", "minuit", "emcee"],
                   help="which fit_ia.py outputs to pack (filename suffix)")
    args = p.parse_args()
    SUFFIX = f"_{args.stat}_{args.model}_{args.sampler}.npz"

    dirs = args.dirs or sorted(d for d in glob.glob("results/fits/nautilus_*")
                               if os.path.isdir(d))
    if not dirs:
        raise SystemExit("no nautilus run dirs found")

    bundle, manifest, runs = {}, [], []
    for d in dirs:
        run = Path(d).name
        runs.append(run)
        files = sorted(glob.glob(os.path.join(d, f"*{SUFFIX}")))
        print(f"{run}: {len(files)} fits")
        for f in files:
            stem = Path(f).name[: -len(SUFFIX)]
            key = f"{run}/{stem}"
            with np.load(f, allow_pickle=True) as nf:
                content = {k: nf[k] for k in nf.files}
            bundle[key] = np.array(content, dtype=object)  # 0-d object array
            manifest.append(key)

    bundle["manifest"] = np.array(sorted(manifest))
    bundle["runs"] = np.array(runs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **bundle)
    size_mb = out.stat().st_size / 1024**2
    print(f"\nwrote {out}  ({len(manifest)} fits from {len(runs)} runs, "
          f"{size_mb:.1f} MB)\n  load with np.load(..., allow_pickle=True)")


if __name__ == "__main__":
    main()
