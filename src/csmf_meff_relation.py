#!/usr/bin/env python3
"""M* - M_eff relation of the lens bins from a saved CSMF fit.

Rebuilds the halo model at the best fit of ``--fit-npz`` (same mass definition,
mass function and bias as recorded in the npz) on the lens bins of ``--in-dir``,
and evaluates per bin the central-weighted effective halo mass
M_eff = <M_h>_cen (h^-1 Msun). Writes an npz with

    rel_logmstar   median log10 M* [h^-2 Msun] of each lens bin (sorted)
    rel_logmeff    log10 M_eff [h^-1 Msun] of the best-fit model in that bin

which plot_fig3_aia_mass_meff.py --relation-npz interpolates (log-log, linearly
extrapolated) to map the stellar-mass bins of the IA fits to M_eff.

    ${HOD_PYTHON} src/csmf_meff_relation.py --config config/data_sources.yaml \
        --fit-npz results/baseline/csmf_fit/csmf_fit.npz \
        --in-dir data/ggl/csmf_input_z5_effective \
        --samples BGS_RED_GMM_VLIM_SNR --out aia_vs_meff_relation.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402
from plot_aia_meff import label_relation  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(gu.CONFIG))
    p.add_argument("--fit-npz", required=True, help="saved CSMF fit (fit_csmf.py --out)")
    p.add_argument("--in-dir", default=None, metavar="DIR",
                   help="csmf-input directory the fit was run on (fit_csmf --in-dir)")
    p.add_argument("--in-name", default=None,
                   help="legacy form: csmf-input subdir under <dest>/ggl (fit_csmf --in-name)")
    p.add_argument("--samples", nargs="+", default=["BGS_RED_GMM_VLIM_SNR"],
                   help="lens samples of the fit")
    p.add_argument("--method", default=None, choices=["minuit", "nautilus", "de"],
                   help="point estimate to evaluate M_eff at (default: whichever the npz "
                        "carries; nautilus = the posterior median)")
    p.add_argument("--out", required=True, help="output npz")
    args = p.parse_args()

    cfg = gu.load_config(args.config)
    rel_x, rel_y = label_relation(cfg, None, args.samples, in_name=args.in_name,
                                  fit_npz=args.fit_npz, in_dir=args.in_dir, method=args.method)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, rel_logmstar=rel_x, rel_logmeff=rel_y,
             fit_npz=str(args.fit_npz), in_dir=str(args.in_dir or args.in_name or ""),
             method=str(args.method or "auto"),
             samples=np.array(args.samples))
    print(f"wrote {out}  ({len(rel_x)} nodes)")


if __name__ == "__main__":
    main()
