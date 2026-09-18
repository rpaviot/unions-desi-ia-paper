#!/usr/bin/env python
"""Appendix figures: NLA best-fit vs measured multipoles for the 12 BGS red
S/N mass bins (nautilus fits, results/fits/nautilus_snr_rmin25-100_zerocross).

Two 4x3 grid figures are written:
  - <out-prefix>_xi0.png : clustering monopole, s^2 xi_0(s)
  - <out-prefix>_xi2.png : IA quadrupole,      s   xi_g+,2(s)

Each panel shows the full measurement (points outside the fitted range in
grey/open), the best-fit model, and the bin's median log M*. The fitted range
is shaded.

Usage:
    python scripts/plot_appendix_multipoles.py \
        [--fits-dir results/fits/nautilus_snr_rmin25-100_zerocross] \
        [--out-prefix plot_for_paper/figA1_bgs_snr]
"""
import argparse
import glob
import os
import re

import numpy as np

import fit_blocks  # per-block layout (split-grid aware)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_style


def _err(cov):
    return np.sqrt(np.clip(np.diag(np.asarray(cov)), 0, None))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits-dir",
                   default="results/fits/nautilus_snr_rmin25-100_zerocross")
    p.add_argument("--pattern",
                   default="BGS_RED_GMM_*massbin*_multipoles_NLA_nautilus.npz")
    p.add_argument("--out-prefix", default="plot_for_paper/figA1_bgs_snr")
    p.add_argument("--xmin", type=float, default=6.0,
                   help="x-axis lower limit [h^-1 Mpc]; the measured grids start "
                        "at ~6, so autoscale would leave the first decades empty")
    p.add_argument("--xmax", type=float, default=110.0,
                   help="x-axis upper limit [h^-1 Mpc]")
    args = p.parse_args()

    files = glob.glob(os.path.join(args.fits_dir, args.pattern))
    files.sort(key=lambda f: int(re.search(r"massbin(\d+)", f).group(1)))
    if not files:
        raise SystemExit(f"no fits matching {args.pattern} in {args.fits_dir}")

    paper_style.apply_style()

    for stat in ("xi0", "xi2"):
        ncols, nrows = 3, int(np.ceil(len(files) / 3))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7, 2.1 * nrows),
                                 sharex=True, squeeze=False)
        for ax, f in zip(axes.flat, files):
            d = np.load(f, allow_pickle=True)
            r_gg, r_gp = fit_blocks.grids(d)
            m_gg, m_gp = fit_blocks.full_model(d)
            if stat == "xi0":
                r = r_gg
                data, sig, model = d["data_gg"], _err(d["cov_xi0_corrected"]), m_gg
                rmin, fac = float(d["rmin_gg"]), r**2
            else:
                r = r_gp
                data, sig, model = d["data_gp"], _err(d["cov_xi2_corrected"]), m_gp
                rmin, fac = float(d["rmin_gp"]), r
            rmax = float(d["rmax"])
            mask = (r >= rmin) & (r <= rmax)

            ax.errorbar(r[~mask], fac[~mask] * data[~mask],
                        yerr=fac[~mask] * sig[~mask], fmt="o", ms=2.5,
                        color="0.6", mfc="white", capsize=1.5, lw=0.8)
            ax.errorbar(r[mask], fac[mask] * data[mask],
                        yerr=fac[mask] * sig[mask], fmt="o", ms=2.5,
                        color="darkorange", capsize=1.5, lw=0.8, zorder=3)
            ax.plot(r, fac * model, "-", color="k", lw=1.2, zorder=4)
            ax.axvspan(rmin, rmax, color="0.93", zorder=0)
            ax.axhline(0, color="0.7", lw=0.6)
            ax.set_xscale("log")
            # The measured grids start at ~6 h^-1 Mpc; autoscale otherwise pads
            # out to 10^0 and wastes most of the panel on empty decades.
            ax.set_xlim(args.xmin, args.xmax)
            mb = int(re.search(r"massbin(\d+)", f).group(1))
            ax.text(0.05, 0.92,
                    rf"bin {mb}, $\log M_\ast={float(d['prop_logmstar_median']):.2f}$",
                    transform=ax.transAxes, fontsize=7, va="top")

        for ax in axes.flat[len(files):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel(r"$s\ [\mathrm{Mpc}/h]$")
        ylab = (r"$s^2\,\xi_0\ [(\mathrm{Mpc}/h)^2]$" if stat == "xi0"
                else r"$s\,\tilde\xi_{g+,2}\ [\mathrm{Mpc}/h]$")
        for row in axes:
            row[0].set_ylabel(ylab)
        fig.tight_layout()
        paper_style.savefig(fig, f"{args.out_prefix}_{stat}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
