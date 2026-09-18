#!/usr/bin/env python
"""Reproduce draft Figs. B.2 & B.3: galaxy-shape correlation functions for the
blue samples, showing the wedge multipole (left) and the projected (right)
statistic side by side with their best-fit models.

Left panel : s * xi_g+,2(s)   from the multipole fit npz  (data_gp = xi2),
             with BOTH best-fit models: NLA (solid) and TATT (dashed).
Right panel: rp * wg+(rp)      from the projected fit npz  (data_gp = wgp),
             with the NLA best fit only -- there is no projected TATT fit.

Sign convention (see the ``wgp`` skill / projected-sign-convention memory): the
projected w_{g+} carries an extra minus sign relative to the multipole
xi_{g+,2}. We flip the sign of the projected panel (data + model) so both panels
display the same physical GI signal, exactly as in the draft figures.

Scales NOT entering the fit are shaded, two-tone in the multipole panel because
the two models have different quadrupole cuts (NLA 10, TATT relaxed to 6):

    light grey  -- dropped by NLA but still fitted by TATT
    darker grey -- outside the fitted range of every model shown

Error bars are the sqrt-diagonal of the Mohammad+21-corrected jackknife
covariance blocks (cov_xi2_corrected / cov_wgp_corrected) carried in the fit npz.

Usage:
    python scripts/plot_ia_corrfunc.py --mode bgsblue --out plot_for_paper/fig4_bgsblue_corrfunc.png
    python scripts/plot_ia_corrfunc.py --mode elg     --out plot_for_paper/fig5_elg_corrfunc.png
"""
import argparse
import os

import numpy as np

import fit_blocks  # per-block layout (split-grid aware)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paper_style

#: shading of the separations that no fit used, and of those the relaxed-cut
#: TATT fit kept while NLA dropped them.  Light -> dark as fewer models use the
#: scale.
SHADE_SOME = "0.93"
SHADE_NONE = "0.85"


def _err(cov):
    return np.sqrt(np.clip(np.diag(np.asarray(cov)), 0, None))


def _fit_lo(d):
    """Lower edge of the fitted gp range, as a bin EDGE (not a bin centre).

    The cut is applied to bin centres, so the boundary that should be drawn is
    the outer edge of the first retained bin -- otherwise the shading clips the
    first fitted point's own bin.
    """
    r = fit_blocks.grids(d)[1]
    keep = np.flatnonzero((r >= float(d["rmin_gp"])) & (r <= float(d["rmax"])))
    if len(keep) == 0:
        return float(r[0])
    i = int(keep[0])
    edges = np.asarray(d["edges_gp"], float) if "edges_gp" in d.files else None
    if edges is not None and len(edges) == len(r) + 1:
        return float(edges[i])
    # pre-``edges_gp`` npz: fall back to the log-midpoint of the straddling bins
    return float(r[0]) if i == 0 else float(np.sqrt(r[i - 1] * r[i]))


def load_multipole(path):
    d = np.load(path, allow_pickle=True)
    r = fit_blocks.grids(d)[1]                          # the gp block's own grid
    data = np.asarray(d["data_gp"], float)              # xi2
    err = _err(d["cov_xi2_corrected"])
    model = fit_blocks.full_model(d)[1]                 # second block = gp
    return r, data, err, model, _fit_lo(d)


def load_model(path):
    """(r, model, fit_lo) of a fit we only want the best-fit curve of."""
    d = np.load(path, allow_pickle=True)
    return fit_blocks.grids(d)[1], fit_blocks.full_model(d)[1], _fit_lo(d)


def load_projected(path, flip=True):
    d = np.load(path, allow_pickle=True)
    r = fit_blocks.grids(d)[1]                          # the gp block's own grid
    data = np.asarray(d["data_gp"], float)              # wgp
    err = _err(d["cov_wgp_corrected"])
    model = fit_blocks.full_model(d)[1]
    s = -1.0 if flip else 1.0
    return r, s * data, err, s * model, _fit_lo(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["bgsblue", "elg"], required=True)
    p.add_argument("--multipole-dir",
                   default="results/fits/nautilus_rmin25-100_zerocross")
    p.add_argument("--projected-dir", default="results/fits_projected_rmin6")
    p.add_argument("--tatt-dir",
                   default="results/fits/nautilus_split_TATT_rmin30-6_bta1",
                   help="TATT multipole fits overplotted on the left panel; "
                        "samples with no file there are silently skipped")
    p.add_argument("--no-tatt", dest="tatt", action="store_false",
                   help="draw the NLA model only")
    p.add_argument("--sampler", default="nautilus",
                   help="fit-file suffix sampler (nautilus / minuit)")
    p.add_argument("--out", default=None)
    p.add_argument("--xmin", type=float, default=5.0,
                   help="x-axis lower limit [h^-1 Mpc]; the measured grids start "
                        "at 6, so this leaves room for the 'unused by any fit' band")
    p.add_argument("--xmax", type=float, default=110.0,
                   help="x-axis upper limit [h^-1 Mpc]")
    args = p.parse_args()

    if args.mode == "bgsblue":
        # (stem, label, colour, marker, filled)
        series = [
            ("BGS_BLUE_GMM_zmin_0.10_zmax_0.50_massbin0",
             r"BGS blue low $M_\ast$", "royalblue", "o", True),
            ("BGS_BLUE_GMM_zmin_0.10_zmax_0.50_massbin1",
             r"BGS blue high $M_\ast$", "darkturquoise", "o", True),
        ]
        default_out = "plot_for_paper/fig4_bgsblue_corrfunc.png"
    else:
        series = [
            ("ELG_LOPnotqso_zmin_0.80_zmax_1.60",
             r"ELG, $0.8<z<1.6$", "royalblue", "o", False),
        ]
        default_out = "plot_for_paper/fig5_elg_corrfunc.png"

    paper_style.apply_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7, 3.5))

    nla_lo, tatt_lo, proj_lo = [], [], []

    for stem, label, colour, marker, filled in series:
        mp = os.path.join(args.multipole_dir,
                          f"{stem}_multipoles_NLA_{args.sampler}.npz")
        pr = os.path.join(args.projected_dir,
                          f"{stem}_projected_NLA_{args.sampler}.npz")
        tt = os.path.join(args.tatt_dir,
                          f"{stem}_multipoles_TATT_{args.sampler}.npz")
        mfc = colour if filled else "none"

        # left: multipole s * xi_g+,2
        r, data, err, model, lo = load_multipole(mp)
        nla_lo.append(lo)
        axL.errorbar(r, r * data, yerr=r * err, fmt=marker, ms=5, color=colour,
                     mfc=mfc, mec=colour, capsize=2, lw=1.1, label=label,
                     zorder=3)
        axL.plot(r, r * model, "-", color=colour, lw=1.8, zorder=4)

        # left: the TATT best fit of the same sample, dashed
        if args.tatt and os.path.exists(tt):
            rt, mt, lo_t = load_model(tt)
            tatt_lo.append(lo_t)
            axL.plot(rt, rt * mt, "--", color=colour, lw=1.6, zorder=4)
        elif args.tatt:
            print(f"no TATT fit for {stem} in {args.tatt_dir} -- NLA only")

        # right: projected rp * wg+ (sign-flipped to the multipole convention)
        rp, wdata, werr, wmodel, lo_p = load_projected(pr, flip=True)
        proj_lo.append(lo_p)
        # Triangles mark the PROJECTED statistic in every mode, so the two
        # estimators are told apart the same way in both corrfunc figures
        # (circles = multipoles, triangles = w_g+), matching Fig. 4.
        rmarker = "^"
        axR.errorbar(rp, rp * wdata, yerr=rp * werr, fmt=rmarker, ms=5,
                     color=colour, mfc=mfc, mec=colour, capsize=2, lw=1.1,
                     zorder=3)
        axR.plot(rp, rp * wmodel, "-", color=colour, lw=1.8, zorder=4)

    # Shade what the fits did not use.  In the multipole panel the two models
    # stop at different scales, so the grey deepens where TATT stops too.
    lo_nla = max(nla_lo)
    lo_tatt = max(tatt_lo) if tatt_lo else lo_nla
    if lo_tatt < lo_nla:
        axL.axvspan(lo_tatt, lo_nla, color=SHADE_SOME, lw=0, zorder=0)
    if args.xmin < lo_tatt:
        axL.axvspan(args.xmin, lo_tatt, color=SHADE_NONE, lw=0, zorder=0)
    if args.xmin < max(proj_lo):
        axR.axvspan(args.xmin, max(proj_lo), color=SHADE_NONE, lw=0, zorder=0)

    for ax in (axL, axR):
        # dotted, not dashed: dashed is the TATT model line in the left panel
        ax.axhline(0.0, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        # measured grids start at 6 h^-1 Mpc -- do not autoscale down to 10^0
        ax.set_xlim(args.xmin, args.xmax)
    axL.set_xlabel(r"$s\ [\mathrm{Mpc}/h]$")
    axL.set_ylabel(r"$s\,\tilde\xi_{g+,2}\ [\mathrm{Mpc}/h]$")
    axR.set_xlabel(r"$r_{\rm p}\ [\mathrm{Mpc}/h]$")
    axR.set_ylabel(r"$r_{\rm p}\,w_{g+}\ [\mathrm{Mpc}/h]$")

    # Two legends rather than one: at slide font scale a single four-entry box
    # has nowhere to sit in the left panel without landing on the data.  The
    # samples key (which applies to both panels) goes to the top of the
    # projected panel, the empty corner of the figure; the model-line key stays
    # in the quadrupole panel, the only panel showing two models.
    fs = 8 * paper_style.FONTSCALE
    samples, _ = axL.get_legend_handles_labels()
    models = [Line2D([], [], color="0.35", ls="-", lw=1.8, label="NLA")]
    if tatt_lo:
        models.append(Line2D([], [], color="0.35", ls="--", lw=1.6, label="TATT"))
    axR.legend(handles=samples, frameon=False, loc="upper left", fontsize=fs)
    axL.legend(handles=models, frameon=False, loc="lower right", fontsize=fs,
               handlelength=2.2)

    fig.tight_layout()
    out = args.out or default_out
    paper_style.savefig(fig, out, dpi=300)


if __name__ == "__main__":
    main()
