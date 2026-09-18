#!/usr/bin/env python
"""Reduced-chi2 distribution of the fiducial NLA minuit fits.

For each sample's fit npz (written by fit_ia.py) this partitions the joint
chi2 into its clustering (monopole, gg) and IA (quadrupole, gp) blocks and
reports the reduced chi2 for the combined, clustering-only and IA-only cases.

Fiducial config (zero_cross_cov=True, no Hartlap, Mohammad+21 corrected cov):
the cut covariance is block diagonal, so chi2_combined = chi2_gg + chi2_gp
exactly, and the dof partition (n_gg-1) + (n_gp-1) = n_total - 2 is consistent
with the joint NLA fit (b1 carried by the monopole, a1 by the quadrupole).

Usage:
    python scripts/chi2_distribution.py [--fits-dir results/fits_zerocross_rmin25]
                                        [--out results/figures/chi2_distribution.png]
"""
import argparse
import glob
import os

import numpy as np

import fit_blocks  # per-block layout (split-grid aware)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ia_family(sample):
    """(tracer, colour) family label for grouping the IA distribution."""
    if sample.startswith("BGS_RED_GMM"):
        return "BGS RED (GMM)"
    if sample.startswith("BGS_BLUE_GMM"):
        return "BGS BLUE (GMM)"
    if sample.startswith("BGS_RED_SFR"):
        return "BGS RED (SFR)"
    if sample.startswith("BGS_BLUE_SFR"):
        return "BGS BLUE (SFR)"
    if sample.startswith("BGS_ANY"):
        return "BGS ANY"
    if sample.startswith("LRG_zmin_0.40"):
        return "LRG 0.4<z<0.75"
    if sample.startswith("LRG_zmin_0.75"):
        return "LRG 0.75<z<1.1"
    if sample.startswith("ELG"):
        return "ELG"
    return "other"


# fixed display order for the grouped IA breakdown
FAMILY_ORDER = ["BGS ANY", "BGS RED (GMM)", "BGS BLUE (GMM)",
                "BGS RED (SFR)", "BGS BLUE (SFR)",
                "LRG 0.4<z<0.75", "LRG 0.75<z<1.1", "ELG"]


def block_chi2(cov_block, diff):
    """diff^T C^-1 diff for one (already zero-cross, already cut) block."""
    return float(diff @ np.linalg.inv(cov_block) @ diff)


def partition(fit_npz):
    """Return dict with combined/clustering/IA chi2, dof and reduced chi2."""
    d = np.load(fit_npz, allow_pickle=True)
    s_gg, s_gp = fit_blocks.grids(d)    # the two blocks may be on different grids
    n = len(s_gg)
    cov = d["cov"]                      # Mohammad-corrected joint cov, (n_gg+n_gp)^2
    data = np.concatenate([d["data_gg"], d["data_gp"]])
    model = d["model_full"]            # model at the full grids, stacked [gg, gp]
    diff = data - model

    rmin_gg, rmin_gp, rmax = float(d["rmin_gg"]), float(d["rmin_gp"]), float(d["rmax"])
    gg = np.where((s_gg >= rmin_gg) & (s_gg <= rmax))[0]      # monopole block idx
    gp = n + np.where((s_gp >= rmin_gp) & (s_gp <= rmax))[0]  # quadrupole block idx

    # zero_cross_cov: blocks are independent, invert each on its own
    chi2_gg = block_chi2(cov[np.ix_(gg, gg)], diff[gg])
    chi2_gp = block_chi2(cov[np.ix_(gp, gp)], diff[gp])
    chi2_comb = chi2_gg + chi2_gp

    n_gg, n_gp = len(gg), len(gp)
    dof_gg, dof_gp = n_gg - 1, n_gp - 1   # b1 <- monopole, a1 <- quadrupole
    dof_comb = n_gg + n_gp - 2            # NLA: 2 free params (b1, a1)

    # cross-check against the chi2 fit_ia stored. The block-wise inversion here
    # differs from fit_ia's whole-matrix inverse only by round-off (these cut
    # covariances are ill-conditioned, cond ~1e4-1e5), so allow a loose tol.
    stored = float(d["chi2"])
    assert abs(stored - chi2_comb) < 1e-3 * max(1.0, abs(stored)), \
        f"{os.path.basename(fit_npz)}: chi2 mismatch {stored} vs {chi2_comb}"

    return {
        "sample": str(d["sample"]),
        "chi2_comb": chi2_comb, "dof_comb": dof_comb,
        "chi2_clust": chi2_gg, "dof_clust": dof_gg,
        "chi2_ia": chi2_gp, "dof_ia": dof_gp,
        "red_comb": chi2_comb / dof_comb,
        "red_clust": chi2_gg / dof_gg,
        "red_ia": chi2_gp / dof_gp,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits-dir", default="results/fits_zerocross_rmin25")
    p.add_argument("--suffix", default="multipoles_NLA_minuit")
    p.add_argument("--out", default="results/figures/chi2_distribution.png")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.fits_dir, f"*_{args.suffix}.npz")))
    rows = [partition(f) for f in files]

    # ---- table ------------------------------------------------------------
    hdr = (f"{'sample':<48} "
           f"{'combined':>16} {'clustering(xi0)':>16} {'IA(xi2)':>16}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['sample']:<48} "
              f"{r['chi2_comb']:7.1f}/{r['dof_comb']:<2d}={r['red_comb']:4.2f} "
              f"{r['chi2_clust']:7.1f}/{r['dof_clust']:<2d}={r['red_clust']:4.2f} "
              f"{r['chi2_ia']:7.1f}/{r['dof_ia']:<2d}={r['red_ia']:4.2f}")

    # ---- distribution summary --------------------------------------------
    def summary(key):
        v = np.array([r[key] for r in rows])
        return (f"  N={len(v):2d}  mean={v.mean():.2f}  median={np.median(v):.2f}  "
                f"std={v.std():.2f}  min={v.min():.2f}  max={v.max():.2f}")

    print("\nReduced-chi2 distribution over", len(rows), "samples:")
    print(" combined  :", summary("red_comb"))
    print(" clustering:", summary("red_clust"))
    print(" IA        :", summary("red_ia"))

    # ---- IA distribution broken out by tracer / colour --------------------
    for r in rows:
        r["family"] = ia_family(r["sample"])
    print("\nIA (xi2) reduced-chi2 by tracer / colour:")
    print(f"  {'family':<16} {'N':>2}  {'median':>6} {'mean':>5} {'std':>5} "
          f"{'min':>5} {'max':>5}")
    groups = {}
    for fam in FAMILY_ORDER:
        v = np.array([r["red_ia"] for r in rows if r["family"] == fam])
        if not len(v):
            continue
        groups[fam] = v
        print(f"  {fam:<16} {len(v):>2d}  {np.median(v):6.2f} {v.mean():5.2f} "
              f"{v.std():5.2f} {v.min():5.2f} {v.max():5.2f}")

    # ---- histogram --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(0, max(r["red_comb"] for r in rows) * 1.05, 16)
    for key, lab, col in [("red_comb", "combined", "k"),
                          ("red_clust", r"clustering ($\xi_0$)", "C0"),
                          ("red_ia", r"IA ($\xi_2$)", "C3")]:
        v = np.array([r[key] for r in rows])
        ax.hist(v, bins=bins, histtype="step", lw=2, color=col,
                label=f"{lab} (med {np.median(v):.2f})")
    ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.set_xlabel(r"reduced $\chi^2 = \chi^2/\mathrm{dof}$")
    ax.set_ylabel("number of samples")
    ax.set_title(f"NLA minuit fiducial (zero-cross, Mohammad, no Hartlap)\n"
                 f"{os.path.basename(args.fits_dir)}, {len(rows)} samples")
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"\nwrote {args.out}")

    # ---- IA-by-group strip plot ------------------------------------------
    fams = [f for f in FAMILY_ORDER if f in groups]
    fig2, ax2 = plt.subplots(figsize=(7.6, 4.4))
    rng = np.random.default_rng(0)
    for i, fam in enumerate(fams):
        v = groups[fam]
        x = i + rng.uniform(-0.12, 0.12, size=len(v))
        ax2.scatter(x, v, s=26, color="C3", alpha=0.8, zorder=3)
        ax2.hlines(np.median(v), i - 0.25, i + 0.25, color="k", lw=2, zorder=4)
    ax2.axhline(1.0, color="grey", ls="--", lw=1)
    ax2.set_xticks(range(len(fams)))
    ax2.set_xticklabels(fams, rotation=30, ha="right")
    ax2.set_ylabel(r"IA ($\xi_2$) reduced $\chi^2$  (dof 9)")
    ax2.set_title("IA reduced-$\\chi^2$ by tracer / colour "
                  "(black bar = median)")
    ax2.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    out2 = os.path.join(os.path.dirname(args.out), "chi2_ia_by_group.png")
    fig2.savefig(out2, dpi=200)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
