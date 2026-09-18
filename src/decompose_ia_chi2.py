#!/usr/bin/env python3
"""Split the IA multipole chi2 into its clustering (xi0) and IA (xi2) parts.

The fit stores a single total chi2. This reconstructs it from the saved data
vector, best-fit model and covariance, and splits it per statistic:

    chi2_tot = chi2_xi0 (clustering monopole) + chi2_xi2 (IA quadrupole)

The split is EXACT, not approximate, whenever the fit ran with
``--zero-cross-cov`` (the standing convention). IALikelihood zeroes the
xi0 x xi2 blocks of the full covariance BEFORE applying the radial cut
(ia_likelihood.py:83-88, then :136), so the cut covariance is block diagonal,
its inverse is block diagonal too, and the quadratic form has no cross term.
Without --zero-cross-cov the blocks are coupled: the two parts then no longer
sum to the total, and the script says so instead of quietly reporting halves.

The reconstruction mirrors set_cut() exactly -- zero cross blocks, cut, invert,
and apply the (n - p - 2)/(n - 1) factor only when the fit used Hartlap
(``jack=not --no-hartlap``, fit_ia.py:163). Every sample's rebuilt total is
checked against the stored chi2 and a mismatch is reported, so a silent
convention drift cannot pass as a result.

Usage:
    python scripts/decompose_ia_chi2.py results/fits/minuit_hs2_rmin30-10_zerocross
    python scripts/decompose_ia_chi2.py DIR_NEW --ref DIR_FIDUCIAL
"""

import argparse
import glob
import os

import numpy as np

import fit_blocks  # per-block layout (split-grid aware)


def decompose(path):
    """Per-statistic chi2 for one fit npz. Returns a dict, or None if unusable."""
    z = np.load(path, allow_pickle=True)
    if str(z["stat"]) != "multipoles":
        return None

    r_gg, r_gp = fit_blocks.grids(z)   # the two blocks may be on different grids
    n = len(r_gg)
    cov = np.asarray(z["cov"]).astype(float).copy()
    zero_cross = bool(z["zero_cross_cov"])

    # --- mirror IALikelihood.__init__: zero the cross blocks on the FULL cov ---
    if zero_cross:
        block = np.zeros_like(cov, dtype=bool)
        block[:n, :n] = True
        block[n:, n:] = True
        cov = cov * block

    # --- mirror set_cut(): per-statistic radial masks, then one joint index set ---
    rmin_gg, rmin_gp, rmax = float(z["rmin_gg"]), float(z["rmin_gp"]), float(z["rmax"])
    m_gg = (r_gg >= rmin_gg) & (r_gg <= rmax)
    m_gp = (r_gp >= rmin_gp) & (r_gp <= rmax)
    idx = np.concatenate([np.where(m_gg)[0], n + np.where(m_gp)[0]])

    inv = np.linalg.inv(cov[np.ix_(idx, idx)])
    if bool(z["hartlap"]):
        nreal = int(z["n_realisations"])
        inv = inv * ((nreal - len(idx) - 2) / (nreal - 1))

    data = np.concatenate([np.asarray(z["data_gg"])[m_gg],
                           np.asarray(z["data_gp"])[m_gp]])
    resid = data - np.asarray(z["model_cut"])

    chi2_tot = float(resid @ inv @ resid)
    k = int(m_gg.sum())
    chi2_gg = float(resid[:k] @ inv[:k, :k] @ resid[:k])
    chi2_gp = float(resid[k:] @ inv[k:, k:] @ resid[k:])

    names = [str(s) for s in z["param_names"]]
    bf, er = np.asarray(z["bestfit"]), np.asarray(z["errors"])
    free = [str(s) for s in z["free_params"]]

    return dict(
        sample=str(z["sample"]),
        chi2_stored=float(z["chi2"]), chi2_tot=chi2_tot,
        chi2_gg=chi2_gg, chi2_gp=chi2_gp,
        n_gg=k, n_gp=int(m_gp.sum()), dof=int(z["dof"]), n_free=len(free),
        a1=float(bf[names.index("a1")]), a1_err=float(er[names.index("a1")]),
        b1=float(bf[names.index("b1")]), b1_err=float(er[names.index("b1")]),
        zero_cross=zero_cross,
        nbins=n, smin=float(r_gg[0]), smax=float(r_gg[-1]),
        # split-grid fits carry a second, independent grid for xi2
        split=fit_blocks.is_split(z),
        nbins_gp=len(r_gp), smin_gp=float(r_gp[0]), smax_gp=float(r_gp[-1]),
        rmin_gg=rmin_gg, rmin_gp=rmin_gp, rmax=rmax,
    )


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*_multipoles_*.npz"))):
        rec = decompose(f)
        if rec is not None:
            out[rec["sample"]] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fitdir")
    ap.add_argument("--ref", default=None, help="reference fit dir to compare against")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="relative tolerance on the rebuilt-vs-stored chi2 check")
    args = ap.parse_args()

    new = load_dir(args.fitdir)
    if not new:
        raise SystemExit(f"no multipole fits in {args.fitdir}")
    ref = load_dir(args.ref) if args.ref else {}

    # ---- validation: the rebuilt total must reproduce the stored chi2 ----------
    bad = [(k, v["chi2_tot"], v["chi2_stored"]) for k, v in {**ref, **new}.items()
           if abs(v["chi2_tot"] - v["chi2_stored"]) > args.tol * max(v["chi2_stored"], 1.0)]
    nocross = [k for k, v in {**ref, **new}.items() if not v["zero_cross"]]
    if bad:
        print("!! rebuilt chi2 disagrees with the stored value -- decomposition NOT trustworthy")
        for k, a, b in bad[:10]:
            print(f"   {k}: rebuilt {a:.6f} vs stored {b:.6f}")
        raise SystemExit(1)
    if nocross:
        print(f"!! {len(nocross)} fit(s) ran WITHOUT --zero-cross-cov: the xi0/xi2 "
              "blocks are coupled and the two parts do NOT sum to the total.\n")

    def _grid(x):
        """One grid, or the two independent grids of a split-binning fit."""
        base = f"{x['nbins']} bins [{x['smin']:.0f},{x['smax']:.0f}]"
        if not x.get("split"):
            return base
        return (f"SPLIT xi0 {base} | xi2 {x['nbins_gp']} bins "
                f"[{x['smin_gp']:.0f},{x['smax_gp']:.0f}]")

    g = next(iter(new.values()))
    print(f"NEW  {args.fitdir}")
    print(f"     grid {_grid(g)}   "
          f"cuts xi0>={g['rmin_gg']:.0f}  xi2>={g['rmin_gp']:.0f}  rmax={g['rmax']:.0f}   "
          f"points {g['n_gg']}+{g['n_gp']}={g['n_gg']+g['n_gp']}  dof={g['dof']}")
    if ref:
        h = next(iter(ref.values()))
        print(f"REF  {args.ref}")
        print(f"     grid {_grid(h)}   "
              f"cuts xi0>={h['rmin_gg']:.0f}  xi2>={h['rmin_gp']:.0f}  rmax={h['rmax']:.0f}   "
              f"points {h['n_gg']}+{h['n_gp']}={h['n_gg']+h['n_gp']}  dof={h['dof']}")
    print(f"\n(rebuilt chi2 reproduces the stored value for all "
          f"{len(new) + len(ref)} fits)\n")

    keys = [k for k in new if k in ref] if ref else list(new)
    keys.sort()

    if ref:
        hdr = (f"{'sample':44s} {'chi2 NEW':>18s} {'chi2 REF':>18s} "
               f"{'A1 NEW':>14s} {'A1 REF':>14s} {'dA1/sig':>8s}")
    else:
        hdr = f"{'sample':44s} {'chi2 tot/dof':>14s} {'xi0':>13s} {'xi2':>13s} {'A1':>16s}"
    print(hdr)
    print("-" * len(hdr))

    acc = []
    for k in keys:
        v = new[k]
        if ref:
            w = ref[k]
            ds = (v["a1"] - w["a1"]) / np.hypot(v["a1_err"], w["a1_err"])
            print(f"{k:44s} "
                  f"{v['chi2_tot']:6.1f}/{v['dof']:<3d}({v['chi2_gg']:4.1f}+{v['chi2_gp']:4.1f}) "
                  f"{w['chi2_tot']:6.1f}/{w['dof']:<3d}({w['chi2_gg']:4.1f}+{w['chi2_gp']:4.1f}) "
                  f"{v['a1']:7.3f}+-{v['a1_err']:5.3f} {w['a1']:7.3f}+-{w['a1_err']:5.3f} "
                  f"{ds:+8.2f}")
            acc.append((v, w, ds))
        else:
            print(f"{k:44s} {v['chi2_tot']:8.1f}/{v['dof']:<5d} "
                  f"{v['chi2_gg']:7.1f}/{v['n_gg']:<5d} {v['chi2_gp']:7.1f}/{v['n_gp']:<5d} "
                  f"{v['a1']:8.3f}+-{v['a1_err']:5.3f}")
            acc.append((v, None, None))

    # ---- aggregate ------------------------------------------------------------
    print()
    def stats(recs, tag):
        red = np.array([r["chi2_tot"] / r["dof"] for r in recs])
        pgg = np.array([r["chi2_gg"] / r["n_gg"] for r in recs])
        pgp = np.array([r["chi2_gp"] / r["n_gp"] for r in recs])
        frac = np.array([r["chi2_gg"] / r["chi2_tot"] for r in recs])
        print(f"{tag}  chi2/dof median {np.median(red):.2f}  mean {red.mean():.2f}  "
              f"max {red.max():.2f}   |   per-point xi0 {np.median(pgg):.2f}  "
              f"xi2 {np.median(pgp):.2f}   |   xi0 share of chi2 {np.median(frac):.0%}")

    stats([a[0] for a in acc], "NEW ")
    if ref:
        stats([a[1] for a in acc], "REF ")
        d = np.array([a[2] for a in acc])
        a1n = np.array([a[0]["a1"] for a in acc])
        a1r = np.array([a[1]["a1"] for a in acc])
        en = np.array([a[0]["a1_err"] for a in acc])
        er = np.array([a[1]["a1_err"] for a in acc])
        print(f"\nA1  median shift {np.median(a1n - a1r):+.3f}  "
              f"median |shift|/sigma {np.median(np.abs(d)):.2f}  max |shift|/sigma {np.abs(d).max():.2f}")
        print(f"A1  median error  NEW {np.median(en):.3f}   REF {np.median(er):.3f}   "
              f"ratio {np.median(en) / np.median(er):.2f}")


if __name__ == "__main__":
    main()
