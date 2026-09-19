#!/usr/bin/env python3
"""Compare lc-materialised outputs with the paper's production results.

Two checks, both printed as tables:

  csmf   the MINUIT CSMF fit (analyses/csmf/results/<universe>/csmf_fit/csmf_fit.npz) against
         the production p2 fit: parameters, Hesse errors and chi2 are expected BIT-IDENTICAL
         (seeded two-pass Migrad on identical inputs). With --nautilus the posterior-median
         keys (nautilus_best_fit / _errors) of a nautilus universe are compared with the MINUIT
         reference in units of the MINUIT error instead.
  ia     every fit npz of an ia_fits output (nla_multipoles / nla_projected / tatt_blue) against
         the same file name in the production nautilus tree: |Delta bestfit| / sigma per free
         parameter, and Delta chi2. nautilus is unseeded, so agreement is expected within the
         posterior noise (a few 0.1 sigma), not bit-for-bit.

Cluster-only (the reference trees live under ~/unions_IA and /n09data); run from the project root:

    python scripts/compare_with_reference.py csmf
    python scripts/compare_with_reference.py csmf --universe csmf_nautilus --nautilus
    python scripts/compare_with_reference.py ia            # all three ia_fits outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REF_CSMF = Path("/n09data/rpaviot/DESIxUnions/ggl/csmf_fit/"
                "csmf_fit_bgs_snr_dr6_rmin05_fhcap_g1prior_z5eff_t10_m200m_p2_minuit.npz")
REF_IA = {
    "nla_multipoles": Path("~/unions_IA/results/fits/nautilus_split_rmin30-10_zerocross_zeff-pair"),
    "nla_projected": Path("~/unions_IA/results/fits/nautilus_split_projected_rmin6_zeff-pair"),
    "tatt_blue": Path("~/unions_IA/results/fits/nautilus_split_TATT_rmin30-6_bta1_a2wider_zeff-pair"),
}


def compare_csmf(universe: str, nautilus: bool) -> int:
    new = Path(f"analyses/csmf/results/{universe}/csmf_fit/csmf_fit.npz")
    if not new.exists():
        print(f"[csmf] {new} missing"); return 1
    a, b = np.load(new, allow_pickle=True), np.load(REF_CSMF, allow_pickle=True)
    names = [str(n) for n in b["minuit_param_names"]]
    ref_bf, ref_err = b["minuit_best_fit"], b["minuit_errors"]
    key = "nautilus" if nautilus else "minuit"
    bf, err = a[f"{key}_best_fit"], a[f"{key}_errors"]
    print(f"[csmf] {new}  ({key})  vs  {REF_CSMF.name}")
    print(f"  {'param':8s} {'lc':>12s} {'paper':>12s} {'diff':>11s} {'diff/sig':>9s} {'err lc':>9s} {'err paper':>9s}")
    worst = 0.0
    for n, x, y, ex, ey in zip(names, bf, ref_bf, err, ref_err):
        d = x - y
        worst = max(worst, abs(d) / ey if ey > 0 else 0)
        print(f"  {n:8s} {x:12.6f} {y:12.6f} {d:11.2e} {d/ey if ey>0 else 0:9.3f} {ex:9.4f} {ey:9.4f}")
    c_new = float(a[f"{key}_chi2"]); c_ref = float(b["minuit_chi2"])
    print(f"  chi2  {c_new:.6f} vs {c_ref:.6f}  (diff {c_new-c_ref:+.2e});  ndof {int(a[f'{key}_ndof'])} vs {int(b['minuit_ndof'])}")
    if nautilus:
        print(f"  nautilus-median vs MINUIT: worst |diff|/sigma_minuit = {worst:.2f}")
        return 0
    same = np.array_equal(bf, ref_bf) and np.array_equal(err, ref_err) and c_new == c_ref
    print(f"  BIT-IDENTICAL: {same}" + ("" if same else f"   (max |diff|/sigma {worst:.2e}; "
          f"max rel param diff {np.max(np.abs(bf-ref_bf)/np.maximum(np.abs(ref_bf),1e-12)):.2e})"))
    return 0 if same else 2


def compare_ia(output: str, universe: str) -> int:
    new_dir = Path(f"analyses/ia_fits/results/{universe}/{output}")
    ref_dir = REF_IA[output].expanduser()
    files = sorted(new_dir.glob("*.npz"))
    if not files:
        print(f"[{output}] no npz under {new_dir}"); return 1
    # match on the stem without the trailing sampler tag (_minuit / _nautilus)
    stem = lambda f: f.name.rsplit("_", 1)[0]
    refs = {stem(r): r for r in ref_dir.glob("*.npz")}
    print(f"\n[{output}] {len(files)} fits in {new_dir}  vs  {ref_dir}")
    print(f"  {'sample':58s} {'param':4s} {'lc':>8s} {'paper':>8s} {'d/sig':>6s} | {'chi2 lc':>8s} {'paper':>8s}")
    worst, missing = 0.0, 0
    for f in files:
        r = refs.get(stem(f))
        if r is None:
            missing += 1; print(f"  {f.name:58s} -- no reference file"); continue
        a, b = np.load(f, allow_pickle=True), np.load(r, allow_pickle=True)
        names = [str(n) for n in a["param_names"]]
        free = {str(n) for n in a["free_params"]}
        first = True
        for i, n in enumerate(names):
            if n not in free:
                continue
            sig = float(b["errors"][i]) or np.inf
            d = (float(a["bestfit"][i]) - float(b["bestfit"][i])) / sig
            worst = max(worst, abs(d))
            tail = f" | {float(a['chi2']):8.2f} {float(b['chi2']):8.2f}" if first else ""
            label = f.name.replace("_multipoles", "").replace("_projected", "").replace(".npz", "")[:58]
            print(f"  {label if first else '':58s} {n:4s} {float(a['bestfit'][i]):8.3f} {float(b['bestfit'][i]):8.3f} {d:6.2f}{tail}")
            first = False
    print(f"  -> worst |Delta bestfit|/sigma_paper = {worst:.2f}; {missing} file(s) without reference")
    return 0 if worst < 1.0 and missing == 0 else 2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["csmf", "ia"])
    p.add_argument("--universe", default="baseline")
    p.add_argument("--nautilus", action="store_true", help="csmf: compare the nautilus posterior median")
    p.add_argument("--outputs", nargs="*", default=list(REF_IA), help="ia: which ia_fits outputs")
    args = p.parse_args()
    if args.what == "csmf":
        raise SystemExit(compare_csmf(args.universe, args.nautilus))
    rc = 0
    for o in args.outputs:
        rc = max(rc, compare_ia(o, args.universe))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
