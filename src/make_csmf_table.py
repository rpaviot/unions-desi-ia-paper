#!/usr/bin/env python3
"""Appendix table of the CSMF (SHMR + satellite) parameters from a saved fit_csmf.py npz.

nautilus fit: one column with the posterior median and its 16-84 % interval, one with the
best fit (the ``--point`` estimate, default the MAP sample of the chain), and the chi2 at
the best fit in the table foot. MINUIT / DE fit: a single "best fit +- parabolic error"
column. Writes a complete A&A ``table`` environment, like make_paper_tables.py.

    ${HOD_PYTHON} src/make_csmf_table.py --fit-npz results/baseline/csmf_fit/csmf_fit.npz \
        --method nautilus --point map --out table_csmf.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csmf_point import POINTS, fit_method, load_point  # noqa: E402

# (tex symbol, description) in the order of the paper's table; parameters the fit does
# not carry (fixed) are skipped.
ROWS = [
    ("M0", r"$\log_{10} M_0$", "SHMR normalisation"),
    ("M1", r"$\log_{10} M_1$", "Characteristic halo mass"),
    ("gamma1", r"$\gamma_1$", "SHMR low-mass slope"),
    ("gamma2", r"$\gamma_2$", "SHMR high-mass slope"),
    ("sigma_c", r"$\sigma_{\rm c}$", "Central SHMR scatter"),
    ("alpha_s", r"$\alpha_{\rm s}$", "Satellite CSMF slope"),
    ("b0", r"$b_0$", "Satellite normalisation"),
    ("b1", r"$b_1$", "Satellite normalisation"),
    ("f_h", r"$f_{\rm h}$", "Matter profile rescaling"),
    ("f_s", r"$f_{\rm s}$", "Satellite profile rescaling"),
]
LABEL = "tab:csmf_bestfit"


def ndigits(err):
    """Decimals for the usual convention: the error to two significant figures when its
    leading digit is 1, else to one (and never fewer than one decimal)."""
    err = abs(float(err))
    if not np.isfinite(err) or err == 0:
        return 2
    mag = int(np.floor(np.log10(err)))
    lead = int(err / 10 ** mag)
    return max(1, -mag + (2 if lead == 1 else 1) - 1)


def asym(v, lo, hi):
    nd = ndigits(min(hi - v, v - lo))
    return rf"${v:.{nd}f}^{{+{hi - v:.{nd}f}}}_{{-{v - lo:.{nd}f}}}$"


def sym(v, e):
    nd = ndigits(e)
    return rf"${v:.{nd}f} \pm {e:.{nd}f}$"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit-npz", required=True)
    p.add_argument("--method", default=None, choices=["minuit", "nautilus", "de"],
                   help="optimiser whose keys to read (default: whichever the npz carries)")
    p.add_argument("--point", default="map", choices=list(POINTS),
                   help="nautilus only: the best-fit column (default map)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = np.load(args.fit_npz, allow_pickle=True)
    m = fit_method(d, args.method)
    names, best, chi2, ndof, point_label = load_point(d, m, args.point)
    idx = {n: i for i, n in enumerate(names)}
    lines = [r"\begin{table}"]

    if m == "nautilus":
        med = np.asarray(d["nautilus_best_fit"], float)
        lo = np.asarray(d["nautilus_lo16"], float)
        hi = np.asarray(d["nautilus_hi84"], float)
        point_short = {"map": "MAP", "mean": "Mean", "median": "Median"}[args.point]
        lines += [
            r"\caption{CSMF parameters: posterior median with the $16$--$84\%$ interval, and "
            + (r"the maximum a posteriori (MAP) sample of the \textsc{nautilus} chain."
               if args.point == "map" else rf"the posterior {args.point}.") + "}",
            rf"\label{{{LABEL}}}",
            r"\centering", r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{l l r r}", r"\hline\hline",
            rf"Parameter & Description & Median & {point_short} \\", r"\hline",
        ]
        for key, sym_tex, desc in ROWS:
            if key not in idx:
                continue
            i = idx[key]
            nd = ndigits(min(hi[i] - med[i], med[i] - lo[i]))
            lines.append(f"{sym_tex} & {desc} & {asym(med[i], lo[i], hi[i])} & ${best[i]:.{nd}f}$ \\\\")
        # The paper's chi2 (Eq. C.1) has no prior term: -2 log L at the point. Saved for
        # the MAP (nautilus_map_log_l); for the median / mean fall back to the objective.
        chi2_data = (-2.0 * float(d["nautilus_map_log_l"])
                     if args.point == "map" and "nautilus_map_log_l" in d else chi2)
        what = {"map": "the maximum a posteriori sample of the chain",
                "mean": "the posterior mean", "median": "the marginal posterior median"}[args.point]
        foot = (r"$M_0$ is in $h^{-2}M_\odot$ and $M_1$ in $h^{-1}M_\odot$ ($M_{200{\rm m}}$). "
                rf"{point_short}: {what}, the best-fit model of Fig.~\ref{{fig:deltasigma_bestfit}}; "
                rf"$\chi^2 = {chi2_data:.1f}$ there "
                rf"(Eq.~\ref{{eq:chi2_csmf}}, {ndof + len(names)} $\Delta\Sigma$ points and "
                rf"$10$ number densities for {len(names)} free parameters).")
    else:
        err_key = f"{m}_errors"
        errs = np.asarray(d[err_key], float) if err_key in d else np.full(len(names), np.nan)
        lines += [
            rf"\caption{{Best-fit CSMF parameters ({'MINUIT, parabolic errors' if m == 'minuit' else m}).}}",
            rf"\label{{{LABEL}}}",
            r"\centering", r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{l l r}", r"\hline\hline",
            r"Parameter & Description & Best fit \\", r"\hline",
        ]
        for key, sym_tex, desc in ROWS:
            if key not in idx:
                continue
            i = idx[key]
            lines.append(f"{sym_tex} & {desc} & {sym(best[i], errs[i])} \\\\")
        foot = (r"$M_0$ is in $h^{-2}M_\odot$ and $M_1$ in $h^{-1}M_\odot$ ($M_{200{\rm m}}$). "
                rf"$\chi^2 = {chi2:.1f}$ at the minimum (Eq.~\ref{{eq:chi2_csmf}}).")

    lines += [r"\hline", r"\end{tabular}", rf"\tablefoot{{{foot}}}", r"\end{table}", ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"[make_csmf_table] {m}, {point_label}: chi2 = {chi2:.3f} / ndof {ndof}  -> {out}")


if __name__ == "__main__":
    main()
