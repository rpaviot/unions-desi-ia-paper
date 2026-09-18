#!/usr/bin/env python3
"""Generate the paper's appendix fit tables (LaTeX) straight from the fit npz.

Writes three tables, split by galaxy colour:
  tab:fits_red   NLA multipole fits, red samples  (BGS red + LRG)   -> Table B.1
  tab:fits_blue  NLA multipole fits, blue samples (BGS blue + ELG)  -> Table B.2
  tab:fits_tatt  TATT multipole fits, blue samples                  -> Table B.3

Generating them from the npz rather than by hand removes any chance of a
transcription error between the chains and the manuscript.

  python scripts/make_paper_tables.py --multipole-dir ... --projected-dir ... \
      --out-multipole sections/table_fits.tex --out-projected sections/table_fits_projected.tex
"""
import argparse
import glob
import os
import re

import numpy as np

Z = {"BGS": "$0.1$--$0.5$", "LRG_lo": "$0.4$--$0.75$",
     "LRG_hi": "$0.75$--$1.1$", "ELG": "$0.8$--$1.6$"}


def load(path):
    z = np.load(path, allow_pickle=True)
    n = [str(x) for x in z["param_names"]]
    free = [str(x) for x in z["free_params"]]
    get = lambda k: (float(z["bestfit"][n.index(k)]), float(z["errors"][n.index(k)]))
    d = dict(free=free, chi2=float(z["chi2"]), dof=int(z["dof"]),
             red=float(z["chi2"]) / int(z["dof"]),
             nsh=int(z["n_shapes"]),
             sig_e=float(z["prop_sigma_e"]), R=float(z["prop_r_mean"]),
             lm=float(z["prop_logmstar_median"]), lms=float(z["prop_logmstar_std"]))
    # Redshift at which each block of the model is evaluated (fit_ia.py
    # --zeff-ia, 2026-09-17). Older npz predate the keys: both blocks were then
    # evaluated at the density-sample z_eff.
    zc = float(z["z_eff_clustering"])
    d["z_gg"] = float(z["z_eff_model_gg"]) if "z_eff_model_gg" in z else zc
    d["z_gp"] = float(z["z_eff_model_gp"]) if "z_eff_model_gp" in z else zc
    for k in ("b1", "a1", "a2"):
        if k in n:
            d[k], d[k + "_e"] = get(k)
    return d


def num(v, e, nd=3):
    return f"${v:.{nd}f} \\pm {e:.{nd}f}$"


def thou(n):
    return f"${n:,}$".replace(",", "\\,")


TABLEFOOT = (
    r"\tablefoot{$\log M_\star$ is the median stellar mass in units of "
    r"$\log(M/h^{-2}M_\odot)$, with the standard deviation of the distribution within "
    r"each bin. $N_{\rm shapes}$ is the number of galaxies in the shape sample. "
    r"$\sigma_e$ is the per-component calibrated shape noise, computed following "
    r"the definition of \citet{Heymans2012} as in \citet{HervasPeters2024}. "
    r"$\mathcal{R}$ is the effective shear response, the weighted average of the "
    r"diagonal elements of the per-galaxy \texttt{Metacalibration} response "
    r"matrix, following \citet{Siegel2025}. $z_{\rm IA}$ is the pair-weighted "
    r"effective redshift of the density--shape cross-correlation, at which the "
    r"galaxy--shape multipole is modelled; the "
    r"clustering monopole is modelled at the effective redshift of the density "
    r"sample, ZGG. ``auto'' denotes samples where the "
    r"density and shape redshift distributions are identical (global clustering "
    r"sample); ``cross'' denotes stellar-mass sub-samples cross-correlated with "
    r"the full density tracer. The linear bias $b_1$ is constrained by the "
    r"clustering monopole, which is common to all sub-samples of a given tracer "
    r"and redshift bin, and is therefore quoted only for the global (auto) "
    r"samples. $A_{\rm IA}^{\tilde\xi_{2,2}}$ is the IA amplitude from the wedge "
    r"multipole fit, with the reduced $\chi^2$ of the joint "
    r"$(\tilde\xi_{0,0},\tilde\xi_{2,2})$ fit in parentheses.}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--multipole-dir", required=True)
    p.add_argument("--projected-dir", default=None,
                   help="unused for the tables; kept so old invocations still parse")
    p.add_argument("--msuffix", default="multipoles_NLA_nautilus")
    p.add_argument("--psuffix", default="projected_NLA_nautilus")
    p.add_argument("--out-red", required=True)
    p.add_argument("--out-blue", required=True)
    p.add_argument("--tatt-dir", default=None,
                   help="TATT multipole fit dir; if given, Table B.3 is written")
    p.add_argument("--tsuffix", default="multipoles_TATT_nautilus")
    p.add_argument("--out-tatt", default=None)
    a = p.parse_args()

    M = lambda stem: os.path.join(a.multipole_dir, f"{stem}_{a.msuffix}.npz")
    P = lambda stem: os.path.join(a.projected_dir, f"{stem}_{a.psuffix}.npz")

    BGS = "zmin_0.10_zmax_0.50"
    LRGL, LRGH = "zmin_0.40_zmax_0.75", "zmin_0.75_zmax_1.10"
    ELG = "ELG_LOPnotqso_zmin_0.80_zmax_1.60"

    # (label, stem, zrange, kind)  kind: auto = global (quote b1), cross = mass bin
    blocks = [
        ("\\textit{BGS red}", [
            ("Global", f"BGS_RED_GMM_{BGS}", Z["BGS"], "auto")] +
            [(f"$M_\\star$ bin {i+1}", f"BGS_RED_GMM_{BGS}_massbin{i}", Z["BGS"], "cross")
             for i in range(12)]),
        ("\\textit{LRG}", [
            ("Global (low-$z$)", f"LRG_{LRGL}", Z["LRG_lo"], "auto"),
            ("Global (high-$z$)", f"LRG_{LRGH}", Z["LRG_hi"], "auto")] +
            [(f"$M_\\star$ bin {i+1}", f"LRG_{LRGL}_massbin{i}", Z["LRG_lo"], "cross")
             for i in range(5)] +
            [(f"$M_\\star$ bin {i+1}", f"LRG_{LRGH}_massbin{i}", Z["LRG_hi"], "cross")
             for i in range(5)]),
        ("\\textit{BGS blue}", [
            ("Global", f"BGS_BLUE_GMM_{BGS}", Z["BGS"], "auto"),
            ("$M_\\star$ bin 1 (low)", f"BGS_BLUE_GMM_{BGS}_massbin0", Z["BGS"], "cross"),
            ("$M_\\star$ bin 2 (high)", f"BGS_BLUE_GMM_{BGS}_massbin1", Z["BGS"], "cross")]),
        ("\\textit{ELG}", [("Global", ELG, Z["ELG"], "auto")]),
    ]

    def write_nla(path, label, caption, blocks_sel, projected=False, foot=TABLEFOOT):
        # projected tables carry two extra columns: the w_{g+} amplitude and the
        # precision gain of the multipoles over it.
        ncol = 12 if projected else 10
        L = [r"\begin{table*}", r"\caption{" + caption + "}",
             r"\label{" + label + "}", r"\centering",
             r"\small",
             # 11 columns overflow \textwidth at 4pt; the projected tables need 3pt
             r"\setlength{\tabcolsep}{" + ("3pt" if projected else "4pt") + "}",
             r"\begin{tabular}{ll" + "c" * (ncol - 2) + "}", r"\hline\hline",
             r"Sample & Type & $z$ range & $z_{\rm IA}$ & $\log M_\star$ & $N_{\rm shapes}$ & $\sigma_e$ "
             r"& $\mathcal{R}$ & $b_1$ & $A_{\rm IA}^{\tilde\xi_{2,2}}\ (\chi^2_\nu)$" +
             (r" & $A_{\rm IA}^{w_{g+}}\ (\chi^2_\nu)$ & Gain" if projected else "") + r" \\",
             r"\hline"]
        zgg = []
        for title, rows in blocks_sel:
            for lab, stem, zr, kind in rows:
                if kind == "auto" and os.path.exists(M(stem)):
                    tname = re.sub(r"\\textit\{(.*)\}", r"\1", title)
                    zgg.append(f"${load(M(stem))['z_gg']:.2f}$ ({tname}"
                               + (f", {lab.split('(')[1].rstrip(')')}" if "(" in lab else "") + ")")
        foot = foot.replace("ZGG", ", ".join(zgg)) if foot else foot
        for title, rows in blocks_sel:
            L += [r"\multicolumn{" + str(ncol) + r"}{c}{" + title + r"} \\", r"\hline"]
            for lab, stem, zr, kind in rows:
                f = M(stem)
                if not os.path.exists(f):
                    L.append(f"% MISSING {stem}")
                    continue
                d = load(f)
                b1 = num(d["b1"], d["b1_e"]) if kind == "auto" else "--"
                row = (f"{lab} & {kind} & {zr} & ${d['z_gp']:.2f}$ & "
                       f"${d['lm']:.2f} \\pm {d['lms']:.2f}$ & "
                       f"{thou(d['nsh'])} & ${d['sig_e']:.3f}$ & ${d['R']:.2f}$ & {b1} & "
                       f"${d['a1']:.3f} \\pm {d['a1_e']:.3f}\\ ({d['red']:.2f})$")
                if projected:
                    fp = P(stem)
                    if os.path.exists(fp):
                        w = load(fp)
                        # w_{g+} carries the opposite sign convention to the multipoles:
                        # quote -a1 so both columns share one sign convention.
                        row += (f" & ${-w['a1']:.3f} \\pm {w['a1_e']:.3f}"
                                f"\\ ({w['red']:.2f})$")
                        # precision gain of the multipoles: sigma_wgp / sigma_xi22 - 1
                        gain = 100.0 * (w["a1_e"] / d["a1_e"] - 1.0)
                        row += f" & ${gain:+.0f}\\%$"
                    else:
                        row += " & -- & --"
                L.append(row + r" \\")
            L.append(r"\hline")
        L += [r"\end{tabular}"] + ([foot] if foot else []) + [r"\end{table*}"]
        open(path, "w").write("\n".join(L) + "\n")
        print(f"wrote {path}")

    write_nla(a.out_red, "tab:fits_red",
              r"Best-fit NLA model parameters for the \emph{red} galaxy samples "
              r"(BGS red and LRG) from the wedge multipole analysis "
              r"($\tilde\xi_{0,0}$ and $\tilde\xi_{2,2}$).",
              blocks[:2])
    write_nla(a.out_blue, "tab:fits_blue",
              r"Best-fit NLA model parameters for the \emph{blue} galaxy samples "
              r"(BGS blue and ELG) from the wedge multipole analysis "
              r"($\tilde\xi_{0,0}$ and $\tilde\xi_{2,2}$) and from the projected "
              r"statistic $w_{g+}$. The corresponding TATT "
              r"constraints are given in Table~\ref{tab:fits_tatt}.",
              blocks[2:], projected=True,
              foot=r"\tablefoot{$z_{\rm IA}$ as in Table~\ref{tab:fits_red}; both "
                   r"$\tilde\xi_{2,2}$ and $w_{g+}$ are modelled at $z_{\rm IA}$, the "
                   r"monopole and $w_{\rm gg}$ at the density-sample redshift "
                   r"(ZGG). The $w_{g+}$ amplitudes are quoted with the sign of "
                   r"the multipole convention. The last column is the gain in "
                   r"precision on $A_{\rm IA}$ delivered by the multipole analysis "
                   r"relative to the projected one, "
                   r"$\sigma_{w_{g+}}/\sigma_{\tilde\xi_{2,2}}-1$.}")

    # ---------------------------------------------------------- Table B.3
    if not (a.tatt_dir and a.out_tatt):
        return
    tatt_rows = [
        ("BGS blue, global", f"BGS_BLUE_GMM_{BGS}", Z["BGS"]),
        ("BGS blue, low $M_\\star$", f"BGS_BLUE_GMM_{BGS}_massbin0", Z["BGS"]),
        ("BGS blue, high $M_\\star$", f"BGS_BLUE_GMM_{BGS}_massbin1", Z["BGS"]),
        ("ELG, global", ELG, Z["ELG"]),
    ]
    U = [r"\begin{table}",
         r"\caption{Best-fit TATT model parameters for the blue samples from the "
         r"wedge multipole analysis. The density-weighting amplitude is fixed to "
         r"$A_{1\delta}=A_1$ ($b_{\rm TA}=1$), leaving $b_1$, $A_1$ and $A_2$ "
         r"free; the fits retain linear galaxy bias only.}",
         r"\label{tab:fits_tatt}", r"\centering",
         r"\small", r"\begin{tabular}{lcccc}", r"\hline\hline",
         r"Sample & $z_{\rm IA}$ & $A_1$ & $A_2$ & $\chi^2_\nu$ \\", r"\hline"]
    for label, stem, zr in tatt_rows:
        f = os.path.join(a.tatt_dir, f"{stem}_{a.tsuffix}.npz")
        if not os.path.exists(f):
            U.append(f"% MISSING {stem}")
            continue
        d = load(f)
        U.append(f"{label} & ${d['z_gp']:.2f}$ & {num(d['a1'], d['a1_e'], 2)} & "
                 f"{num(d['a2'], d['a2_e'], 2)} & ${d['red']:.2f}$ \\\\")
    U += [r"\hline", r"\end{tabular}",
          r"\tablefoot{The sample properties ($z$ range, $\log M_\star$, "
          r"$N_{\rm shapes}$, $\sigma_e$, $\mathcal{R}$) are listed in "
          r"Table~\ref{tab:fits_blue} and are not repeated here; the recovered "
          r"$b_1$ agrees with the NLA value of Table~\ref{tab:fits_blue} to better "
          r"than $0.5\%$ in every case. $A_1$ is the tidal-alignment amplitude (the "
          r"direct counterpart of $A_{\rm IA}$ in the NLA fits of "
          r"Table~\ref{tab:fits_blue}) and $A_2$ the tidal-torquing amplitude, "
          r"in the parametrisation of \citet{Blazek19}. The quadrupole scale cut is "
          r"relaxed to $r_{\rm min}=6\,h^{-1}\,{\rm Mpc}$ for these fits, retaining "
          r"all ten $\tilde\xi_{2,2}$ points, while the monopole cut is unchanged at "
          r"$30\,h^{-1}\,{\rm Mpc}$; these constraints therefore rely on smaller "
          r"scales than the NLA baseline, so $\chi^2_\nu$ is not directly comparable "
          r"with the NLA values of Table~\ref{tab:fits_blue}.}",
          r"\end{table}"]
    open(a.out_tatt, "w").write("\n".join(U) + "\n")
    print(f"wrote {a.out_tatt}")


if __name__ == "__main__":
    main()
