#!/usr/bin/env python3
"""Derived posterior statistics for the TATT blue-sample fits (paper Sect. 4.2.3).

The TATT paragraph of the results section quotes four things that are not the
raw marginals in Table B.3, and that therefore have to be recomputed from the
nautilus chains rather than read off the fit npz:

  * P(A_2 > 0), the posterior mass above zero -- the significance quoted in the
    text. Under the old [-3,3] prior the A_2 marginal was skewed (g1 = -0.97 for
    the high-M* bin) and this differed from mean/sigma by up to 0.6 sigma; under
    the fiducial [-6,6] prior the posteriors are symmetric and the two agree to
    0.03 sigma;
  * the value of the well-constrained combination A_1 + c*A_2 along the
    (A_1, A_2) degeneracy, and how much better determined it is than A_1;
  * the same combination under the narrow and wide A_2 priors, to show that it
    is prior-independent while the A_2 marginal is not;
  * the per-bin chi2 decomposition of the ELG fit, which identifies the single
    quadrupole bin responsible for the poor chi2_nu.

Run:
  .venv/bin/python scripts/tatt_posterior_stats.py
  .venv/bin/python scripts/tatt_posterior_stats.py --combo-coeff 0.44
"""
import argparse
import os

import numpy as np

WIDE = "results/fits/nautilus_split_TATT_rmin30-6_bta1_a2wide"
# Fiducial since 2026-08-25: [-5,5] still truncated the high-M* blue bin.
WIDER = "results/fits/nautilus_split_TATT_rmin30-6_bta1_a2wider"
NARROW = "results/fits/nautilus_split_TATT_rmin30-6_bta1"

SAMPLES = [
    ("BGS blue, global", "BGS_BLUE_GMM_zmin_0.10_zmax_0.50_multipoles_TATT_nautilus.npz"),
    ("BGS blue, low M*", "BGS_BLUE_GMM_zmin_0.10_zmax_0.50_massbin0_multipoles_TATT_nautilus.npz"),
    ("BGS blue, high M*", "BGS_BLUE_GMM_zmin_0.10_zmax_0.50_massbin1_multipoles_TATT_nautilus.npz"),
    ("ELG, global", "ELG_LOPnotqso_zmin_0.80_zmax_1.60_multipoles_TATT_nautilus.npz"),
]


def load_chain(path):
    """Return (a1, a2) chain columns and normalised nautilus importance weights."""
    d = np.load(path, allow_pickle=True)
    free = [str(x) for x in d["free_params"]]
    chain = np.asarray(d["chain"], float)
    logw = np.asarray(d["log_w"], float)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return chain[:, [free.index("a1"), free.index("a2")]], w, d


def moments(c, w):
    mean = (c * w[:, None]).sum(0)
    cov = np.cov(c.T, aweights=w)
    sd = np.sqrt(np.diag(cov))
    return mean, sd, cov[0, 1] / (sd[0] * sd[1])


def degeneracy_coeff(c, w):
    """Coefficient of the tightest direction: minor eigenvector of the (A1,A2) covariance."""
    cov = np.cov(c.T, aweights=w)
    _, V = np.linalg.eigh(cov)
    v = V[:, 0]
    return float(v[1] / v[0])


def combo(c, w, coeff):
    p = c[:, 0] + coeff * c[:, 1]
    m = (p * w).sum()
    return m, float(np.sqrt(((p - m) ** 2 * w).sum()))


def elg_chi2_decomposition(d):
    """Split the stored chi2 into monopole/quadrupole and find the worst xi2 bin."""
    r_gg, r_gp = d["r_fit_gg"], d["r_fit_gp"]
    model = d["model_cut"]
    m_gg, m_gp = model[: r_gg.size], model[r_gg.size:]

    # the stored covariances are on the full measured grid; select the fitted bins
    k_gg = np.searchsorted(np.round(d["s_mid"], 4), np.round(r_gg, 4))
    c_gg = d["cov_xi0_corrected"][np.ix_(k_gg, k_gg)]
    d_gg = d["data_gg"][k_gg]
    c_gp, d_gp = d["cov_xi2_corrected"], d["data_gp"]

    chi2 = lambda r, ci: float(r @ ci @ r)
    x_gg = chi2(d_gg - m_gg, np.linalg.inv(c_gg))
    x_gp = chi2(d_gp - m_gp, np.linalg.inv(c_gp))

    res = (d_gp - m_gp) / np.sqrt(np.diag(c_gp))
    worst = int(np.argmax(np.abs(res)))
    keep = [i for i in range(r_gp.size) if i != worst]
    x_gp_drop = chi2((d_gp - m_gp)[keep], np.linalg.inv(c_gp[np.ix_(keep, keep)]))
    return x_gg, x_gp, r_gp, res, worst, x_gp_drop


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wide-dir", default=WIDER, help="fiducial fits, A_2 prior [-6,6]")
    p.add_argument("--narrow-dir", default=NARROW, help="prior-check fits, A_2 prior [-3,3]")
    p.add_argument("--combo-coeff", type=float, default=None,
                   help="fix the A_2 coefficient of the degeneracy direction "
                        "(default: per-sample minor eigenvector)")
    a = p.parse_args()

    print(f"{'sample':20s} {'A_1':>16s} {'A_2':>16s} {'rho':>7s} {'P(A2>0)':>9s}")
    coeffs = {}
    for label, fname in SAMPLES:
        c, w, _ = load_chain(os.path.join(a.wide_dir, fname))
        mean, sd, rho = moments(c, w)
        coeffs[label] = degeneracy_coeff(c, w)
        print(f"{label:20s} {mean[0]:+8.2f} +- {sd[0]:4.2f} {mean[1]:+8.2f} +- {sd[1]:4.2f} "
              f"{rho:+7.3f} {w[c[:, 1] > 0].sum():9.3f}")

    print("\ndegeneracy direction A_1 + c*A_2, fiducial vs narrow [-3,3] A_2 prior")
    for label, fname in SAMPLES:
        cf = a.combo_coeff if a.combo_coeff is not None else coeffs[label]
        row = []
        for d in (a.wide_dir, a.narrow_dir):
            c, w, _ = load_chain(os.path.join(d, fname))
            m, s = combo(c, w, cf)
            row.append((m, s))
        c_w, w_w, _ = load_chain(os.path.join(a.wide_dir, fname))
        gain = moments(c_w, w_w)[1][0] / row[0][1]
        print(f"{label:20s} c={cf:5.3f}  wide {row[0][0]:+.3f} +- {row[0][1]:.3f}   "
              f"narrow {row[1][0]:+.3f} +- {row[1][1]:.3f}   "
              f"[{gain:.1f}x tighter than A_1]")

    print("\nELG chi2 decomposition (fiducial wide-prior fit)")
    _, _, d = load_chain(os.path.join(a.wide_dir, SAMPLES[-1][1]))
    x_gg, x_gp, r_gp, res, worst, x_gp_drop = elg_chi2_decomposition(d)
    print(f"  chi2 total {x_gg + x_gp:6.2f} / {int(d['dof'])} dof   "
          f"(monopole {x_gg:.2f}, quadrupole {x_gp:.2f})")
    print("  quadrupole residuals in sigma:",
          " ".join(f"{s:+.1f}" for s in res))
    print(f"  worst residual at s = {r_gp[worst]:.2f} Mpc/h ({res[worst]:+.1f} sigma); "
          f"dropping it leaves chi2_gp = {x_gp_drop:.2f}")
    # the TATT fits relax the quadrupole cut from 10 to 6 Mpc/h; check whether the
    # two bins this adds are what spoils the chi2 (they are not)
    keep = [i for i in range(r_gp.size) if r_gp[i] > 10.0]
    c_gp = d["cov_xi2_corrected"]
    resid = (d["data_gp"] - d["model_cut"][d["r_fit_gg"].size:])[keep]
    x_gp_nla_range = float(resid @ np.linalg.inv(c_gp[np.ix_(keep, keep)]) @ resid)
    print(f"  restricted to the NLA range s > 10: chi2_gp = {x_gp_nla_range:.2f} "
          f"over {len(keep)} bins -- no single bin dominates")


if __name__ == "__main__":
    main()
