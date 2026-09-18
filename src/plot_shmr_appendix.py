#!/usr/bin/env python3
"""Paper figure: CSMF central SHMR vs Dvornik+23 and the ZM15 Fig. 11 red samples.

Our fiducial minuit fit (bgs_snr_dr6_rmin05_fhcap_g1prior_z5) marginalised
central relation with its 68% band, the Dvornik+23 fiducial SHMR, and the
red/early-type halo-to-stellar-mass constraints collected in Zu & Mandelbaum
(2015, Fig. 11): the ZM15 iHOD fitting formula (their eq. 38, <log Mh | M*>),
the Velander+14 CFHTLenS red points and the Mandelbaum+06 SDSS early-type
points. NOTE: the literature curve/points are mean halo mass at fixed stellar
mass, not the inverse SHMR — the two differ at the high-mass end (ZM15 Sec 6).

Units: M* in h^-2 Msun, Mh in h^-1 Msun everywhere.
  Velander+14 quote h70 units:  M h70^-1 -> x0.7 h^-1 ; M h70^-2 -> x0.49 h^-2.
  Mandelbaum+06 M* are physical (h=0.7): x0.49 to h^-2; Mh already h^-1;
  their 95% CL errors are halved to ~1sigma.

Run from the NRV venv:
  /home/rpaviot/NRV_HOD/.venv_hod/bin/python plot_for_paper/plot_shmr_appendix.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggl_utils as gu  # noqa: E402
import paper_style  # noqa: E402
from plot_csmf_shmr_vs_dvornik import (  # noqa: E402
    DVORNIK, DVORNIK_ERR, load_curve, mstar_central)

paper_style.apply_style()

# Fiducial (2026-09-17): M200m halo masses, Tinker10 HMF + bias, n_gal over the
# DESI effective area (--label overrides; the pre-09-17 paper fit was
# bgs_snr_dr6_rmin05_fhcap_g1prior_z5, 200c / Tinker08 / geometric area).
LABEL = "bgs_snr_dr6_rmin05_fhcap_g1prior_z5eff_t10_m200m_p2"
METHOD = "minuit"

# PIP/IIP companion fit: identical configuration (minuit, rp_min 0.5, f_h<=1 cap,
# Gaussian gamma1 prior, n_gal anchor, z5 window, 10 VLIM_SNR bins); the only
# difference is the DESI fibre-collision treatment of the lenses and randoms.
# Overlaid with --pip PATH (the fit is not part of the release: the PIP/IIP
# appendix was reduced to a qualitative statement, 2026-08).

# --- Zu & Mandelbaum 2015 eq. 38 (comment coefficients in their source) ------
ZM15 = dict(a=1.821, b=11.177, c=4.413, d=11.115, e=23.366, f=-0.121)


def zm15_lgmh(lgms):
    p = ZM15
    return (p["c"] / (1.0 + np.exp(-p["a"] * (lgms - p["b"])))
            + p["d"] * np.sin(p["f"] * (lgms - p["e"])))


# --- Velander+14 Table (stellar mass bins, red lenses, S3-S7) ----------------
# M* [1e10 h70^-2], Mh [1e11 h70^-1] with asymmetric 1sigma errors.
VEL_MS = np.array([1.97, 5.64, 13.0, 22.6, 38.6])
VEL_MH = np.array([5.81, 26.3, 81.2, 160.0, 388.0])
VEL_MH_LO = np.array([1.20, 2.88, 8.91, 24.2, 67.1])
VEL_MH_HI = np.array([1.67, 3.23, 12.1, 28.3, 90.7])

# --- Mandelbaum+06 Table 3 (early types) -------------------------------------
# M* [1e10 Msun, h=0.7], Mcent [1e11 h^-1 Msun], 95% CL errors.
M06_MS = np.array([3.0, 5.8, 11.2, 21.3, 39.6])
M06_MH = np.array([4.9, 14.1, 34.0, 158.0, 716.0])
M06_MH_LO = np.array([3.2, 5.3, 9.0, 33.0, 190.0]) / 2.0
M06_MH_HI = np.array([4.7, 5.6, 10.0, 37.0, 123.0]) / 2.0


def asym_log_err(m, lo, hi):
    return np.array([np.log10(m / (m - lo)), np.log10((m + hi) / m)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pip", default=None, metavar="NPZ",
                    help="overlay the inverse-probability-weighted (PIP/IIP) companion fit")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=LABEL, help="CSMF minuit fit label")
    ap.add_argument("--fit-npz", default=None,
                    help="explicit path to the fit npz (default: "
                         "<dest>/ggl/csmf_fit/csmf_fit_<label>_minuit.npz)")
    ap.add_argument("--config", default=str(gu.CONFIG))
    ap.add_argument("--method", default=METHOD, choices=["minuit", "nautilus", "de"],
                    help="minuit / de: band from the parameter covariance; nautilus: band "
                         "from the posterior samples")
    args = ap.parse_args()
    method = args.method

    cfg = gu.load_config(args.config)
    dest = Path(cfg["dest"])
    rng = np.random.default_rng(42)
    logMh = np.linspace(11.0, 15.0, 300)
    n_mc = 4000

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # This work
    fit_npz = (Path(args.fit_npz) if args.fit_npz
               else dest / "ggl" / "csmf_fit" / f"csmf_fit_{args.label}_{method}.npz")
    y, lo, hi, pb = load_curve(fit_npz, method, logMh, n_mc, rng)
    ax.fill_betweenx(logMh, lo, hi, color="C0", alpha=0.25, lw=0)
    ax.plot(y, logMh, "C0", lw=1.8,
            label="This work (BGS)" + (", close-pair" if args.pip else ""))

    if args.pip:
        yq, loq, hiq, _ = load_curve(Path(args.pip), method, logMh, n_mc,
                                     np.random.default_rng(43))
        ax.fill_betweenx(logMh, loq, hiq, color="C4", alpha=0.25, lw=0)
        ax.plot(yq, logMh, color="C4", ls=":", lw=1.8,
                label="This work (BGS), inv.-prob.")

    # Dvornik+23 fiducial with MC band
    yd = mstar_central(logMh, DVORNIK)
    keys = ["M0", "M1", "gamma1", "gamma2"]
    draws = rng.normal(0, 1, size=(n_mc, 4))
    curves = np.empty((n_mc, logMh.size))
    for i in range(n_mc):
        pi = {k: DVORNIK[k] + draws[i, j] * DVORNIK_ERR[k]
              for j, k in enumerate(keys)}
        curves[i] = mstar_central(logMh, pi)
    dlo, dhi = np.nanpercentile(curves, [16, 84], axis=0)
    ax.fill_betweenx(logMh, dlo, dhi, color="0.3", alpha=0.15, lw=0)
    ax.plot(yd, logMh, "k--", lw=1.5, label="Dvornik et al. (2023)")

    # ZM15 iHOD <log Mh | M*>
    lgms = np.linspace(9.8, 11.8, 200)
    ax.plot(lgms, zm15_lgmh(lgms), color="C2", ls="-.", lw=1.5,
            label="Zu & Mandelbaum (2015)")

    # Velander+14 red (h70 -> h units)
    v_ms = 10.0 + np.log10(VEL_MS) + 2.0 * np.log10(0.7)
    v_mh = 11.0 + np.log10(VEL_MH) + np.log10(0.7)
    v_err = asym_log_err(VEL_MH, VEL_MH_LO, VEL_MH_HI)
    ax.errorbar(v_ms, v_mh, yerr=v_err, fmt="s", ms=3.5, color="C3", lw=0.9,
                capsize=1.5, label="Velander et al. (2014) red")

    # Mandelbaum+06 early types
    m_ms = 10.0 + np.log10(M06_MS) + 2.0 * np.log10(0.7)
    m_mh = 11.0 + np.log10(M06_MH)
    m_err = asym_log_err(M06_MH, M06_MH_LO, M06_MH_HI)
    ax.errorbar(m_ms, m_mh, yerr=m_err, fmt="^", ms=3.5, color="C1", lw=0.9,
                capsize=1.5, mfc="none",
                label="Mandelbaum et al. (2006) early")

    ax.set_xlabel(r"$\log_{10}(M_\ast\,/\,h^{-2}M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(M_{\rm h}\,/\,h^{-1}M_\odot)$")
    ax.set_xlim(9.9, 11.7)
    ax.set_ylim(11.2, 15.0)
    ax.legend(loc="upper left", frameon=False, fontsize=7 * paper_style.FONTSCALE)

    out = Path(args.out) if args.out else (
        REPO / "plot_for_paper" /
        ("figB_shmr_literature_pip.png" if args.pip else "figB_shmr_literature.png"))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")
    print({k: pb[k] for k in ("M0", "M1", "gamma1", "gamma2")})


if __name__ == "__main__":
    main()
