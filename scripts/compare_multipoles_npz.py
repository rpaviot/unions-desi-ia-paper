#!/usr/bin/env python3
"""Compare two <stem>_GPU_multipoles.npz (e.g. compute_correlations_cucount.py output vs
the release): grids, xi0 / xi2 means, jackknife sigmas and correlation matrices.

    python scripts/compare_multipoles_npz.py NEW.npz REF.npz
"""
import sys
import numpy as np

a, b = (np.load(f, allow_pickle=True) for f in sys.argv[1:3])


def rel(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.nanmax(np.abs(x - y)) / np.nanmax(np.abs(y)))


print(f"s_bins identical: {np.allclose(a['s_bins'], b['s_bins'])}   "
      f"n_shapes {int(a['n_shapes'])} vs {int(b['n_shapes'])}   "
      f"n_clustering {int(a['n_clustering'])} vs {int(b['n_clustering'])}")
for key in ("xi0", "xi2"):
    print(f"{key:4s} mean : max rel diff = {rel(a[key], b[key]):.3e}")
    for ck in (f"cov_{key}", f"cov_{key}_corrected"):
        sa, sb = np.sqrt(np.diag(a[ck])), np.sqrt(np.diag(b[ck]))
        ra, rb = a[ck] / np.outer(sa, sa), b[ck] / np.outer(sb, sb)
        print(f"     {ck:20s} sigma: max rel diff = {rel(sa, sb):.3e}   "
              f"corr matrix: max abs diff = {np.nanmax(np.abs(ra - rb)):.3e}")
for ck in ("cov_combined", "cov_combined_corrected"):
    print(f"{ck:22s} Frobenius |A-B|/|B| = "
          f"{np.linalg.norm(a[ck] - b[ck]) / np.linalg.norm(b[ck]):.3e}")
print(f"z_eff_IA {float(a['z_eff_IA']):.6f} vs {float(b['z_eff_IA']):.6f};  "
      f"z_eff_clustering {float(a['z_eff_clustering']):.6f} vs {float(b['z_eff_clustering']):.6f}")
