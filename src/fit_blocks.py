"""Block layout of a fit_ia.py result npz, split-grid aware.

A fit stacks two statistics -- the density block ("gg": xi0 or wp) followed by
the IA block ("gp": xi2 or wgp). Historically both shared one separation grid,
so readers did ``r = d["r_mid"]`` and sliced ``model_full`` at ``len(r)``.

Since --corr-dir-gp the two blocks may live on DIFFERENT grids (split binning:
xi0 on 19 bins, xi2 on 10), and that slicing is wrong. Such a fit omits
``r_mid`` on purpose and writes ``r_mid_gg`` / ``r_mid_gp`` / ``n_gg`` / ``n_gp``
instead. Use these helpers rather than indexing by hand.
"""
import numpy as np


def grids(d):
    """(r_gg, r_gp): the two blocks' separations. Equal arrays for a single-grid fit."""
    if "r_mid_gg" in d:
        return np.asarray(d["r_mid_gg"], float), np.asarray(d["r_mid_gp"], float)
    r = np.asarray(d["r_mid"], float)          # pre-split-grid fit npz
    return r, r


def sizes(d):
    """(n_gg, n_gp): full-grid length of each block, i.e. how ``cov`` is laid out."""
    r_gg, r_gp = grids(d)
    return len(r_gg), len(r_gp)


def full_model(d):
    """``model_full`` split into its (gg, gp) blocks, on the full (uncut) grids."""
    n_gg, _ = sizes(d)
    m = np.asarray(d["model_full"], float)
    return m[:n_gg], m[n_gg:]


def cut_model(d):
    """``model_cut`` split into its (gg, gp) blocks, on the fitted (cut) grids."""
    k = len(np.asarray(d["r_fit_gg"]))
    m = np.asarray(d["model_cut"], float)
    return m[:k], m[k:]


def is_split(d):
    """True when the two blocks were measured on different separation grids."""
    return bool(d["split_grid"]) if "split_grid" in d else False
