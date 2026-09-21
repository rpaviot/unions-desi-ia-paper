"""Point estimates of a saved CSMF fit (fit_csmf.py npz).

A fit npz carries one ``<method>_param_names / _best_fit / _chi2 / _ndof`` key set per
optimiser (``minuit`` / ``de``: the minimum; ``nautilus``: the posterior MEDIAN, plus
``nautilus_map`` and ``nautilus_mean`` with their own chi2 keys). The downstream scripts
(decompose_csmf_chi2, plot_csmf_bestfit, csmf_meff_relation, make_csmf_table) pick the
point they evaluate the model at through :func:`load_point`; ``point`` only matters for
a nautilus fit.
"""
from __future__ import annotations

import numpy as np

METHODS = ("minuit", "nautilus", "de")
POINTS = ("median", "map", "mean")

# (values key, chi2 key) of each nautilus point estimate
_NAUTILUS_KEYS = {
    "median": ("nautilus_best_fit", "nautilus_chi2"),
    "map": ("nautilus_map", "nautilus_map_chi2"),
    "mean": ("nautilus_mean", "nautilus_mean_chi2"),
}
_LABELS = {
    "minuit": "MINUIT best fit",
    "de": "differential-evolution best fit",
    ("nautilus", "median"): "posterior median",
    ("nautilus", "map"): "MAP sample (highest posterior of the chain)",
    ("nautilus", "mean"): "posterior mean",
}


def fit_method(fit, method=None):
    """Which optimiser's keys a fit npz carries: 'minuit' / 'nautilus' / 'de'
    (auto-detected in that order when ``method`` is None)."""
    if method:
        if f"{method}_best_fit" not in fit:
            raise KeyError(f"fit has no {method}_best_fit")
        return method
    for m in METHODS:
        if f"{m}_best_fit" in fit:
            return m
    raise KeyError("fit npz has no minuit_/nautilus_/de_ best fit")


def load_point(fit, method=None, point="median"):
    """(names, values, chi2, ndof, label) of one point estimate of a saved fit.

    ``chi2`` is the fitter's objective at that point (-2 log L + 2 x prior penalty,
    the MINUIT convention) or NaN when the npz predates the key."""
    if point not in POINTS:
        raise ValueError(f"point must be one of {POINTS}, got {point!r}")
    m = fit_method(fit, method)
    names = [str(s) for s in fit[f"{m}_param_names"]]
    if m == "nautilus":
        vkey, ckey = _NAUTILUS_KEYS[point]
        label = _LABELS[(m, point)]
    else:
        vkey, ckey = f"{m}_best_fit", f"{m}_chi2"
        label = _LABELS[m]
    if vkey not in fit:
        raise KeyError(f"fit has no {vkey} (re-run fit_csmf.py to get the {point} point)")
    values = np.asarray(fit[vkey], float)
    chi2 = float(fit[ckey]) if ckey in fit else float("nan")
    ndof = int(fit[f"{m}_ndof"])
    return names, values, chi2, ndof, label


def posterior_draws(fit, n, rng):
    """``n`` parameter vectors drawn (with replacement, importance-weighted) from the
    nautilus posterior stored in a fit npz, and the parameter names."""
    names = [str(s) for s in fit["param_names"]]
    pts = np.asarray(fit["points"], float)
    logw = np.asarray(fit["log_w"], float)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return names, pts[rng.choice(pts.shape[0], size=n, p=w)]
