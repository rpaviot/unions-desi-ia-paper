#!/usr/bin/env python3
"""Imaging-systematics weights for the DESI clustering samples (BGS).

Ported from the user's old_codes/get_IA_samples.py (compute_systematics_weights
and its helpers, plus the DESI random-reproduction routine). Used by
build_ia_samples.py for tracers listed under ia_samples.systematics.enabled_for.

Why BGS only: in DR1 the official WEIGHT_SYS was derived for the bright BGS
selection only -- for the BGS samples used here it is 1, folded into the total
WEIGHT. We therefore recompute it: a linear/Ridge regression of galaxy density
contrast (delta = weighted data / weighted randoms, per nside=256 healpix pixel)
against a handful of imaging maps, fit separately for the N and S photometric
systems (PHOTSYS), AFTER the redshift cut and over the FULL DESI footprint (not
the UNIONS overlap -- the regression needs the whole-survey density field).

    weight_sys(pixel) = 1 / max(y_pred, floor)

is then assigned per object to both data and randoms, and WEIGHT_CORR = WEIGHT *
WEIGHT_SYS. Because the BGS WEIGHT carried a unit WEIGHT_SYS, this multiplication
is clean (it would double-count for LRG/ELG, which is why they keep DESI's).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV


def ang2pix_nested(ra, dec, nside):
    """RA/Dec (deg) -> nested healpix pixel index."""
    import healpy as hp
    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))
    return hp.ang2pix(nside, theta, phi, nest=True)


def load_systematics_maps(filepath, map_names):
    """Load named healpix columns from a DESI hpmaps FITS file. HI is rescaled by
    1e20 (its native units), matching the old pipeline."""
    from astropy.table import Table
    t = Table.read(filepath)
    maps = {}
    for name in map_names:
        if name not in t.colnames:
            raise KeyError(f"Map '{name}' not found in {filepath}")
        maps[name] = np.array(t[name])
    if 'HI' in maps:
        maps['HI'] = maps['HI'] / 1e20
    return maps


def compute_delta_map(gal_ra, gal_dec, gal_w, ran_ra, ran_dec, ran_w, nside, region_name):
    """Weighted density contrast per pixel: delta = (data/random) / mean, defined on
    pixels with positive random weight."""
    gal_pix = ang2pix_nested(gal_ra, gal_dec, nside)
    ran_pix = ang2pix_nested(ran_ra, ran_dec, nside)

    n_pix = 12 * nside ** 2
    gal_pix_sum = np.bincount(gal_pix, weights=gal_w, minlength=n_pix)
    ran_pix_sum = np.bincount(ran_pix, weights=ran_w, minlength=n_pix)

    mask_valid = ran_pix_sum > 0
    pix_idx = np.where(mask_valid)[0]

    obs_density = np.full(n_pix, np.nan)
    obs_density[mask_valid] = gal_pix_sum[mask_valid] / ran_pix_sum[mask_valid]

    mean_density = np.nanmean(obs_density[mask_valid])
    delta = np.full(n_pix, np.nan)
    delta[mask_valid] = obs_density[mask_valid] / mean_density

    return {'pix_idx': pix_idx, 'delta': delta, 'mean_density': mean_density}


def create_binned_data(info, maps, map_names, n_bins=20, bin_method='percentile'):
    """Bin pixels independently along each systematic and average delta per bin --
    a denoised (delta, maps) dataset for the regression, weighted by pixel count."""
    pix_idx = info['pix_idx']
    delta = info['delta'][pix_idx]

    X_pixel = []
    for name in map_names:
        vals = maps[name][pix_idx].astype(float)
        if not np.isfinite(vals).all():
            median_val = np.nanmedian(vals)
            vals = np.where(np.isfinite(vals), vals, median_val)
        X_pixel.append(vals)
    X_pixel = np.vstack(X_pixel).T

    X_binned_list, y_binned_list, bin_weights_list = [], [], []
    bin_info = {'map_names': [], 'bin_edges': [], 'bin_counts': []}

    for feat_idx, name in enumerate(map_names):
        sys_vals = X_pixel[:, feat_idx]
        if bin_method == 'percentile':
            bin_edges = np.percentile(sys_vals, np.linspace(0.5, 99.5, n_bins + 1))
        else:
            bin_edges = np.linspace(sys_vals.min(), sys_vals.max(), n_bins + 1)
        bin_edges = np.unique(bin_edges)
        actual_n_bins = len(bin_edges) - 1
        bin_indices = np.digitize(sys_vals, bin_edges[1:-1])

        for bin_idx in range(actual_n_bins):
            mask = (bin_indices == bin_idx)
            if mask.sum() == 0:
                continue
            X_binned_list.append(X_pixel[mask, :].mean(axis=0))
            y_binned_list.append(delta[mask].mean())
            bin_weights_list.append(mask.sum())

        bin_info['map_names'].append(name)
        bin_info['bin_edges'].append(bin_edges)
        bin_info['bin_counts'].append(actual_n_bins)

    return (np.array(X_binned_list), np.array(y_binned_list),
            np.array(bin_weights_list), bin_info, X_pixel)


def compute_vif(X, feature_names):
    """Variance inflation factor per feature (collinearity diagnostic)."""
    vif_data = {}
    for i in range(X.shape[1]):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        lr = LinearRegression()
        lr.fit(X_others, y)
        r2 = lr.score(X_others, y)
        vif_data[feature_names[i]] = 1 / (1 - r2) if r2 < 0.999 else np.inf
    return vif_data


def fit_ridge_regression(info, maps, map_names, nside, region_name,
                         alpha_range=(1e-3, 1e3), n_alphas=50,
                         use_ridge=True, floor_frac=0.001,
                         use_binning=True, n_bins=20, bin_method='percentile'):
    """Fit delta ~ maps (standardised, RidgeCV by default), predict per pixel, and
    return weight_sys = 1/max(pred, floor) on a full-sky healpix array."""
    pix_idx = info['pix_idx']
    delta = info['delta']

    if use_binning:
        X_binned, y_binned, bin_weights, bin_info, X_pixel = create_binned_data(
            info, maps, map_names, n_bins=n_bins, bin_method=bin_method)
        X_fit, y_fit, sample_weights = X_binned, y_binned, bin_weights
    else:
        X_list = []
        for name in map_names:
            vals = maps[name][pix_idx].astype(float)
            if not np.isfinite(vals).all():
                vals = np.where(np.isfinite(vals), vals, np.nanmedian(vals))
            X_list.append(vals)
        X_pixel = np.vstack(X_list).T
        X_fit, y_fit, sample_weights = X_pixel, delta[pix_idx], None

    X_mean = X_fit.mean(axis=0)
    X_std = X_fit.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X_fit - X_mean) / X_std

    vif_data = compute_vif(X_norm, map_names)
    max_vif = max(vif_data.values())

    if use_ridge:
        alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas)
        model = RidgeCV(alphas=alphas, fit_intercept=True, cv=5, scoring='r2')
        model.fit(X_norm, y_fit, sample_weight=sample_weights)
    else:
        model = LinearRegression(fit_intercept=True)
        model.fit(X_norm, y_fit, sample_weight=sample_weights)

    X_pixel_norm = (X_pixel - X_mean) / X_std
    y_pred = model.predict(X_pixel_norm)
    if use_binning:
        r2 = model.score(X_norm, y_fit, sample_weight=sample_weights)
    else:
        r2 = model.score(X_pixel_norm, y_fit)

    floor_val = floor_frac * np.median(y_pred)
    y_pred_floored = np.maximum(y_pred, floor_val)
    weight_sys = np.ones(12 * nside ** 2)
    weight_sys[pix_idx] = 1.0 / y_pred_floored

    diagnostics = {'r2': r2, 'vif': vif_data, 'max_vif': max_vif,
                   'n_pixels': len(pix_idx), 'method': 'Ridge' if use_ridge else 'OLS'}
    if use_ridge:
        diagnostics['alpha'] = model.alpha_
    return weight_sys, y_pred_floored, diagnostics


def fit_sysweights(data_df, randoms_df, map_file_north, map_file_south,
                   map_names, nside, *, n_bins=20, bin_method='percentile',
                   alpha_range=(1e-3, 1e3), n_alphas=50, floor_frac=0.001):
    """Fit the N and S weight_sys(pixel) healpix arrays from data + randoms (split by
    PHOTSYS). Run AFTER the z-cut, over the FULL footprint. The cheap per-object
    assignment is apply_sysweights() -- separated so the same fit can be applied to
    the footprint-masked data and to the reproduced randoms without copying the full
    (large) random catalogue. Returns (ws_north, ws_south, diagnostics)."""
    maps_north = load_systematics_maps(map_file_north, map_names)
    maps_south = load_systematics_maps(map_file_south, map_names)

    gal_ra = np.asarray(data_df['RA']); gal_dec = np.asarray(data_df['DEC'])
    gal_w = np.asarray(data_df['WEIGHT']); gal_photsys = np.asarray(data_df['PHOTSYS'])
    ran_ra = np.asarray(randoms_df['RA']); ran_dec = np.asarray(randoms_df['DEC'])
    ran_w = np.asarray(randoms_df['WEIGHT']); ran_photsys = np.asarray(randoms_df['PHOTSYS'])

    is_n = gal_photsys == 'N'; is_s = gal_photsys == 'S'
    rn = ran_photsys == 'N'; rs = ran_photsys == 'S'
    print(f"    galaxies: {len(gal_ra):,} (N:{is_n.sum():,}, S:{is_s.sum():,}); "
          f"randoms: {len(ran_ra):,} (N:{rn.sum():,}, S:{rs.sum():,})")

    info_n = compute_delta_map(gal_ra[is_n], gal_dec[is_n], gal_w[is_n],
                               ran_ra[rn], ran_dec[rn], ran_w[rn], nside, "North")
    info_s = compute_delta_map(gal_ra[is_s], gal_dec[is_s], gal_w[is_s],
                               ran_ra[rs], ran_dec[rs], ran_w[rs], nside, "South")

    ws_n, _, diag_n = fit_ridge_regression(
        info_n, maps_north, map_names, nside, "North",
        alpha_range=alpha_range, n_alphas=n_alphas, floor_frac=floor_frac,
        use_ridge=True, use_binning=True, n_bins=n_bins, bin_method=bin_method)
    ws_s, _, diag_s = fit_ridge_regression(
        info_s, maps_south, map_names, nside, "South",
        alpha_range=alpha_range, n_alphas=n_alphas, floor_frac=floor_frac,
        use_ridge=True, use_binning=True, n_bins=n_bins, bin_method=bin_method)
    print(f"    North R2={diag_n['r2']:.4f} (alpha={diag_n.get('alpha')}), "
          f"South R2={diag_s['r2']:.4f} (alpha={diag_s.get('alpha')})")
    return ws_n, ws_s, {'north': diag_n, 'south': diag_s}


def apply_sysweights(df, ws_n, ws_s, nside):
    """Assign per-object WEIGHT_SYS from the fitted N/S healpix arrays (by PHOTSYS) and
    set WEIGHT_CORR = WEIGHT * WEIGHT_SYS. Returns a copy."""
    out = df.copy()
    photsys = np.asarray(out['PHOTSYS'])
    pix = ang2pix_nested(out['RA'], out['DEC'], nside)
    ws = np.ones(len(out))
    is_n = photsys == 'N'; is_s = photsys == 'S'
    ws[is_n] = ws_n[pix[is_n]]
    ws[is_s] = ws_s[pix[is_s]]
    out['WEIGHT_SYS'] = ws
    out['WEIGHT_CORR'] = out['WEIGHT'] * out['WEIGHT_SYS']
    return out


def compute_systematics_weights(data_df, randoms_df, map_file_north, map_file_south,
                                map_names, nside, **kw):
    """Convenience wrapper: fit on (data, randoms) then assign WEIGHT_SYS/WEIGHT_CORR to
    both. build_ia_samples.py uses fit_sysweights + apply_sysweights directly so it can
    fit on the full footprint but assign only on the UNIONS-masked catalogues."""
    print("  computing systematics weights...")
    ws_n, ws_s, _ = fit_sysweights(data_df, randoms_df, map_file_north, map_file_south,
                                   map_names, nside, **kw)
    data_df = apply_sysweights(data_df, ws_n, ws_s, nside)
    randoms_df = apply_sysweights(randoms_df, ws_n, ws_s, nside)
    print(f"    WEIGHT_SYS mean: data={data_df['WEIGHT_SYS'].mean():.3f}, "
          f"randoms={randoms_df['WEIGHT_SYS'].mean():.3f}")
    return data_df, randoms_df


def galactic_cap(df):
    """NGC/SGC per row. Prefer DESI's own CAP column (authoritative -- it comes from the
    LSS file organisation, *_NGC_* / *_SGC_*); else fall back to the sign of galactic
    latitude (NGC: b > 0). This is the galactic hemisphere, distinct from PHOTSYS (N/S),
    which is the photometric system used for the imaging-systematics regression."""
    if 'CAP' in df.columns:
        return np.asarray(df['CAP'])
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    b = SkyCoord(ra=np.asarray(df['RA']) * u.deg,
                 dec=np.asarray(df['DEC']) * u.deg).galactic.b.deg
    return np.where(b > 0, 'NGC', 'SGC')


def create_clustering_randoms_desi(clustering_cat, clustering_randoms, P0=7000):
    """Rebuild the random catalogue following DESI methodology: draw (Z, weights,
    PHOTSYS, NX, NTILE) from the data, recompute NX/FKP via the per-NTILE assignment
    completeness, build WEIGHT from WEIGHT_COMP*WEIGHT_ZFAIL*WEIGHT_SYS / <WEIGHT_COMP>,
    and balance the NGC/SGC weight sums. Returns the new randoms DataFrame."""
    print(f"  reproducing clustering randoms (P0={P0})...")
    random_z = clustering_cat.sample(n=len(clustering_randoms), replace=True).copy()
    out = clustering_randoms.copy()

    out['Z'] = random_z['Z'].values
    out['WEIGHT_COMP'] = random_z['WEIGHT_COMP'].values
    out['WEIGHT_SYS'] = random_z['WEIGHT_SYS'].values
    out['WEIGHT_ZFAIL'] = random_z['WEIGHT_ZFAIL'].values
    out['PHOTSYS'] = random_z['PHOTSYS'].values
    out['NX_DATA'] = random_z['NX'].values
    out['NTILE_DATA'] = random_z['NTILE'].values

    mean_wcomp_ntile = clustering_cat.groupby('NTILE').apply(lambda g: g['WEIGHT_COMP'].mean())
    mean_Cassign_ntile = 1.0 / mean_wcomp_ntile
    Cassign_random = mean_Cassign_ntile.loc[out['NTILE']].values
    Cassign_data = mean_Cassign_ntile.loc[out['NTILE_DATA']].values

    out['NX'] = out['NX_DATA'] * (Cassign_random / Cassign_data)
    out['WEIGHT_FKP'] = 1.0 / (1.0 + out['NX'] * P0)

    mean_wcomp_for_randoms = mean_wcomp_ntile.loc[out['NTILE']].values
    w_prime = out['WEIGHT_COMP'] * out['WEIGHT_ZFAIL'] * out['WEIGHT_SYS']
    out['WEIGHT'] = w_prime / mean_wcomp_for_randoms
    if 'FRAC_TLOBS_TILES' in out.columns:
        out['WEIGHT'] *= out['FRAC_TLOBS_TILES']
    out = out.drop(['NX_DATA', 'NTILE_DATA'], axis=1)

    clustering_cat = clustering_cat.copy()
    clustering_cat['GALCAP'] = galactic_cap(clustering_cat)
    out['GALCAP'] = galactic_cap(out)
    ratios = {}
    for cap in ['NGC', 'SGC']:
        dws = clustering_cat[clustering_cat['GALCAP'] == cap]['WEIGHT'].sum()
        rws = out[out['GALCAP'] == cap]['WEIGHT'].sum()
        ratios[cap] = dws / rws if rws > 0 else 0
    target = max(ratios.values())
    for cap in ['NGC', 'SGC']:
        if ratios[cap] > 0:
            out.loc[out['GALCAP'] == cap, 'WEIGHT'] *= ratios[cap] / target
    out = out.drop(columns='GALCAP')
    print(f"    data: {len(clustering_cat):,}, randoms: {len(out):,}")
    return out
