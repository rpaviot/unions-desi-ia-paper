#!/usr/bin/env python3
"""Build the UNIONS x DESI position-shape (IA) samples.

Sample structure (config `ia_samples.tracers`): ONE clustering catalogue + ONE
randoms per TRACER (BGS=BGS_ANY, LRG, ELG) define the density field. The colour /
mass `shape_samples` under a tracer are matched subsets of that tracer's galaxies --
SHAPE-only catalogues that, downstream, reuse the tracer's clustering, randoms and
jackknife patch centres. Each shape sample also gets its own shape randoms.

For every tracer and redshift bin this script writes the density catalogues:

  1. redshift-cuts the tracer clustering data and randoms;
  2. footprint-masks both to the UNIONS overlap -- a galaxy is kept iff it lands in a
     healpix (nside = footprint_nside) pixel occupied by a UNIONS shape;
     ->  <tracer>_<ztag>_clustering.parquet , <tracer>_<ztag>_randoms.parquet

and, for every shape sample under that tracer (+ each LOGMSTAR bin):

  3. redshift-cuts + footprint-masks the shape sample's DESI galaxies;
  4. sky-matches each to its nearest UNIONS shape within `match_sep_arcsec`
     (astropy match_coordinates_sky);
  5. calibrates the matched ellipticities with the sample's weighted mean metacal
     responsivity,  e = e_uncal / <w (R_g11 + R_g22) / 2>;
  6. splits into LOGMSTAR bins where `n_mass_bins` is set (mass_floor cut, outer
     edges at the `mass_pct` percentiles), each bin recalibrated on its own
     galaxies. Default `bin_scheme: equal_n` is equipopulated; `equal_snr` targets
     equal expected A_IA significance per bin instead (see stellar_mass_bins_snr):
     equal-N below `snr_logm_break`, thinning as 10^(-2*snr_beta_bright*dlogM)
     above it, floored at `snr_n_min`, outer edges from `mass_pct_snr`;
  7. builds shape randoms = `shape_randoms_factor` x the shape sample, with positions
     drawn from the tracer randoms (the survey footprint) and n(z)/weight resampled
     from the shapes (used or not downstream).
     ->  <shape>_<ztag>_shapes[_massbin{i}].parquet , <shape>_<ztag>_shape_randoms[_massbin{i}].parquet

Clustering randoms are never synthesised (just the DESI randoms in the overlap
footprint + z bin, capped at `randoms_factor` x data) except for BGS, where they are
reproduced following DESI methodology after the WEIGHT_SYS recompute (`systematics`).

The UNIONS for_IA sample is 634M rows; to keep the cross-match cheap we read it
once, record which nside pixels it occupies (-> DESI footprint mask), then keep only
the UNIONS shapes whose pixel is occupied by some tracer galaxy (+ healpix neighbours,
>> the 1" match radius) before building the match tree. Shape samples are colour
subsets of their tracer, so the tracer pixels already cover them.

Usage
-----
    python scripts/build_ia_samples.py [--config CONFIG] [--tracers NAME ...]
                                       [--head-unions N] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent

# UNIONS for_IA columns we read (positions in f8, shapes/weights in f4).
UNIONS_COLS = ["RA", "DEC", "WEIGHT", "e1_uncal", "e2_uncal", "R_g11", "R_g22"]
# DESI columns carried onto the matched shape catalogue (intersected with what's there).
DESI_KEEP = ["TARGETID", "RA", "DEC", "Z", "WEIGHT", "LOGMSTAR", "PHOTSYS", "CAP", "NTILE"]
RANDOM_KEEP = ["RA", "DEC", "Z", "WEIGHT", "WEIGHT_FKP", "CAP", "NTILE", "FRAC_TLOBS_TILES"]
# FKP weight carried onto the density catalogues (clustering data + randoms) so the
# correlations can optionally apply WEIGHT*WEIGHT_FKP on the density side (compute_
# correlations.py --use-fkp). Density-only: it is NOT propagated to the shape samples.
# For BGS the reproduced randoms recompute WEIGHT_FKP = 1/(1+NX*P0) themselves; here it
# is the raw DESI column carried onto the clustering data and the LRG/ELG randoms.
CLUSTERING_EXTRA = ["WEIGHT_FKP"]
# DESI clustering-weight components, carried onto the shape catalogue so the shape
# randoms can be rebuilt with the DESI per-NTILE completeness reconstruction (the same
# machinery as the clustering randoms) -- see build_shape_randoms. Intersected with
# what each catalogue carries; WEIGHT_SYS is recomputed in place for BGS.
SHAPE_DATA_EXTRA = ["WEIGHT_COMP", "WEIGHT_ZFAIL", "WEIGHT_SYS", "NX", "FRAC_TLOBS_TILES"]
# Extra columns needed only for the imaging-systematics recompute + DESI random
# reproduction (BGS); intersected with what each catalogue actually carries.
SYS_DATA_EXTRA = ["WEIGHT_COMP", "WEIGHT_ZFAIL", "WEIGHT_SYS", "NX", "FRAC_TLOBS_TILES"]
SYS_RANDOM_KEEP = ["RA", "DEC", "Z", "WEIGHT", "WEIGHT_COMP", "WEIGHT_ZFAIL", "WEIGHT_SYS",
                   "NX", "NTILE", "PHOTSYS", "FRAC_TLOBS_TILES", "CAP"]


def ang2pix(ra, dec, nside):
    """RA/Dec (deg) -> nested healpix pixel index."""
    import healpy as hp
    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))
    return hp.ang2pix(nside, theta, phi, nest=True)


def calibrate(df, weight_col="WEIGHT"):
    """Add calibrated e1/e2 = e_uncal / R_mean, with R_mean the weighted mean of
    (R_g11 + R_g22)/2 over finite-shape, positive-weight objects. Drops non-finite."""
    e1, e2 = df["e1_uncal"].to_numpy(), df["e2_uncal"].to_numpy()
    r11, r22 = df["R_g11"].to_numpy(), df["R_g22"].to_numpy()
    w = df[weight_col].to_numpy()
    good = (np.isfinite(e1) & np.isfinite(e2) & np.isfinite(r11)
            & np.isfinite(r22) & np.isfinite(w) & (w > 0))
    r_i = (r11 + r22) / 2.0
    r_mean = np.sum(w[good] * r_i[good]) / np.sum(w[good])
    out = df.copy()
    out["e1"] = np.where(good, e1 / r_mean, np.nan)
    out["e2"] = np.where(good, e2 / r_mean, np.nan)
    out = out[np.isfinite(out["e1"]) & np.isfinite(out["e2"])].reset_index(drop=True)
    return out, r_mean


def write_shapes_parquet(df, path, r_mean):
    """Write a calibrated shape catalogue with its responsivity r_mean recorded in
    the parquet schema metadata (key b"r_mean"), so downstream code can recover the
    per-sample calibration without recomputing it."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"r_mean": str(r_mean).encode()})
    pq.write_table(table, path)


def stellar_mass_bins(df, n_bins, col, floor, pct):
    """Equipopulated LOGMSTAR bins: drop col<=floor, outer edges at pct percentiles,
    equipopulated edges in between. Returns list of (bin_df, (lo, hi))."""
    d = df[df[col] > floor]
    d = d[d[col].notna()]
    m = d[col].to_numpy()
    p_lo, p_hi = np.percentile(m, pct[0]), np.percentile(m, pct[1])
    inner = np.percentile(m[(m >= p_lo) & (m <= p_hi)], np.linspace(0, 100, n_bins + 1))
    edges = np.unique(np.concatenate([[p_lo], inner[1:-1], [p_hi]]))
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (d[col] >= lo) & (d[col] <= hi) if i == len(edges) - 2 else \
              (d[col] >= lo) & (d[col] < hi)
        out.append((d[sel].reset_index(drop=True), (lo, hi)))
    return out


def stellar_mass_bins_snr(df, n_bins, col, floor, pct, logm_break, beta_faint,
                          beta_bright, n_min):
    """Equal-expected-S/N LOGMSTAR bins (bin_scheme: equal_snr).

    Per-bin error on A_IA ~ 1/sqrt(N) while the signal follows a two-regime power
    law A_IA ~ 10^(beta*(logM-break)) (flat faint regime + steep bright tail,
    Fortuna+21), so equal expected significance means a target count per bin
    N_t = C * 10^(-2*beta(logM_med)*(logM_med - break)),  clamped at n_min.
    With beta_faint = 0 this reduces exactly to equal-N below the break. Greedy
    sweep from the faint edge (bin median via a fixed-point pass); C is
    binary-searched so the sweep yields exactly n_bins boxes. Binning is a design
    choice: a mis-set break/beta costs S/N uniformity across bins, never bias.
    Returns the same [(bin_df, (lo, hi)), ...] as stellar_mass_bins."""
    d = df[(df[col] > floor) & df[col].notna()]
    m = np.sort(d[col].to_numpy())
    p_lo, p_hi = np.percentile(m, pct[0]), np.percentile(m, pct[1])
    m = m[(m >= p_lo) & (m <= p_hi)]

    def target(C, logm_med):
        beta = beta_faint if logm_med < logm_break else beta_bright
        return max(int(C * 10.0 ** (-2.0 * beta * (logm_med - logm_break))), int(n_min))

    def build(C):
        edges, i = [p_lo], 0
        while i < len(m):
            nt = target(C, m[i])
            for _ in range(2):                       # fixed point on the bin median
                nt = target(C, m[min(i + nt // 2, len(m) - 1)])
            j = i + nt
            if len(m) - j < n_min:                   # remainder too small -> top box
                break
            edges.append(float(m[j]))
            i = j
        edges.append(p_hi)
        return edges

    # bins(C) is non-increasing in C; take the LARGEST C that still yields n_bins so
    # the excess lands in the faint equal-N bins, not as a bloated top box.
    lo, hi = int(n_min), len(m)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(build(mid)) - 1 >= n_bins:
            lo = mid + 1
        else:
            hi = mid
    edges = build(max(lo - 1, int(n_min)))
    if len(edges) - 1 != n_bins:
        print(f"      WARNING equal_snr: {len(edges)-1} bins built, {n_bins} requested "
              f"(n_min={n_min} too tight for this sample?)")
    out = []
    for i in range(len(edges) - 1):
        e_lo, e_hi = edges[i], edges[i + 1]
        sel = (d[col] >= e_lo) & (d[col] <= e_hi) if i == len(edges) - 2 else \
              (d[col] >= e_lo) & (d[col] < e_hi)
        out.append((d[sel].reset_index(drop=True), (e_lo, e_hi)))
    return out


def build_shape_randoms(shapes, tracer_randoms, factor, p0, reconstruct, rng):
    """Shape randoms for one shape sample, following DESI methodology (the same
    reconstruction as the clustering randoms, sysweights.create_clustering_randoms_desi):
    `factor` x the shape-sample size of positions are drawn from the tracer randoms (the
    survey footprint), while Z and the DESI clustering-weight components are resampled
    from the shapes and the random WEIGHT is rebuilt per-NTILE as
    WEIGHT_COMP*WEIGHT_ZFAIL*WEIGHT_SYS / <WEIGHT_COMP>_NTILE (x FRAC_TLOBS_TILES), then
    NGC/SGC-balanced. This reproduces old_codes/get_IA_samples.create_shape_randoms_shuffle
    (and fixes its NGC/SGC scaling bug). The shape sample's lens density weight
    WEIGHT_CLUSTERING plays the role of the data WEIGHT.

    Returns RA/DEC/Z/WEIGHT. `reconstruct` is sysweights.create_clustering_randoms_desi;
    P0 only enters WEIGHT_FKP (dropped here), so its value is immaterial to the output."""
    if len(shapes) == 0 or len(tracer_randoms) == 0:
        return pd.DataFrame(columns=["RA", "DEC", "Z", "WEIGHT"])
    cols = ["Z", "NTILE", "PHOTSYS", "CAP", "WEIGHT_COMP", "WEIGHT_ZFAIL", "WEIGHT_SYS",
            "NX", "FRAC_TLOBS_TILES"]
    dv = shapes[[c for c in cols if c in shapes.columns]].copy()
    dv["WEIGHT"] = shapes["WEIGHT_CLUSTERING"].to_numpy()   # lens clustering weight as the data WEIGHT
    # the per-NTILE completeness lookup needs every random NTILE present in the data;
    # sparse mass bins may miss rare NTILE values, so restrict the footprint randoms.
    r = tracer_randoms[tracer_randoms["NTILE"].isin(set(dv["NTILE"]))]
    n = int(factor * len(shapes))
    r = r.sample(n=n, replace=n > len(r), random_state=rng).reset_index(drop=True)
    sr = reconstruct(dv, r, P0=p0)
    return sr[["RA", "DEC", "Z", "WEIGHT"]].reset_index(drop=True)


def ztag(name, zlo, zhi):
    return f"{name}_zmin_{zlo:.2f}_zmax_{zhi:.2f}"


def expected_outputs(ia):
    """The full set of parquet filenames this config is meant to produce in out_dir
    (every tracer, z-bin, shape sample and mass bin). Computed from the WHOLE config
    (not the --tracers subset) so cleaning never deletes another tracer's outputs."""
    names = set()
    for t in ia["tracers"]:
        for zlo, zhi in t["zbins"]:
            tag = ztag(t["name"], zlo, zhi)
            names |= {f"{tag}_clustering.parquet", f"{tag}_randoms.parquet"}
            for s in t["shape_samples"]:
                stag = ztag(s["name"], zlo, zhi)
                names |= {f"{stag}_shapes.parquet", f"{stag}_shape_randoms.parquet"}
                for i in range(int(s["n_mass_bins"] or 0)):
                    names |= {f"{stag}_shapes_massbin{i}.parquet",
                              f"{stag}_shape_randoms_massbin{i}.parquet"}
    return names


def clean_outputs(ia, out_dir):
    """Remove orphaned IA parquet (any *.parquet in out_dir not in expected_outputs --
    e.g. the old per-colour BGS clustering/randoms or ELG_LOPnotqso_* from a previous
    layout) and clear the cached jackknife patch centres (recomputed from the rebuilt
    randoms downstream). Idempotent: a no-op once the directory matches the config."""
    keep = expected_outputs(ia)
    removed = 0
    for p in sorted(out_dir.glob("*.parquet")):
        if p.name not in keep:
            print(f"  removing orphan {p.name}", flush=True)
            p.unlink(); removed += 1
    pc = out_dir / "patch_centers"
    if pc.is_dir():
        for f in sorted(pc.glob("*.txt")):
            print(f"  removing stale patch centres {f.name}", flush=True)
            f.unlink(); removed += 1
    print(f"  --clean: removed {removed} file(s)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config" / "data_sources.yaml"))
    ap.add_argument("--tracers", nargs="+", help="Restrict to these tracer names (BGS LRG ELG).")
    ap.add_argument("--head-unions", type=int, default=0,
                    help="Read only the first N UNIONS rows (smoke test; matches are regional).")
    ap.add_argument("--clean", action="store_true",
                    help="Before building, remove orphaned IA parquet (old layouts) and the "
                         "cached patch centres. Computed from the whole config, so safe with --tracers.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None, help="output directory (default: the config location under <dest>)")
    args = ap.parse_args()

    import healpy as hp
    import pyarrow.parquet as pq
    from astropy.coordinates import SkyCoord, match_coordinates_sky
    import astropy.units as u
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sysweights as sw

    cfg = yaml.safe_load(open(args.config))
    dest = Path(cfg["dest"])
    ia = cfg["ia_samples"]
    cat_dir = dest / ia["cat_subdir"]
    out_dir = Path(args.out_dir) if args.out_dir else dest / ia["out_subdir"]
    nside = int(ia["footprint_nside"])
    sep = float(ia["match_sep_arcsec"]) * u.arcsec
    mass_col = ia["mass_column"]
    floor = float(ia["mass_floor"])
    pct = ia["mass_pct"]
    snr_kw = dict(pct=ia.get("mass_pct_snr", pct),
                  logm_break=float(ia.get("snr_logm_break", 11.0)),
                  beta_faint=float(ia.get("snr_beta_faint", 0.0)),
                  beta_bright=float(ia.get("snr_beta_bright", 0.5)),
                  n_min=int(ia.get("snr_n_min", 30000)))
    rfac = int(ia["randoms_factor"])
    srfac = int(ia.get("shape_randoms_factor", ia["randoms_factor"]))
    tracers = [t for t in ia["tracers"] if not args.tracers or t["name"] in args.tracers]
    rng = np.random.default_rng(42)

    # Imaging-systematics recompute (per-tracer `systematics: true`, BGS only). The
    # regression is fit ONCE on the tracer clustering catalogue vs its randoms over the
    # FULL DESI footprint, then WEIGHT_SYS/WEIGHT_CORR are applied to the masked
    # clustering, the masked randoms (reproduced) and every shape sample matched from it.
    sysc = ia.get("systematics") or {}
    any_sys = any(t.get("systematics") for t in tracers)
    sys_kw = dict(n_bins=int(sysc.get("n_bins", 20)),
                  bin_method=sysc.get("bin_method", "percentile"),
                  alpha_range=tuple(float(a) for a in sysc.get("ridge_alpha_range", [1e-3, 1e3])),
                  n_alphas=int(sysc.get("n_alphas", 50)),
                  floor_frac=float(sysc.get("floor_frac", 0.001))) if any_sys else {}
    sys_nside = int(sysc.get("map_nside", 256))
    sys_names = sysc.get("map_names", [])
    sys_p0 = float(sysc.get("p0", 7000))
    sys_reproduce = bool(sysc.get("reproduce_randoms", False))

    def sys_map_files():
        d = dest / sysc["map_dir"]
        tpl = sysc["map_template"]
        return str(d / tpl.format(cap="N")), str(d / tpl.format(cap="S"))

    print(f"DESI catalogues : {cat_dir}")
    print(f"UNIONS for_IA   : {dest / ia['unions_for_ia']}")
    print(f"out             : {out_dir}")
    print(f"match {sep}, footprint nside={nside}, clustering randoms<= {rfac}x data, "
          f"shape randoms = {srfac}x shapes\n")
    print("Tracers:")
    for t in tracers:
        for zlo, zhi in t["zbins"]:
            ss = ", ".join(f"{s['name']}(m={s['n_mass_bins']})" for s in t["shape_samples"])
            print(f"  {ztag(t['name'], zlo, zhi)}  sys={bool(t.get('systematics'))}  shapes=[{ss}]")
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        print("Cleaning orphaned outputs + cached patch centres:", flush=True)
        clean_outputs(ia, out_dir)
    npix = hp.nside2npix(nside)

    # ---- 1. Read UNIONS for_IA, record occupied pixels -------------------------
    t0 = time.time()
    upath = dest / ia["unions_for_ia"]
    if args.head_unions:
        ut = pq.ParquetFile(upath).read_row_groups(
            range(min(args.head_unions // 1_000_000 + 1,
                      pq.ParquetFile(upath).num_row_groups)), columns=UNIONS_COLS)
        u_df = ut.to_pandas().iloc[:args.head_unions]
    else:
        u_df = pq.read_table(upath, columns=UNIONS_COLS).to_pandas()
    u_ra = u_df["RA"].to_numpy(np.float64)
    u_dec = u_df["DEC"].to_numpy(np.float64)
    u_pix = ang2pix(u_ra, u_dec, nside)
    unions_occ = np.zeros(npix, dtype=bool)
    unions_occ[u_pix] = True
    print(f"[{time.time()-t0:.0f}s] UNIONS: {len(u_df):,} shapes, "
          f"{unions_occ.sum():,} occupied pixels")

    # ---- 2. Per tracer x z-bin: build the clustering data (z-cut, sys, footprint mask)
    #         and collect occupied pixels (+ neighbours) for the UNIONS prefilter. Shape
    #         samples are colour subsets of the tracer, so the tracer pixels cover them.
    tracer_data: dict[tuple, pd.DataFrame] = {}     # (tracer, zlo, zhi) -> masked clustering
    tracer_ws: dict[tuple, tuple] = {}              # (tracer, zlo, zhi) -> (ws_n, ws_s) for sys
    sys_random_masked: dict[tuple, pd.DataFrame] = {}
    desi_pix = np.zeros(npix, dtype=bool)
    map_n, map_s = (sys_map_files() if any_sys else (None, None))
    for t in tracers:
        use_sys = bool(t.get("systematics"))
        clu_full = pd.read_parquet(cat_dir / t["clustering"])
        keep = [c for c in (DESI_KEEP + CLUSTERING_EXTRA + (SYS_DATA_EXTRA if use_sys else []))
                if c in clu_full.columns]
        for zlo, zhi in t["zbins"]:
            key = (t["name"], zlo, zhi)
            d = clu_full[(clu_full["Z"] >= zlo) & (clu_full["Z"] <= zhi)][keep].reset_index(drop=True)

            if use_sys:
                rf = pd.read_parquet(cat_dir / t["randoms"])
                if "PHOTSYS" not in rf.columns:
                    raise SystemExit(
                        f"{t['randoms']} has no PHOTSYS column -- rebuild randoms with the "
                        "PHOTSYS fix in build_catalogues.py before running systematics.")
                rcols = [c for c in SYS_RANDOM_KEEP if c in rf.columns]
                rf = rf[(rf["Z"] >= zlo) & (rf["Z"] <= zhi)][rcols].reset_index(drop=True)
                # Fit on the full-footprint z-cut clustering vs its randoms (split N/S).
                print(f"  fitting WEIGHT_SYS on {t['clustering']} ({len(d):,} gal) z=[{zlo},{zhi}]")
                ws_n, ws_s, _ = sw.fit_sysweights(d, rf, map_n, map_s, sys_names, sys_nside, **sys_kw)
                tracer_ws[key] = (ws_n, ws_s)
                d = sw.apply_sysweights(d, ws_n, ws_s, sys_nside)
                rpix = ang2pix(rf["RA"], rf["DEC"], nside)
                rmask = rf[unions_occ[rpix]].reset_index(drop=True)
                sys_random_masked[key] = sw.apply_sysweights(rmask, ws_n, ws_s, sys_nside)
                del rf

            pix = ang2pix(d["RA"], d["DEC"], nside)
            d = d[unions_occ[pix]].reset_index(drop=True)
            tracer_data[key] = d
            desi_pix[ang2pix(d["RA"], d["DEC"], nside)] = True
            print(f"  {ztag(*key)}: {len(d):,} clustering galaxies in footprint")
        del clu_full
    # dilate DESI pixels by their healpix neighbours so a UNIONS match (<=1") near a
    # pixel edge is never prefiltered away (neighbour pixels span ~nside scale >> 1").
    occ_idx = np.flatnonzero(desi_pix)
    neigh = hp.get_all_neighbours(nside, occ_idx, nest=True)  # (8, Nocc)
    desi_pix[neigh[neigh >= 0]] = True

    # ---- 3. Prefilter UNIONS to DESI-occupied pixels, build the match tree once -
    keepu = desi_pix[u_pix]
    ru = u_df[keepu].reset_index(drop=True)
    print(f"[{time.time()-t0:.0f}s] UNIONS prefiltered to DESI footprint: "
          f"{len(ru):,} shapes ({100*len(ru)/len(u_df):.2f}%)")
    del u_df, u_ra, u_dec, u_pix, desi_pix
    u_coords = SkyCoord(ra=ru["RA"].to_numpy() * u.degree,
                        dec=ru["DEC"].to_numpy() * u.degree)

    def match_shapes(gal, wcol):
        """Match DESI galaxies `gal` to nearest UNIONS shape within sep; return the
        calibrated shape catalogue. `wcol` is the lens density-weight column on `gal`
        (WEIGHT_CORR for sys tracers, else WEIGHT); the UNIONS lensing WEIGHT and
        ellipticities/responsivities then overwrite WEIGHT/e*/R_g* before calibration."""
        gc = SkyCoord(ra=gal["RA"].to_numpy() * u.degree, dec=gal["DEC"].to_numpy() * u.degree)
        idx, d2d, _ = match_coordinates_sky(gc, u_coords)
        ok = np.asarray(d2d < sep)
        m = gal[ok].reset_index(drop=True)
        us = ru.iloc[idx[ok]].reset_index(drop=True)
        m["WEIGHT_CLUSTERING"] = m[wcol].to_numpy()
        m = m.drop(columns=[c for c in ("WEIGHT", "WEIGHT_CORR") if c in m.columns])
        for c in ["WEIGHT", "e1_uncal", "e2_uncal", "R_g11", "R_g22"]:
            m[c] = us[c].to_numpy()
        shapes, r_mean = calibrate(m, weight_col="WEIGHT")
        return shapes, r_mean, int(ok.sum())

    # ---- 4. Write clustering + randoms per tracer; match + mass-bin + shape randoms
    #         per shape sample. Clustering randoms are read+footprint-cut once per
    #         (randoms file, z-bin) and reused across a tracer's shape samples.
    random_cache: dict[tuple, pd.DataFrame] = {}
    for t in tracers:
        use_sys = bool(t.get("systematics"))
        for zlo, zhi in t["zbins"]:
            key = (t["name"], zlo, zhi)
            tag = ztag(t["name"], zlo, zhi)
            d = tracer_data[key]

            # --- density catalogue (the clustering tracer) ---
            d.to_parquet(out_dir / f"{tag}_clustering.parquet", index=False)
            if use_sys:
                r = sys_random_masked[key]
                cap = rfac * len(d)
                if len(r) > cap:
                    r = r.sample(n=cap, replace=False, random_state=42).reset_index(drop=True)
                if sys_reproduce:
                    r = sw.create_clustering_randoms_desi(d, r, P0=sys_p0)
            else:
                rkey = (t["randoms"], zlo, zhi)
                if rkey not in random_cache:
                    rr = pd.read_parquet(cat_dir / t["randoms"])
                    rr = rr[(rr["Z"] >= zlo) & (rr["Z"] <= zhi)]
                    rkeep = [c for c in RANDOM_KEEP if c in rr.columns]
                    rpix = ang2pix(rr["RA"], rr["DEC"], nside)
                    random_cache[rkey] = rr[unions_occ[rpix]][rkeep].reset_index(drop=True)
                r = random_cache[rkey]
                cap = rfac * len(d)
                if len(r) > cap:
                    r = r.sample(n=cap, replace=False, random_state=42).reset_index(drop=True)
            r.to_parquet(out_dir / f"{tag}_randoms.parquet", index=False)
            print(f"  {tag}: clustering {len(d):,}, randoms {len(r):,}")

            # --- shape samples (colour / mass subsets matched to UNIONS) ---
            for s in t["shape_samples"]:
                stag = ztag(s["name"], zlo, zhi)
                sdf = pd.read_parquet(cat_dir / s["data"])
                skeep = [c for c in (DESI_KEEP + SHAPE_DATA_EXTRA) if c in sdf.columns]
                sd = sdf[(sdf["Z"] >= zlo) & (sdf["Z"] <= zhi)][skeep].reset_index(drop=True)
                if use_sys:
                    sd = sw.apply_sysweights(sd, *tracer_ws[key], sys_nside)
                spix = ang2pix(sd["RA"], sd["DEC"], nside)
                sd = sd[unions_occ[spix]].reset_index(drop=True)
                wcol = "WEIGHT_CORR" if "WEIGHT_CORR" in sd.columns else "WEIGHT"

                shapes, r_mean, n_match = match_shapes(sd, wcol)
                print(f"    {stag}: matched {n_match:,}/{len(sd):,} "
                      f"({100*n_match/max(len(sd),1):.1f}%)  R={r_mean:.4f}  -> shapes {len(shapes):,}")
                write_shapes_parquet(shapes, out_dir / f"{stag}_shapes.parquet", r_mean)
                build_shape_randoms(shapes, r, srfac, sys_p0, sw.create_clustering_randoms_desi,
                                    rng).to_parquet(out_dir / f"{stag}_shape_randoms.parquet", index=False)

                if s["n_mass_bins"]:
                    if s.get("bin_scheme", "equal_n") == "equal_snr":
                        kw = {**snr_kw, **{k[4:]: s[k] for k in
                              ("snr_logm_break", "snr_beta_faint", "snr_beta_bright", "snr_n_min")
                              if k in s}}
                        bins = stellar_mass_bins_snr(shapes, int(s["n_mass_bins"]),
                                                     mass_col, floor, **kw)
                    else:
                        bins = stellar_mass_bins(shapes, int(s["n_mass_bins"]), mass_col, floor, pct)
                    for i, (bdf, (lo, hi)) in enumerate(bins):
                        bcal, br = calibrate(bdf, weight_col="WEIGHT")
                        write_shapes_parquet(bcal, out_dir / f"{stag}_shapes_massbin{i}.parquet", br)
                        build_shape_randoms(bcal, r, srfac, sys_p0, sw.create_clustering_randoms_desi,
                                            rng).to_parquet(
                            out_dir / f"{stag}_shape_randoms_massbin{i}.parquet", index=False)
                        print(f"      massbin{i} [{lo:.2f},{hi:.2f}]: {len(bcal):,}  R={br:.4f}")
                del sdf, sd

    print(f"\n[{time.time()-t0:.0f}s] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
