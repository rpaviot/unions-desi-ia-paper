#!/usr/bin/env python3
"""Build analysis-ready DESI catalogues for the UNIONS x DESI IA analysis.

Per tracer, assemble one parquet catalogue from the raw downloads:

  clustering.dat  (positions, redshift, weights)            NGC + SGC combined
     <- full.dat  (colours / morphology, joined on TARGETID)
     <- fastspec  (stellar mass etc., joined on TARGETID; concatenated over the
                   12 nside=1 HEALPix files of the tracer's program)

For BGS_ANY two red/blue splits are additionally produced and written as separate
catalogues:
  * GMM   -- Gaussian-mixture colour bimodality in (mag_r, g-r) per redshift slice
  * sSFR  -- specific star-formation rate threshold (sSFR <= thr -> red)

Randoms are assembled separately (positions + weights only) into one parquet per
tracer (NGC + SGC, all configured random indices).

Driven entirely by config/data_sources.yaml (`catalogue_build:` block). This script
performs NO downloads; run scripts/fetch_desi.py first. It is intended to be launched
via scripts/build_catalogues.slurm, not interactively (it is memory/IO heavy).

Usage
-----
    python scripts/build_catalogues.py [--config CONFIG]
                                       [--tracer NAME [NAME ...]]
                                       [--randoms / --no-randoms]
                                       [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# FITS IO  (fitsio reads only the requested columns -- essential for the 70+ GB
# fastspec catalogues, where we want ~15 of several hundred columns).
# --------------------------------------------------------------------------- #
def read_fits_columns(path: Path, columns: list[str], ext=1) -> pd.DataFrame:
    """Read a subset of columns from a FITS binary table into a DataFrame.

    Only columns that exist and are 1-D are kept. Byte order is normalised to
    native so the result is pandas/parquet-friendly.
    """
    import fitsio

    with fitsio.FITS(str(path)) as f:
        hdu = f[ext]
        present = set(hdu.get_colnames())
        want = [c for c in columns if c in present]
        missing = [c for c in columns if c not in present]
        if missing:
            print(f"    [warn] {path.name}: missing columns {missing}")
        data = hdu.read(columns=want)

    out = {}
    for c in want:
        arr = data[c]
        if arr.ndim != 1:
            print(f"    [warn] {path.name}: skipping {c} (shape {arr.shape})")
            continue
        out[c] = np.asarray(arr).astype(arr.dtype.newbyteorder("="), copy=False)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_clustering(lss_dir: Path, tracer: str, caps: list[str],
                    columns: list[str]) -> pd.DataFrame:
    """Load + vertically stack the clustering data catalogue across caps."""
    parts = []
    for cap in caps:
        path = lss_dir / f"{tracer}_{cap}_clustering.dat.fits"
        df = read_fits_columns(path, columns)
        df["CAP"] = cap
        print(f"    {path.name}: {len(df):,} rows")
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    print(f"  clustering total: {len(out):,} rows")
    return out


def attach_colours(df: pd.DataFrame, lss_dir: Path, tracer: str,
                   columns: list[str]) -> pd.DataFrame:
    """Left-join colour/morphology columns from the full catalogue on TARGETID."""
    path = lss_dir / f"{tracer}_full.dat.fits"
    full = read_fits_columns(path, columns).drop_duplicates("TARGETID")
    print(f"    {path.name}: {len(full):,} unique targets")
    merged = df.merge(full, on="TARGETID", how="left", suffixes=("", "_full"))
    assert len(merged) == len(df), "row count changed attaching colours"
    n = merged["FLUX_R"].notna().sum() if "FLUX_R" in merged else 0
    print(f"  colours attached: {n:,}/{len(merged):,} matched")
    return merged


def load_fastspec(fastspec_dir: Path, program: str, columns: list[str],
                  ext: str, healpix: list[str]) -> pd.DataFrame:
    """Concatenate the per-HEALPix fastspec files of one program (bright/dark)."""
    parts = []
    for hp in healpix:
        path = fastspec_dir / f"fastspec-iron-main-{program}-nside1-{hp}.fits"
        if not path.exists():
            print(f"    [warn] missing fastspec file {path.name}; skipping")
            continue
        df = read_fits_columns(path, columns, ext=ext)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).drop_duplicates("TARGETID")
    print(f"  fastspec ({program}): {len(out):,} unique targets")
    return out


def merge_fastspec(df: pd.DataFrame, fastspec: pd.DataFrame) -> pd.DataFrame:
    """Left-join fastspec stellar columns on TARGETID, preserving all DESI rows."""
    merged = df.merge(fastspec, on="TARGETID", how="left", suffixes=("", "_fastspec"))
    assert len(merged) == len(df), "row count changed merging fastspec"
    if "LOGMSTAR" in merged:
        n = merged["LOGMSTAR"].notna().sum()
        print(f"  fastspec merged: {n:,}/{len(merged):,} have LOGMSTAR "
              f"({100 * n / len(merged):.1f}%)")
    return merged


# --------------------------------------------------------------------------- #
# Photometry helpers
# --------------------------------------------------------------------------- #
def flux_to_mag(flux, mw_transmission=1.0):
    """Nanomaggie flux -> AB magnitude, de-reddened by MW_TRANSMISSION."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return 22.5 - 2.5 * np.log10(flux / mw_transmission)


def add_photometry(df: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    """Add dereddened AB magnitudes MAG_<band> and the g-r colour GR_DERED.

    Extinction is applied here once (via MW_TRANSMISSION) so no downstream code has
    to recompute it -- the catalogue carries the corrected photometry directly.
    """
    for b in bands:
        fcol, tcol = f"FLUX_{b}", f"MW_TRANSMISSION_{b}"
        if fcol in df and tcol in df:
            df[f"MAG_{b}"] = flux_to_mag(df[fcol].values, df[tcol].values)
        else:
            print(f"    [warn] cannot deredden band {b} (missing {fcol}/{tcol})")
    if "MAG_G" in df and "MAG_R" in df:
        df["GR_DERED"] = df["MAG_G"] - df["MAG_R"]
    return df


# --------------------------------------------------------------------------- #
# BGS red/blue splits
# --------------------------------------------------------------------------- #
def fit_colour_sequences(z, mag_r, color, z_bins, n_components, random_state):
    """Fit a GMM in (mag_r, g-r) per redshift slice; return per-slice red/blue
    sequence centres and widths in colour."""
    from sklearn.mixture import GaussianMixture

    centers, mean_red, std_red, mean_blue, std_blue = [], [], [], [], []
    for i in range(len(z_bins) - 1):
        sel = (z >= z_bins[i]) & (z < z_bins[i + 1]) & np.isfinite(mag_r) & np.isfinite(color)
        if np.count_nonzero(sel) < 50:
            continue
        X = np.vstack([mag_r[sel], color[sel]]).T
        gmm = GaussianMixture(n_components=n_components,
                              random_state=random_state).fit(X)
        cmeans = gmm.means_[:, 1]
        b, r = int(np.argmin(cmeans)), int(np.argmax(cmeans))
        centers.append(0.5 * (z_bins[i] + z_bins[i + 1]))
        mean_blue.append(gmm.means_[b, 1]);  std_blue.append(np.sqrt(gmm.covariances_[b, 1, 1]))
        mean_red.append(gmm.means_[r, 1]);   std_red.append(np.sqrt(gmm.covariances_[r, 1, 1]))
    return (np.array(centers), np.array(mean_red), np.array(std_red),
            np.array(mean_blue), np.array(std_blue))


def gmm_colour_split(df, gmm_cfg):
    """Add is_red_GMM / is_blue_GMM boolean columns. Classify each galaxy by which
    smoothed colour sequence it is closer to, within sigma_cut."""
    from scipy.interpolate import UnivariateSpline

    mag_r = df["MAG_R"].values if "MAG_R" in df else \
        flux_to_mag(df["FLUX_R"].values, df["MW_TRANSMISSION_R"].values)
    z = df["Z"].values
    color = df["GR_DERED"].values if "GR_DERED" in df else \
        flux_to_mag(df["FLUX_G"].values, df["MW_TRANSMISSION_G"].values) - mag_r

    centers, m_red, s_red, m_blue, s_blue = fit_colour_sequences(
        z, mag_r, color, np.asarray(gmm_cfg["z_bins"], float),
        gmm_cfg["n_components"], gmm_cfg["random_state"])
    if len(centers) < 4:
        raise RuntimeError("too few populated redshift slices to fit colour sequences")

    spl = lambda y: UnivariateSpline(centers, y, s=0.0)
    mr, sr = spl(m_red), spl(s_red)
    mb, sb = spl(m_blue), spl(s_blue)

    valid = np.isfinite(z) & np.isfinite(color)
    zc, cc = z[valid], color[valid]
    sig = float(gmm_cfg["sigma_cut"])
    d_red = np.abs(cc - mr(zc)) / sr(zc)
    d_blue = np.abs(cc - mb(zc)) / sb(zc)

    red = np.zeros(len(df), bool)
    blue = np.zeros(len(df), bool)
    red[valid] = (d_red < d_blue) & (d_red <= sig)
    blue[valid] = (d_blue < d_red) & (d_blue <= sig)
    df["is_red_GMM"] = red
    df["is_blue_GMM"] = blue
    print(f"  GMM split: {red.sum():,} red, {blue.sum():,} blue, "
          f"{len(df) - red.sum() - blue.sum():,} unclassified")
    return df


def ssfr_split(df, threshold):
    """Add is_red_sSFR / is_blue_sSFR via sSFR = SFR / 10**LOGMSTAR."""
    if "SFR" not in df or "LOGMSTAR" not in df:
        print("  [warn] SFR/LOGMSTAR absent; skipping sSFR split")
        return df
    with np.errstate(invalid="ignore", divide="ignore"):
        ssfr = df["SFR"].values / np.power(10.0, df["LOGMSTAR"].values)
    valid = np.isfinite(ssfr)
    df["is_blue_sSFR"] = valid & (ssfr > threshold)
    df["is_red_sSFR"] = valid & (ssfr <= threshold)
    print(f"  sSFR split: {df['is_red_sSFR'].sum():,} red, "
          f"{df['is_blue_sSFR'].sum():,} blue (valid {valid.sum():,})")
    return df


# --------------------------------------------------------------------------- #
# Randoms
# --------------------------------------------------------------------------- #
def assemble_randoms(lss_dir: Path, out_dir: Path, tracer: str, caps: list[str],
                     indices: list[int], columns: list[str], dry: bool) -> None:
    # positions + weights, and PHOTSYS (N/S split needed to recompute imaging-systematics
    # weights on the randoms; see scripts/sysweights.py).
    ran_cols = list(columns)
    parts = []
    for cap in caps:
        for i in indices:
            path = lss_dir / f"{tracer}_{cap}_{i}_clustering.ran.fits"
            if dry:
                print(f"    would read {path.name}")
                continue
            df = read_fits_columns(path, ran_cols)
            df["CAP"] = cap
            df["RAN_IDX"] = i
            parts.append(df)
    if dry:
        return
    out = pd.concat(parts, ignore_index=True)
    dest = out_dir / f"{tracer}_clustering.ran.parquet"
    out.to_parquet(dest, index=False)
    print(f"  randoms -> {dest}  ({len(out):,} rows)")


# --------------------------------------------------------------------------- #
# Per-tracer driver
# --------------------------------------------------------------------------- #
def build_tracer(tracer: str, spec: dict, cfg: dict, lss_dir: Path,
                 fastspec_dir: Path, out_dir: Path, healpix: list[str],
                 dry: bool, with_fastspec: bool = True) -> None:
    cb = cfg["catalogue_build"]
    print(f"\n{'=' * 64}\n{tracer}\n{'=' * 64}")
    if dry:
        steps = ["load clustering", "attach colours", "deredden photometry"]
        if with_fastspec:
            steps.append("merge fastspec")
        if spec.get("splits"):
            steps.append("split red/blue")
        steps.append("write parquet")
        print(f"  [dry-run] would: {', '.join(steps)}")
        return

    df = load_clustering(lss_dir, tracer, cb["caps"], cb["clustering_columns"])
    df = attach_colours(df, lss_dir, tracer, cb["full_columns"])
    df = add_photometry(df, cb["dered_bands"])
    if with_fastspec:
        fastspec = load_fastspec(fastspec_dir, spec["fastspec_program"],
                                 cb["fastspec_columns"], cb["fastspec_ext"], healpix)
        df = merge_fastspec(df, fastspec)
    else:
        print("  [skip] fastspec merge (--no-fastspec): no stellar-mass columns")

    base = out_dir / f"{tracer}_clustering.dat.parquet"
    df.to_parquet(base, index=False)
    print(f"  catalogue -> {base}  ({len(df):,} rows, {len(df.columns)} cols)")

    if spec.get("splits"):
        df = gmm_colour_split(df, cb["gmm"])
        kinds = [("GMM", "GMM")]
        if with_fastspec:
            df = ssfr_split(df, float(cb["ssfr_threshold"]))
            kinds.append(("sSFR", "SFR"))
        df.to_parquet(base, index=False)  # rewrite with split flags
        for kind, prefix in kinds:
            for color in ("red", "blue"):
                col = f"is_{color}_{kind}"
                sub = df[df[col]].copy()
                dest = out_dir / f"{tracer.replace('_ANY', '')}_{color.upper()}_{prefix}_clustering.dat.parquet"
                sub.to_parquet(dest, index=False)
                print(f"    {col}: {len(sub):,} -> {dest.name}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config" / "data_sources.yaml"))
    ap.add_argument("--tracer", nargs="+", help="Restrict to these tracers.")
    ap.add_argument("--randoms", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--randoms-only", action="store_true",
                    help="Only (re)assemble randoms; skip the data catalogue build "
                         "(e.g. to add PHOTSYS to existing randoms).")
    ap.add_argument("--fastspec", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-fastspec skips the stellar-mass merge (fast; colour "
                         "catalogues + GMM split only).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None, help="output directory (default: the config location under <dest>)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dest = Path(cfg["dest"])
    lss_dir = dest / cfg["sources"]["desi_lss"]["subdir"]
    fastspec_dir = dest / cfg["sources"]["fastspec"]["subdir"]
    healpix = cfg["sources"]["fastspec"]["healpix"]
    cb = cfg["catalogue_build"]
    out_dir = Path(args.out_dir) if args.out_dir else dest / cb["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tracers = cb["tracers"]
    wanted = args.tracer or list(tracers)
    print(f"Building catalogues for: {', '.join(wanted)}")
    print(f"  LSS      : {lss_dir}")
    print(f"  fastspec : {fastspec_dir}")
    print(f"  out      : {out_dir}\n")

    for tracer in wanted:
        spec = tracers[tracer]
        if not args.randoms_only:
            build_tracer(tracer, spec, cfg, lss_dir, fastspec_dir, out_dir, healpix,
                         args.dry_run, with_fastspec=args.fastspec)
        if args.randoms or args.randoms_only:
            assemble_randoms(lss_dir, out_dir, tracer, cb["caps"],
                             cb["random_indices"], cb["clustering_columns"],
                             args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
