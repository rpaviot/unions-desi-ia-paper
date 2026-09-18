# Data layer

Every path the analysis reads or writes lives under this directory (`dest: data` in
`config/data_sources.yaml`; recipes run from the project root). It has two kinds of
content.

## Released snapshots (committed, 3.4 MB)

The outputs of the UNIONS-dependent stages, as used for the paper. They are the
`release_*` inputs of `astra.yaml` and are all that `analyses/csmf`, `analyses/ia_fits`
and the paper figures need.

| path | contents | produced by |
|---|---|---|
| `correlations_highscales/<tracer>/` | xi_0, xi~22 on s = geomspace(6, 100, 20) (`*_GPU_multipoles.npz`) and wp, wg+ on the matching r_p grid (`*_CPU_projected.npz`), 70 jackknife realisations + covariances (raw and Mohammad & Percival corrected), n(z) of both samples, z_eff, counts; 39 samples | ia_measurements.correlations_fine |
| `correlations_xi2coarse/<tracer>/` | the same on geomspace(6, 100, 11); the fits take xi~22 / wg+ from here | ia_measurements.correlations_coarse |
| `ia/sample_properties/` | per shape sample sigma_e, n_eff, z_eff, LOGMSTAR statistics (json); `footprint_area_<tracer>.json` | samples.sample_properties, samples.footprint_area |
| `ia/patch_centers/` | the 70 k-means patch centres of each measured sample (init `random`; the `*_np70_random_*` files are the fiducial tessellation, the others are the earlier kmeans++ / tree tessellations kept for the sensitivity test) | ia_measurements (cached by compute_correlations.py) |
| `ggl/nz/` | clustering-z n(z) of the `bgs` and `lrg` source bins, `theta` and `rp` scale modes, plus the variants of the robustness tests (dz, ODDS cut, window widths); `nz_<bin>.npz` = the theta run | ggl.clustering_nz |
| `ggl/deltasigma_dr6/` | Delta Sigma(r_p) of the 10 BGS_RED_GMM_VLIM_SNR lens bins, DR6 photo-z sources, real-h Planck18, 100-region jackknife | ggl.deltasigma |
| `ggl/magnification_alphas.{csv,json}` | magnification-bias alpha per lens bin (json: the finite-difference curves) | ggl.magnification_alphas |
| `ggl/completeness/` | Wright et al. 2017-style M*_lim(z) of the BGS red and LRG samples | samples.mstar_completeness |
| `ggl/csmf_input_<window>_<area>/` | per-bin CSMF fitter input `csmf_BGS_RED_GMM_VLIM_SNR_massbin{0-9}.npz` (r_p, Delta Sigma, covariance, n_gal +- err, alpha, magnification template) for the three (ngal_zwindow, ngal_area) combinations, with the matching `ngal_BGS_RED_GMM_VLIM_SNR.npz` of ggl.ngal alongside. `z5_effective` is the paper fiducial | ggl.ngal + ggl.csmf_input |
| `ggl/csmf_fit_seed/` | the converged MINUIT fit (M200m, Tinker08, z5 / geometric-area n_gal) every production CSMF fit starts from | csmf.csmf_fit (an earlier universe) |
| `desi_download_manifest.json` | URL, expected size and check status of the 91 DESI DR1 files (155 GB) fetched by `src/fetch_desi.py` | samples (fetch) |

## Cluster-only products (symlinks, not committed)

`scripts/link_cluster_data.sh` links the large products into place from the cluster copy
(`/n09data/rpaviot/DESIxUnions`); without it these paths do not exist and the three
upstream stages cannot run:

| path | contents | size |
|---|---|---|
| `desi/` | DESI DR1 LSS clustering catalogues + randoms (iron v1.5) and FastSpecFit VAC, laid out as the release | 155 GB, public (`src/fetch_desi.py`) |
| `catalogues/` | analysis-ready per-tracer parquet (clustering + full + FastSpecFit join, dereddened photometry, GMM / sSFR colour splits, randoms) | 21 GB, regenerable from `desi/` |
| `hpmaps/` | DESI imaging-systematics healpix maps (per cap) | 199 MB, public |
| `unions/` | UNIONS shape samples `unions_shape_for_IA.parquet`, `unions_shape_for_GGL.parquet` | 39 GB, **proprietary** |
| `ia/*.parquet` | IA samples: density, randoms, shapes, shape randoms per tracer / redshift bin / mass bin, and the volume-limited lens bins | 124 files, UNIONS-matched |
| `ggl/reference/` | LRG+ELG combined clustering-z reference (0.8 < z < 1.1) | 236 MB, DESI-only |

## The UNIONS shape catalogue

UNIONS / CFIS ShapePipe comprehensive catalogue **v1.6.9** (PSF model: PSFEx, Bertin 2011;
ngmix metacalibration shapes), with the **DR6** photo-z (BPZ on the ugriz GAaP photometry,
`Z_B`, `Z_B_MIN`, `Z_B_MAX`, `T_B`, `ODDS`), 644 507 262 rows in one hdf5 file
(`unions_shapepipe_comprehensive_struc_ugriz_2024_v1.6.c.DR6.hdf5`, ~390 GB). It is
proprietary to the UNIONS collaboration and is not distributed here. `src/build_unions_shape.py`
streams it once and writes the two samples the analysis uses
(`config/data_sources.yaml: unions_shape.samples`):

- **for_IA** (no quality cut, Dec > 22): `RA`, `Dec`, `w_iv` -> WEIGHT, `e1_uncal`, `e2_uncal`,
  and the metacal responsivities `R_g11`, `R_g22` derived from the `e{1,2}{P,M}` ladder
  (dg = 0.01). Shapes are calibrated per IA sample by the weighted mean responsivity.
- **for_GGL** (ShapePipe cond1: 0.5 < T/T_psf < 3, MAG_WIN > 20, 10 < snr < 200, Dec > 22,
  veto of the mask bit-planes 1, 2, 4, 8, 64, 1024 of `data_ext`): the for_IA columns plus
  `snr`, `MAG_WIN`, `NGMIX_T_NOSHEAR`, `NGMIX_Tpsf_NOSHEAR`, `e1_PSF`, `e2_PSF`, `fwhm_PSF`,
  `Z_B`, `Z_B_MIN`, `Z_B_MAX`, `T_B`, `ODDS`.

The IA correlation estimator is normalised by R_D S (shape randoms are not used) and
e_+ > 0 denotes radial alignment (e_+ = -e_t). Access to UNIONS data: contact the UNIONS
collaboration (https://www.skysurvey.cc).
