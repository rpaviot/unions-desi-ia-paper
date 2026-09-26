# Project Notes for Claude

ASTRA project (lightcone-cli 0.4.2 / astra-spec 0.0.14) for the paper
*UNIONS x DESI: intrinsic alignments of DESI BGS, LRG and ELG galaxies* (Paviot et al.).
It was scoped with `/lc-from-code` from the analysis repository `~/unions_IA` on
2026-09-18; `astra.yaml` + `analyses/*/astra.yaml` are the spec (the paper's figures live in
`analyses/paper`, whose inputs alias the ia_fits / csmf outputs; root outputs re-export
them -- lc's executor does not resolve a root output re-export used as an input, astra
does), `src/` the code,
`data/` the inputs and released products, `config/data_sources.yaml` the sample
definitions.

```
lc run --universe baseline          # everything runnable (csmf, ia_fits, paper figures)
lc run csmf_fit --universe baseline # one output
lc status / lc verify
astra validate astra.yaml           # after EVERY spec change
```

## Environments (no container runtime on this cluster: `runtime: none`)

Recipes run on the host from the project root and pick their interpreter from two
environment variables (default `python`):

| variable | what it must provide | local value |
|---|---|---|
| `IA_PYTHON` | ia2pt (github.com/rpaviot/IA2pt), pyccl 3.3 + camb, fast-pt, iminuit, nautilus, treecorr, pycorr, dsigma 1.2, healpy, pyarrow; cucount 0.2.7 + lsstypes (GitHub) for the GPU pair counts | `~/unions_IA/.venv/bin/python` |
| `HOD_PYTHON` | HOD_NRV (github.com/rpaviot/NRV_HOD, public `main` @ 8145e2e, editable as `nrvpy`), jax, dark_emulator, iminuit, nautilus, pyccl | `~/NRV_HOD/.venv_hod/bin/python` |

`export IA_PYTHON=... HOD_PYTHON=...` before `lc run`; `scripts/env_check.sh` checks both.
`requirements.txt` + `Containerfile` (IA) and `containers/Containerfile.hod` document the
same two stacks for people with docker. On the cluster `lc run` must sit inside a SLURM
allocation (it srun-launches its Dask workers): `sbatch --export=ALL,OUTPUTS="csmf_fit"
scripts/lc_run.slurm` (see the header for the options). Logs land in `results/logs/` (ignored,
and outside the image-identity hash: the Containerfile COPYs only src/ config/ scripts/,
because lc folds the hash of the COPY sources into every output's code_version -- a
log or README edit under a `COPY . .` would mark every output stale).

Operating `lc run` under SLURM (learned 2026-09-18/19):

- Cancelled or wall-killed runs leave Snakemake "incomplete" markers, and the next `lc run`
  fails at once with `IncompleteFilesException` (lc does not pass `--rerun-incomplete`).
  The markers are `.snakemake/incomplete/<base64 of the output path>` (`.snakemake` is a
  symlink into the scratch root); remove the markers of the affected outputs and their
  emptied output dirs, then resubmit.
- Snakemake wipes an output dir before re-running its rule, so an interrupted rule leaves
  an empty dir ("missing") -- and destroys any checkpoint inside it. The nautilus CSMF fit
  (the `baseline` fiducial since 2026-09-21; `csmf_minuit` is the MINUIT cross-check;
  ~26 h on 20 cores at ~2 logL/s, so a 24 h wall is too short)
  checkpoints to `<output>/csmf_fit.npz.h5`; the slowness is the f_h/f_s NFW rescaling
  re-evaluating sici-based u(k|M) per call, not a misconfiguration -- measured 2.1 logL/s
  for both the paper and the July-default halo model). Copy that file OUT of the output dir before
  the job dies and resume with `sbatch --export=ALL,...,CSMF_NAUTILUS_RESUME=<copy>
  --time=48:00:00 scripts/lc_run.slurm` (fit_csmf `--resume-from`, recipe passthrough).
  Same seed / n_live / `--vectorized` are required for the resume. A FINISHED checkpoint
  resumes in minutes (nautilus returns at once; fit_csmf only recomputes the summary
  keys), which is how the baseline is re-materialised after a code change without
  re-sampling: the finished chain is kept at
  `/n09data/rpaviot/DESIxUnions/ggl/csmf_fit/paper_dir_ckpt/csmf_nautilus_872393_done/csmf_fit.npz.h5`.
- One `lc run` per project at a time (exclusive run lock): chain jobs with
  `--dependency=afterany:<jobid>`.
- Snakemake launches independent rules concurrently and each HOD / IA python spawns ~20
  OpenBLAS + XLA threads; on a node shared with other jobs of the same user the default
  per-user thread limit (RLIMIT_NPROC 1280) ran out and four scripts died at import
  (OpenBLAS `pthread_create failed`, exit -2; run 874068). `lc_run.slurm` raises the soft
  limit to the hard one. The IA fits are effectively single-threaded, so
  `--cpus-per-task=14 --mem=64G` (sbatch flags override the script's `#SBATCH` lines)
  is enough for the baseline chain and backfills much sooner than 20 cores / 110 GB.
- **A commit invalidates every materialised output for Snakemake** (not for `lc status`):
  the rule `params` dict carries `git_sha` (and `lc_version`), and lc runs Snakemake with
  `--rerun-triggers code,input,mtime,params`, so after any new commit the next `lc run`
  reports "Params have changed", DELETES the output dirs it targets and re-runs them
  (verified in `.snakemake/metadata`: the stored params record the sha). `lc status` /
  `lc verify` use code_version (recipe + image hash + decisions, no sha) and stay `ok`.
  After commit-only changes run `lc run ... --rerun-triggers code,input,mtime`, and check
  `lc status` first: what it calls `ok` is not re-run by that form.

## Runnable boundary

The UNIONS shape catalogue is proprietary, so `analyses/samples`, `analyses/ia_measurements`
and `analyses/ggl` can only run with data access (here: `data/unions`, `data/ia/*.parquet`,
`data/catalogues`, `data/hpmaps` are symlinks into `/n09data/rpaviot/DESIxUnions`, created
by `scripts/link_cluster_data.sh`; elsewhere they dangle). Their outputs are shipped as
snapshots — the `release_*` inputs of the root spec — under `data/`:

| release input | path | produced by |
|---|---|---|
| correlations, fine grid | `data/correlations_highscales/` | ia_measurements.correlations_fine (geomspace(6,100,20)) |
| correlations, coarse grid | `data/correlations_xi2coarse/` | ia_measurements.correlations_coarse (geomspace(6,100,11)) |
| sample properties + footprint areas | `data/ia/sample_properties/` | samples.sample_properties / footprint_area |
| jackknife patch centres | `data/ia/patch_centers/` | ia_measurements (cached by compute_correlations) |
| clustering-z n(z) | `data/ggl/nz/` | ggl.clustering_nz |
| Delta Sigma (DR6 sources) | `data/ggl/deltasigma_dr6/` | ggl.deltasigma |
| magnification alphas | `data/ggl/magnification_alphas.csv` | ggl.magnification_alphas |
| M* completeness | `data/ggl/completeness/` | samples.mstar_completeness |
| CSMF inputs | `data/ggl/csmf_input_{z5_effective,z5_geometric,full_geometric}/` | ggl.csmf_input for the (ngal_zwindow, ngal_area) combinations |
| CSMF minuit seed | `data/ggl/csmf_fit_seed/` | a converged 200m / Tinker08 / z5-geometric fit |

`analyses/csmf`, `analyses/ia_fits` and the root figures / tables run for anyone from
these snapshots.

## Script inventory (src/, flat; one copy of the analysis repo's scripts)

| script | stage | purpose | reads | writes | invocation notes |
|---|---|---|---|---|---|
| fetch_desi.py | samples | download DESI DR1 LSS + FastSpecFit from the PIC mirror, size-verified | config | data/desi/**, download manifest | `--source --workers --dry-run` |
| build_catalogues.py | samples | join clustering + full + FastSpecFit per tracer, deredden, BGS GMM (3 comp., 2 sigma) and sSFR splits, randoms | data/desi | data/catalogues/*.parquet | DESI-only; `--tracer --randoms --dry-run` |
| build_unions_shape.py | samples | stream the UNIONS hdf5 once -> for_IA / for_GGL parquet, metacal responsivity | UNIONS hdf5 (proprietary) | data/unions/*.parquet | `--sample --max-chunks --dry-run` |
| sysweights.py | samples (module) | BGS imaging-systematics weights (RidgeCV on 6 maps, 20 percentile bins, floor 1e-3) + DESI-style random reconstruction | hpmaps | — | imported by build_*_samples |
| build_ia_samples.py | samples | footprint mask, 1" match, responsivity calibration, mass bins (equal-S/N or equal-N), shape randoms | catalogues, unions for_IA, hpmaps | data/ia/*.parquet | `--tracers --clean`; seeds 42 |
| build_vlim_samples.py | samples | volume-limited S/N staircase of the BGS red sample (BGS_RED_GMM_VLIM_SNR) | ia shapes, completeness | data/ia/*_VLIM_SNR_* | `--samples --no-plot` |
| compute_mstar_completeness.py | samples | Wright+17-style M*_lim(z) (count turn-over), monotonic | catalogues, ia | data/ggl/completeness | `--samples --no-plot` |
| compute_footprint_area.py | samples | density-ratio area per tracer (2500 randoms/deg2/file) | unions for_IA, randoms | data/ia/sample_properties/footprint_area_*.json | `--tracers` |
| sample_properties.py | samples | sigma_e, n_eff, z_eff, M* stats per shape sample | ia shapes | data/ia/sample_properties/*.json | `--only --overwrite` |
| build_combined_reference.py | samples | LRG+ELG reference 0.8-1.1 for clustering-z | desi combined | data/ggl/reference | — |
| compute_correlations.py (+ djk/) | ia_measurements | xi0 / xi~22 (cucount GPU) and wp / wg+ (pycorr/treecorr) with the 70-patch delete-one jackknife + Mohammad & Percival correction | ia samples, patch centres | data/correlations*/<tracer>/<stem>_{GPU_multipoles,CPU_projected}.npz | `--tracer --zmin --zmax --sample --with-projected --s-nbins/--s-min/--s-max --rp-match-s --n-patches --kmeans-init --fkp fkp|none --sysweights recomputed|desi --out`; `--s-nbins` counts BINS (fine grid 19, coarse 10); GPU node |
| compute_correlations_cucount.py | ia_measurements | the same xi0 / xi~22 with cucount's native split jackknife (lsstypes Count2Jackknife), no djk; estimators + projection written out | ia samples, patch centres | <out>/<tracer>/<stem>_GPU_multipoles.npz (same keys) | same flags minus the projected ones; validated vs the release: xi0 1e-13, xi2 6e-6 (`scripts/validate_cucount_measurement.slurm`) |
| compute_gaussian_cov.py / _projected.py | ia_measurements | Gaussian covariance of [xi0, xi~22] / [wp, wg+] (ia2pt.gausscov) | correlations npz, shapes, area | *_gaussian_cov*.npz | `--corr-dir --tracers --samples --a1 --kmax` |
| compute_magnification.py (+ ggl_utils.py) | ggl | finite-difference magnification alpha per lens bin | catalogues, ia | data/ggl/magnification_alphas.csv | — |
| estimate_nz.py | ggl | clustering-z n(z) of the source bins (treecorr NN vs DESI slices, CCL bias correction) | ia LRG/ELG, reference, unions for_GGL | data/ggl/nz/nz_<bin>_<mode>.npz | `--scale-mode theta|rp` (production: rp) |
| compute_deltasigma.py | ggl | dsigma 1.2 Delta Sigma with clustering-z table_n, random subtraction, 100-region JK; real-h Planck18 (NOT H0=100) | vlim lenses + randoms, unions for_GGL, nz | data/ggl/deltasigma_dr6/*.npz | `--samples --massbins --n-jobs --out-name` |
| compute_ngal.py | ggl | n_gal per lens bin over the footprint area (+JK error), abundance anchor | vlim lenses, area | data/ggl/csmf_input*/ngal_*.npz | `--zlo-percentile 5 --area-mode geometric|effective --suffix` |
| build_csmf_input.py | ggl | per-bin CSMF fitter input (Delta Sigma + n_gal + alpha; magnification subtracted for LRG only) | deltasigma, ngal, alphas | data/ggl/csmf_input*/csmf_*_massbin*.npz | `--in-name --out-name --ngal-suffix` |
| fit_csmf.py | csmf | CSMFFitter (HOD_NRV) minuit / nautilus / DE fit of Delta Sigma [+ n_gal] | csmf inputs, seed fit | csmf_fit_<label>_<method>.npz | `--method --rp-min --in-dir --mass-def --mass-function tinker10 --halo-bias --gamma1-prior gaussian|flat --ngal-anchor ngal|none --start-from --label --out`; HOD env; the fitter's ndof = Delta Sigma points - free params (90), the n_gal terms add to chi2 only |
| csmf_meff_relation.py | csmf | M*-M_eff relation of the lens bins from a fit npz (plot_aia_meff.label_relation) | csmf fit, inputs | aia_vs_meff_relation.npz (rel_logmstar / rel_logmeff) | `--fit-npz --in-dir --samples --out`; HOD env; reproduces the paper's relation npz bit-for-bit |
| decompose_csmf_chi2.py, plot_csmf_bestfit.py, plot_shmr_appendix.py, plot_csmf_shmr*.py | csmf | chi2 split, Fig B.3, Fig B.2 | csmf fit | txt / png | HOD env |
| fit_ia.py (+ ia2pt) | ia_fits | NLA / TATT fits of [xi0, xi~22] or [wp, wg+] on the split grids, minuit / nautilus / emcee | correlations npz (fine + coarse), sample properties | <out-dir>/<stem>_<stat>_<model>_<sampler>.npz | see the fiducial invocation below |
| tatt_posterior_stats.py, decompose_ia_chi2.py, chi2_distribution.py, fit_blocks.py, bundle_nautilus_fits.py | ia_fits | posterior stats, chi2 splits, split-grid slicing helper, bundling | fit npz | — | — |
| plot_fig3_aia_mass_meff.py, plot_ia_corrfunc.py, plot_aia_mass_blue_compare.py, plot_appendix_multipoles.py, plot_tatt_a1a2_gtc.py, plot_wgp_vs_multipoles_gtc.py, plot_colour_redshift.py, make_paper_tables.py, paper_style.py | paper | Figs 2-7, A1, A, tables B.1-B.3 | fit npz, M_eff relation | png / tex | every `--*-dir` default in these scripts is STALE: always pass the dirs explicitly (the recipes do) |

## Fiducial invocations (what produced the paper numbers)

```
# IA fits (SLURM array, ~/unions_IA/scripts/fit_nautilus_arrays.slurm, 2026-09-17)
fit_ia.py --stat multipoles --model NLA --sampler nautilus \
  --corr-dir <fine> --corr-dir-gp <coarse> --exclude SFR \
  --no-hartlap --zero-cross-cov --zeff-ia pair --rmin 30 10 --rmax 100
fit_ia.py --stat projected  --model NLA  ... --rmin 6 6
fit_ia.py --stat multipoles --model TATT --fix-b2 --fix-bta 1 --prior a2 -6 6 \
  --sample BLUE_GMM --sample ELG_ --exclude SFR ... --rmin 30 6
# CSMF fit (run_csmf_fit_m200m_variants.slurm, two seeded minuit passes; label *_p2)
fit_csmf.py --method minuit --seed 42 --rp-min 0.5 --in-name csmf_input_z5eff \
  --mass-def MassDef200m --mass-function Tinker10 --halo-bias Tinker10 \
  --samples BGS_RED_GMM_VLIM_SNR --start-from <seed npz>
# n_gal for those inputs: compute_ngal.py --zlo-percentile 5 --area-mode effective
# Delta Sigma inputs: build_csmf_input.py --in-name deltasigma_dr6
```

## Plumbing added for the recipes (2026-09-18)

Every recipe flag exists and was exercised: `--out-dir` on all builders, explicit
`--in-dir` / `--fit-npz` on the CSMF scripts, `--properties-dir` and `--fix-bta free` on
fit_ia (which now imports `ia2pt` directly), `--zlo-percentile z5|full`,
`--fkp` / `--sysweights` decision forms, `--jk-seed` on compute_ngal / compute_deltasigma
(default None = the unseeded production draws: dsigma's jackknife KMeans is random, so
the released n_gal_err / Delta Sigma covariance are one draw and re-runs reproduce them
to ~10 %, not bit-for-bit). Reference checks: M_eff relation bit-identical, chi2
decomposition 134.65 = saved, tables B.1-B.3 byte-identical, LRG z1 MINUIT chi2 20.32 =
nautilus tree, sample_properties / csmf_input rebuilds bit-identical.

## CSMF fiducial = the nautilus posterior (2026-09-21)

`csmf_optimizer` defaults to nautilus and `csmf_point_estimate` to `map`. fit_csmf saves,
next to the chain, `nautilus_best_fit` (marginal MEDIAN) / `_lo16` / `_hi84` / `_errors` /
`_covariance` / `_chi2` (at the median), `nautilus_map` / `_map_chi2` (the chain sample
maximising log_l MINUS the gamma1 prior penalty -- nautilus's log_l is the likelihood
alone, the Gaussian prior sits in its Prior object -- i.e. MINUIT's objective) and
`nautilus_mean` / `_mean_chi2`. `src/csmf_point.py::load_point(fit, method, point)` is the
one place the downstream scripts read a point estimate from; `--point` on
decompose_csmf_chi2, plot_csmf_bestfit, csmf_meff_relation (+ `--n-samples 1000`: 16-84 %
band of log M_eff per bin from posterior draws, drawn as x error bars on Fig. 3 and not
propagated into its broken power-law fit) and make_csmf_table (Table B.4: median with
16-84 % + MAP column, chi2 at the MAP in the foot). Numbers: MAP chi2 135.4 = 105.8
Delta Sigma + 28.5 n_gal + 1.1 prior; median chi2 140.8; mean chi2 150 (skewed
posterior -- never call the mean a best fit); MAP vs MINUIT minimum within 0.4 sigma
(chi2 134.65); f_h = 0.85 +0.11 -0.16, f_s = 0.81 +0.13 -0.24 (MINUIT rails both at 1).
Fig. B.2 is unchanged (already the marginal median curve + 68 % band for nautilus).

## Validation of the release directory (2026-09-21, runs 874103 baseline / 874104 csmf_minuit)

`scripts/compare_with_reference.py`: all 39 IA fits (30 NLA multipoles, 5 projected, 4
TATT) within 0.02 sigma of the production `nautilus_split_*_zeff-pair` trees, chi2 equal
to 2 decimals (nautilus is unseeded: the re-drawn posteriors move the paper's quoted
numbers in the last digit only, e.g. LRG high-z A_IA 2.94 -> 2.95 +- 0.38); the nautilus
MAP within 0.41 sigma of the MINUIT minimum (chi2 135.38 vs 134.65), the median within
1.06 sigma; the csmf_minuit universe reproduces the production p2 fit to 3e-7 relative
in every parameter and 2e-5 in chi2 (134.653338 vs 134.653315) -- bit-identical when run
on the same CPU type and thread count as the production fit (872379 on n09 / 20 cores),
XLA's floating-point reduction order otherwise (874104 on n27 / 14 threads). Figures regenerate (Fig. 3 broken power
law beta1 = -0.26 +- 0.10, beta2 = 1.61 +- 0.29, log Mb = 12.91 +- 0.04, chi2 28.1/18);
tables B.1-B.3 regenerate up to last-digit posterior noise; Table B.4 (table_csmf) is
new and generated. `lc verify --universe baseline` and `--universe csmf_minuit` pass.
The draft (IA_draft_repo, commit f445603) carries exactly these outputs.

## Conventions that bite

- Every IA fit: `--no-hartlap --zero-cross-cov` (a delete-one jackknife over-corrects with
  Hartlap; cross blocks are noise). TATT: always `--fix-b2`. Mismatched conventions make
  TATT look worse than NLA.
- Projected `w_g+` fits return `a1` with the opposite sign to the multipole fits; tables
  and figures flip it (`-a1`). e_+ > 0 is radial alignment (= -e_t).
- Units: h^-1 Mpc everywhere; M* is h^-2 Msun (FastSpecFit, ~0.34 dex above the
  Dvornik h^-2 scale), M_h / M_eff h^-1 Msun.
- Delta Sigma uses the real-h Planck18 cosmology (H0 = 67.66); dsigma 1.2 with table_n
  and H0 = 100 inflates Delta Sigma by 1/h.
- Fig. 3: the broken power law is the LINEAR-A_IA MINUIT fit; never `--mc-band`
  (that branch plots the log-space seed).
- `fit_ia.py`'s own defaults (rmin 10/6, Hartlap on, zero-cross off, z_ia clustering) are
  NOT the paper's; the recipes pass everything explicitly.

## Report

`index.md` + `myst.yml` are a template MyST report wired to the MySTRA plugin. Reference
analysis elements by path (`{astra}` role / directive, `{astra:value}`); never hard-type
a measured value in the prose. Preview with `myst start` (needs `npm i -g mystmd`).
