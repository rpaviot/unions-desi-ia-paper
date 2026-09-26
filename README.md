# UNIONS x DESI: intrinsic alignments of DESI BGS, LRG and ELG galaxies

Reproducibility directory of Paviot et al. (2026), an [ASTRA](https://github.com/lightcone-cli)
analysis orchestrated by `lightcone-cli`: the specification (`astra.yaml` + one per
sub-analysis), the code (`src/`), the released data products (`data/`) and the recipes
that turn them into the paper's figures and tables.

```
astra.yaml                      root: inputs, 3 decisions, re-exports of the sub-analysis outputs
analyses/samples                DESI catalogues -> colour / stellar-mass samples, shapes, randoms   [UNIONS]
analyses/ia_measurements        xi_0 / xi~22 and wp / wg+ with the jackknife covariance            [UNIONS, GPU]
analyses/ggl                    magnification, clustering-z n(z), Delta Sigma, n_gal, CSMF inputs  [UNIONS]
analyses/csmf                   CSMF halo-model fit -> SHMR -> M* - M_eff relation                  [HOD env]
analyses/ia_fits                NLA / TATT fits of the IA statistics (ia2pt)
analyses/paper                  the figures and tables (root outputs re-export them)
config/data_sources.yaml        every sample definition and analysis setting the scripts read
data/                           released snapshots (3.4 MB) + links to the large products (data/README.md)
src/                            the analysis scripts, one flat directory (inventory in CLAUDE.md)
universes/baseline.yaml         the paper's choices; alternatives are one decision away
```

## Running

Two Python environments (this cluster has no container runtime; the recipes run on the
host and pick their interpreter from two variables, default `python`):

| variable | stack | build |
|---|---|---|
| `IA_PYTHON` | ia2pt, pyccl, fast-pt, iminuit, nautilus, treecorr, pycorr, dsigma, healpy; cucount + lsstypes for the GPU pair counts | `requirements.txt` / `Containerfile` |
| `HOD_PYTHON` | HOD_NRV @ `main` (8145e2e), jax, Dark Emulator | `containers/Containerfile.hod` |

```
export IA_PYTHON=/path/to/ia/venv/bin/python HOD_PYTHON=/path/to/hod/venv/bin/python
scripts/env_check.sh                       # both stacks importable?
lc run --universe baseline                 # everything that can run from the release
lc run csmf_fit --universe csmf_minuit     # one output (two seeded MINUIT passes, ~15 min)
lc run csmf_fit --universe baseline        # the nautilus posterior: ~26 h on 20 cores (see CLAUDE.md)
lc run nla_multipoles --universe baseline  # 30 nautilus fits, hours: use a compute node
lc status ; lc verify
```

### The runnable boundary

The UNIONS shape catalogue is proprietary, so `analyses/samples`, `analyses/ia_measurements`
and `analyses/ggl` need data access: on the analysis cluster
`scripts/link_cluster_data.sh` links the large products into `data/`; elsewhere those
paths dangle. Their outputs are shipped as snapshots under `data/` (the `release_*`
inputs of `astra.yaml`, 3.4 MB), from which `analyses/csmf`, `analyses/ia_fits` and every
paper figure and table run for anyone. `data/README.md` lists what is in each snapshot,
and the DESI download itself is public (`src/fetch_desi.py`, manifest in
`data/desi_download_manifest.json`).

### Decisions

Each `astra.yaml` declares the analysis choices as decisions with options; `universes/`
fix them. The baseline universe is the paper. Root: `ngal_zwindow` (z5 | full),
`ngal_area` (effective | geometric), `aia_mass_fit_space` (linear | log). csmf:
`mass_definition` (200m | 200c), `mass_function` (tinker10 | tinker08), `csmf_rp_min`
(0.5 | 0.1 Mpc/h), `gamma1_prior` (gaussian | flat), `ngal_anchor` (ngal | none),
`csmf_optimizer` (nautilus | minuit; `universes/csmf_minuit.yaml` is the MINUIT cross-check),
`csmf_point_estimate` (map | median | mean: the single model behind the chi2
decomposition, Fig. B.3, the M_eff mapping and the best-fit column of Table B.4). ia_fits: `z_ia` (pair | clustering), `rmin_gg`
(30 | 25), `rmin_gp_nla` (10 | 6), `rmin_gp_tatt` (6 | 10), `a2_prior` (6 | 5 | 3),
`tatt_bta` (1 | free), `sampler` (nautilus | minuit), `covariance_convention`.
ia_measurements: `jackknife_tessellation` (random | kmeanspp), `fkp_weights`,
`bgs_sysweights`. ggl: `nz_scale_mode` (rp | theta). To run an alternative, copy
`universes/baseline.yaml`, change a value, `lc run --universe <name>`.

### Conventions that bite (details in CLAUDE.md)

- Every IA fit: `--no-hartlap --zero-cross-cov`; TATT always `--fix-b2`.
- Projected w_g+ amplitudes have the opposite sign to the multipoles (tables flip them).
- M* in h^-2 Msun, M_h / M_eff in h^-1 Msun, separations in h^-1 Mpc.
- Delta Sigma uses the real-h Planck18 cosmology (H0 = 67.66), not H0 = 100.
- The jackknife covariances depend on the tessellation; the patch centres are shipped.

## Code and data releases

- [ia2pt](https://github.com/rpaviot/IA2pt) 0.2.0 -- IA two-point model (NLA / TATT), likelihood,
  samplers, Gaussian covariance; [docs](https://ia2pt.readthedocs.io/en/latest/).
- [HOD_NRV](https://github.com/rpaviot/NRV_HOD) public `main` @ `8145e2e` -- CSMF halo model and fitter.
- [cucount](https://github.com/adematti/cucount) / [lsstypes](https://github.com/adematti/lsstypes) -- GPU pair counts with the native split jackknife (A. de Mattia).
- DESI DR1 LSS catalogues and FastSpecFit VAC (DESI Collaboration 2025); UNIONS ShapePipe v1.6.9 (proprietary).

## References the methods rest on

Singh et al. 2023 (arXiv:2307.02545, wedge-free multipole estimator); Grieb et al. 2016
(Gaussian covariance) and Kurita & Takada 2022 (arXiv:2202.11839, spin-2 kernel);
Mohammad & Percival 2022 (arXiv:2109.07071, jackknife cross-pair correction);
Maion et al. 2024 (arXiv:2307.13754, TATT one-loop terms); the DESI clustering-redshift
analysis (arXiv:2510.23565); Dvornik et al. 2023 (CSMF halo model, gamma1 prior);
Wright et al. 2017 (stellar-mass completeness).
