#!/bin/bash
#SBATCH --job-name=cucount_val
#SBATCH --partition=comp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=110G
#SBATCH --output=results/logs/validate_cucount_%j.out
# Validation of src/compute_correlations_cucount.py against the released LRG z1 multipoles
# (the paper-directory equivalent of ~/unions_IA cross-check job 870968).
#   sbatch scripts/validate_cucount_measurement.slurm            (from the project root)
set -euo pipefail
module purge
module load cuda/12.4
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PY=${IA_PYTHON:-$HOME/unions_IA/.venv/bin/python}
OUT=${OUT:-results/validate_cucount}
echo "host=$(hostname) start=$(date -Is) python=$PY out=$OUT"
$PY src/compute_correlations_cucount.py --config config/data_sources.yaml \
    --tracer LRG --zmin 0.4 --zmax 0.75 --sample LRG_zmin_0.40_zmax_0.75 --no-massbins \
    --s-nbins 19 --s-min 6 --s-max 100 --n-patches 70 --kmeans-init random \
    --fkp fkp --sysweights recomputed --out "$OUT"
echo "--- comparison with the release (data/correlations_highscales) ---"
$PY scripts/compare_multipoles_npz.py \
    "$OUT/LRG/LRG_zmin_0.40_zmax_0.75_GPU_multipoles.npz" \
    data/correlations_highscales/LRG/LRG_zmin_0.40_zmax_0.75_GPU_multipoles.npz
echo "done=$(date -Is)"
