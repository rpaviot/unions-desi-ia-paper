#!/bin/bash
#SBATCH --job-name=lc_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --mem=110G
#SBATCH --output=results/logs/lc_run_%j.out
# Materialise ASTRA outputs through `lc run` inside a SLURM allocation (lc dispatches
# its Dask workers with srun when it finds itself in one). Usage, from the project root:
#
#   sbatch --export=ALL,OUTPUTS="csmf_fit meff_relation",UNIVERSE=baseline scripts/lc_run.slurm
#   sbatch --export=ALL,OUTPUTS="nla_multipoles" --time=24:00:00 scripts/lc_run.slurm
#
# Logs land in results/logs/ (create it first: sbatch does not). IA_PYTHON / HOD_PYTHON are inherited from the submitting shell (--export=ALL) or
# default to the two local venvs below.
set -euo pipefail
module purge
export TMPDIR=/n09data/rpaviot/tmp
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export IA_PYTHON=${IA_PYTHON:-$HOME/unions_IA/.venv/bin/python}
export HOD_PYTHON=${HOD_PYTHON:-$HOME/NRV_HOD/.venv_hod/bin/python}
LC=${LC:-$HOME/unions_IA/.venv/bin/lc}
UNIVERSE=${UNIVERSE:-baseline}
OUTPUTS=${OUTPUTS:?set OUTPUTS=\"output_id ...\"}
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "host=$(hostname) start=$(date -Is) universe=$UNIVERSE outputs=[$OUTPUTS]"
echo "IA_PYTHON=$IA_PYTHON HOD_PYTHON=$HOD_PYTHON lc=$($LC --version 2>/dev/null || echo '?')"
# shellcheck disable=SC2086
$LC run $OUTPUTS --universe "$UNIVERSE" --verbose
$LC status --universe "$UNIVERSE" | grep -E "$(echo $OUTPUTS | sed 's/ /|/g')" || true
echo "done=$(date -Is)"
