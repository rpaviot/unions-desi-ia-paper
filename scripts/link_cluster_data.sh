#!/usr/bin/env bash
# Link the large, non-distributed products of the analysis into data/ so that the
# UNIONS-dependent stages (analyses/samples, ia_measurements, ggl) can run here.
#
# Everything this script creates is a symlink into the cluster copy and is NOT
# committed (.gitignore); elsewhere the links simply dangle and only the released
# snapshots under data/ (correlations_*, ia/sample_properties, ia/patch_centers,
# ggl/*) are available -- which is all analyses/csmf, analyses/ia_fits and the paper
# figures need.
#
# Usage: scripts/link_cluster_data.sh [SOURCE_ROOT]   (default: /n09data/rpaviot/DESIxUnions)
set -euo pipefail
SRC=${1:-/n09data/rpaviot/DESIxUnions}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DATA=$HERE/data
[ -d "$SRC" ] || { echo "source root $SRC not found" >&2; exit 1; }

link() {  # link <target> <linkname>
  local tgt=$1 lnk=$2
  [ -e "$tgt" ] || { echo "  skip (missing) $tgt"; return; }
  if [ -L "$lnk" ] || [ ! -e "$lnk" ]; then ln -sfn "$tgt" "$lnk"; echo "  $lnk -> $tgt";
  else echo "  keep (real path) $lnk"; fi
}

echo "DESI download (LSS + FastSpecFit), analysis-ready catalogues, imaging maps:"
link "$SRC/desi"        "$DATA/desi"
link "$SRC/catalogues"  "$DATA/catalogues"
link "$SRC/hpmaps"      "$DATA/hpmaps"

echo "UNIONS shape samples (proprietary):"
link "$SRC/unions"      "$DATA/unions"

echo "Clustering-z combined reference:"
mkdir -p "$DATA/ggl"
link "$SRC/ggl/reference" "$DATA/ggl/reference"

echo "IA samples (density, randoms, shapes, shape randoms, mass bins), file by file:"
mkdir -p "$DATA/ia"   # real directory: sample_properties/ and patch_centers/ are committed
n=0
for f in "$SRC"/ia/*.parquet; do
  ln -sfn "$f" "$DATA/ia/$(basename "$f")"; n=$((n+1))
done
echo "  $n parquet files linked into data/ia/"
