#!/usr/bin/env bash
# One-shot remote GPU training: pull -> train -> push weights back.
#
# Usage (from the repo root on the GPU box):
#   bash gpu_train.sh                 # trains 'blended' on GPU 0
#   bash gpu_train.sh chunk           # any mode
#   bash gpu_train.sh blended 0       # mode + device (0 = first CUDA GPU, or 'cpu')
#   MODE=spill DEVICE=0 bash gpu_train.sh
#
# Modes: standard | spill | logo | chunk | unmixed | blended
# Then on the Mac: git pull && cp the printed best.pt into active_pipeline/checkpoints/
set -euo pipefail

MODE="${1:-${MODE:-blended}}"
DEVICE="${2:-${DEVICE:-0}}"
RUN="${MODE}-nano-v1"
BEST="training/runs/${MODE}-seg/${RUN}/weights/best.pt"

cd "$(dirname "$0")"                       # repo root, wherever this lives

echo ">> [1/3] pull latest dataset + code"
git pull --ff-only

echo ">> [2/3] train ${MODE} on device ${DEVICE}"
( cd training && python train_multi.py --mode "${MODE}" --device "${DEVICE}" )

if [ ! -f "${BEST}" ]; then
  echo "!! expected weights not found: ${BEST}" >&2
  echo "!! (did training finish? check training/runs/${MODE}-seg/)" >&2
  exit 1
fi

echo ">> [3/3] push weights back"
git add "${BEST}"
git commit -m "Train ${MODE}-seg on GPU (device ${DEVICE})" \
  || { echo "nothing to commit (weights unchanged)"; exit 0; }
git push

echo ">> done. On the Mac:  git pull && cp ${BEST} active_pipeline/checkpoints/yolo_${MODE}_seg.pt"
