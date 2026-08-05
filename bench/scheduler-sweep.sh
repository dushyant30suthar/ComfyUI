#!/usr/bin/env bash
# Sweep every scheduler ComfyUI offers, against the one we have always used.
#
# `simple` has been the scheduler for every render ever done on this box, for
# exactly the reason `euler` was: it sat in matrix.PROBE, a config written to
# benchmark throughput, and nobody re-examined it. The sampler turned out to be
# the largest quality lever here and it was essentially free. This is the same
# shape of unexamined default, so it gets the same treatment.
#
# Held fixed so the scheduler is the only variable: same prompt, same seed 777,
# same res_multistep at 10 steps, no cache, 124 frames at native 1344x768 --
# identical to A/B arm B, which is therefore directly comparable.
#
# A note on expectations, recorded BEFORE the run so it cannot be rationalised
# afterwards: H3 is a flow-matching model, and karras/exponential were designed
# for variance-preserving diffusion schedules. They may be badly mismatched. The
# prediction is that simple/sgm_uniform/beta/kl_optimal cluster and the two
# VP-oriented ones are visibly worse. Running them anyway -- a cheap measurement
# beats a confident guess, which is the lesson this whole project keeps
# re-teaching.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCHEDULERS=(simple sgm_uniform karras exponential ddim_uniform beta normal linear_quadratic kl_optimal)
P=bench/journey/ab-prompt.txt
[[ -f "$P" ]] || { echo "missing $P"; exit 64; }

echo "=== $(date +%H:%M:%S)  scheduler sweep: ${#SCHEDULERS[@]} arms, res_multistep 10 steps, 124f ==="

for s in "${SCHEDULERS[@]}"; do
  if ls output/sched/${s}_*.mp4 >/dev/null 2>&1; then
    echo "[$s] already rendered, skipping"; continue
  fi
  echo "=== $(date +%H:%M:%S)  $s ==="
  : > bench/gen-server.log
  timeout 2400 bench/memguard.sh -- .venv/bin/python bench/gen.py \
    --prompt-file "$P" --name "sched/$s" --seed 777 \
    --length 124 --sampler res_multistep --steps 10 --nocache --scheduler "$s" \
    --chunk 4 --tiled 32 --lowmem
  rc=$?
  fb=$(grep -c "using pytorch attention instead" bench/gen-server.log 2>/dev/null | head -1)
  [[ "$fb" =~ ^[0-9]+$ ]] || fb=0
  [[ $rc -eq 0 ]] && echo "  OK$([[ $fb -gt 0 ]] && echo "  WARNING: $fb attention fallbacks")" \
                  || echo "  FAILED rc=$rc — continuing"
done

echo "=== $(date +%H:%M:%S)  done ==="
ls -lh output/sched/*.mp4 2>/dev/null | awk '{print "  "$5, $9}'
