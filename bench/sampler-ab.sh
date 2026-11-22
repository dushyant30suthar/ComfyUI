#!/usr/bin/env bash
# A/B the sampler question that the journey film raised: is the noise the step
# count, the solver, or the cache?
#
# Same prompt, same seed, same canvas, 124 frames (5.2s) so three arms cost ~25
# min total instead of ~2h at 362. Noise character is a per-step property, so it
# reads the same at either length.
#
#   A  euler,          9 real steps (EasyCache 0.4)   <- exactly what the film used
#   B  res_multistep, 10 steps, no cache              <- same cost as A
#   C  res_multistep, 20 steps, no cache              <- the official reference
#
# The claim under test, from the H3 sampling analysis: a second-order method's
# local error per step is the square of a first-order one's, so res_multistep at
# 20 steps ~ euler at 40-50, at one DiT call per step either way. If that holds,
# B should visibly beat A at equal cost and C should be clearly the best.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

P=bench/journey/ab-prompt.txt
cat > "$P" <<'EOF'
A still lake at dawn in Udaipur, white palaces rising from the water. Mist lies on the surface and the far shore is a soft silhouette of domes and hills. The water barely moves; a single boat crosses far off. Camera: slow aerial descent from high above the lake, settling toward the water. Timing: the sun clears the far ridge at the halfway point and light floods across the water. Style: cinematic, warm gold and pale blue, anamorphic, soft haze. Audio: distant temple bells, water lapping, birds waking.
EOF

run () {  # name steps sampler extra...
  local name=$1 steps=$2 samp=$3; shift 3
  echo "=== $(date +%H:%M:%S)  $name: $samp, $steps steps $* ==="
  : > bench/gen-server.log
  bench/memguard.sh -- .venv/bin/python bench/gen.py \
    --prompt-file "$P" --name "ab/$name" --seed 777 \
    --length 124 --steps "$steps" --sampler "$samp" \
    --chunk 4 --tiled 32 --lowmem "$@"
  grep -o "EasyCache - skipped [0-9]*/[0-9]*" bench/gen-server.log | tail -1
  grep -c "using pytorch attention instead" bench/gen-server.log \
    | xargs -I{} echo "  attention fallbacks: {}"
}

run A-euler-cache      20 euler
run B-resmulti-10      10 res_multistep --nocache
run C-resmulti-20      20 res_multistep --nocache

echo "=== $(date +%H:%M:%S)  done ==="
ls -lh output/ab/*.mp4 2>/dev/null | awk '{print "  "$5, $9}'
