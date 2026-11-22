#!/usr/bin/env bash
# Sol-Attn: does training-free sparse attention beat SageAttention on this box?
#
# MiniMax's ecosystem chart claims "sol-attn / sage-attn / EasyCache, ~20-35%
# each, and they stack". At the ComfyUI node level they do NOT stack: sol-attn
# and our sage_chunked node both claim
# transformer_options["optimized_attention_override"], so the last one applied
# wins and the other silently does nothing. This tests sol-attn INSTEAD OF sage.
#
# The upstream repo claims 3.93x at 8k tokens and 9.00x at 32k against SDPA.
# Note the comparison is against SDPA, not against SageAttention -- our baseline
# is already accelerated, so expect far less than those headline numbers.
#
# It is an APPROXIMATION (sparse KV routing). Adoption therefore requires a
# quality check against the dense baseline, exactly as v4-exact-attn got, not
# adoption on a speed number alone. tau is the sparsity knob: higher skips more
# KV blocks, so 1.5 should be faster and worse than 0.8.
#
# All arms: 124f native, seed 777, res_multistep 10 steps, no cache.
# Baseline is the CURRENT production config (sage + chunk 4).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f "venv/bin/python bench/gen.py" >/dev/null 2>&1; do sleep 15; done
sleep 3

P=bench/journey/ab-prompt.txt
COMMON=(--prompt-file "$P" --seed 777 --length 124 --sampler res_multistep --steps 10
        --nocache --tiled 32 --lowmem)

run () {
  local name=$1; shift
  ls output/solattn/${name}_*.mp4 >/dev/null 2>&1 && { echo "[$name] exists, skipping"; return; }
  echo "=== $(date +%H:%M:%S)  $name  $* ==="
  : > bench/gen-server.log
  timeout 3600 bench/memguard.sh -- .venv/bin/python bench/gen.py \
    --name "solattn/$name" "${COMMON[@]}" "$@"
  local rc=$?
  # sol-attn falling back to SDPA is the failure that would look like success
  grep -iE "sol.?attn|SDPA|fallback|sm_?120" bench/gen-server.log | head -3
  local fb; fb=$(grep -c "using pytorch attention instead" bench/gen-server.log 2>/dev/null | head -1)
  [[ "$fb" =~ ^[0-9]+$ ]] || fb=0
  [[ $rc -eq 0 ]] && echo "  OK  (pytorch-attn fallbacks: $fb)" || echo "  FAILED rc=$rc"
}

run baseline-sage --chunk 4
run sol-tau0.8    --solattn 0.8
run sol-tau1.0    --solattn 1.0
run sol-tau1.5    --solattn 1.5

echo "=== $(date +%H:%M:%S)  done ==="
ls -lh output/solattn/*.mp4 2>/dev/null | awk '{print "  "$5, $9}'
