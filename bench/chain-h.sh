#!/usr/bin/env bash
# Wait for the in-flight gen.py clip, then run phase H.
#
# Phase H is the last benchmark lever pointed at something real. Phase N showed
# 73.3 s of the ~84 s fixed per-clip cost is the VAE decode + mux, and H is the
# phase that acts on it: --bf16-vae, --fp16-intermediates, --cpu-vae,
# --fp8_e4m3fn-text-enc. Phase C, by contrast, targets load/evict transitions,
# which N showed are almost nothing.
#
# H is a LAUNCH phase — it restarts the server per row — so it is immune to both
# harness biases found today (the ~24 s first-row penalty and the three-rows-per-
# server host-RAM limit). No --warmup needed.
cd "$(dirname "$0")/.."
LOGS=bench/h3-logs
PAT="ge""n.py"
while pgrep -f "$PAT" > /dev/null 2>&1; do sleep 30; done
sleep 20                       # let the CUDA context hand VRAM back

echo "=== $(date +%H:%M:%S)  phase H — encoder & VAE precision ==="
: > bench/sweep-server.log
timeout 5400 bench/memguard.sh -- .venv/bin/python bench/sweep.py H \
  "--base-flags=--use-sage-attention --fast" --length 124 --steps 10 \
  > "$LOGS/phaseH.log" 2>&1
tail -10 "$LOGS/phaseH.log"

echo "=== $(date +%H:%M:%S)  regenerating findings ==="
.venv/bin/python bench/analyse.py 2>&1 | tail -4
