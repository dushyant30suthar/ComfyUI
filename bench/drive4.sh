#!/usr/bin/env bash
# Unattended continuation for 2026-08-04, in value order. The GPUs are a serial
# resource, so everything that touches them is chained here rather than run
# concurrently; each stage is independently logged so a failure in one does not
# cost the others. Every heavy stage goes through memguard (rule 1).
#
#   bench/drive4.sh            # all stages
#   bench/drive4.sh q l        # only the named stages
cd "$(dirname "$0")/.."
LOGS=bench/h3-logs
mkdir -p "$LOGS"
GUARD=bench/memguard.sh
PY=.venv/bin/python
BASE="--base-flags=--use-sage-attention"

# shift the stage name off FIRST, then an empty remainder means "no selection
# given, run everything". Testing $# before the shift counts the stage name
# itself, so every want() returns false and the whole driver silently no-ops.
want() { local s="$1"; shift; [[ $# -eq 0 ]] && return 0; for a in "$@"; do [[ "$a" == "$s" ]] && return 0; done; return 1; }
stage() { echo; echo "=== $(date +%H:%M:%S)  $* ==="; }

SEL=("$@")

# 1. Attention at 15s shape. Cheap, and it decides whether sage3 is worth the
#    FP4 error at the length where attention is ~80% of the work.
if want attn "${SEL[@]}"; then
  stage "attn-micro @ 106,930 tokens (15.1s clip shape)"
  timeout 900 $GUARD -- $PY bench/attn-micro.py --tokens 106930 --iters 4 \
    > "$LOGS/attn-micro-15s.log" 2>&1
  grep -E "backend|sage|pytorch \(ref" "$LOGS/attn-micro-15s.log" || tail -3 "$LOGS/attn-micro-15s.log"
fi

# 2. sage3 end-to-end, now that the selector's recursion guard is fixed. K10/K11
#    also write real clips, which is what the FP4 error has to be judged on.
if want k "${SEL[@]}"; then
  stage "K10/K11 — sage3 alone, then sage3 + EasyCache 0.4"
  : > bench/sweep-server.log
  timeout 3600 $GUARD -- $PY bench/sweep.py K --rows K10,K11 "$BASE" \
    --length 124 --steps 10 > "$LOGS/phaseK-sage3.log" 2>&1
  tail -5 "$LOGS/phaseK-sage3.log"
fi

# 3. The two-GPU split. Success = Q1 ok AND gpu1 busy in the sampler; the
#    sampler here records utilisation so "both cards computing" is evidence,
#    not an assumption.
if want q "${SEL[@]}"; then
  stage "Q1 — pipeline split 22/28 + EasyCache 0.4 (T2 validation)"
  : > bench/sweep-server.log
  ( while true; do
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits >> "$LOGS/q1-gpu.csv"; sleep 2
    done ) & SAMP=$!
  timeout 2400 $GUARD -- $PY bench/sweep.py Q "$BASE" \
    --length 124 --steps 10 > "$LOGS/phaseQ-fixed.log" 2>&1
  kill $SAMP 2>/dev/null
  tail -4 "$LOGS/phaseQ-fixed.log"
  awk -F, '{gsub(/ /,"")} $1==0&&$2>50{a++} $1==1&&$2>50{b++} $1==0{n++} $1==1{m++}
    END{printf "gpu0 busy %d/%d samples, gpu1 busy %d/%d samples\n", a, n, b, m}' "$LOGS/q1-gpu.csv"
  grep -E "H3PipelineSplit|BufferError|Error" bench/sweep-server.log | tail -8
fi

# 4. What the ~84s of non-sampling time per clip actually is. N0-N1 isolates
#    VAE decode + mux from load/evict transitions; N2 confirms the slope.
if want n "${SEL[@]}"; then
  stage "N — fixed-overhead decomposition"
  : > bench/sweep-server.log
  timeout 2400 $GUARD -- $PY bench/sweep.py N "$BASE" --length 124 \
    > "$LOGS/phaseN.log" 2>&1
  tail -5 "$LOGS/phaseN.log"
fi

# 5. Resolution. The largest untested lever: tokens fall with W*H and attention
#    is quadratic in tokens, so this buys more than any flag — for a quality
#    price that has to be looked at, not inferred.
if want l "${SEL[@]}"; then
  stage "L — resolution scaling"
  : > bench/sweep-server.log
  timeout 3600 $GUARD -- $PY bench/sweep.py L --rows L1,L2,L3,L4 "$BASE" \
    --length 124 --steps 10 > "$LOGS/phaseL.log" 2>&1
  tail -6 "$LOGS/phaseL.log"
fi

stage "done — regenerating findings"
$PY bench/analyse.py 2>&1 | tail -5
