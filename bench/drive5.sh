#!/usr/bin/env bash
# Second unattended batch. Run AFTER drive4.sh has exited — the GPUs are serial
# and two sweeps would fight over both the cards and port 8191.
#
# Ordered by expected value per GPU-hour:
#   D  device placement — the idle card has 15.5 GB free while the 15.7 GB text
#      encoder and 5 GB VAE contend with the DiT for cuda:0. Costs no quality,
#      and it attacks BOTH terms: less contention means less per-step spill, and
#      not evicting/reloading those models between stages is exactly the 83.7 s
#      of non-sampling time. Never run at clip length; the highest-value gap.
#   H  encoder & VAE precision — the other angle on the same fixed cost.
#   C  offload & memory flags — 13 singles, slowest, run last.
#
#   bench/drive5.sh          # all
#   bench/drive5.sh d h      # selected stages
cd "$(dirname "$0")/.."
LOGS=bench/h3-logs
mkdir -p "$LOGS"
GUARD=bench/memguard.sh
PY=.venv/bin/python
BASE="--base-flags=--use-sage-attention --fast"

want() { local s="$1"; shift; [[ $# -eq 0 ]] && return 0; for a in "$@"; do [[ "$a" == "$s" ]] && return 0; done; return 1; }
stage() { echo; echo "=== $(date +%H:%M:%S)  $* ==="; }
SEL=("$@")

# The 15.1s attention shape, retried now that the reference is kept as a token
# slice rather than a full fp32 copy (the first attempt OOMed a 15.5 GiB card
# after producing the torch number).
if want attn "${SEL[@]}"; then
  stage "attn-micro @ 106,930 tokens (15.1s shape), fixed memory handling"
  timeout 900 $GUARD -- $PY bench/attn-micro.py --tokens 106930 --iters 3 \
    > "$LOGS/attn-micro-15s.log" 2>&1
  grep -E "backend|sage|pytorch \(ref|tokens=" "$LOGS/attn-micro-15s.log" || tail -3 "$LOGS/attn-micro-15s.log"
fi

# K11 was killed at its VAE decode by the watchdog's old hard floor, with swap
# grown only 173 MiB. It is the one row that says whether sage3 and EasyCache
# actually stack — the log says the cache skips 3/10 steps under sage3 vs 4/10
# under sage2, which would make the pair a wash.
if want k11 "${SEL[@]}"; then
  stage "K11 re-run — sage3 + EasyCache 0.4"
  : > bench/sweep-server.log
  timeout 2400 $GUARD -- $PY bench/sweep.py K --rows K11 "$BASE" \
    --length 124 --steps 10 > "$LOGS/phaseK11.log" 2>&1
  tail -3 "$LOGS/phaseK11.log"
  grep "EasyCache - skipped" bench/sweep-server.log | tail -1
fi

# Q1 sampled 10/10 with both cards alternating and no BufferError, then the
# watchdog's old hard floor killed it at the VAE decode (swap had actually
# FALLEN 67 MiB), so no record was written. Re-run under the corrected rules to
# get the number.
if want q "${SEL[@]}"; then
  stage "Q1 re-run — pipeline split 22/28, for the record"
  : > bench/sweep-server.log
  rm -f "$LOGS/q1-gpu.csv"
  ( while true; do
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits >> "$LOGS/q1-gpu.csv"; sleep 2
    done ) & SAMP=$!
  timeout 2400 $GUARD -- $PY bench/sweep.py Q "$BASE" --length 124 --steps 10 \
    > "$LOGS/phaseQ-rerun.log" 2>&1
  kill $SAMP 2>/dev/null
  tail -3 "$LOGS/phaseQ-rerun.log"
  awk -F, '{gsub(/ /,"")} $1==0{n++; if($2>50)a++} $1==1{m++; if($2>50)b++}
    END{printf "gpu0 busy %d/%d (%.0f%%)  gpu1 busy %d/%d (%.0f%%)\n",a,n,100*a/n,b,m,100*b/m}' \
    "$LOGS/q1-gpu.csv"
fi

# Phase N, second attempt. The first run's N0 landed at 112.2s against a
# predicted 83.7 + 28.3 = 112.0 (the linear model is sound), but N1 died on
# SaveLatent — H3's sampler emits a packed audio+video NestedTensor — and would
# have reused N0's cached sampler output anyway. Both fixed in matrix.py.
if want n "${SEL[@]}"; then
  stage "N re-run — fixed-overhead decomposition"
  : > bench/sweep-server.log
  timeout 2400 $GUARD -- $PY bench/sweep.py N "$BASE" --length 124 \
    > "$LOGS/phaseN-fixed.log" 2>&1
  tail -5 "$LOGS/phaseN-fixed.log"
fi

# ONE SERVER PER ROW, deliberately. A workflow phase normally amortises the
# server across its rows, but on this 31 GB box host RAM accumulates across
# prompts within a server and the third execution reliably thrashes: phase N was
# killed at its third row on BOTH attempts, having completed rows 1 and 2 each
# time. D's decisive comparison is D0 (everything on cuda:0) against D1
# (encoder+VAE on the idle card) — rows 1 and 2 — so sharing a server would risk
# exactly the measurement this phase exists for. A fresh server per row costs a
# ~90 s boot each and buys a clean memory baseline; it also makes each row
# independently cold, so no --warmup is needed or wanted.
if want d "${SEL[@]}"; then
  for row in D0 D1 D2 D3 D4; do
    stage "D/$row — device placement"
    : > bench/sweep-server.log
    timeout 2400 $GUARD -- $PY bench/sweep.py D --rows $row "$BASE" \
      --length 124 --steps 10 > "$LOGS/phaseD-$row.log" 2>&1
    tail -3 "$LOGS/phaseD-$row.log"
  done
fi

# Split balance, AFTER phase D: Q1 proved the split runs, but at 22/28 it made a
# single clip slower, and cuda:1 sat at 15.65 GB of 15.5 GiB usable — full
# enough that its arena must evict and re-fault, reintroducing the host-RAM
# streaming the split exists to remove. Each row needs its OWN server: applying
# H3PipelineSplit to a model ComfyUI still has cached would rebind blocks that
# are already rebound.
if want r "${SEL[@]}"; then
  for row in R1 R2 R3 R4; do
    stage "R/$row — split balance"
    : > bench/sweep-server.log
    timeout 2400 $GUARD -- $PY bench/sweep.py R --rows $row "$BASE" \
      --length 124 --steps 10 > "$LOGS/phase-$row.log" 2>&1
    tail -3 "$LOGS/phase-$row.log"
    grep -E "H3PipelineSplit\] vbar-rebound" bench/sweep-server.log | tail -1
  done
fi

# L re-run WITH warmup. drive4 ran L before the first-run penalty was known, so
# its L1 row absorbed ~24s of checkpoint staging and reads ~24s slower than it
# is — which matters because L's whole purpose is ranking resolutions against
# each other. L0 (native) is not re-run here: it is the same geometry as the
# K rows and is already measured warm.
# Also one server per row, same reason as D. (The drive4 L rows DID all complete
# on one server — smaller canvases hold less — but the first-row penalty landed
# on L1 there, and a fresh server per row fixes that too: every row is equally
# cold, which is how the launch phases have always been fair.)
if want l "${SEL[@]}"; then
  for row in L1 L2 L3 L4; do
    stage "L/$row — resolution scaling, own server"
    : > bench/sweep-server.log
    timeout 2400 $GUARD -- $PY bench/sweep.py L --rows $row "$BASE" \
      --length 124 --steps 10 > "$LOGS/phaseL-$row.log" 2>&1
    tail -3 "$LOGS/phaseL-$row.log"
  done
fi

# Three of the biggest remaining wins (sage3 FP4, EasyCache 0.6, 960x544) are
# blocked on the same unanswerable-by-timing question. One prompt, one seed,
# one server, six renders — after which they can be adopted or dropped on
# evidence instead of staying permanently "measured but unjudged".
if want quality "${SEL[@]}"; then
  stage "quality — same prompt/seed under each quality-gated config"
  timeout 7200 $GUARD -- $PY bench/quality.py > "$LOGS/quality.log" 2>&1
  tail -12 "$LOGS/quality.log"
fi

# The 15.1s clip is the actual production target and has NEVER been run — every
# 15s figure in the docs is extrapolated. At 10 steps this costs about half a
# full-quality run while still answering the two questions that matter: does the
# per-step cost follow the (now verified) quadratic model, and does the VAE
# decode OOM at 362 frames. If it OOMs, the remedies to try are --reserve-vram
# 2.0 (costs resident weights) and expandable_segments (may cost nothing).
if want 15s "${SEL[@]}"; then
  stage "15s validation — 362 frames, 10 steps, tuned config"
  : > bench/sweep-server.log
  timeout 5400 $GUARD -- $PY bench/sweep.py K --rows K2 "$BASE" \
    --length 362 --steps 10 > "$LOGS/phase15s.log" 2>&1
  tail -4 "$LOGS/phase15s.log"
  grep -iE "out of memory|OutOfMemory|allocat" bench/sweep-server.log | tail -3
fi

if want h "${SEL[@]}"; then
  stage "H — encoder & VAE precision"
  : > bench/sweep-server.log
  timeout 5400 $GUARD -- $PY bench/sweep.py H "$BASE" --warmup --length 124 --steps 10 \
    > "$LOGS/phaseH.log" 2>&1
  tail -8 "$LOGS/phaseH.log"
fi

# C12 (--high-ram) is deliberately EXCLUDED. It asks ComfyUI to keep more in
# host RAM, and the arithmetic is the one that got --novram banned: 21 GB DiT +
# 15.7 GB encoder against 31 GB. The watchdog would very likely catch it — it
# fired correctly on the sage3+cache row at MemAvailable=125 MiB — but the guard
# is a last resort, not a plan, and this box has already needed one hard reset.
# Run it deliberately and attended if the number is ever wanted.
if want c "${SEL[@]}"; then
  stage "C — offload & memory flags (12 rows, restart each; C12 --high-ram skipped as unsafe)"
  : > bench/sweep-server.log
  timeout 9000 $GUARD -- $PY bench/sweep.py C \
    --rows C0,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11 \
    "$BASE" --length 124 --steps 10 > "$LOGS/phaseC.log" 2>&1
  tail -16 "$LOGS/phaseC.log"
fi

stage "done — regenerating findings"
$PY bench/analyse.py 2>&1 | tail -5
