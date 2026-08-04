#!/usr/bin/env bash
# Overnight batch, 2026-08-04. Runs unattended; nothing here needs a human.
#
# Ordered by what is actually still open. The benchmark matrix is spent — phase
# H was the last lever aimed at something real and returned 0.8% (noise), so
# what remains is the two-GPU work and one completeness sweep.
#
# SAFETY. Everything goes through memguard, which is now calibrated on eight
# recorded events (duration + total, never an instantaneous reading, and it
# type-checks /proc/meminfo before comparing). --novram stays banned and phase
# C's C12 (--high-ram) is excluded: 21 GB DiT + 15.7 GB encoder against 31 GB is
# the arithmetic that froze this box once already.
cd "$(dirname "$0")/.."
LOGS=bench/h3-logs
mkdir -p "$LOGS"
G=bench/memguard.sh
PY=.venv/bin/python
BASE="--base-flags=--use-sage-attention --fast"
stage() { echo; echo "=== $(date +%H:%M:%S)  $* ==="; }

# 1. The split, with the prefetch guard now scoped to cuda:1's blocks instead of
#    the whole model. Phase R found the penalty FLAT across cuts (284/276/277s
#    at 25/28/31) — a cut-independent cost, which is the shape of losing
#    prefetch everywhere rather than a memory-balance problem. R2 at cut 28 is
#    the comparison: beat 276.2s and the guard was the cause; match it and the
#    penalty is structural and the split is finished as a latency play.
stage "R2 re-run — split 28/22 with the prefetch guard scoped"
: > bench/sweep-server.log
timeout 2400 $G -- $PY bench/sweep.py R --rows R2 "$BASE" --length 124 --steps 10 \
  > "$LOGS/night-R2-scoped.log" 2>&1
tail -3 "$LOGS/night-R2-scoped.log"
grep -E "H3PipelineSplit\] (armed|vbar-rebound)" bench/sweep-server.log | tail -2

# 2. Phase D showed placement frees 4 GB on cuda:0 and gains nothing, because
#    the allocator does not expand into vacated space. --reserve-vram is the
#    only lever that might make it claim that space. Untested combination; in no
#    phase. If it works it is worth the ~50s offload tax.
stage "D1 + --reserve-vram 0.5 — does the allocator claim freed space?"
: > bench/sweep-server.log
timeout 2400 $G -- $PY bench/sweep.py D --rows D1 "$BASE --reserve-vram 0.5" \
  --length 124 --steps 10 > "$LOGS/night-D1-reserve.log" 2>&1
tail -3 "$LOGS/night-D1-reserve.log"

# 3. Completeness only. Phase N put 73.3s of the ~84s fixed cost in the VAE
#    decode, so C's offload/memory flags act on a term that barely exists. Run
#    it so the matrix is finished, not because anything is expected.
stage "C — offload & memory flags (12 rows; C12 --high-ram excluded as unsafe)"
: > bench/sweep-server.log
timeout 9000 $G -- $PY bench/sweep.py C --rows C0,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11 \
  "$BASE" --length 124 --steps 10 > "$LOGS/night-C.log" 2>&1
tail -16 "$LOGS/night-C.log"

stage "regenerating findings"
$PY bench/analyse.py 2>&1 | tail -4
stage "overnight batch complete"
