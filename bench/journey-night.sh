#!/usr/bin/env bash
# Render the 12-shot Udaipur -> Himachal film overnight, then assemble it.
#
# This version uses all three generation modes together, which is the point:
#   t2v   for places we have no photograph of
#   i2v   where a real photograph IS the shot (frame 0 is the photo itself)
#   r2v   to put the real car somewhere we have no photograph of
#   @chain to join clip pairs into continuous 30s takes past the 15.1s ceiling
#
# Shot 2 (the first r2v clip) is rendered separately and already in flight when
# this starts — it doubled as the proof that reference tokens fit alongside a
# maxed 362-frame clip. produce.sh skips any shot whose output already exists,
# so this picks up around it.
#
# Profile `fast`: native 1344x768, 362 frames, EasyCache halving the steps.
# The resolution ladder showed canvas dominates perceived quality and step
# count is second, so native shots beat more numerous draft ones.
#
# Assembly runs after EVERY shot from the second onward, so there is always a
# current, watchable film on disk — not only if the batch reaches the end.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=bench/h3-logs/journey-night.log

# Wait out any clip still rendering; two ComfyUI instances need ~43 GB against
# this box's 31 GB and would thrash. The bracket in the pattern keeps pgrep
# from matching this script's own command line.
while pgrep -f "[b]ench/gen.py" >/dev/null 2>&1; do sleep 30; done

echo "=== $(date +%H:%M:%S)  Udaipur -> Himachal, 12 shots, profile=fast ===" | tee -a "$LOG"
bench/produce.sh bench/journey-prompts.txt fast journey 2>&1 | tee -a "$LOG"

echo "=== $(date +%H:%M:%S)  final assembly ===" | tee -a "$LOG"
bench/assemble.sh journey 2>&1 | tee -a "$LOG"
echo "=== $(date +%H:%M:%S)  done ===" | tee -a "$LOG"
