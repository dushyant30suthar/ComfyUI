#!/usr/bin/env bash
# Render the 12-shot Udaipur -> Himachal film, then assemble it. Unattended.
#
# Profile choice, from today's measurements:
#   native 56.7 min/clip -> 11.3 h for 12 shots. Too long for one night.
#   fast   29.6 min/clip ->  5.9 h.              <- chosen
#   draft  10.3 min/clip ->  2.1 h.
# `fast` keeps native 1344x768 and lets EasyCache halve the steps. The
# resolution ladder showed canvas dominates perceived quality and step count is
# second, so 12 native shots beats 24 draft ones for something meant to be
# watched end to end.
#
# Assembly runs even if shots failed — a 9-shot film beats no film. It also runs
# after each shot from shot 3 onward, so there is always a current cut on disk
# rather than only at the end.
cd "$(dirname "$0")/.."
LOG=bench/h3-logs/journey-night.log

echo "=== $(date +%H:%M:%S)  Udaipur -> Himachal, 12 shots, profile=fast ===" | tee -a "$LOG"
bench/produce.sh bench/journey-prompts.txt fast journey 2>&1 | tee -a "$LOG"

echo "=== $(date +%H:%M:%S)  assembling ===" | tee -a "$LOG"
bench/assemble.sh journey 2>&1 | tee -a "$LOG"
echo "=== $(date +%H:%M:%S)  done ===" | tee -a "$LOG"
