#!/usr/bin/env bash
# Wait for drive4 to finish, then start drive5. The GPUs are serial: two sweeps
# would fight over the cards and over port 8191.
#
# The pattern is assembled at runtime so pgrep cannot match this script's own
# command line — matching it would make the wait loop see "still running"
# forever (and a pkill on the same literal would kill the shell issuing it).
cd "$(dirname "$0")/.."
PAT="driv""e4.sh"
while pgrep -f "$PAT" > /dev/null 2>&1; do sleep 30; done
sleep 20                     # let the last CUDA context hand VRAM back
exec bench/drive5.sh
