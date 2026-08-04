#!/usr/bin/env bash
# Wait for the in-flight 15s validation to finish, then generate the user's clip
# BEFORE the remaining benchmark phases, then let those resume.
#
# The 15s row is the last thing in drive5 the user is waiting on results for; H
# and C are low-value singles that can wait. So: watch for drive5 to move past
# the 15s stage, stop it there, run the clip, then relaunch it with just h and c.
cd "$(dirname "$0")/.."
LOG=bench/h3-logs/drive5g.log
PROMPT="$1"
[[ -f "$PROMPT" ]] || { echo "usage: chain-gen.sh <prompt-file>"; exit 64; }

# The 15s stage is done once drive5 announces the next one (H).
while ! grep -q "H — encoder & VAE precision" "$LOG" 2>/dev/null; do
  # ...or once drive5 exits altogether, in which case nothing to interrupt.
  pgrep -f "driv""e5.sh" >/dev/null 2>&1 || break
  sleep 15
done

# Pattern assembled at runtime so pkill cannot match this script's own cmdline.
PAT="driv""e5.sh"
pkill -f "$PAT" 2>/dev/null
sleep 2
pkill -f "memg""uard.sh" 2>/dev/null
sleep 1
for p in $(pgrep -f "ma""in.py --po""rt"); do kill -TERM "$p" 2>/dev/null; done
sleep 2
pkill -f "swe""ep.py" 2>/dev/null
sleep 5

echo "=== $(date +%H:%M:%S)  generating the requested clip ==="
bench/memguard.sh -- .venv/bin/python bench/gen.py \
  --prompt-file "$PROMPT" --name incident --seed 42
rc=$?
echo "=== $(date +%H:%M:%S)  clip done (rc=$rc); resuming benchmark queue ==="
exec bench/drive5.sh h c
