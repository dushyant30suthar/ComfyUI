#!/usr/bin/env bash
# Work the remaining cheap levers from OPEN-LEVERS.md back to back, so the box
# never idles between them. Waits for any render already in flight first.
#
# All arms: 124 frames at native 1344x768, seed 777, res_multistep 10 steps, no
# cache -- the config settled by the A/B and the scheduler sweep. Only the one
# variable under test moves.
#
#   ref-size    match vs max, same r2v prompt and refs. Targets a MEASURED
#               failure: reference identity survives when the subject fills the
#               frame and collapses to a generic silhouette when it is distant.
#               `max` uses a 2048px short edge instead of scaling refs to the
#               canvas. CLAIMED "several times slower" by the node tooltip --
#               that claim is the second thing this measures.
#
#   cfg         1.0 (control) vs 2.0 vs 3.0. H3 is guidance-distilled, so cfg>1
#               is expected to cost double for nothing. That is a reasonable
#               inference which has NEVER been measured here, and "reasonable
#               inference, never measured" is precisely the category that cost
#               this project two days on euler.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f "[b]ench/gen.py" >/dev/null 2>&1; do sleep 20; done
sleep 5

COMMON=(--length 124 --sampler res_multistep --steps 10 --nocache --chunk 4 --tiled 32 --lowmem)

cat > bench/journey/refsize-test.txt <<'EOF'
<Picture 1> and <Picture 2> show the same car: a silver-grey Tata Altroz hatchback with a black contrast roof. That exact car is parked on a wet cobbled street in a European old town at dusk, warm shop windows glowing behind it, the cobbles reflecting the light. Camera: slow low dolly along the car's flank from front three-quarter to rear. Timing: a tram passes in the background at the halfway point. Style: cinematic, warm amber against cool blue dusk, shallow depth of field, wet reflections. Audio: distant tram bell, footsteps on stone, light rain.
EOF

run () {  # name, then extra flags
  local name=$1; shift
  ls output/levers/${name}_*.mp4 >/dev/null 2>&1 && { echo "[$name] done already, skipping"; return; }
  echo "=== $(date +%H:%M:%S)  $name  $* ==="
  : > bench/gen-server.log
  timeout 3600 bench/memguard.sh -- .venv/bin/python bench/gen.py \
    --name "levers/$name" --seed 777 "${COMMON[@]}" "$@"
  local rc=$?
  local fb; fb=$(grep -c "using pytorch attention instead" bench/gen-server.log 2>/dev/null | head -1)
  [[ "$fb" =~ ^[0-9]+$ ]] || fb=0
  local oom; oom=$(grep -c "out of memory" bench/gen-server.log 2>/dev/null | head -1)
  [[ "$oom" =~ ^[0-9]+$ ]] || oom=0
  if [[ $rc -eq 0 ]]; then
    echo "  OK$([[ $fb -gt 0 ]] && echo "  WARNING: $fb attention fallbacks")"
  else
    echo "  FAILED rc=$rc  (oom lines: $oom) — continuing"
  fi
}

REFS=j-car-dirt-track.png,j-car-lane.png
run refsize-match --prompt-file bench/journey/refsize-test.txt --ref "$REFS" --ref-size match
run refsize-max   --prompt-file bench/journey/refsize-test.txt --ref "$REFS" --ref-size max

run cfg-1.0 --prompt-file bench/journey/ab-prompt.txt
run cfg-2.0 --prompt-file bench/journey/ab-prompt.txt --cfg 2.0
run cfg-3.0 --prompt-file bench/journey/ab-prompt.txt --cfg 3.0

echo "=== $(date +%H:%M:%S)  queue done ==="
ls -lh output/levers/*.mp4 2>/dev/null | awk '{print "  "$5, $9}'
