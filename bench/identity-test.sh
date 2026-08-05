#!/usr/bin/env bash
# Can reference conditioning give the ACTUAL car rather than a likeness?
#
# The user's verdict on the best r2v output so far: "it looks legit but it's not
# what my car looks like". That is the honest ceiling of attribute-carrying
# references, and this tests the cheapest lever that might raise it.
#
# WHY CROPPING MATTERS. Reference images are scaled before encoding:
#   match -> down to the generation's pixel area
#   max   -> to a 2048px short edge, never upscaled
# On the full 2252x4000 photo, `match` scales to 0.339, so most of the token
# budget is spent encoding dirt, sky and scrub. Cropping to the car spends the
# same budget on the car. Measured on these two references, cropped:
#   dirt-track  match 1225x841  vs  max 1689x1160  = 1.90x the pixels
#   lane        match 1227x840  vs  max 2162x1480  = 3.10x the pixels
#
# Arms, all else identical (124f native, seed 777, res_multistep 10, no cache):
#   A  full photo,  match   <- what the journey film effectively used
#   B  full photo,  max
#   C  cropped car, match
#   D  cropped car, max     <- the hypothesis: most identity tokens on the subject
#
# A separate script, deliberately: the last extension was appended to a RUNNING
# lever-queue.sh, which is the same mistake this project already documented
# against produce.sh. Bash reads scripts by byte offset; do not edit one mid-run.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f "[b]ench/gen.py" >/dev/null 2>&1; do sleep 20; done
sleep 5

COMMON=(--length 124 --sampler res_multistep --steps 10 --nocache --chunk 4 --tiled 32 --lowmem)
FULL=jf-car-dirt-track.png,jf-car-lane.png
CROP=jf-car-dirt-track-crop.png,jf-car-lane-crop.png

# A prompt that ASKS for the identity-bearing detail, rather than describing the
# car generically. If the references carry identity at all, naming what to look
# at is how it surfaces.
cat > bench/journey/identity-prompt.txt <<'EOF'
<Picture 1> and <Picture 2> show the same specific car from two angles. Reproduce that exact vehicle faithfully: its precise body shape and proportions, its silver-grey paint with the black contrast roof, the exact shape of its headlights and tail lights, the pattern of its alloy wheels, and the trim along its flank. That car stands on a wet cobbled street in an old town at dusk, warm shop windows glowing behind it. Camera: slow dolly along the car's flank from front three-quarter to rear, the car filling most of the frame throughout. Timing: reflected light sweeps along the bodywork at the halfway point. Style: cinematic, warm amber against cool blue dusk, shallow depth of field, wet reflections. Audio: light rain, distant footsteps on stone.
EOF

run () {
  local name=$1; shift
  ls output/identity/${name}_*.mp4 >/dev/null 2>&1 && { echo "[$name] exists, skipping"; return; }
  echo "=== $(date +%H:%M:%S)  $name  $* ==="
  : > bench/gen-server.log
  timeout 3600 bench/memguard.sh -- .venv/bin/python bench/gen.py \
    --prompt-file bench/journey/identity-prompt.txt \
    --name "identity/$name" --seed 777 "${COMMON[@]}" "$@"
  local rc=$?; local fb
  fb=$(grep -c "using pytorch attention instead" bench/gen-server.log 2>/dev/null | head -1)
  [[ "$fb" =~ ^[0-9]+$ ]] || fb=0
  [[ $rc -eq 0 ]] && echo "  OK$([[ $fb -gt 0 ]] && echo "  WARNING: $fb fallbacks")" \
                  || echo "  FAILED rc=$rc — continuing"
}

run A-full-match  --ref "$FULL" --ref-size match
run B-full-max    --ref "$FULL" --ref-size max
run C-crop-match  --ref "$CROP" --ref-size match
run D-crop-max    --ref "$CROP" --ref-size max

echo "=== $(date +%H:%M:%S)  identity test done ==="
ls -lh output/identity/*.mp4 2>/dev/null | awk '{print "  "$5, $9}'
