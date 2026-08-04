#!/usr/bin/env bash
# Extract the last frame of a rendered clip into input/, so the next shot can
# start exactly where this one ended.
#
#   bench/lastframe.sh output/journey/shot-002_00001.mp4 chain-003
#     -> input/chain-003.png
#
# This is what makes clips joinable past the model's 15.1s ceiling: gen.py's
# --first-frame reproduces its seed image at frame 0 to within 2.7/255, so a
# cut on that boundary is invisible and two clips read as one 30s take.
#
# Uses the SECOND-to-last frame deliberately. The final frame of a VAE-decoded
# clip is the most likely to carry temporal-tile edge artifacts, and any
# artifact here is inherited as the seed of the whole next clip.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:?usage: lastframe.sh <video> <output-name>}"
NAME="${2:?usage: lastframe.sh <video> <output-name>}"
[[ -f "$SRC" ]] || { echo "no such video: $SRC"; exit 64; }

N=$(ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$SRC")
[[ "$N" =~ ^[0-9]+$ && "$N" -gt 2 ]] || { echo "cannot read frame count from $SRC"; exit 1; }

ffmpeg -v error -i "$SRC" -vf "select=eq(n\,$((N-2)))" -vframes 1 \
       -y "input/${NAME}.png"

SZ=$(identify -format "%wx%h" "input/${NAME}.png" 2>/dev/null || echo "?")
echo "input/${NAME}.png  (frame $((N-2)) of $N, $SZ)"
