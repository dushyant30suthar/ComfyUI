#!/usr/bin/env bash
# Concatenate the rendered shots, in order, into one film.
#
#   bench/assemble.sh journey            # -> output/journey/JOURNEY.mp4
#
# Uses ffmpeg's concat demuxer with stream copy: no re-encode, so no generation
# loss and it takes seconds. Every clip comes from the same pipeline at the same
# resolution, framerate and codec, which is the condition concat requires.
#
# MISSING SHOTS ARE SKIPPED, not fatal. A batch that renders 9 of 12 still
# assembles into a 9-shot film, in the right order, with the gaps simply absent.
# Better to wake up to something watchable than to nothing.
set -uo pipefail
cd "$(dirname "$0")/.."

DIR="${1:-journey}"
SRC="output/$DIR"
OUT="$SRC/${DIR^^}.mp4"
[[ -d "$SRC" ]] || { echo "no such folder: $SRC"; exit 64; }

LIST=$(mktemp); trap 'rm -f "$LIST"' EXIT
n=0
for f in $(ls "$SRC"/shot-*.mp4 2>/dev/null | sort); do
  echo "file '$(realpath "$f")'" >> "$LIST"
  n=$((n+1))
done
[[ "$n" -gt 0 ]] || { echo "no shots in $SRC"; exit 1; }

echo "assembling $n shots from $SRC"
ffmpeg -v error -f concat -safe 0 -i "$LIST" -c copy "$OUT" -y || {
  # Fall back to a re-encode if stream copy refuses (mismatched parameters).
  echo "  stream copy failed — re-encoding"
  ffmpeg -v error -f concat -safe 0 -i "$LIST" \
    -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k "$OUT" -y
}

dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUT")
printf 'done: %s\n  %d shots, %.1f s (%.1f min)\n' "$OUT" "$n" "$dur" "$(echo "$dur/60" | bc -l)"
ls -lh "$OUT" | awk '{print "  "$5}'
