#!/usr/bin/env bash
# Overnight batch production. Renders every prompt in a file, unattended.
#
#   bench/produce.sh prompts.txt                 # native 15.1s, 20 real steps (~57 min each)
#   bench/produce.sh prompts.txt draft           # 960x544 15.1s (~10 min each)
#   bench/produce.sh prompts.txt fast            # native 15.1s, cache on (~30 min each)
#   bench/produce.sh prompts.txt draft-short     # 960x544 5.2s (~2 min each) — for iterating
#
# PROMPT FILE FORMAT: prompts separated by BLANK LINES. A prompt may span
# multiple lines. Lines starting with # are ignored.
#
# ONE SERVER PER CLIP, deliberately. Host RAM accumulates across prompts inside
# a single ComfyUI server and the third execution reliably thrashes on this 31 GB
# box — measured, twice. A ~90s boot per clip is cheap next to losing the batch
# at 3am.
#
# RESUMABLE: a prompt whose output already exists is skipped, so re-running after
# an interruption picks up where it stopped.
#
# A failure does not stop the batch. The next prompt still runs.
set -uo pipefail
cd "$(dirname "$0")/.."

PROMPTS="${1:-}"
PROFILE="${2:-native}"
PREFIX="${3:-batch}"   # subfolder under output/ — filename_prefix carries the path
[[ -f "$PROMPTS" ]] || { echo "usage: produce.sh <prompts-file> [native|fast|draft|draft-short] [output-subfolder]"; exit 64; }

case "$PROFILE" in
  native)      ARGS=(--length 362 --steps 20 --nocache --chunk 4 --tiled 32 --lowmem) ;;
  fast)        ARGS=(--length 362 --steps 20            --chunk 4 --tiled 32 --lowmem) ;;
  draft)       ARGS=(--length 362 --steps 20 --draft    --chunk 4 --tiled 32 --lowmem) ;;
  draft-short) ARGS=(--length 124 --steps 20 --draft) ;;
  *) echo "unknown profile: $PROFILE"; exit 64 ;;
esac

OUT=bench/h3-logs/produce-$(date +%Y%m%d-%H%M%S).log
mkdir -p bench/h3-logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Split on blank lines into one temp file per prompt.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
awk -v d="$TMP" '
  /^#/ {next}
  /^[[:space:]]*$/ { if (n>0 && len>0) {n++; len=0}; next }
  { if (n==0) n=1; print >> (d "/" sprintf("%03d", n) ".txt"); len++ }
' "$PROMPTS"
COUNT=$(ls "$TMP" 2>/dev/null | wc -l)
[[ "$COUNT" -gt 0 ]] || { echo "no prompts found in $PROMPTS"; exit 64; }

echo "=== $(date +%H:%M:%S)  batch start: $COUNT prompts, profile=$PROFILE ===" | tee -a "$OUT"
echo "    flags: ${ARGS[*]}" | tee -a "$OUT"

i=0; ok=0; fail=0
for f in "$TMP"/*.txt; do
  i=$((i+1))
  name="${PREFIX}/shot-$(printf '%03d' "$i")"
  if ls output/${name}_*.mp4 >/dev/null 2>&1; then
    echo "[$i/$COUNT] $name — already rendered, skipping" | tee -a "$OUT"; ok=$((ok+1)); continue
  fi
  # A prompt may open with directive lines, which are stripped before it is sent:
  #   @image: f.png        image-to-video, animate FROM input/f.png
  #   @ref: a.png,b.png    reference-to-video, carry those subjects into a NEW
  #                        scene. Address them in the prompt as <Picture 1>, ...
  #   @refsize: max        2048px references — better identity, much slower
  #   @chain               image-to-video seeded with the LAST FRAME of the
  #                        previous shot, so the two clips play as one take.
  #                        This is how a sequence runs past the model's 15.1s
  #                        ceiling; the seam is invisible because gen.py
  #                        reproduces its seed frame to within 2.7/255.
  IMG=""; REF=""; REFSIZE="match"; CHAIN=""
  while head -1 "$f" | grep -q '^@'; do
    D=$(head -1 "$f")
    case "$D" in
      @image:*)   IMG=$(echo "$D"   | sed 's/^@image:[[:space:]]*//') ;;
      @ref:*)     REF=$(echo "$D"   | sed 's/^@ref:[[:space:]]*//') ;;
      @refsize:*) REFSIZE=$(echo "$D" | sed 's/^@refsize:[[:space:]]*//') ;;
      @chain)     CHAIN=1 ;;
      *) echo "      unknown directive: $D — ignoring" | tee -a "$OUT" ;;
    esac
    tail -n +2 "$f" > "$f.p" && mv "$f.p" "$f"
  done
  # Resolve @chain against whatever the previous shot actually produced. If it
  # is missing (that shot failed), fall back to text-to-video rather than
  # aborting: a standalone clip beats a hole in the film.
  if [[ -n "$CHAIN" ]]; then
    PREV=$(ls -t output/${PREFIX}/shot-$(printf '%03d' $((i-1)))_*.mp4 2>/dev/null | head -1)
    if [[ -n "$PREV" ]] && bench/lastframe.sh "$PREV" "chain-$(printf '%03d' "$i")" >>"$OUT" 2>&1; then
      IMG="chain-$(printf '%03d' "$i").png"
    else
      echo "      @chain: shot $((i-1)) unavailable — rendering standalone" | tee -a "$OUT"
    fi
  fi
  MODE="t2v"; [[ -n "$IMG" ]] && MODE="i2v from $IMG"; [[ -n "$REF" ]] && MODE="r2v via $REF"
  echo "[$i/$COUNT] $(date +%H:%M:%S) $name  [$MODE] — $(head -c 55 "$f")..." | tee -a "$OUT"
  : > bench/gen-server.log
  timeout 7200 bench/memguard.sh -- .venv/bin/python bench/gen.py \
    --prompt-file "$f" --name "$name" --seed $((42 + i)) \
    ${IMG:+--first-frame "$IMG"} ${REF:+--ref "$REF" --ref-size "$REFSIZE"} "${ARGS[@]}" \
    >> "$OUT" 2>&1
  rc=$?
  # grep -c PRINTS "0" and EXITS 1 when there are no matches, so a `|| echo 0`
  # fires as well and yields "0\n0", which [[ -gt ]] cannot parse.
  fb=$(grep -c "using pytorch attention instead" bench/gen-server.log 2>/dev/null | head -1)
  [[ "$fb" =~ ^[0-9]+$ ]] || fb=0
  if [[ $rc -eq 0 ]] && ls output/${name}_*.mp4 >/dev/null 2>&1; then
    ok=$((ok+1)); echo "      OK$([[ $fb -gt 0 ]] && echo "  WARNING: $fb attention fallbacks — ran the slow path")" | tee -a "$OUT"
    # Re-cut the film after every shot. Costs a couple of seconds (stream copy,
    # no re-encode) and means there is always a current, watchable film on disk
    # rather than only after the last shot lands hours later.
    [[ $ok -ge 2 ]] && bench/assemble.sh "$PREFIX" >>"$OUT" 2>&1
  else
    fail=$((fail+1)); echo "      FAILED (rc=$rc) — continuing" | tee -a "$OUT"
  fi
done

echo "=== $(date +%H:%M:%S)  batch done: $ok rendered, $fail failed ===" | tee -a "$OUT"
ls -lh output/${PREFIX}/shot-*.mp4 2>/dev/null | awk '{print "  "$5, $9}' | tee -a "$OUT"
