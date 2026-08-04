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
  echo "[$i/$COUNT] $(date +%H:%M:%S) $name — $(head -c 70 "$f")..." | tee -a "$OUT"
  : > bench/gen-server.log
  timeout 7200 bench/memguard.sh -- .venv/bin/python bench/gen.py \
    --prompt-file "$f" --name "$name" --seed $((42 + i)) "${ARGS[@]}" \
    >> "$OUT" 2>&1
  rc=$?
  fb=$(grep -c "using pytorch attention instead" bench/gen-server.log 2>/dev/null || echo 0)
  if [[ $rc -eq 0 ]] && ls output/${name}_*.mp4 >/dev/null 2>&1; then
    ok=$((ok+1)); echo "      OK$([[ $fb -gt 0 ]] && echo "  WARNING: $fb attention fallbacks — ran the slow path")" | tee -a "$OUT"
  else
    fail=$((fail+1)); echo "      FAILED (rc=$rc) — continuing" | tee -a "$OUT"
  fi
done

echo "=== $(date +%H:%M:%S)  batch done: $ok rendered, $fail failed ===" | tee -a "$OUT"
ls -lh output/${PREFIX}/shot-*.mp4 2>/dev/null | awk '{print "  "$5, $9}' | tee -a "$OUT"
