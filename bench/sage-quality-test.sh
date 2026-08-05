#!/usr/bin/env bash
# Arm D: is SageAttention's int8 QK path costing us quality on H3?
#
# comfy/ldm/minimax/model.py:181 calls optimized_attention WITHOUT
# low_precision_attention=False. A dozen other models in comfy/ldm pass that
# opt-out; H3 does not, so its DiT attention is quantized to int8. Upstream
# issue #15263 reports this produces pure noise. Ours is coherent, so we are not
# hitting the total failure -- but quality degradation is the same mechanism,
# further down the same road.
#
# This runs arm C's exact config with the opt-out applied, so C vs D isolates
# what the int8 path costs. Same prompt, same seed, same everything else.
#
# NOTE WHAT THE FIX COSTS: low_precision_attention=False routes the DiT to exact
# pytorch attention -- ~2.9x slower per call, measured on this box. SageAttention
# is the foundation of the 15.1s-at-native unlock (the whole head-chunking node
# exists to fit sage's int8 scratch). If D wins on quality, the machine's entire
# speed/quality trade has to be re-derived, not just tweaked.
#
# The patch is applied to a git-tracked file and reverted at the end, including
# on failure. Nothing is left modified.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

F=comfy/ldm/minimax/model.py
LINE='        out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)'
PATCHED='        out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, low_precision_attention=False, transformer_options=transformer_options)'

git diff --quiet -- "$F" || { echo "$F already modified — refusing to patch on top"; exit 1; }
grep -qF "$LINE" "$F" || { echo "expected line not found in $F — upstream may have changed it"; exit 1; }

restore () { git checkout -- "$F" && echo "  reverted $F"; }
trap restore EXIT

python3 - "$F" "$LINE" "$PATCHED" <<'PYEOF'
import sys
f,old,new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(f).read()
assert s.count(old) == 1, f"expected exactly one match, found {s.count(old)}"
open(f,'w').write(s.replace(old,new))
print(f"  patched {f}: low_precision_attention=False")
PYEOF

echo "=== $(date +%H:%M:%S)  D-exact-attn: res_multistep, 20 steps, exact attention ==="
: > bench/gen-server.log
bench/memguard.sh -- .venv/bin/python bench/gen.py \
  --prompt-file bench/journey/ab-prompt.txt --name "ab/D-exact-attn" --seed 777 \
  --length 124 --steps 20 --sampler res_multistep --nocache \
  --chunk 4 --tiled 32 --lowmem

echo "  pytorch-attention notices (EXPECTED to be >0 here — that is the point):"
grep -c "using pytorch attention instead" bench/gen-server.log
echo "=== $(date +%H:%M:%S)  done ==="
