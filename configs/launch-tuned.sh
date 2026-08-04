#!/usr/bin/env bash
# ComfyUI web UI with the measured-best flags for this box.
#   configs/launch-tuned.sh          -> http://127.0.0.1:8188
#
# expandable_segments is set here because it is free when unneeded and is
# REQUIRED for 15.1s clips at native (it reclaims ~2.4 GB stranded in allocator
# fragments). --disable-smart-memory is NOT set: it is needed for long clips but
# costs reloads on short ones, so add it only for a long session:
#   configs/launch-tuned.sh 8188 --disable-smart-memory --fp16-intermediates
#
# In the graph, for anything past ~13s at native, add the
# "SageAttention (chunked heads)" node with groups=4 between the model loader
# and the sampler, and swap VAEDecode for VAEDecodeTiled (temporal_size 32).
# Without those a long clip silently falls back to a ~3x slower attention path.
cd "$(dirname "$0")/.."
PORT="${1:-8188}"; shift 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python main.py --port "$PORT" --use-sage-attention --fast "$@"
