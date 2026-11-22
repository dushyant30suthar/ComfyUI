#!/usr/bin/env python3
"""Time the attention backends at H3's REAL shape, without generating a clip.

    .venv/bin/python bench/attn-micro.py [--tokens 37710]

Why this exists: a full sweep row costs ~4 minutes and buries the attention
kernel under model loading, 49 other blocks, VAE decode and mux. Attention is
58% of FLOPs at a 5.2s clip and ~80% at 15s, so it deserves to be measured on
its own — and a backend that quantises V (sage3, FP4) needs its error measured
next to its speed, in the same run, or the clock alone will sell a bad trade.

Shape comes from comfy/ldm/minimax/model.py: hidden 5376 = 56 heads x 128 dim,
one packed sequence per forward (video + audio + text), 50 blocks. Token counts
match bench/matrix.tokens(): 37,710 at 5.2s and 106,930 at 15.1s, both 1344x768.

Reference is attention_pytorch in fp32 accumulation — not another approximation.
"""
import argparse, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# Parse BEFORE clearing sys.argv: comfy.cli_args parses the real command line at
# import time and rejects flags it does not know, so the line has to be emptied
# for it — but emptying it first silently discards these flags and the script
# then benchmarks the default shape while claiming it ran the requested one.
_ap = argparse.ArgumentParser()
_ap.add_argument("--tokens", type=int, default=37710,
                 help="packed sequence length (37710 = 5.2s clip, 106930 = 15.1s)")
_ap.add_argument("--iters", type=int, default=8)
ARGS = _ap.parse_args()
sys.argv = [sys.argv[0]]

from comfy.ldm.modules.attention import (  # noqa: E402
    get_attention_function, attention_pytorch,
)

HEADS, DIM = 56, 128


def bench(fn, q, k, v, iters, warmup=2):
    for _ in range(warmup):
        fn(q, k, v, HEADS)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(q, k, v, HEADS)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters, out


def main():
    args = ARGS
    n = args.tokens
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    # skip_reshape layout: (B, H, N, D) — what the DiT hands the backend.
    shape = (1, HEADS, n, DIM)
    q, k, v = (torch.randn(shape, dtype=torch.bfloat16, device=dev) / DIM ** 0.25
               for _ in range(3))

    gb = 3 * q.numel() * q.element_size() / 1024 ** 3
    print(f"tokens={n:,}  heads={HEADS} dim={DIM}  qkv={gb:.2f} GB  iters={args.iters}\n")

    def call(fname):
        f = get_attention_function(fname, None)
        if f is None:
            return None
        # _inside_attn_wrapper keeps wrap_attn from consulting any override and
        # is the guard that must survive into internal fallbacks; see
        # custom_nodes/sage3_select.
        return lambda q, k, v, h: f(q, k, v, h, skip_reshape=True,
                                    _inside_attn_wrapper=True)

    # Keep only a token slice of the reference in fp32. A full fp32 copy is 2x
    # the bf16 output — at the 15.1s shape that is 3.1 GB on top of 4.6 GB of
    # q/k/v, and the next backend's own output then OOMs a 15.5 GiB card. The
    # slice is plenty to characterise elementwise quantisation error.
    ERR_TOK = 4096

    ref_t, ref = bench(lambda q, k, v, h: attention_pytorch(
        q, k, v, h, skip_reshape=True, _inside_attn_wrapper=True),
        q, k, v, args.iters)
    ref32 = ref[:, :, :ERR_TOK].float().clone()
    del ref
    torch.cuda.empty_cache()

    print(f"{'backend':22} {'ms/call':>9} {'vs torch':>9}   error vs torch SDPA")
    print(f"{'pytorch (reference)':22} {ref_t*1000:>9.1f} {'—':>9}   —")

    for name, label in (("sage", "sage2 (INT8 QK)"), ("sage3", "sage3 (FP4 QKV)")):
        f = call(name)
        if f is None:
            print(f"{label:22} {'not registered':>9}")
            continue
        try:
            t, out = bench(f, q, k, v, args.iters)
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"{label:22} FAILED: {str(e)[:70]}")
            continue
        d = (out[:, :, :ERR_TOK].float() - ref32).abs()
        rel = d.mean().item() / ref32.std().item()
        print(f"{label:22} {t*1000:>9.1f} {(t/ref_t-1)*100:>+8.1f}%   "
              f"mean |err| {d.mean().item():.5f} = {rel*100:.1f}% of signal sd, "
              f"max {d.max().item():.4f}")
        del out, d
        torch.cuda.empty_cache()

    print("\nNote: random gaussian q/k/v is the pessimal case for a quantised "
          "backend — real activations have structure to exploit. Treat the error "
          "column as an upper bound and confirm on real clips before adopting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
