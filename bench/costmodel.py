#!/usr/bin/env python3
"""Predict a run's server time from geometry, and flag where reality disagrees.

    .venv/bin/python bench/costmodel.py            # every measured row
    .venv/bin/python bench/costmodel.py L K        # selected phases

The model, entirely from measurements on this box:

    server_s  =  FIXED  +  real_steps x step(tokens)
    step(n)   =  ATTN_FRAC x base_step x (n/n0)^2  +  (1-ATTN_FRAC) x base_step x (n/n0)

Attention is quadratic in sequence length and everything else (projections, FFN,
norms) is linear, so a geometry change moves the two terms by different amounts
and a single scale factor cannot describe it.

Constants, and where each comes from:
  BASE_STEP  28.3 s   (366.7 - 112.2)/9 — K0 at 10 steps vs N0 at 1 step, both
                      the first row on a fresh server so the constant cancels
  FIXED      84.0 s   N0 - BASE_STEP; VAE decode, mux, load/evict transitions
  ATTN_FRAC  0.54     50 blocks x 304 ms sage2 (bench/attn-micro.py) / 28.3 s,
                      which independently matches the 58%-of-FLOPs estimate
  N0_TOKENS  37,710   matrix.tokens(1344, 768, 124)

The quadratic term is verified: 2.836x the tokens cost 8.03x the time against a
predicted 8.04x. What this script is FOR is the residual — a row that misses its
prediction is either a knob doing something the model does not know about, or a
measurement artefact (the ~24s first-row penalty being the usual culprit).
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matrix  # noqa: E402

RESULTS = os.path.join(HERE, "results.jsonl")

BASE_STEP = 28.3
FIXED = 84.0
ATTN_FRAC = 0.54
N0_TOKENS = 37710
FIRST_ROW_PENALTY = 24.0

# The constants above were fitted to rows launched with `--use-sage-attention`
# ONLY. The deployed configuration adds `--fast`, which phase B measured at
# about -5%, so this model runs ~5% pessimistic against any row that carried it.
# Rows are not self-describing about this — `flags` is recorded per row, so the
# report below prints it and you should not difference two rows whose flags
# differ. (Learned the hard way: bench/drive4.sh used the bare flag and
# bench/drive5.sh added --fast, which makes N0 land at 112.2s in one and 103.1s
# in the other for reasons that have nothing to do with the thing being tested.)
CALIBRATION_FLAGS = ["--use-sage-attention"]

# Skipped steps per threshold at 10 steps. Recovered from the rows themselves —
# (server_s - FIXED) / BASE_STEP gives the real step count, and every one lands
# within 0.1 of an integer, which is its own confirmation that the model is
# sound. Without these a cached row looks 40% faster than the model for reasons
# that have nothing to do with geometry.
#   0.2 -> (284.7-84)/28.3 = 7.09 real, so 3 skipped
#   0.3 -> 6.04 -> 4     0.4 -> 5.99 -> 4     0.6 -> 5.07 -> 5
# Note 0.5 buys nothing over 0.4: same 4 skips, which is the saturation in §1.
SKIPPED = {0.2: 3, 0.3: 4, 0.4: 4, 0.5: 4, 0.6: 5}


# sage3 vs sage2 on the attention term alone, from bench/attn-micro.py at the
# real shape: 245.0 ms against 304.3 ms.
SAGE3_ATTN = 245.0 / 304.3


def step_cost(tokens, sage3=False):
    r = tokens / N0_TOKENS
    attn = ATTN_FRAC * r * r * (SAGE3_ATTN if sage3 else 1.0)
    return BASE_STEP * (attn + (1 - ATTN_FRAC) * r)


def predict(tokens, steps, cache_threshold=None, sage3=False):
    real = steps - SKIPPED.get(cache_threshold, 0) if cache_threshold else steps
    return FIXED + max(1, real) * step_cost(tokens, sage3)


def row_config(phase, row):
    """The row's actual overrides, from matrix.py rather than from its prose.

    Recovering a cache threshold by grepping the note string silently fails for
    every row whose note says "aggressive" instead of "0.4", and the model then
    predicts ten real steps where six ran — reporting a 113 s residual on a row
    that is in fact within a second of the model. The definitions are right
    there; use them.
    """
    try:
        _, _, rows = matrix.phase(phase)
    except KeyError:
        return {}
    for entry in rows:
        if entry[0] == row and isinstance(entry[1], dict):
            return entry[1]
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phases", nargs="*")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    rows = [r for r in rows if r.get("ok") and r.get("server_s")]
    if args.phases:
        rows = [r for r in rows if r["phase"] in args.phases]

    seen = {}
    for r in rows:
        seen[(r["phase"], r["row"], r.get("width"), r.get("steps"))] = r

    print(f"{'row':6} {'geometry':13} {'tokens':>8} {'st':>3} "
          f"{'measured':>9} {'predicted':>10} {'residual':>9}  note")
    print("-" * 96)
    for (ph, row, _, _), r in sorted(seen.items()):
        tok = r.get("tokens") or matrix.tokens(r["width"], r["height"], r["length"])
        cfg = row_config(ph, row)
        thr = cfg.get("cache_threshold") if cfg.get("cache") else None
        p = predict(tok, r["steps"], thr, sage3=bool(cfg.get("sage3")))
        m = r["server_s"]
        resid = m - p
        tags = []
        if cfg.get("cache"):
            tags.append(f"{cfg['cache']} {thr}")
        if cfg.get("sage3"):
            tags.append("sage3")
        if cfg.get("dualpipe"):
            tags.append(f"split@{cfg['dualpipe']}")
        flag = ""
        if abs(resid) > 15:
            flag = "off model"
            if abs(resid - FIRST_ROW_PENALTY) < 12:
                flag = "~first-row penalty"
        # Mark rows launched with flags the constants were not fitted to; the
        # model reads ~5% pessimistic for them and they must not be differenced
        # against rows that lack them.
        if sorted(r.get("flags") or []) != sorted(CALIBRATION_FLAGS):
            extra = [f for f in (r.get("flags") or []) if f not in CALIBRATION_FLAGS]
            tags.append("+" + " ".join(extra) if extra else "flags!")
        print(f"{ph}/{row:4} {r['width']}x{r['height']:<7} {tok:>8,} {r['steps']:>3} "
              f"{m:>8.1f}s {p:>9.1f}s {resid:>+8.1f}s  {' '.join(tags):<22} {flag}")

    print(f"\nconstants: step={BASE_STEP}s at {N0_TOKENS:,} tokens, fixed={FIXED}s, "
          f"attention share={ATTN_FRAC:.0%}")
    print("A residual near +24s on a phase's first row is the checkpoint-staging "
          "penalty, not the setting under test (see sweep.py --warmup).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
