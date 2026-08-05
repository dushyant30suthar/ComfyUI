#!/usr/bin/env python3
"""Versioned render profiles — the single source of truth for how a clip is made.

    .venv/bin/python bench/profiles.py                 # list them
    .venv/bin/python bench/profiles.py v2-fast         # show one in full
    .venv/bin/python bench/profiles.py v2-fast --flags # the argv for gen.py

Profiles are VERSIONED AND NEVER OVERWRITTEN. When a better configuration is
found the old one stays, marked SUPERSEDED, with the measurement that retired it
and the date. Two reasons: a config that was wrong is still the config that
produced everything rendered before it, so output stays explicable; and a
"better" setting that turns out worse on other material can be reverted to
without re-deriving anything.

`status` is the only thing that changes on an existing profile.
  CURRENT     — use this unless you have a reason not to
  REFERENCE   — matches the upstream template; the quality bar, not the default
  SUPERSEDED  — kept for provenance; `retired_by` says what beat it
  CANDIDATE   — under measurement, do not use for production
"""
import argparse, json, sys

# Canvas and length are properties of the shot, not the profile, so they are
# passed separately. Everything here is a quality/cost decision.
PROFILES = {

    "v1-euler-cache": dict(
        status="SUPERSEDED",
        added="2026-08-04", retired="2026-08-05",
        flags=["--sampler", "euler", "--steps", "20"],
        real_steps=9,
        cost_124f="324 s", cost_362f="~27 min",
        summary="euler + EasyCache 0.4. What the Udaipur->Himachal film was rendered with.",
        reasoning="""
            Inherited from matrix.PROBE, which was written to benchmark
            THROUGHPUT and never intended as a quality config. EasyCache at 0.4
            skips ~11 of 20 steps, so `--steps 20` bought 9 real first-order
            steps. Nobody checked it against the shipped template for two days.
        """,
        retired_by="""
            v2-fast, at +8.6% time, on the same prompt and seed 777. At 1:1 with
            no resampling this profile renders distant architecture as blocky
            checkerboard artifact where v2 resolves windows and roof lines.
            Upstream guidance is also explicit that cache tiers are
            approximations whose output "must not be used as a reference".
        """),

    "v2-fast": dict(
        status="CURRENT",
        added="2026-08-05",
        flags=["--sampler", "res_multistep", "--steps", "10", "--nocache"],
        real_steps=10,
        cost_124f="352 s", cost_362f="~29 min",
        summary="res_multistep, 10 real steps, no cache. Best quality per minute measured.",
        reasoning="""
            A second-order solver's local error per step is the SQUARE of a
            first-order one's, so ~20 res_multistep steps accumulate roughly what
            euler accumulates in 40-50 — at one DiT call per step either way.
            Ten second-order steps therefore land near euler-20/25 while costing
            what euler-9-plus-cache cost.

            Measured 2026-08-05, 124 frames at 1344x768, seed 777: 352 s against
            v1's 324 s (+8.6%), and not close on quality.
        """),

    "v3-reference": dict(
        status="REFERENCE",
        added="2026-08-05",
        flags=["--sampler", "res_multistep", "--steps", "20", "--nocache"],
        real_steps=20,
        cost_124f="638 s", cost_362f="~53 min",
        summary="What the shipped ComfyUI r2v template uses. The quality bar.",
        reasoning="""
            video_minimax_h3_r2v.json ships KSamplerSelect=res_multistep and
            BasicScheduler=simple/20/1.0 with no cache node. Upstream measures
            res_multistep at 20 DiT calls as matching euler at 49 with no visible
            difference on the same seed.

            1.81x the cost of v2-fast (638 s vs 352 s at 124 frames). Worth it
            for a hero shot; expensive across a whole film.
        """),

    "v4-exact-attn": dict(
        status="CANDIDATE",
        added="2026-08-05",
        flags=["--sampler", "res_multistep", "--steps", "20", "--nocache"],
        patch="comfy/ldm/minimax/model.py:181 low_precision_attention=False",
        real_steps=20,
        cost_124f="measuring", cost_362f="unknown",
        summary="v3 plus opting the H3 DiT out of SageAttention's int8 QK path.",
        reasoning="""
            comfy/ldm/minimax/model.py:181 calls optimized_attention WITHOUT
            low_precision_attention=False, so the DiT's attention is quantized to
            int8. A dozen other models in comfy/ldm pass that opt-out, and
            upstream issue #15263 reports the missing flag makes H3 produce pure
            noise. Our output is coherent, so we are not hitting the total
            failure — but the same mechanism degrading fine detail is consistent
            with what we see.

            NOT A FREE FIX: the opt-out routes the DiT to exact pytorch
            attention, ~2.9x slower per call on this box. SageAttention is the
            foundation of the 15.1s-at-native unlock — the whole head-chunking
            node exists to fit sage's int8 scratch. If this wins on quality the
            machine's entire speed/quality trade has to be re-derived.

            Requires a source patch, so it is not selectable by flags alone.
            See bench/sage-quality-test.sh.
        """),
}

# Flags every long clip needs regardless of profile. These are memory
# correctness, not quality choices — see RUNBOOK section 2.
LONG_CLIP = ["--chunk", "4", "--tiled", "32", "--lowmem"]


def show(name, p):
    print(f"{name}   [{p['status']}]")
    print(f"  {p['summary']}")
    print(f"  flags       {' '.join(p['flags'])}")
    if p.get("patch"):
        print(f"  patch       {p['patch']}")
    print(f"  real steps  {p['real_steps']}")
    print(f"  cost        {p['cost_124f']} at 124f   |   {p['cost_362f']} at 362f native")
    for key in ("reasoning", "retired_by"):
        if p.get(key):
            print(f"  {key}:")
            for line in p[key].strip().split("\n"):
                print(f"    {line.strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--flags", action="store_true", help="print gen.py argv and exit")
    ap.add_argument("--long", action="store_true", help="append the long-clip memory flags")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if not a.name:
        cur = [n for n, p in PROFILES.items() if p["status"] == "CURRENT"]
        for n, p in PROFILES.items():
            print(f"  {n:18s} {p['status']:11s} {p['real_steps']:>3} steps  "
                  f"{p['cost_362f']:>9}  {p['summary'][:52]}")
        print(f"\ndefault: {cur[0] if cur else '(none marked CURRENT)'}")
        return 0

    p = PROFILES.get(a.name)
    if not p:
        print(f"unknown profile: {a.name}\nknown: {', '.join(PROFILES)}", file=sys.stderr)
        return 64
    if a.json:
        print(json.dumps(p, indent=2)); return 0
    if a.flags:
        print(" ".join(p["flags"] + (LONG_CLIP if a.long else []))); return 0
    show(a.name, p)
    if p["status"] == "SUPERSEDED":
        print("\n  WARNING: superseded. Use it only to reproduce old output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
