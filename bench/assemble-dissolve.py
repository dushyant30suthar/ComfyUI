#!/usr/bin/env python3
"""Assemble the shots with a short dissolve at chained boundaries.

    .venv/bin/python bench/assemble-dissolve.py journey
    .venv/bin/python bench/assemble-dissolve.py journey --fade 0.4
    .venv/bin/python bench/assemble-dissolve.py journey --all

Why this exists, and when NOT to use it
---------------------------------------
`assemble.sh` concatenates with stream copy: no re-encode, no generation loss,
seconds to run. That is the default and usually the right answer.

But a @chain boundary is not the invisible splice it was designed to be.
Measured 2026-08-05: seeding a clip with a *generated* frame reproduces it to
8.6/255, not the 2.7/255 that a clean photograph gets. At the shot-002 ->
shot-003 boundary the car shifted left and the framing widened — small, but a
jump cut. A ~0.25 s dissolve there hides it and reads as deliberate.

The trade is real and it is not only cost:
  * it re-encodes the whole film (concat stream-copy cannot dissolve), so it
    takes minutes rather than seconds and loses a generation of quality;
  * a dissolve is different GRAMMAR. A cut says "same moment, new angle"; a
    dissolve says "time passed". On a chained pair the first is what actually
    happened, so use the shortest fade that hides the jump, not a luxurious one.

By default only CHAINED boundaries dissolve — the ones with a jump to hide. Hard
cuts stay hard, which is what you want between sequences. `--all` overrides.

Chained boundaries are read from the prompt file, so this stays correct if the
shot list changes.
"""
import argparse, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# A one-frame xfade is visually a cut, and keeping every boundary the same
# operation means one uniform filtergraph instead of splicing concat and xfade
# together, which ffmpeg makes genuinely awkward.
CUT = 1.0 / 24


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def chained_shots(prompts):
    """1-based indices of shots whose prompt opens with @chain."""
    if not os.path.exists(prompts):
        return set()
    blocks, cur = [], []
    for line in open(prompts):
        if line.startswith("#"):
            continue
        if not line.strip():
            if cur:
                blocks.append(cur); cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)
    return {i for i, b in enumerate(blocks, 1)
            if any(l.strip() == "@chain" for l in b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--fade", type=float, default=0.25,
                    help="dissolve length in seconds at chained boundaries (default 0.25)")
    ap.add_argument("--all", action="store_true",
                    help="dissolve EVERY boundary, not only chained ones")
    ap.add_argument("--prompts", default="bench/journey-prompts.txt")
    ap.add_argument("--crf", type=int, default=17)
    args = ap.parse_args()

    src = os.path.join(ROOT, "output", args.folder)
    shots = sorted(f for f in os.listdir(src)
                   if re.match(r"shot-\d+_.*\.mp4$", f))
    if len(shots) < 2:
        print(f"need at least 2 shots in {src}, found {len(shots)}")
        return 1
    paths = [os.path.join(src, f) for f in shots]
    nums = [int(re.match(r"shot-(\d+)_", f).group(1)) for f in shots]
    durs = [duration(p) for p in paths]
    chain = chained_shots(os.path.join(ROOT, args.prompts))

    # Boundary k joins shots[k] and shots[k+1]. It dissolves when the RIGHT-hand
    # shot was rendered with @chain — that is the one whose frame 0 was seeded
    # from its predecessor and so the one carrying the jump.
    fades = [args.fade if (args.all or nums[k + 1] in chain) else CUT
             for k in range(len(shots) - 1)]

    print(f"{len(shots)} shots from {src}")
    for k, f in enumerate(fades):
        kind = f"dissolve {f:.2f}s" if f > CUT else "cut"
        print(f"  {shots[k]} -> {shots[k+1]}   {kind}")

    # xfade offsets are absolute positions in the ACCUMULATED stream, and each
    # dissolve overlaps the two clips, so the running length grows by
    # (next duration - fade) rather than by the full next duration.
    vf, af, acc = [], [], durs[0]
    vlast, alast = "[0:v]", "[0:a]"
    for k in range(len(shots) - 1):
        f = fades[k]
        off = acc - f
        vo, ao = f"[v{k}]", f"[a{k}]"
        vf.append(f"{vlast}[{k+1}:v]xfade=transition=fade:duration={f:.4f}:offset={off:.4f}{vo}")
        af.append(f"{alast}[{k+1}:a]acrossfade=d={f:.4f}{ao}")
        vlast, alast = vo, ao
        acc += durs[k + 1] - f

    out = os.path.join(src, f"{args.folder.upper()}-dissolve.mp4")
    cmd = ["ffmpeg", "-v", "error", "-stats"]
    for p in paths:
        cmd += ["-i", p]
    cmd += ["-filter_complex", ";".join(vf + af),
            "-map", vlast, "-map", alast,
            "-c:v", "libx264", "-crf", str(args.crf), "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-y", out]
    print(f"\nre-encoding to {os.path.basename(out)} "
          f"(expect ~{acc:.0f}s of video, this is not a stream copy)...")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"ffmpeg failed rc={rc}")
        return rc
    print(f"\ndone: {out}")
    print(f"  {duration(out):.1f}s  ({os.path.getsize(out)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
