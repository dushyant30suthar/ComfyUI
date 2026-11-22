#!/usr/bin/env python3
"""Contact sheet of a rendered film — every shot, three frames each.

    .venv/bin/python bench/storyboard.py journey

Lets you judge a twelve-shot film at a glance instead of scrubbing three
minutes of video, and makes it obvious which shots are worth re-rendering.

Each row is one shot, labelled with the mode it was generated in (read back
from bench/gen.jsonl, so it reflects what actually ran rather than what was
planned) and its render time.
"""
import argparse, json, os, re, subprocess, sys
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TW, TH, LABEL = 380, 217, 26


def frame(video, n, out):
    subprocess.run(["ffmpeg", "-v", "error", "-i", video,
                    "-vf", f"select=eq(n\\,{n})", "-vframes", "1", "-y", out],
                   check=True)


def count(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_entries", "stream=nb_read_frames",
                        "-of", "default=nw=1:nk=1", video],
                       capture_output=True, text=True).stdout.strip()
    return int(r) if r.isdigit() else 0


def manifest():
    """name -> (mode, minutes). Last record wins, so a re-render supersedes."""
    m = {}
    p = os.path.join(ROOT, "bench", "gen.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("name") and r.get("ok"):
                mode = r.get("mode", "?")
                # A chained shot is i2v as far as the graph is concerned, but the
                # distinction matters when judging: its seed was generated, not
                # photographed, so it inherits the previous shot's framing.
                if mode == "i2v" and str(r.get("first_frame", "")).startswith("chain-"):
                    mode = "i2v<-chain"
                elif mode == "i2v":
                    mode = "i2v<-photo"
                m[r["name"]] = (mode, (r.get("server_s") or 0) / 60)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = os.path.join(ROOT, "output", args.folder)
    shots = sorted(f for f in os.listdir(src) if re.match(r"shot-\d+_.*\.mp4$", f))
    if not shots:
        print(f"no shots in {src}")
        return 1
    man, tmp = manifest(), "/tmp/_sb.png"

    sheet = Image.new("RGB", (TW * 3, len(shots) * (TH + LABEL)), (14, 14, 16))
    dr = ImageDraw.Draw(sheet)
    for row, f in enumerate(shots):
        path = os.path.join(src, f)
        n = count(path)
        num = re.match(r"shot-(\d+)_", f).group(1)
        mode, mins = man.get(f"{args.folder}/shot-{num}", ("?", 0))
        y = row * (TH + LABEL)
        for col, pos in enumerate((1, n // 2, max(0, n - 3))):
            frame(path, pos, tmp)
            sheet.paste(Image.open(tmp).resize((TW - 3, TH - 3), Image.LANCZOS),
                        (col * TW + 1, y + 1))
        dr.rectangle([0, y + TH, TW * 3, y + TH + LABEL], fill=(14, 14, 16))
        dr.text((6, y + TH + 7),
                f"shot {num}   [{mode}]   {mins:.1f} min   "
                f"start / middle / end", fill=(205, 205, 210))
        print(f"  shot {num}  {mode}")

    out = args.out or os.path.join(src, f"STORYBOARD-{args.folder}.png")
    sheet.save(out)
    os.path.exists(tmp) and os.remove(tmp)
    print(f"\n{out}\n  {sheet.size[0]}x{sheet.size[1]}, {len(shots)} shots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
