#!/usr/bin/env python3
"""Generate one clip from a prompt given on the command line.

    .venv/bin/python bench/gen.py --prompt "..."
    .venv/bin/python bench/gen.py --prompt "..." --length 209 --draft
    .venv/bin/python bench/gen.py --prompt-file p.txt --seed 7

Unlike clips.py (which cycles a fixed prompt list) and quality.py (which varies
the config against a fixed prompt), this varies the PROMPT against a fixed,
measured-good config. It is the "just make me this video" path.

Defaults are the deployed configuration, warm cost ~5.6 min per clip:
    1344x768, 20 steps, EasyCache 0.4, cfg 1.0, euler/simple, sage2 + --fast
`--draft` switches to 960x544, which measured 2.5x faster (~2.2 min) for a
quality trade that has not been judged yet — see bench/quality.py output.

Files land in output/video/ and a manifest line goes to bench/gen.jsonl.
"""
import argparse, json, os, subprocess, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import matrix  # noqa: E402

PY = os.path.join(ROOT, ".venv", "bin", "python")
PORT = 8195                      # 8188 panel, 8191 sweep, 8192 clips, 8193 quality
BASE = f"http://127.0.0.1:{PORT}"
MANIFEST = os.path.join(HERE, "gen.jsonl")
LOG = os.path.join(HERE, "gen-server.log")
FLAGS = ["--use-sage-attention", "--fast"]


def api(path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def alive():
    try:
        api("/system_stats", timeout=4)
        return True
    except Exception:
        return False


def start_server(flags=FLAGS):
    subprocess.run(["pkill", "-f", f"[m]ain.py --port {PORT}"], capture_output=True)
    time.sleep(2)
    log = open(LOG, "a")
    proc = subprocess.Popen([PY, "main.py", "--port", str(PORT)] + list(flags),
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    for _ in range(300):
        if proc.poll() is not None:
            raise RuntimeError(f"server died rc={proc.returncode}; see {LOG}")
        if alive():
            return proc
        time.sleep(1)
    raise RuntimeError("server did not boot")


def server_seconds(entry):
    start = end = None
    for event, data in entry.get("status", {}).get("messages", []):
        ts = data.get("timestamp")
        if ts is None:
            continue
        if event == "execution_start":
            start = ts
        elif event in ("execution_success", "execution_error", "execution_interrupted"):
            end = ts
    return (end - start) / 1000.0 if (start and end) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--length", type=int, default=124,
                    help="frames; 124=5.2s, 209=8.7s, 362=15.1s (trained range 124-362)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draft", action="store_true", help="960x544 instead of native 1344x768")
    ap.add_argument("--name", default="gen", help="filename prefix")
    ap.add_argument("--first-frame", default="",
                    help="filename in input/ to animate FROM (image-to-video). The node is "
                         "called MiniMaxH3ImageToVideo but its image inputs are optional — "
                         "omitting them is plain text-to-video")
    ap.add_argument("--last-frame", default="",
                    help="filename in input/ to animate TO. With --first-frame this "
                         "interpolates between the two")
    ap.add_argument("--ref", default="",
                    help="comma-separated filenames in input/ to use as IDENTITY references "
                         "(reference-to-video). Switches to the ref2va checkpoint. Unlike "
                         "--first-frame, these are not animated: their subject is carried into "
                         "a new scene. Address them in the prompt as <Picture 1>, <Picture 2>, "
                         "... in the order given — an unnamed reference still costs tokens on "
                         "every step but steers nothing. Max 9")
    ap.add_argument("--ref-video", default="",
                    help="comma-separated video filenames in input/ used as references. A "
                         "previously rendered clip here carries motion and look forward into "
                         "the next shot. Addressed as <Video 1>, ... Max 3, 2-15s each")
    ap.add_argument("--ref-size", default="match", choices=["match", "max"],
                    help="'match' scales references to the generation's pixel area; 'max' uses "
                         "a 2048px short edge for best identity fidelity and is several times "
                         "slower, because reference tokens ride through every sampling step")
    ap.add_argument("--sampler", default="res_multistep",
                    help="res_multistep is the OFFICIAL H3 default and what the shipped "
                         "ComfyUI r2v template uses. It is second order, so its local error "
                         "per step is the square of euler's: ~20 steps accumulate what euler "
                         "accumulates in 40-50, at the same one DiT call per step. euler here "
                         "is a benchmark-harness inheritance, not a considered choice")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--chunk", type=int, default=0, metavar="GROUPS",
                    help="run SageAttention over GROUPS head groups (custom_nodes/sage_chunked). "
                         "Exact — heads are independent. Cuts sage's int8 scratch to ~1/GROUPS, "
                         "which is what makes 15.1s at native 1344x768 possible at all: "
                         "unchunked it needs 748 MiB more than the card has and silently drops "
                         "to pytorch attention. 4 is the tested value")
    ap.add_argument("--nocache", action="store_true",
                    help="disable EasyCache. With it on at 0.4 the sampler SKIPS half the "
                         "steps (10 of 20 measured), so --steps 20 buys 10 real ones. Off, "
                         "20 means 20 — roughly double the time and the single biggest "
                         "fidelity gain available at any length")
    ap.add_argument("--lowmem", action="store_true",
                    help="--disable-smart-memory + --fp16-intermediates. Needed for long "
                         "clips on 31 GB: comfy stages ~39 GB of models (15 GB encoder + "
                         "20 GB DiT + 5 GB VAE) and normally keeps them resident between "
                         "stages. That survives sampling because it is mmap-backed, but the "
                         "VAE decode needs real anonymous memory and MemAvailable hits 0. "
                         "Evicting between stages frees the DiT before the decode runs")
    ap.add_argument("--tiled", type=int, default=0, metavar="FRAMES",
                    help="decode the video VAE FRAMES at a time instead of the whole clip "
                         "at once. REQUIRED past ~200 frames: the one-shot decode of a 362-"
                         "frame clip materialises the entire pixel tensor and exhausted host "
                         "RAM (MemAvailable hit 0, swap +4.3 GB) AFTER sampling had already "
                         "succeeded. 32 is a safe value")
    ap.add_argument("--reserve", type=float, default=0.0,
                    help="--reserve-vram N: hold N GiB back so attention has room for its "
                         "workspace. REQUIRED at 362 frames — sage2 wants ~1.5 GiB of scratch "
                         "there, and without the reservation comfy silently substitutes "
                         "pytorch attention, which is ~3x slower per call")
    args = ap.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = open(args.prompt_file).read().strip()
    if not prompt:
        print("need --prompt or --prompt-file")
        return 2

    flags = list(FLAGS)
    if args.lowmem:
        flags += ["--disable-smart-memory", "--fp16-intermediates"]
    if args.reserve:
        flags += ["--reserve-vram", str(args.reserve)]
    w, h = (960, 544) if args.draft else (1344, 768)
    refs = [s.strip() for s in args.ref.split(",") if s.strip()]
    rvids = [s.strip() for s in args.ref_video.split(",") if s.strip()]
    if len(refs) > 9 or len(rvids) > 3:
        print("at most 9 --ref images and 3 --ref-video")
        return 2
    params = dict(matrix.PROBE, width=w, height=h, length=args.length, steps=args.steps,
                  sampler=args.sampler, scheduler=args.scheduler,
                  cfg=1.0, prompt=prompt, seed=args.seed,
                  cache=None if args.nocache else "easy", cache_threshold=0.4,
                  tiled_decode=args.tiled, chunk_heads=args.chunk,
                  first_frame=args.first_frame or None,
                  last_frame=args.last_frame or None,
                  refs=refs or None, ref_videos=rvids or None, ref_size=args.ref_size)
    # references live in the other 20 GB checkpoint; the fl2va one has no
    # reference pathway and would silently ignore them
    if refs or rvids:
        params["unet"] = matrix.REF2VA
    wf = matrix.build_workflow(**params)
    # a name containing "/" writes into that subfolder under output/
    wf["13"]["inputs"]["filename_prefix"] = args.name if "/" in args.name else f"video/{args.name}"

    tok = matrix.tokens(w, h, args.length)
    mode = ("r2v" if (refs or rvids) else
            "i2v" if args.first_frame else "t2v")
    print(f"[{mode}] {w}x{h}, {args.length} frames ({args.length/24:.1f}s), {args.steps} steps, "
          f"seed {args.seed}, {tok:,} tokens")
    if refs:
        print("  refs: " + ", ".join(f"<Picture {i+1}>={f}" for i, f in enumerate(refs))
              + f"  [{args.ref_size}]")
    if rvids:
        print("  ref videos: " + ", ".join(f"<Video {i+1}>={f}" for i, f in enumerate(rvids)))
    if args.first_frame:
        print(f"  from frame: {args.first_frame}")
    print(f"prompt: {prompt[:110]}{'...' if len(prompt) > 110 else ''}\n")

    proc = start_server(flags)
    t0 = time.time()
    try:
        try:
            pid = api("/prompt", {"prompt": wf}).get("prompt_id")
        except urllib.error.HTTPError as e:
            print(f"REJECTED: {e.read().decode()[:400]}")
            return 1
        while time.time() - t0 < 10800:
            time.sleep(5)
            try:
                hist = api(f"/history/{pid}", timeout=20)
            except Exception:
                continue
            if pid not in hist:
                continue
            entry = hist[pid]
            ok = entry.get("status", {}).get("status_str") == "success"
            files = [f["filename"] for out in (entry.get("outputs") or {}).values()
                     for kind in ("images", "video", "gifs", "audio")
                     for f in (out.get(kind) or []) if isinstance(f, dict) and f.get("filename")]
            secs = server_seconds(entry)
            rec = {"ok": ok, "mode": mode, "sampler": args.sampler, "server_s": secs, "files": files,
                   "width": w, "height": h, "length": args.length, "steps": args.steps,
                   "seed": args.seed, "refs": refs, "ref_videos": rvids,
                   "first_frame": args.first_frame, "name": args.name,
                   "prompt": prompt, "ts": int(time.time())}
            if not ok:
                msgs = [str(m) for m in entry.get("status", {}).get("messages", [])]
                rec["detail"] = " | ".join(msgs[-2:])[:400]
            with open(MANIFEST, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if ok:
                print(f"done in {secs:.0f}s ({secs/60:.1f} min)")
                sub = args.name.rsplit("/", 1)[0] if "/" in args.name else "video"
                for f in files:
                    print(f"  {os.path.join(ROOT, 'output', sub, f)}")
            else:
                print(f"FAILED: {rec.get('detail','')[:200]}")
            return 0 if ok else 1
        print("timeout")
        return 1
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            proc.terminate()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        subprocess.run(["pkill", "-f", f"[m]ain.py --port {PORT}"], capture_output=True)
        sys.exit(130)
