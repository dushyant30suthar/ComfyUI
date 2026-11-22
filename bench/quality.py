#!/usr/bin/env python3
"""Generate the same clip under each quality-gated config, for side-by-side judging.

    .venv/bin/python bench/quality.py                 # all configs, 5.2s clips
    .venv/bin/python bench/quality.py --only deployed,sage3
    .venv/bin/python bench/quality.py --prompt-index 2 --seed 7

Three of the largest remaining speedups are blocked on the same question — does
the output still look right — and none of them can be settled by a timing table:

  EasyCache 0.6   -9.9% vs the deployed 0.4, by tolerating more trajectory drift
  960x544         2.4x measured at 5.2s, by generating 0.51x the tokens
  SageAttention 3 -8.3% at 5.2s uncached, by quantising Q, K AND V to FP4

960x544 is the one that matters — at 2.4x it is a different production profile,
not an increment. sage3 is rendered for the record rather than for adoption: it
OOMs at 15s (silently falling back to pytorch, which is SLOWER than sage2) and
swaps the box when paired with the deployed cache. It is included because its
blocker is workspace size, which scales with token count — so if the draft
canvas wins on quality, sage3 may fit alongside it where it cannot fit at
native, and knowing whether FP4 looks acceptable is what decides if that is
worth chasing.

So this renders ONE prompt at ONE seed under each, plus a no-cache 20-step
reference as the quality ceiling and the deployed config as the incumbent. Fixed
prompt and seed is the whole point: any difference between two outputs is the
knob, not the sampler's luck.

Everything runs on a single server — sage3 and the caches are workflow-level
nodes, not launch flags — so this costs one boot plus the generations. Files land
in output/video/quality-<config>.mp4 with a manifest in bench/quality.jsonl.
"""
import argparse, json, os, subprocess, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import matrix  # noqa: E402
import clips as clipsmod  # noqa: E402  (prompt list; written to H3's documented order)

PY = os.path.join(ROOT, ".venv", "bin", "python")
PORT = 8193                      # 8188 panel, 8191 sweep, 8192 clips
BASE = f"http://127.0.0.1:{PORT}"
MANIFEST = os.path.join(HERE, "quality.jsonl")
LOG = os.path.join(HERE, "quality-server.log")
FLAGS = ["--use-sage-attention", "--fast"]

# Every row is the deployed config with exactly ONE thing changed, so a
# difference has a single cause. reference is the exception and is deliberately
# the most expensive thing here: without a no-cache ceiling to compare against,
# "the cache looks fine" is unfalsifiable.
CONFIGS = {
    "reference":  dict(steps=20, note="20 steps, NO cache, sage2, native — quality ceiling"),
    "deployed":   dict(steps=20, cache="easy", cache_threshold=0.4,
                       note="the shipping config: EasyCache 0.4, sage2, native"),
    "cache06":    dict(steps=20, cache="easy", cache_threshold=0.6,
                       note="EasyCache 0.6 — is the extra 9.9% visible?"),
    # sage3 runs WITHOUT the cache on purpose. The two together were killed twice
    # by the watchdog, the second time at MemAvailable=125 MiB with swap +3.1 GB
    # — sage3's FP4 workspace plus EasyCache's retained activations exceed the
    # card, comfy evicts weights to a 31 GB host that cannot hold them, and the
    # machine pages. Uncached sage3 (row K10) completes fine at 336 s.
    "sage3":      dict(steps=20, sage3=True,
                       note="SageAttention 3 FP4, NO cache — 4x sage2's attention error"),
    "res960":     dict(steps=20, cache="easy", cache_threshold=0.4, width=960, height=544,
                       note="960x544 — 0.51x tokens; judge detail, not framing"),
    "steps10":    dict(steps=10, cache="easy", cache_threshold=0.4,
                       note="half the steps — the cheapest possible win if it holds"),
}


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


def start_server():
    # bracket the pattern so pkill cannot match the shell that issues it
    subprocess.run(["pkill", "-f", f"[m]ain.py --port {PORT}"], capture_output=True)
    time.sleep(2)
    log = open(LOG, "a")
    proc = subprocess.Popen([PY, "main.py", "--port", str(PORT)] + FLAGS,
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


def run(name, cfg, prompt, seed, length):
    note = cfg.pop("note", "")
    params = dict(matrix.PROBE, length=length, cfg=1.0, prompt=prompt, seed=seed)
    params.update(cfg)
    wf = matrix.build_workflow(**params)
    wf["13"]["inputs"]["filename_prefix"] = f"video/quality-{name}"

    t0 = time.time()
    try:
        pid = api("/prompt", {"prompt": wf}).get("prompt_id")
    except urllib.error.HTTPError as e:
        return {"config": name, "ok": False, "error": e.read().decode()[:300]}

    while time.time() - t0 < 7200:
        time.sleep(3)
        try:
            hist = api(f"/history/{pid}", timeout=20)
        except Exception:
            continue
        if pid not in hist:
            continue
        entry = hist[pid]
        ok = entry.get("status", {}).get("status_str") == "success"
        files = []
        for out in (entry.get("outputs") or {}).values():
            for kind in ("images", "video", "gifs", "audio"):
                for f in out.get(kind, []) or []:
                    if isinstance(f, dict) and f.get("filename"):
                        files.append(f["filename"])
        rec = {"config": name, "note": note, "ok": ok,
               "server_s": server_seconds(entry), "files": files,
               "width": params["width"], "height": params["height"],
               "steps": params["steps"], "seed": seed, "prompt": prompt[:90]}
        if not ok:
            msgs = [str(m) for m in entry.get("status", {}).get("messages", [])]
            rec["detail"] = " | ".join(msgs[-2:])[:300]
        return rec
    return {"config": name, "ok": False, "error": "timeout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated config names")
    ap.add_argument("--prompt-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--length", type=int, default=124)
    args = ap.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        print(f"unknown config(s): {unknown}\nknown: {list(CONFIGS)}")
        return 2

    prompt = clipsmod.PROMPTS[args.prompt_index % len(clipsmod.PROMPTS)]
    print(f"prompt #{args.prompt_index}: {prompt[:80]}...")
    print(f"seed {args.seed}, length {args.length} frames, {len(names)} configs\n")

    # A FRESH SERVER PER CONFIG. Sharing one would be faster — the 21 GB
    # checkpoint gets staged once instead of six times — but host RAM
    # accumulates across prompts on this 31 GB box and the third execution
    # reliably thrashes (phase N died at its third row on both attempts). Losing
    # render 3 of 6 would take the 960x544 comparison with it, which is the
    # whole reason this script exists. Boots are cheap next to a lost answer.
    results = []
    for name in names:
        print(f"  -- {name}: {CONFIGS[name].get('note','')}")
        proc = start_server()
        try:
            rec = run(name, dict(CONFIGS[name]), prompt, args.seed, args.length)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), 15)
            except Exception:
                proc.terminate()
            time.sleep(5)          # let the CUDA context hand VRAM back
        rec["ts"] = int(time.time())
        results.append(rec)
        with open(MANIFEST, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if rec.get("ok"):
            print(f"     {rec['server_s']:.0f}s -> {', '.join(rec['files']) or '(no file)'}")
        else:
            print(f"     FAILED: {str(rec.get('detail') or rec.get('error'))[:120]}")

    print(f"\n{'config':12} {'server s':>9}  file")
    for r in results:
        s = f"{r['server_s']:.0f}" if r.get("server_s") else "—"
        print(f"{r['config']:12} {s:>9}  {(r.get('files') or ['FAILED'])[0]}")
    print(f"\nClips in {os.path.join(ROOT,'output','video')}/quality-*.mp4 — "
          f"same prompt and seed, so any difference is the knob. Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        subprocess.run(["pkill", "-f", f"[m]ain.py --port {PORT}"], capture_output=True)
        sys.exit(130)
