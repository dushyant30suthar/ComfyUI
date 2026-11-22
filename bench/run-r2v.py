#!/usr/bin/env python3
"""Run the reference-to-video workflow — the path that has never generated.

    .venv/bin/python bench/run-r2v.py --image h3-i2v-seed.png --prompt "..."

r2v is a SEPARATE 21 GB checkpoint (`minimax_h3_ref2va_pruned_int8_convrot`)
that has sat on this disk since the first session without ever being loaded. It
is not the fl2va model with a different node: fl2va does text-to-video and
first/last-frame conditioning, ref2va carries identity or style from reference
images, videos and audio into a new generation.

The workflow at workflows/h3/h3-r2v.json is already correctly wired — it
validates, it has just never been executed. This runs it with a real image
substituted in, on its own server, under the same flags production uses.

Note the autogrow input naming: reference slots are `ref_image_0`,
`ref_image_1`, ... not a list. Extra references mean extra keys.
"""
import argparse, json, os, subprocess, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "bin", "python")
PORT = 8197
BASE = f"http://127.0.0.1:{PORT}"
LOG = os.path.join(HERE, "r2v-server.log")
WF = os.path.join(ROOT, "workflows", "h3", "h3-r2v.json")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="h3-i2v-seed.png", help="filename in input/")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--length", type=int, default=124)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    wf = json.load(open(WF))
    # the LoadImage feeding ref_image_0
    for k, v in wf.items():
        if v["class_type"] == "LoadImage":
            v["inputs"]["image"] = args.image
    # execute() iterates (ref_images or {}).values(), so this must be a dict keyed
    # by the autogrow socket name. A flat "ref_image_0" key reaches execute() as
    # an unexpected kwarg and the node dies — which is exactly why the shipped
    # r2v workflow validated but had never generated a frame.
    n6 = wf["6"]["inputs"]
    if "ref_image_0" in n6:
        n6["ref_images"] = {"ref_image_0": n6.pop("ref_image_0")}
    if args.prompt:
        wf["6"]["inputs"]["prompt"] = args.prompt
    wf["6"]["inputs"]["length"] = args.length
    wf["8"]["inputs"]["steps"] = args.steps
    wf["8"]["inputs"]["seed"] = args.seed
    wf["8"]["inputs"]["cfg"] = 1.0          # H3 is guidance-distilled; >1 doubles cost for nothing
    wf["13"]["inputs"]["filename_prefix"] = "video/r2v-test"

    print(f"checkpoint: {wf['1']['inputs']['unet_name']}")
    print(f"reference:  input/{args.image}")
    print(f"{wf['6']['inputs']['width']}x{wf['6']['inputs']['height']}, "
          f"{args.length} frames ({args.length/24:.1f}s), {args.steps} steps")
    print(f"prompt: {wf['6']['inputs']['prompt'][:100]}\n")

    subprocess.run(["pkill", "-f", f"[m]ain.py --port {PORT}"], capture_output=True)
    time.sleep(2)
    log = open(LOG, "w")
    proc = subprocess.Popen([PY, "main.py", "--port", str(PORT)] + FLAGS,
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    for _ in range(300):
        if proc.poll() is not None:
            print(f"server died rc={proc.returncode}; see {LOG}")
            return 1
        if alive():
            break
        time.sleep(1)

    t0 = time.time()
    try:
        try:
            pid = api("/prompt", {"prompt": wf}).get("prompt_id")
        except urllib.error.HTTPError as e:
            print(f"REJECTED: {e.read().decode()[:500]}")
            return 1
        while time.time() - t0 < 7200:
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
            if ok:
                print(f"done in {time.time()-t0:.0f}s -> {', '.join(files)}")
            else:
                msgs = [str(m) for m in entry.get("status", {}).get("messages", [])]
                print(f"FAILED: {' | '.join(msgs[-2:])[:500]}")
            return 0 if ok else 1
        print("timeout")
        return 1
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
