#!/usr/bin/env python3
"""Find the longest clip that still runs the FAST attention path, per canvas.

    .venv/bin/python bench/probe-length.py                    # native 1344x768
    .venv/bin/python bench/probe-length.py --draft            # 960x544

The binding constraint on this box is not clip length or resolution — it is the
token count they jointly produce, because SageAttention has to allocate a
workspace that scales with it. Past some count it cannot, and comfy SILENTLY
substitutes attention_pytorch, which is ~2.9x slower per call. A run that does
this still succeeds; it just takes three times as long and nothing says why.

Measured endpoints before this script existed:
     37,710 tokens (5.2s native)   -> clean, 6 clips, 0 fallbacks
     55,776 tokens (15.1s @960x544)-> clean
    109,062 tokens (15.1s native)  -> 250 fallbacks
Everything between 55,776 and 109,062 was unknown, which is a wide band to be
guessing inside one 20-minute render at a time.

So: one server, one sampling step per length, latent saved instead of decoded
(the VAE decode is a separate memory problem and irrelevant to this question).
A single step runs attention 50 times — ample to trigger the allocation. The
server log's fallback count is the answer.
"""
import argparse, json, os, subprocess, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import matrix  # noqa: E402

PY = os.path.join(ROOT, ".venv", "bin", "python")
PORT = 8196
BASE = f"http://127.0.0.1:{PORT}"
LOG = os.path.join(HERE, "probe-server.log")
FLAGS = ["--use-sage-attention", "--fast"]
FALLBACK = "using pytorch attention instead"


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
    subprocess.run(["pkill", "-f", f"[m]ain.py --port {PORT}"], capture_output=True)
    time.sleep(2)
    open(LOG, "w").close()
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


def fallbacks():
    try:
        with open(LOG, errors="ignore") as fh:
            return fh.read().count(FALLBACK)
    except FileNotFoundError:
        return 0


def probe(w, h, length, chunk=0, timeout=1800):
    """One sampling step at this geometry. Returns (ok, new_fallbacks, seconds)."""
    before = fallbacks()
    params = dict(matrix.PROBE, width=w, height=h, length=length, steps=1, cfg=1.0,
                  seed=42, stop_after="sampler", chunk_heads=chunk)
    wf = matrix.build_workflow(**params)
    t0 = time.time()
    try:
        pid = api("/prompt", {"prompt": wf}).get("prompt_id")
    except urllib.error.HTTPError as e:
        return False, -1, 0.0, e.read().decode()[:160]
    while time.time() - t0 < timeout:
        time.sleep(4)
        try:
            hist = api(f"/history/{pid}", timeout=20)
        except Exception:
            continue
        if pid not in hist:
            continue
        ok = hist[pid].get("status", {}).get("status_str") == "success"
        return ok, fallbacks() - before, time.time() - t0, ""
    return False, fallbacks() - before, time.time() - t0, "timeout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--lengths", default="209,277,311,362")
    ap.add_argument("--chunk", type=int, default=0,
                    help="head groups for SageAttentionChunked (0 = off)")
    args = ap.parse_args()

    w, h = (960, 544) if args.draft else (1344, 768)
    lengths = [int(x) for x in args.lengths.split(",")]

    print(f"probing {w}x{h}, 1 step each, latent saved"
          + (f", head-chunked x{args.chunk}" if args.chunk else ", unchunked"))
    print(f"{'frames':>7} {'secs':>6} {'tokens':>9} {'fallbacks':>10}  verdict")
    proc = start_server()
    best = None
    try:
        for n in lengths:
            tok = matrix.tokens(w, h, n)
            ok, fb, secs, err = probe(w, h, n, args.chunk)
            if err:
                print(f"{n:>7} {n/24:>5.1f}s {tok:>9,} {'—':>10}  ERROR {err[:60]}")
                continue
            if fb == 0 and ok:
                verdict = "CLEAN — fast attention path"
                best = n
            elif fb > 0:
                verdict = f"FELL BACK ({fb} calls) — too big"
            else:
                verdict = "failed for another reason"
            print(f"{n:>7} {n/24:>5.1f}s {tok:>9,} {fb:>10}  {verdict}")
            if fb > 0:
                break        # everything longer is worse; stop burning time
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            proc.terminate()

    print()
    if best:
        print(f"LONGEST CLEAN at {w}x{h}: {best} frames = {best/24:.1f}s "
              f"({matrix.tokens(w,h,best):,} tokens)")
    else:
        print(f"nothing in {lengths} ran clean at {w}x{h}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        subprocess.run(["pkill", "-f", f"[m]ain.py --port {PORT}"], capture_output=True)
        sys.exit(130)
