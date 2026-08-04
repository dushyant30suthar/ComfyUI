# H3 video — the working configuration

_Operational runbook. Everything here is measured on this box (2× RTX 5060 Ti
16 GB, 31 GB RAM). The reasoning lives in the `opencode-llama.cpp` docs; this
file is only what to type and what must be true._

**Recovery points**

| repo | commit | branch |
|---|---|---|
| `~/Projects/comfyui` | `694abe26` | `h3-2gpu-v0.30.1` |
| `~/Projects/opencode-llama.cpp` | `5689935` | `main` |

If anything is ever lost: `git checkout h3-2gpu-v0.30.1` in comfyui restores
every node, tool and config below. **Never work on master** — it is stock
upstream ComfyUI.

---

## 1. Just make me a clip

```bash
cd ~/Projects/comfyui

# best quality this machine can do — 15.1s, native, all 20 steps, ~57 min
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bench/memguard.sh -- .venv/bin/python bench/gen.py \
  --prompt "your prompt here" \
  --length 362 --steps 20 --nocache --chunk 4 --tiled 32 --lowmem

# everyday: 15.1s, native, cache on (~10 real steps), ~30 min
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bench/memguard.sh -- .venv/bin/python bench/gen.py \
  --prompt "your prompt here" \
  --length 362 --steps 20 --chunk 4 --tiled 32 --lowmem

# draft, for iterating on a shot — 15.1s at 960x544, ~10 min
bench/memguard.sh -- .venv/bin/python bench/gen.py \
  --prompt "your prompt here" \
  --length 362 --steps 20 --draft --tiled 32 --lowmem

# short and quick — 5.2s native, ~6 min
bench/memguard.sh -- .venv/bin/python bench/gen.py --prompt "your prompt here"
```

Use `--prompt-file some.txt` for long prompts. Output lands in
`output/video/`, and a record of every run goes to `bench/gen.jsonl`.

---

## 2. What the flags do, and when each is REQUIRED

| flag | required when | what happens without it |
|---|---|---|
| `--chunk 4` | any clip past ~13 s at native | SageAttention cannot allocate its 3.64 GiB scratch, comfy **silently** swaps in pytorch attention at ~2.9× slower per call. The run still succeeds — it just takes three times as long and says nothing. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 15.1 s at native | OOM on a 1.09 GiB allocation with ~875 MiB free and 2.4 GiB stranded in fragments. This is a *fragmentation* fix — it does nothing against genuine exhaustion. |
| `--tiled 32` | past ~200 frames | The one-shot VAE decode materialises the whole pixel tensor and exhausts host RAM **after sampling has already succeeded**. Worst possible time to fail. |
| `--lowmem` | past ~200 frames | Comfy keeps ~39 GB of models staged (15 GB encoder + 20 GB DiT + 5 GB VAE) through the decode. This evicts between stages; it moved MemAvailable from 2.0 GB to 4.4 GB. |
| `--nocache` | when you want the steps you asked for | EasyCache 0.4 skips half of them — `--steps 20` buys 10 real ones. |

**`--chunk 4`, not 8.** Four frees 1.64 GiB for ~6.6% time; eight frees 1.91 GiB
for ~15%. The extra headroom is never needed.

Everything runs under `bench/memguard.sh`. It has killed the box's benchmarks
many times and prevented one repeat of a hard reset; do not skip it on long runs.

---

## 3. What is actually possible

Measured, same prompt and seed:

| resolution | length | real steps | time |
|---|---|---:|---:|
| 960×544 | 15.1 s | 10 | 10.3 min |
| 1344×768 | 13.0 s | 20 | 43.7 min |
| 1344×768 | 15.1 s | 10 | 29.6 min |
| **1344×768** | **15.1 s** | **20** | **56.7 min** |

**15.1 s is the model's limit, not the machine's.** H3 is trained on 124–362
frames; the node accepts more but it is untrained territory.

Continuous generation, warm (model already staged), deployed config:

| canvas | per clip | clips/hour |
|---|---:|---:|
| 1344×768 | 331.8 s | 10.8 |
| 960×544 | 132.5 s | 27.0 |
| 832×480 | 105.2 s | 34.2 |

Valid frame counts must satisfy `n % 17 == 5`: 124 (5.2 s), 209 (8.7 s),
277 (11.5 s), 311 (13.0 s), 362 (15.1 s).

---

## 4. Verify it actually worked

The one failure mode that does not announce itself:

```bash
grep -c "using pytorch attention instead" bench/gen-server.log
```

**Must be 0.** Anything above zero means attention fell back and the clip took
~3× longer than it should have. This is the single most useful check after any
long run.

Also worth a look:

```bash
grep -c "out of memory" bench/gen-server.log        # must be 0
grep "EasyCache - skipped" bench/gen-server.log     # how many steps were real
grep TRIPPED bench/h3-logs/*.log                    # watchdog kills
```

---

## 5. The pieces, and what breaks if one goes missing

| path | why it matters |
|---|---|
| `custom_nodes/sage_chunked/` | **The one that changed what this box can do.** Head-grouped attention; without it native tops out at 13 s. |
| `custom_nodes/h3_dualpipe/` | Two-GPU pipeline split. Works, but ~11% slower than one GPU — not for production. |
| `custom_nodes/sage3_select/` | SageAttention 3 selector. Measured and **not adoptable** — OOMs at 15 s, swaps the box with the cache. Kept for the record. |
| `bench/gen.py` | Prompt → clip. All three modes (t2v / i2v / r2v) and every flag above, documented in `--help`. |
| `bench/produce.sh` | Batch renderer. Directives `@image` / `@ref` / `@refsize` / `@chain`; one server per clip; resumable; re-cuts the film after every shot. |
| `bench/lastframe.sh` | Pulls a clip's second-to-last frame into `input/` — the mechanism for joining clips past 15.1 s. |
| `bench/assemble.sh` | ffmpeg concat of the rendered shots; tolerates missing ones. |
| `bench/memguard.sh` | The watchdog. Calibrated on eight recorded events. |
| `bench/costmodel.py` | Predicts a run's time before you spend it. Within 2.6 s on every single-GPU row. |
| `bench/probe-length.py` | Finds the longest clip that keeps the fast attention path, at any canvas. |
| `bench/quality.py` | Renders one prompt under every quality-relevant config for side-by-side judging. |
| `comfy/ldm/minimax/audio_vae.py` | Carries a local one-line fix (`.to(x.dtype)` was missing `device=`). **A `git pull` on master would lose this.** |

`custom_nodes/` is in `.gitignore`, so the nodes were force-added. If you ever
add another, it needs `git add -f`.

---

## 6. Known limits

- **15.1 s** — the model's trained maximum, not the hardware's.
- **1344×768** — `adapt_canvas()` caps area at 1.03 MP. 1080p was never reachable.
- **Two clips at once: no.** Each ComfyUI instance stages ~35 GB into host RAM;
  two need ~43 GB against 31 GB and thrash. **64 GB of DDR4 fixes this** and is
  the only unlock money can buy — it would roughly double throughput.
- **The second GPU does not help a single clip.** Attention must fit on one card;
  the split distributes weights, not attention.
- **Do not use `--novram` or `--high-ram`.** 21 GB DiT + 15.7 GB encoder against
  31 GB froze this machine once and cost a hard reset.

---

## 7. If something breaks

**Clip is 3× slower than the table says** → attention fell back. Check §4, add
`--chunk 4`.

**OOM during sampling** → add `expandable_segments`, then `--chunk 8`, then drop
to `--draft`.

**Dies after sampling, at the decode** → missing `--tiled 32 --lowmem`. The
sampling work is lost; there is no resume.

**Watchdog killed it** → read the line. `MemAvailable` low with swap flat is
usually a healthy run and the rules are tuned for that; low *and* swap climbing
is real. Do not retune it without reading §6 of the session-2 doc — it took
eight kills, seven of them wrong, to get right.

**Everything is broken** → `git checkout h3-2gpu-v0.30.1 && git reset --hard 694abe26`.

---

## 8. Where the reasoning lives

In `~/Projects/opencode-llama.cpp/docs/`:

- `h3-levers-2026-08-04-session2.md` — everything measured. §10 is the result.
- `HANDOVER-2026-08-04b-h3-video.md` — cold-readable state of the work.
- `h3-multigpu-and-attention-memory-research-2026-08-04.md` — research, kept
  separate from measurement, with every claim tagged VERIFIED / REPORTED /
  UNTESTED.
- `minimax-h3-video-2026-08-03.md` — the first session: weights, geometry, setup.

---

## 9. The three generation modes

All three run through `bench/gen.py`, so every memory flag in §2 applies
unchanged to all of them.

| mode | flag | what it does | checkpoint |
|---|---|---|---|
| text-to-video | *(none)* | prompt only | fl2va |
| image-to-video | `--first-frame f.png` | animates **from** that exact image | fl2va |
| | `--last-frame f.png` | animates **to** it; with both, interpolates | fl2va |
| reference-to-video | `--ref a.png,b.png` | carries the **identity** of those subjects into a **new** scene | ref2va |
| | `--ref-video clip.mp4` | carries motion and look forward from a rendered clip | ref2va |

The distinction that matters: `--first-frame` reproduces the image and moves it,
so the shot is locked to that photograph's composition and location. `--ref`
does not reproduce anything — it puts the *subject* somewhere it has never
been. To get a specific real car onto a road you have no photo of, you need
`--ref`.

**Reference prompts must address their references.** The node presents them as
`<Picture 1>`, `<Picture 2>`, … (1-based, in the order given; videos are
`<Video 1>`, audio `<Audio 1>`). A reference the prompt never names is still
encoded and still costs tokens on every sampling step, but steers nothing.
Limits: 9 images, 3 videos, 3 audio.

`--ref-size` is `match` (scale references to the generation's pixel area) or
`max` (2048px short edge — better identity, several times slower, because
reference tokens ride through every step).

**Measured 2026-08-05:** r2v with two references at `match`, 1344×768 × 362
frames, `--chunk 4 --tiled 32 --lowmem`, ran **167 s/step** against ~169 s/step
for plain t2v at the same shape. Two references at `match` size are effectively
free. `max` has not been measured at 362 frames and should be assumed to be the
thing that breaks the memory budget.

### Joining clips past the 15.1 s ceiling

```bash
bench/lastframe.sh output/journey/shot-002_00001.mp4 chain-003
.venv/bin/python bench/gen.py --first-frame chain-003.png --prompt "..." ...
```

`gen.py` reproduces its seed frame at frame 0 to within 2.7/255, so the cut is
invisible and the two clips play as one 30 s take. `lastframe.sh` deliberately
takes the *second*-to-last frame: the final frame of a VAE-decoded clip is the
most likely to carry temporal-tile edge artifacts, and any artifact there would
be inherited as the seed of the entire next clip.

Chain two or three clips, not ten — each generation inherits the previous one's
drift, and it compounds.

---

## 10. Batch production (overnight films)

```bash
bench/produce.sh <prompts-file> <profile> <output-subfolder>
bench/assemble.sh <output-subfolder>          # concatenates into one film
```

**Prompt file format:** prompts separated by BLANK LINES; a prompt may span
several lines; `#` starts a comment.

A prompt may open with directive lines, stripped before it is sent:

| directive | effect |
|---|---|
| `@image: f.png` | image-to-video from `input/f.png` |
| `@ref: a.png,b.png` | reference-to-video; address them as `<Picture 1>`, `<Picture 2>` |
| `@refsize: max` | 2048px references — better identity, much slower |
| `@chain` | image-to-video seeded with the **previous shot's last frame** |

`@chain` resolves at run time against whatever the previous shot actually
produced. If that shot failed, the clip renders standalone as text-to-video
rather than aborting — a self-contained clip beats a hole in the film.

Write each prompt in H3's documented order —
Subject → Scene → Action → Camera → Timing → Style → Audio — and name an
explicit camera move and a timing beat. 15 s is long enough that an unspecified
shot drifts or stalls halfway.

| profile | canvas / length | real steps | per clip | 12 shots |
|---|---|---:|---:|---:|
| `native` | 1344×768, 15.1 s | 20 | 56.7 min | 11.3 h |
| **`fast`** | **1344×768, 15.1 s** | **10** | **29.6 min** | **5.9 h** |
| `draft` | 960×544, 15.1 s | 10 | 10.3 min | 2.1 h |
| `draft-short` | 960×544, 5.2 s | 10 | ~2 min | 25 min |

`fast` is usually the right pick for a finished piece: the resolution ladder
showed canvas dominates perceived quality and step count is second, so twelve
native shots beat twenty-four draft ones.

**Properties that matter at 3am:**

- **One server per clip.** Host RAM accumulates across prompts and the third
  execution thrashes. A ~90 s boot per clip is cheap against losing the batch.
- **A failure does not stop the batch.** It logs and moves on.
- **Resumable** — already-rendered shots are skipped, so re-running continues
  where it stopped.
- **Silent-fallback detection** — if a clip drops to the slow attention path the
  log says so, rather than you inferring it from a strange timestamp.
- **Assembly tolerates gaps.** 9 of 12 shots still produces a 9-shot film in the
  right order. Better to wake to something watchable.

### Continuity, since the model has no memory between clips

Nothing carries across shots automatically. Describe the recurring subject
**identically every time** — same colour, same distinguishing features — or it
will change between cuts. For tighter continuity use `--first-frame` to chain
from the previous shot's last frame, or reference-to-video when a subject must
persist across scenes that are not continuous.

### Editing a batch that is already running

`produce.sh` splits the prompt file into temp files **at launch**, so editing the
source file mid-run changes nothing. To correct a prompt in flight, patch the
live copies:

```bash
TD=$(ls -dt /tmp/tmp.* | while read d; do [ -f "$d/001.txt" ] && echo "$d" && break; done)
sed -i 's/old text/new text/' $TD/*.txt
```
