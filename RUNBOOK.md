# H3 video — the working configuration

_Operational runbook. Everything here is measured on this box (2× RTX 5060 Ti
16 GB, 31 GB RAM). The reasoning lives in the `opencode-llama.cpp` docs; this
file is only what to type and what must be true._

**Recovery points**

| repo | remote | branch |
|---|---|---|
| `~/Projects/comfyui` | `fork` = github.com/dushyant30suthar/ComfyUI | `h3-2gpu-v0.30.1` |
| `~/Projects/opencode-llama.cpp` | `origin` = github.com/dushyant30suthar/opencode-llama.cpp | `main` |

**`origin` in the comfyui repo is UPSTREAM comfyanonymous/ComfyUI, not yours.**
Push to `fork`. Pushing this branch to `origin` would be a pull request against
upstream ComfyUI, which is not what you want.

If anything is ever lost: `git checkout h3-2gpu-v0.30.1` in comfyui restores
every node, tool and config below. **Never work on master** — it is stock
upstream ComfyUI.

---

## 0a. Rebuilding this from nothing

Everything below is what git CANNOT give you. The repo has the code, the nodes,
the tooling and this document; it does not have 72 GB of weights, the Python
environment, or the knowledge of which versions actually work. That is here.

**Verified working environment** (captured 2026-08-05 from the box that rendered
the Udaipur→Himachal film):

| | version |
|---|---|
| ComfyUI | 0.30.1 |
| Python | 3.12.12 |
| torch | **2.13.0+cu130** (CUDA 13.0) |
| sageattention | **2.2.0** |
| triton | 3.7.1 |
| numpy / safetensors / transformers | 2.5.1 / 0.8.0 / 5.14.1 |
| comfyui-frontend-package | 1.47.12 |
| comfyui-workflow-templates | 0.11.27 |
| NVIDIA driver | 610.43.03 |
| GPU | 2× RTX 5060 Ti 16 GB, **sm_120** (Blackwell) |
| host RAM | 31 GB |

**`sageattention` and `triton` are NOT in `requirements.txt`** — they are
installed by hand and everything in this runbook depends on sage being present.
A fresh `pip install -r requirements.txt` gives you a ComfyUI that runs H3 at
roughly a third of the speed and never tells you why.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install sageattention==2.2.0 triton==3.7.1
```

sm_120 is recent enough that torch/sage wheels must match it; older sage builds
compile but produce garbage on Blackwell.

**Model files** — 72 GB, none in git. The tree must look exactly like this,
because the filenames are hardcoded in `bench/matrix.py`:

```
models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors   20G   t2v + i2v
models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors  20G   r2v
models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors          15G
models/vae/minimax_h3_video_vae_fp16.safetensors                          4.9G
models/vae/minimax_h3_audio_vae_fp32.safetensors                          578M
```

Get them from the Comfy-Org MiniMax H3 repackaged repo on Hugging Face (the
int8 `convrot` builds — **not** the NVFP4 ones; int8 measurably holds shape
better through fast motion). The other files in `models/` belong to LTX/Wan and
are unrelated to H3.

**A local source fix that a `git pull` on master would destroy:**
`comfy/ldm/minimax/audio_vae.py` carries a one-line correction (`.to(x.dtype)`
was missing `device=`). It is committed on this branch. **Never work on master.**

**Verify a rebuild actually works** before trusting it:

```bash
.venv/bin/python -c "import torch,sageattention; \
  print(torch.__version__, torch.cuda.get_device_capability(0))"   # expect 2.13.0+cu130 (12,0)
bench/memguard.sh -- .venv/bin/python bench/gen.py --prompt "a still lake at dawn" \
  --length 124 --draft --sampler res_multistep --steps 10 --nocache
grep -c "using pytorch attention instead" bench/gen-server.log      # MUST be 0
```

If that last number is not 0, sage is not active and nothing else in this
document will reproduce.

---

## 0. Your workflow, script to final

```
  shot list (.txt)  ->  produce.sh  ->  storyboard.py  ->  fix shots  ->  assemble.sh
     what to make       render it       judge it         re-render      cut the film
```

**1 — Write the shot list.** One text file. Prompts separated by blank lines,
`#` for comments, directives on the first lines of a prompt. This file IS the
film; everything else is mechanical.

```
@ref: j-car-dirt-track.png,j-car-lane.png
<Picture 1> and <Picture 2> show the same car: a silver-grey hatchback...
Camera: slow aerial descent. Timing: the sun crests at two-thirds.
Style: cinematic, ochre and warm white. Audio: wind over dry grass.

@chain
The car begins to roll forward, dust lifting behind it. Camera: the aerial
rises and swings behind. Timing: the plume catches the light near the end.
```

**2 — Pick a profile.** `bench/profiles.py` lists them with cost and reasoning.
`v2-fast` unless you have a reason.

**3 — Render.** One command, unattended, resumable, re-cuts after every shot.

```bash
bench/produce.sh myshots.txt v2-fast myfilm
```

**4 — Judge.** Do not scrub three minutes of video.

```bash
.venv/bin/python bench/storyboard.py myfilm      # every shot, start/middle/end
```

**5 — Fix individual shots.** Delete the bad one, re-render just it, re-cut.
Keep the old take — a retake is not automatically better.

```bash
mkdir -p output/myfilm/takes
cp output/myfilm/shot-004_*.mp4 output/myfilm/takes/shot-004_take1.mp4
rm output/myfilm/shot-004_*.mp4
.venv/bin/python bench/gen.py --prompt-file fixed.txt --name myfilm/shot-004 \
  --length 362 $(.venv/bin/python bench/profiles.py v2-fast --flags --long)
bench/assemble.sh myfilm
```

**6 — Assemble.** `produce.sh` already does this after every shot. Run it
manually after a retake. Add dissolves only if a chained cut reads badly.

```bash
bench/assemble.sh myfilm                                   # hard cuts, stream copy, seconds
.venv/bin/python bench/assemble-dissolve.py myfilm         # dissolves at chained joins, re-encodes
```

---

## 0b. What you control at prompt time

**Choosing the mode is the real decision.** Everything else is a dial.

| you want | use | why |
|---|---|---|
| a place you have no photo of | *(nothing)* — t2v | prompt only |
| this exact photo, moving | `--first-frame f.png` | frame 0 IS the photo |
| this real subject somewhere new | `--ref a.png,b.png` | carries identity into a new scene |
| longer than 15.1 s | `@chain` | seeds the next clip from this one's end |

**The flags, grouped by what they're for:**

| flag | default | what it decides |
|---|---|---|
| `--prompt` / `--prompt-file` | — | the shot |
| `--length` | 124 | frames. Must satisfy `n % 17 == 5`: 124 (5.2 s), 209, 277, 311, 362 (15.1 s max) |
| `--draft` | off | 960×544 instead of native 1344×768 |
| `--seed` | 42 | change it to reroll the same prompt |
| `--name` | gen | output name; a `/` in it writes to that subfolder |
| **quality** | | |
| `--sampler` | `res_multistep` | leave it. euler needs 40–50 steps for the same result |
| `--steps` | 20 | real steps, since caching is off by default |
| `--nocache` | — | already implied by the profiles; EasyCache costs fidelity |
| **mode** | | |
| `--first-frame` / `--last-frame` | — | i2v; both = interpolate between them |
| `--ref` | — | r2v, up to 9 images. Address them as `<Picture 1>`… |
| `--ref-video` | — | r2v from a rendered clip, up to 3 |
| `--ref-size` | `match` | `max` is 2048 px, better identity, much slower, untested at 362 f |
| **memory — REQUIRED past ~200 frames** | | |
| `--chunk 4` | 0 | without it, attention silently falls back to a 3× slower path |
| `--tiled 32` | 0 | without it the VAE decode exhausts host RAM *after* sampling succeeded |
| `--lowmem` | off | evicts models between stages |

Plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the environment for
15.1 s at native. The profiles and `produce.sh` set all of the memory ones for
you; you only need them when calling `gen.py` by hand.

**Writing the prompt itself.** H3's documented order, and every part earns its
place:

```
Subject -> Scene -> Action -> Camera -> Timing -> Style -> Audio
```

Name an explicit **camera move** and a **timing beat**. 15 s is long enough that
an unspecified shot drifts or stalls halfway. Audio is generated too — describe
it or you get whatever the model infers.

Three rules learned the hard way, each of which cost a render:

- **Say what must stay VISIBLE**, not only what an effect does. "The mist
  thickens and hides the ridge" produced a blank grey screen.
- **Frame the subject large when using `--ref`.** Identity you paid for is
  invisible in a shot where the subject is forty pixels wide.
- **A shot that will be chained FROM should end wide**, not on a push-in. You
  don't choose the frame you hand to the next clip.

---

## 1. Just make me a clip

```bash
cd ~/Projects/comfyui

# see the profiles, what they cost and why
.venv/bin/python bench/profiles.py

# everyday: 15.1s native, res_multistep 10 real steps, ~29 min
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bench/memguard.sh -- .venv/bin/python bench/gen.py \
  --prompt "your prompt here" --length 362 \
  $(.venv/bin/python bench/profiles.py v2-fast --flags --long)

# hero shot: 20 real steps, the upstream reference config, ~53 min
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bench/memguard.sh -- .venv/bin/python bench/gen.py \
  --prompt "your prompt here" --length 362 \
  $(.venv/bin/python bench/profiles.py v3-reference --flags --long)

# iterating on wording: 5.2s, ~1 min, same sampler so it is representative
bench/memguard.sh -- .venv/bin/python bench/gen.py \
  --prompt "your prompt here" --length 124 --draft \
  --sampler res_multistep --steps 10 --nocache
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

`lastframe.sh` deliberately takes the *second*-to-last frame: the final frame of
a VAE-decoded clip is the most likely to carry temporal-tile edge artifacts, and
any artifact there would be inherited as the seed of the entire next clip.

**How exact the join is depends on what you seed with — measured 2026-08-05:**

| seed | frame-0 error vs seed | reads as |
|---|---:|---|
| a clean photograph | 2.7/255 | exact |
| a previously generated frame | **8.6/255** | a small reframe — a mild jump cut |

Seeding from generated output does **not** reproduce the frame. In the measured
case the subject shifted left and the framing widened slightly; the difference
image lit up along the whole subject silhouette, which is displacement rather
than noise. What *does* carry reliably is the scene, the subject's identity, the
lighting and the action — so the two clips read as continuous footage with an
edit in it, not as one unbroken take.

Treat a chained boundary as a **cut you control**, not as an invisible splice.
If the jump reads badly, a ~0.25 s crossfade at that boundary hides it and looks
deliberate — at the cost of re-encoding (concat stream-copy cannot dissolve).

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
