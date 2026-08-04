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
| `bench/gen.py` | Prompt → clip. All the flags above, documented in `--help`. |
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
