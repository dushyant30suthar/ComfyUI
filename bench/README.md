# H3 tuning matrix — 2× RTX 5060 Ti (sm_120), i5-9400F, 31 GiB RAM

## What the research turned up

**There are no published sampler settings for H3.** MiniMax's blog, the ComfyUI
day-0 post and the docs tutorial all omit steps/cfg/sampler, and
`comfyui-workflow-templates` has no H3 template as of 0.11.28. The only
community claim found: `res_multistep` + `beta`/`normal` scheduler reportedly
beats `simple` for reference-heavy prompts. Everything else below is either read
out of ComfyUI v0.30.1's own `--help` and source (authoritative) or is an
explicit hypothesis to measure.

**The one number that matters.** H3 runs *full dense attention* —
`optimized_attention(q, k, v, mask=None)` in `comfy/ldm/minimax/model.py`, no
sparsity, no windowing — over one packed text+video+audio sequence. By FLOP
count that is **58% of the work at 5s and 80% at 15s**. So the attention backend
is not one knob among many; it is the knob. Everything in phase A exists for
that reason, and phase A is the only one also run at CONFIRM length, because a
short probe systematically understates an attention win.

**What is installed here right now**

| | |
|---|---|
| torch | 2.13.0+cu130, sm_120, flash/mem-efficient/cuDNN SDPA all available |
| triton | 3.7.1 ✅ (SageAttention needs it) |
| nvcc | ✅ `/usr/local/cuda-13.3/bin/nvcc` (rpm `cuda-nvcc-13-3`) — **not on PATH**, must be added |
| sageattention | built from source by `install-accel.sh` |
| sageattn3 (Blackwell FP4) | built from source, but see the reachability note |
| flash_attn | opt-in only: `WITH_FLASH=1 bench/install-accel.sh` |

Run `bench/install-accel.sh` first. Rows whose module is absent are skipped
automatically rather than silently reported as a default-config result.

**Two traps, both hit while setting this up.**

*PyPI's `sageattention` is version 1.0.6* — the old pure-Triton implementation
that exports only `sageattn`/`sageattn_varlen` and declares no `__version__`.
It installs and imports perfectly, which is exactly what makes it dangerous: an
"is it importable?" check passes and you benchmark **v1** while believing you
measured v2. The 30–35%-over-SDPA figures refer to the INT8-QK **v2.2** kernels,
which only exist in a source build of `thu-ml/SageAttention`. The install script
now uninstalls the wheel and builds from source unconditionally.

*nvcc is present but not on PATH.* The NVIDIA driver alone cannot compile CUDA
kernels, and `command -v nvcc` returning nothing looks identical to "no toolkit
installed". It is at `/usr/local/cuda-13.3/bin/nvcc`; the script now searches for
it. Without this, both sage2 and sage3 silently fall back to being skipped.

**SageAttention 2 vs 3.** ComfyUI dispatches both, differently
(`comfy/ldm/modules/attention.py`): `sageattention` → `sageattn`, selected by
the `--use-sage-attention` launch flag; `sageattn3` → `sageattn3_blackwell`,
registered as attention function `"sage3"` but **not reachable from any launch
flag or stock node in v0.30.1** — it needs a custom node setting
`transformer_options["optimized_attention_override"]`. So sage2 is the clean
experiment; sage3 (FP4, built for sm_120, reported 2–5× over FlashAttention) is
a stretch goal needing `ComfyUI-SageAttention3`. Installing it now only takes
the build off the critical path.

## Design

Two axes with very different costs:

- **Launch flags** (attention, memory, precision) are read at import, so each row
  needs a server restart — ~10s boot plus ~30–60s to reload 21 GB off NVMe.
- **Workflow params** (sampler, steps, cfg, device placement, geometry) are just
  a different graph against a resident model, and are free.

`sweep.py` nests them accordingly: one restart amortises over a whole workflow
phase.

Two stages, mirroring the `knob-ab.sh` / `deep-confirm.sh` split from the LLM work:

| | geometry | steps | cfg | tokens | why |
|---|---|---|---|---|---|
| **PROBE** | 1344×768, len 22 | 10 | 1.0 | 7,130 | throughput only — deliberately below the trained 124–362 frame range, so output is garbage by design. Keeps a row near ~2 min instead of ~40. |
| **CONFIRM** | 1344×768, len 124 | 20 | 1.0 | 37,710 | 5.2s, bottom of the trained range. Earned by the top few configs only. |

## Running it

```sh
cd ~/Projects/comfyui
bench/install-accel.sh                       # once — unlocks rows A2/A3

.venv/bin/python bench/sweep.py --list       # 64 rows across 10 phases
.venv/bin/python bench/sweep.py A            # attention backend first
.venv/bin/python bench/sweep.py A --confirm  # then re-rank the winners at 5.2s

# workflow phases share one server; pin the phase-A winner as the base
.venv/bin/python bench/sweep.py D E F G J --base-flags --use-sage-attention
```

It binds **port 8191**, never 8188, so a sweep never fights the server the
opencode panel owns. Results append to `results.jsonl` — an interrupted sweep
resumes by re-running.

## Timing rule

Every number comes from ComfyUI's own `execution_start` → `execution_success`
messages, which carry server-side millisecond timestamps (`execution.py`
`add_message`). Client-side stream timing is **never** used — that is exactly the
mistake that inflated the first exllamav3 decode numbers by ~1.7× and rigged the
whole llama.cpp comparison. `wall_s` is recorded alongside purely as a sanity
check; when the two disagree, the server is right.

## Order to run, and what each phase is worth

1. **A — attention backend.** Expect the largest single win. Run at both stages.
2. **F — cfg and steps.** Not a tuning knob so much as arithmetic: `cfg=1.0`
   skips the negative pass and **halves** total compute. Establish whether H3
   needs CFG at all before optimising anything else.
3. **C — offload and memory.** The DiT is 20.97 GB against 15.5 GiB per card, so
   something spills every step across PCIe. `--async-offload`, `--fast-disk`
   (this box has NVMe) and `--reserve-vram` are the candidates.
4. **D — device placement.** Free to test. Does keeping cuda:0 clear for the DiT
   actually pay, given GPU1 sits on the x4 PCH slot at a measured 2.76 GB/s?
5. **B — `--fast`.** `fp16_accumulation`, `fp8_matrix_mult`, `cublas_ops`,
   `autotune`. ComfyUI's own help calls these untested and quality-affecting.
6. **J — geometry scaling.** Validates the quadratic model and finds the real
   ceiling. Predicted: 3× the length costs ~6× the time.
7. **E, G — sampler/scheduler and sigma shift.** Quality-affecting, so these
   need eyes on the output, not just a stopwatch.
8. **H, I — precision and single-card control.**

## Predictions to falsify

Written down first so the measurements can embarrass them:

- SageAttention 2 gives **>20%** at PROBE length and **more** at CONFIRM, because
  attention's share grows with sequence length.
- `cfg=1.0` is within noise of exactly **2×** faster than `cfg>1`.
- `--highvram` (C4) **OOMs** — 20.97 GB will not sit in 15.5 GiB.
- D1 (DiT alone on cuda:0) beats D0 (everything on cuda:0), but by less than
  hoped, because the spill still crosses PCIe either way.
- D4 (DiT on GPU1) is clearly worst — that card is on the x4 slot.
- J is superlinear: `len 362` costs **~6×** `len 124`, not 3×.
