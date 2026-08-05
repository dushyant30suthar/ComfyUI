# Open levers — what has not been tested yet

_Backlog for the H3 video setup on this box (2× RTX 5060 Ti 16 GB, 31 GB RAM).
Companion to `RUNBOOK.md`, which is only what is known to work._

Ranked by **expected value ÷ cost to find out**, not by how interesting they are.

Every entry says what is **MEASURED** versus **CLAIMED** versus **GUESSED**, because
the most expensive mistake of this project was carrying a number measured in one
context into another and treating it as fact. Two days were spent on a config
(`euler` + EasyCache) that was never a quality decision at all — it was a
benchmarking default nobody re-examined. Assume everything below is wrong until
it has a measurement next to it.

**Status legend:** 🔴 untested · 🟡 partially tested · ⚫ closed (kept so it is
not re-litigated)

---

## 1. 🔴 Scheduler sweep — the cheapest untested quality lever

**Cost to test:** ~1 h (6 arms × ~6 min at 124 frames) · **Payoff:** unknown, possibly free quality

We have used `scheduler="simple"` for every single render ever done here, for the
same reason we used `euler`: it was in `matrix.PROBE` and nobody questioned it.
The sampler turned out to be the single biggest quality lever on the box and it
was free. **`simple` is exactly the same class of unexamined default.**

Available and untried: `sgm_uniform`, `karras`, `exponential`, `ddim_uniform`,
`beta`, `normal`, `linear_quadratic`, `kl_optimal`.

MEASURED: the shipped ComfyUI r2v template uses `BasicScheduler: simple`, so
`simple` is at least the upstream choice — this is weaker evidence than it looks,
since the template also just picks defaults.

```bash
for s in simple karras beta kl_optimal linear_quadratic sgm_uniform; do
  .venv/bin/python bench/gen.py --prompt-file bench/journey/ab-prompt.txt \
    --name "sched/$s" --seed 777 --length 124 --sampler res_multistep \
    --steps 10 --nocache --scheduler "$s"
done
```

**Do this one first.** It is the same shape of oversight as the sampler, the test
is cheap, and if it pays it pays on every render forever.

---

## 2. 🔴 `--ref-video` — plumbing built, never once executed

**Cost to test:** ~30 min · **Payoff:** potentially the best continuity tool we have

`gen.py --ref-video` was implemented (`LoadVideo` → `GetVideoComponents` →
`ref_videos` dict) and **has never been run**. The node accepts up to 3 reference
videos of 2–15 s.

Why it matters: our clip chaining seeds the next shot from a **single frame**, and
MEASURED that costs 8.6/255 with a visible reframe. A reference *video* gives the
model motion and look across many frames instead of one. GUESSED: continuity
should be substantially better than single-frame chaining.

Note the same trap as `--ref` images: the prompt must address it as `<Video 1>`.

**Untested code is not working code** — that is the exact lesson of the r2v
workflow, which validated for two days and could never have run.

---

## 3. 🔴 `--ref-size max` — the fix for identity at distance

**Cost to test:** ~1 h · **Payoff:** fixes a MEASURED failure

MEASURED on the journey film: r2v held the car convincingly in shot 2 (car fills
the frame) and produced a generic silhouette in shot 8 (car distant), with three
references instead of two and identical cost. Reference identity only survives
when the subject is framed large.

`ref_size="max"` uses a 2048 px short edge instead of scaling references to the
canvas. CLAIMED by the node's own tooltip: better identity fidelity, "several
times slower", because reference tokens ride through every sampling step.

MEASURED: at `match`, two refs cost 167 s/step against t2v's 169 — free. At `max`
the token count rises sharply and **we have zero VRAM headroom at 362 frames**
(15.6 GB of 16). Test at 124 frames first; assume it OOMs at 362 until shown
otherwise.

---

## 4. 🔴 sol-attn — the only speed lever that might not cost quality

**Cost to test:** ~2 h including install · **Payoff:** CLAIMED 20–35%

From MiniMax's own ecosystem chart: *"sol-attn · sage-attn · EasyCache — ~20–35%
each on consumer cards, and they stack."* We run sage-attn; **sol-attn is the one
we do not have.**

`github.com/KingGore/ComfyUI_sol-attn_Blackwell` targets **SM120 / RTX 5090**.
Our cards are sm_120 Blackwell, so the target matches — GUESSED that a 5060 Ti
works where a 5090 is named, UNVERIFIED.

Why this is the lever that unlocks the others: `v3-reference` quality costs
~53 min/clip, which is too expensive for a twelve-shot film. At −25% that becomes
~40 min and reference quality gets affordable overnight. **Every other quality
lever is gated on making time cheaper.**

Sparse attention is an approximation, so it must be quality-tested the same way
v4 was, not adopted on a speed number alone.

---

## 5. 🔴 LoRA on the car and the face — the real identity fix

**Cost to test:** hours to days · **Payoff:** solves the problem references only approximate

MEASURED: i2v does not hold a face for 15 s (drifts to a generic man in the same
shirt), and r2v identity only reads when the subject is framed large. Both are
workarounds for not having the subject *in* the model.

Trainers exist per MiniMax's chart: AI Toolkit (T2V + I2V), DiffSynth-Studio
(including NF4 low-memory variants), Musubi Tuner (CLAIMED 48 GB, "shrinking").

**The blocker is our 31 GB of host RAM**, not the GPUs. DiffSynth's NF4 training
path is the one worth investigating first. UNVERIFIED whether any of them fit.

This is the only route to a car that is reliably *your* car in every shot rather
than a good likeness in the close-ups.

---

## 6. 🟡 The multishot workflow — compare against ours

**Cost to test:** ~1 h to read, ~30 min to run · **Payoff:** possibly a better seam

MiniMax's chart lists a community "Multishot workflow — chains shots past the 15 s
single-clip limit, up to 30 s with audio". That is precisely what
`bench/lastframe.sh` does.

MEASURED for ours: 8.6/255 at the join, a small but visible reframe; consistent
at 8.62 and 8.71 across two different chains, so it does not compound.

Worth reading theirs specifically for **how they condition the join**. If they
overlap latents or condition on several frames rather than seeding from one, their
seam should beat ours and we should adopt it. If they seed from a single frame
too, we already know exactly what that costs and can stop wondering.

---

## 7. 🔴 cfg > 1.0

**Cost to test:** ~20 min · **Payoff:** probably nothing, but it is cheap

Every render has used `cfg=1.0` on the grounds that H3 is guidance-distilled, so
guidance above 1 doubles cost for nothing. That is a reasonable inference and it
has **never been measured here**. The upstream sampling doc does not discuss cfg
at all.

Cheap to falsify at 124 frames. Low expected value; listed because "reasonable
inference, never measured" is the category that already cost us two days.

---

## 8. 🔴 `shift_video` / `shift_audio`

**Cost to test:** ~1 h · **Payoff:** unknown, and it invalidates other measurements

Fixed at 12.0 / 3.0 throughout. CLAIMED by the upstream doc: the two "must move
together", and changing them "disqualifies the profile tiers".

That warning means a shift sweep makes previously measured numbers incomparable.
Only worth running as a deliberate, isolated experiment — not casually.

---

## 9. 🟡 Phase C — offload and memory flags

**Cost to test:** several hours, 13 rows, restart per row · **Payoff:** low

0 of 13 rows measured. Died on the first row at `MemAvailable=0`. `C4 --highvram`
should join `C12 --high-ram` on the permanently excluded list — `--novram` and
`--high-ram` froze this box once and cost a hard reset.

MEASURED (phase N) that this targets a near-nonexistent term. Kept only for
completeness. **`--novram` remains BANNED on this box.**

---

## 10. 🟡 Two-GPU: rebound vbar re-fault path

**Cost to test:** a day of work · **Payoff:** enables an approach already measured as slower

MEASURED: the pipeline split runs, both cards compute, and it is **~11% slower**
than one GPU — structurally, not from the cut point or the prefetch guard. Its
value is VRAM headroom and unbuilt micro-batching, not speed.

Giving rebound vbar allocations a working re-fault path would unblock split memory
pressure. Hard to justify while the split loses to a single card.

---

## Closed — do not re-litigate

| lever | verdict |
|---|---|
| ⚫ `v4-exact-attn` (sage int8 opt-out) | MEASURED: 1.95× cost, **no visible gain**. Sage is quality-safe here. See `bench/profiles.py`. |
| ⚫ Spectrum acceleration | Needs ~6.1 GiB history at native; we have ~0.4 GiB spare. Also an approximation that harms fine detail. Could only work at draft resolution. |
| ⚫ SageAttention 3 | MEASURED not adoptable — OOMs at 15 s, swaps the box when combined with the cache. |
| ⚫ torch.compile | Impossible here: every clip length is a new shape, so it recompiles constantly. |
| ⚫ EasyCache | MEASURED as the main cause of the journey film's artifacting, together with euler. Upstream: cache output "must not be used as a reference". |
| ⚫ Device placement / `--reserve-vram` | MEASURED 0.8% and 1.3%. Noise. |
| ⚫ Resolution above 1344×768 | `adapt_canvas()` caps area at 1.03 MP. 2K was never reachable locally. |
| ⚫ Clips longer than 15.1 s in one shot | 362 frames is the model's trained maximum. Chaining is the only route past it. |

---

## The one thing money fixes

**64 GB of DDR4.** MEASURED: two ComfyUI instances need ~43 GB against our 31 GB
and thrash, so we render strictly one clip at a time. With 64 GB, two concurrent
instances would roughly double throughput — which matters more than any software
lever here, because it is what makes `v3-reference` affordable across a whole
film.

The GPUs are not the bottleneck for throughput. Host RAM is.
