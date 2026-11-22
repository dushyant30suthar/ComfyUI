# Flow: text-to-video

_Prompt in, clip out. No images involved. The simplest mode and the one to use
whenever the shot is a place, not a specific thing._

**Use t2v when:** you have no photograph of what you want, and nothing in the
shot has to be a *specific* real object or person. Landscapes, establishing
shots, weather, crowds, anything generic.

**Do NOT use t2v when:** the shot must contain your car, your face, or any real
subject. It will invent a plausible one. See `FLOW-R2V.md`.

---

## 1. Setup — once

Nothing beyond a working install. See `RUNBOOK.md` §0a if rebuilding from
scratch; the checkpoint used here is `fl2va`, which is also what i2v uses, so
t2v and i2v can share a batch without reloading anything.

```bash
cd ~/Projects/comfyui
.venv/bin/python bench/profiles.py        # confirm v2-fast is CURRENT
```

---

## 2. Write the prompt

H3's documented order. Every part earns its place:

```
Subject -> Scene -> Action -> Camera -> Timing -> Style -> Audio
```

**Name an explicit camera move and a timing beat.** 15 s is long enough that an
unspecified shot drifts or stalls halfway. "Camera: slow aerial descent" and
"Timing: the sun crests the ridge at the two-thirds mark" are doing real work.

**Audio is generated too.** Describe it or you get whatever the model infers.
Speech runs ~2.5 words/second, so a 15.1 s shot holds about **38 words** of
dialogue — write less and the model invents sound to fill the silence.

Three rules that each cost a render to learn:

- **Say what must stay VISIBLE**, not only what an effect does. "The mist
  thickens and hides the ridge" produced a blank grey screen for the last
  seconds of a shot.
- **A shot you will chain FROM should end wide**, not on a push-in — you do not
  choose the frame you hand to the next clip.
- Keep recurring subjects described **identically** across prompts. The model has
  no memory between clips; consistency comes from the words. (For a *specific*
  subject this is not enough — use r2v.)

Example:

```
A still lake at dawn in Udaipur, white palaces rising from the water. Mist lies
on the surface, the far shore a soft silhouette of domes and hills. The water
barely moves; a single boat crosses far off. Camera: slow aerial descent from
high above the lake, settling toward the water. Timing: the sun clears the far
ridge at the halfway point and light floods across the water. Style: cinematic,
warm gold and pale blue, anamorphic, soft haze. Audio: distant temple bells,
water lapping, birds waking.
```

---

## 3. Test cheap before you commit

A 15.1 s native shot costs ~29 min. A 5.2 s test costs ~6 min and tells you
whether the wording works.

```bash
.venv/bin/python bench/gen.py --prompt-file shot.txt --name test/shot1 \
  --length 124 --sampler res_multistep --steps 10 --nocache
```

Iterate on wording here, not at full length.

---

## 4. Produce

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bench/memguard.sh -- .venv/bin/python bench/gen.py \
  --prompt-file shot.txt --name myfilm/shot-001 --length 362 \
  $(.venv/bin/python bench/profiles.py v2-fast --flags --long)
```

Or a whole shot list at once — see `RUNBOOK.md` §10 for `produce.sh`.

**The memory flags are not optional past ~200 frames.** `--chunk 4 --tiled 32
--lowmem` plus `expandable_segments`. The profile supplies them via `--long`.

---

## 5. Verify

```bash
grep -c "using pytorch attention instead" bench/gen-server.log     # MUST be 0
```

Non-zero means attention silently fell back and the clip took ~3× longer than
it should have. This is the one failure that does not announce itself.

---

## Cost

| | length | real steps | time |
|---|---|---:|---:|
| test | 5.2 s | 10 | ~6 min |
| `v2-fast` | 15.1 s | 10 | ~29 min |
| `v3-reference` | 15.1 s | 20 | ~53 min |

15.1 s (362 frames) is the model's trained maximum. Longer needs chaining — see
`FLOW-I2V.md`.

---

## Traps

- **`--steps 20` does not mean 20 steps** if a cache is on. The retired
  `v1-euler-cache` profile skipped 11 of 20. All current profiles disable it.
- **Frame counts must satisfy `n % 17 == 5`**: 124, 209, 277, 311, 362.
- **`--novram` and `--high-ram` are BANNED on this box** — they froze it once and
  cost a hard reset.
