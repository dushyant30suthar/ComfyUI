# Flow: reference-to-video

_Your photographs are shown to the model as conditioning, and it generates a NEW
scene containing that subject. The only way to put a specific real thing
somewhere it was never photographed._

**Use r2v when:** the shot must contain YOUR car, YOUR face, YOUR object, in a
place no photo of it exists.

**Do NOT use r2v when:** the photo itself is the shot (use `FLOW-I2V.md` — it is
cheaper and exact at frame 0), or when nothing in the shot needs to be a specific
real thing (use `FLOW-T2V.md`).

---

## 0. READ THIS FIRST — the bug that made this mode useless

Until 2026-08-05, **every r2v run on this box silently discarded its
references.** Clips generated fine, cost the same as plain t2v, and were produced
from prompt text alone. An entire 12-shot film was rendered that way.

Cause: ComfyUI autogrow inputs need **dot-prefixed** API keys.

```
{"ref_images": {"ref_image_0": [..]}}    validates, runs, DISCARDS silently
{"ref_image_0": [..]}                    TypeError: unexpected keyword argument
{"ref_images.ref_image_0": [..]}         correct
```

Fixed in `bench/matrix.py` (commit `b365a5f3`). Recorded here because the failure
mode is invisible: **a completed run is not a run that used your references.**

**Always verify references landed** — see §5. Do not assume.

---

## 1. Setup — once

r2v uses a **separate 21 GB checkpoint**, `minimax_h3_ref2va_pruned_int8_convrot`.
It is a different file from the `fl2va` one that t2v and i2v use, so switching
modes forces a model reload (~60–90 s). Batch your r2v shots together.

`gen.py` selects it automatically whenever `--ref` or `--ref-video` is present.

```bash
ls models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
```

---

## 2. Prepare references — MEASURED, and it contradicts intuition

**Use whole photographs. Do NOT crop to the subject. Use `--ref-size match`.**

That is the opposite of what seems sensible, and the opposite of what an earlier
draft of this document said. Measured 2026-08-05, 124 f native, seed 777,
res_multistep 10, two references of the same car, identical prompt:

| arm | reference | sizing | cost | result |
|---|---|---|---:|---|
| **A** | whole photo | `match` | **414 s** | ✅ **best value** — plate legible, roof contrast, good proportions |
| B | whole photo | `max` | 967 s | good, but 2.3× the cost for a marginal gain |
| C | cropped to car | `match` | 413 s | ❌ **worst** — plate blurred illegible, contrast lost |
| D | cropped to car | `max` | 521 s | good; recovers what C lost |

Two things fall out of this:

**Cropping then using `match` throws detail away twice.** A crop is already
smaller, and `match` scales it again — 3,952 tokens against the whole photo's
4,032, with the extra loss of everything outside the crop. If you do crop, you
must pay for `max` to get back to parity, and then you have spent 521 s to match
what 414 s gave you.

**Token count is not the whole story.** A gets 4,032 tokens and produces a
legible number plate; C gets 3,952 and produces a smudge. Near-identical budgets,
very different results. The whole-scene reference is doing something the
cut-out is not.

*SPECULATIVE, not established:* the full photo shows the car sitting on ground,
with a horizon, a shadow and real-world scale, and H3 reads references through a
vision-language encoder that likely handles a natural photograph better than a
floating cut-out. A moderate crop — subject large but ground and horizon kept —
has NOT been tested and would distinguish this from sampling noise.

**Caveat: n=1 per arm.** Changing the reference changes the whole sampling
trajectory, so some of the A-vs-C gap could be luck. The gap was visible enough
to act on; it is not a law.

Practical preparation, then, is almost nothing:

```python
from PIL import Image, ImageOps
ImageOps.exif_transpose(Image.open("photo.jpg")).convert("RGB").save("input/car-ref.png")
```

Fix EXIF rotation, save as PNG, done. **Do not resize. Do not crop to canvas** —
canvas-sized references make `match` and `max` compute the same scale, which
silently turns `--ref-size` into a no-op.

**Give 2–3 angles.** Up to 9 images and 3 videos are accepted. Different angles
that show identity-bearing detail beat more shots of the same view.

---

## 3. Write the prompt — you MUST address the references

References the prompt never names are still encoded, still cost tokens on every
sampling step, and **steer nothing**. Address them as `<Picture 1>`,
`<Picture 2>`… in the order given; videos are `<Video 1>`.

**Name the identity-bearing detail explicitly.** Not "a grey hatchback" — say
what to look at.

```
<Picture 1> and <Picture 2> show the same specific car from two angles.
Reproduce that exact vehicle faithfully: its body shape and proportions, its
silver-grey paint with the black contrast roof, the shape of its headlights and
tail lights, the pattern of its alloy wheels, and the trim along its flank.
That car stands on a wet cobbled street in an old town at dusk...
Camera: slow dolly along the car's flank, the car filling most of the frame.
```

**Frame the subject LARGE.** Reference identity is invisible in a shot where the
subject is forty pixels wide — a distant car reads as a generic silhouette no
matter how many references you paid for.

---

## 4. Generate

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bench/memguard.sh -- .venv/bin/python bench/gen.py \
  --prompt-file shot.txt --ref car-front.png,car-rear.png --ref-size match \
  --name myfilm/shot-002 --length 362 \
  $(.venv/bin/python bench/profiles.py v2-fast --flags --long)
```

`--ref-video clip.mp4` uses a rendered clip as a reference instead — the model
sees the subject in motion from several angles. Max 3, 2–15 s each.

---

## 5. VERIFY the references actually landed

Non-negotiable, given §0. The cheapest signal is **cost**:

| | 124 f, res_multistep 10 |
|---|---:|
| no references (t2v) | 354 s |
| **2 references, `match`** | **414 s** |

**If an r2v run costs the same as t2v, your references are being dropped.**
That single number would have caught the bug months of work earlier.

Then look at the clip. Fine, distinctive detail is the tell — a number plate or a
badge transferring is proof the pathway is live.

---

## 6. Test cheap, then produce

Never discover an identity problem five hours into a batch.

```bash
# ~7 min identity check at 5.2s, native resolution
.venv/bin/python bench/gen.py --prompt-file test.txt \
  --ref car-front.png,car-rear.png --ref-size match \
  --length 124 --sampler res_multistep --steps 10 --nocache \
  --chunk 4 --tiled 32 --lowmem --name test/id
```

Judge it. Good → run the film at `--length 362`. Not good → re-crop, swap angles,
name different details in the prompt, retry. Each loop is ~7 min against ~29 for
a real shot.

---

## What r2v holds — MEASURED 2026-08-05, with references actually working

- ✅ **Fine distinctive detail transfers.** A number plate was reproduced legibly
  from a reference photo, along with the grille badge, contrast roof and
  headlight signature.
- ⚠️ **Not a copy.** Repeating fine geometry — alloy wheel spoke patterns — drifts
  first, and proportions can read slightly off. The model generates a very good
  likeness; it does not clone.
- ❌ **Nothing transfers if the subject is small in frame.** Measured on the
  journey film: the same car held convincingly filling the frame and became a
  generic silhouette when distant, with *more* references and identical cost.

**Structural note:** H3 encodes references as Qwen3-VL vision tokens plus VAE
latents — there is no face-recognition component anywhere in the path. That is
why it works on a car at all, unlike the entire InstantID/PuLID/PhotoMaker family
which requires a detectable face and crashes without one.

---

## Cost

| config | 124 f | notes |
|---|---:|---|
| no refs | 354 s | baseline |
| 2 refs, `match` | 414 s | **+17%** |
| 2 refs, `max` | 967 s | **2.3×** — 2048px short edge; reference tokens ride through every sampling step |

`--ref-size max` was long recorded as free. That was measured while references
were being discarded entirely. Measured properly it costs **2.3×** and, on whole
photographs, buys only a marginal gain. **`match` is the default and the right
choice**; `max` is only worth it if you are cropping (see §2) or a specific
detail refuses to transfer.

At 362 frames native we run at 15.6 GB of 16 GB with no headroom — **assume `max`
will OOM at full length until shown otherwise.** Test at 124 frames first.
