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

## 2. Prepare references — the step that decides everything

**Crop to the SUBJECT. Keep the resolution high. Do NOT crop to 1344×768.**

This is the exact opposite of i2v seed preparation, and getting it backwards is a
mistake already made here: canvas-sized references make `--ref-size max` a no-op,
because both sizing modes then compute a scale of 1.0.

Why cropping matters — the reference is scaled before encoding, so a wide shot
spends its token budget on sky and tarmac instead of on your subject:

| reference | `match` gives | `max` gives |
|---|---|---|
| full 2252×4000 photo | 768×1344 (car is ~⅓ of it) | 2048×3648 |
| cropped to the car | car fills the frame | car fills the frame, 2048px |

```python
from PIL import Image, ImageOps
im = ImageOps.exif_transpose(Image.open("photo.jpg"))    # fix rotation
im.crop((left, top, right, bottom)).convert("RGB").save("input/car-ref.png")
```

**Give 2–3 angles.** Up to 9 images and 3 videos are accepted. Angles that show
the identity-bearing detail — badge, plate, lights, wheels — beat more images of
the same view.

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
| 2 refs, `max` | *measuring* | 2048px short edge; substantially slower — reference tokens ride through every sampling step |

`--ref-size max` was long assumed free. That was measured while references were
being discarded. **`match` is the default and the safe choice**; reach for `max`
only when a specific detail refuses to transfer, and expect to pay for it.

At 362 frames native we run at 15.6 GB of 16 GB with no headroom — **assume `max`
will OOM at full length until shown otherwise.** Test at 124 frames first.
