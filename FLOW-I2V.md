# Flow: image-to-video

_Your photograph becomes frame 0, and the model animates forward from it._

**Use i2v when:** the photograph IS the shot. Your actual cabin, your actual
mountain slope, your actual street. Frame 0 is genuinely your image, so
everything in it is real — and i2v reproduces detail from the seed that no
prompt described.

**Also use i2v for chaining:** seeding a clip with the previous clip's last frame
is how a sequence gets past the model's hard 15.1 s ceiling.

**Do NOT use i2v when:** you want a real subject somewhere it was never
photographed — you are locked to the seed image's composition and location. Use
`FLOW-R2V.md`. And do not use it to hold a *face* for 15 s; it does not.

---

## 1. Setup — once

Same `fl2va` checkpoint as t2v, so the two can share a batch. Nothing extra.

---

## 2. Prepare the seed image — THIS IS THE STEP THAT GOES WRONG

**Crop and resize to EXACTLY the generation canvas: 1344×768** (or 960×544 for
`--draft`). The seed becomes frame 0, so it must match the output geometry.

**This is the opposite of what r2v wants.** r2v references should be cropped to
the *subject* and left at high resolution; i2v seeds must match the *canvas*.
Preparing one set of images and using it for both is a real mistake that has
been made here — at canvas size, r2v's `--ref-size max` silently becomes a no-op.

Also: **honour EXIF rotation**, or a phone photo lands sideways.

```python
from PIL import Image, ImageOps
im = ImageOps.exif_transpose(Image.open("photo.jpg"))       # fix rotation
im = ImageOps.fit(im, (1344, 768), Image.LANCZOS, centering=(0.5, 0.5))
im.convert("RGB").save("input/seed.png")
```

The file must live in `input/`. Pass the **filename only**, not a path.

---

## 3. Write the prompt

Describe **what happens next**, and describe the scene as it actually is in the
photo. Fighting the image wastes the conditioning — if the photo shows a green
hillside, do not prompt "ochre desert".

Same structure as t2v (Subject → Scene → Action → Camera → Timing → Style →
Audio), but the Scene half is already established by the image, so spend the
words on **Action** and **Camera**.

---

## 4. Generate

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bench/memguard.sh -- .venv/bin/python bench/gen.py \
  --prompt-file shot.txt --first-frame seed.png \
  --name myfilm/shot-004 --length 362 \
  $(.venv/bin/python bench/profiles.py v2-fast --flags --long)
```

`--last-frame` also exists. With both, the model interpolates between them.

---

## 5. Chaining — getting past 15.1 s

```bash
bench/lastframe.sh output/myfilm/shot-002_00001_.mp4 chain-003
# -> input/chain-003.png, then use it as --first-frame for the next shot
```

In a batch, put `@chain` as the first line of a prompt and `produce.sh` does it.

**It takes the SECOND-to-last frame deliberately.** The final frame of a
VAE-decoded clip is the likeliest to carry temporal-tile edge artifacts, and here
an artifact is not one bad frame — it seeds the entire next clip.

**MEASURED: the join is a cut, not an invisible splice.**

| seed | frame-0 error |
|---|---:|
| a clean photograph | 2.7/255 |
| a previously generated frame | **8.6/255** |

Seeding from generated output does not reproduce the frame — expect a small
reframe. What *does* carry is scene, subject identity, lighting and action. So a
chained pair reads as continuous footage with an edit in it. It does not
compound (measured 8.62 then 8.71 down a chain), but chain two or three, not ten
— *content* drift accumulates even though the seam error does not.

If a join reads badly: `.venv/bin/python bench/assemble-dissolve.py myfilm`
puts a ~0.25 s dissolve on chained boundaries only.

---

## 6. Verify

```bash
grep -c "using pytorch attention instead" bench/gen-server.log     # MUST be 0
```

And **look at frame 0** — it should be your image.

---

## What i2v does and does not hold, over 15 s

**MEASURED on the journey film:**

- ✅ **Recovers detail no prompt described.** A cabin seed reproduced the sticker
  on the sun visor, the bell hanging from the mirror, the vent surrounds, the
  driver's shirt and his wristband. As a way to get a specific real *place* on
  screen it is far stronger than describing it.
- ❌ **Does not hold a face.** The driver was recognisably the man in the seed
  photo for a few seconds and a generic man in the same shirt by the end. Scene,
  car and wardrobe survive; identity does not.
- ⚠️ **Conditioning is strongest at frame 0 and decays fast.** A degraded seed
  costs the opening beat, not the clip — one shot recovered from a badly smeared
  seed within a second. Prompt an immediate camera move AWAY from the seed and
  the bad frames pass while the camera is already leaving them.

**For a shot whose point is a specific person, prefer r2v with a face reference.**
i2v is right when the point is the *place* and a person merely happens to be in
it.

---

## Cost

Identical to t2v — the seed encode is negligible. ~29 min at `v2-fast`, 15.1 s
native. Chaining adds ~0.8 min per clip for the seed encode.
