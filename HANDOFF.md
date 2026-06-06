Yes — this is a **great candidate for a Codex handoff**.

And I think the handoff should be framed very explicitly as:

> **The stock ImageQuantize node is adaptive-palette quantization.
> What we need is fixed-palette remapping with clean alpha handling.**

That distinction is the whole story.

---

## My recommendation

Ask Codex to build a **new ComfyUI custom node**, not to “improve” the existing behavior mentally.

A good node name would be something like:

- `FixedPaletteQuantize`
- `PaletteRemap`
- `ImageToFixedPalette`

I slightly prefer **`FixedPaletteQuantize`** because it says exactly what it does.

---

# Ready-to-paste Codex handoff

````md
# Codex Handoff: ComfyUI Custom Node for Fixed-Palette Quantization

## Context

I am working in ComfyUI on a sprite/pixel-art workflow.

The built-in/custom `ImageQuantize` node I tested is not sufficient for my needs because it does **adaptive quantization**: it chooses a palette automatically based on the image. In practice, this wastes many palette entries on slight variations of white/gray and does not let me force an image into a deliberately chosen small palette.

My use case is pixel-art/sprite cleanup from an RGBA source image (e.g. a cat sprite), where I want:

- strict palette control
- pure transparency handling
- no accidental auto-generated palette
- low color count (e.g. 7–8 visible colors + transparency)
- predictable, repeatable output

This is for a ComfyUI custom node.

---

## Goal

Create a new ComfyUI custom node that takes an input image and remaps every non-transparent pixel to the **nearest color from a user-specified fixed palette**.

This is **not** k-means/adaptive quantization.

This is a **fixed-palette remapper**.

---

## Node name

`FixedPaletteQuantize`

---

## Functional requirements

### Inputs

Please implement these inputs:

1. `image` (IMAGE)
   - The source image to quantize.

2. `mask` (optional MASK)
   - Optional alpha/transparency mask input.
   - If provided, pixels below `alpha_threshold` should become fully transparent in output.
   - If not provided, the node should still work on the image alone.

3. `palette_text` (STRING, multiline)
   - Multiline text input containing one color per line.
   - Accept formats like:
     - `#RRGGBB`
     - `R,G,B`
   - Example:
     ```text
     #FFFFFF
     #DDE3F2
     #AAB3C8
     #3A3F6D
     #9B9EB6
     #F6B3C2
     #A8C81E
     #6DB44A
     ```
   - These are the only colors that output pixels should use.

4. `distance_metric` (COMBO)
   - Options:
     - `rgb`
     - `lab`
   - `lab` is preferred default if feasible, since perceptual matching is better.

5. `alpha_threshold` (FLOAT)
   - Default something like `0.01` or `0.05`.
   - Pixels with alpha/mask below threshold become fully transparent.

6. `dither` (COMBO)
   - Options:
     - `none`
     - `floyd-steinberg`
   - Default `none`.
   - Dithering is optional and should only affect non-transparent pixels.

7. `preserve_exact_matches` (BOOLEAN)
   - If enabled, pixels already exactly equal to a palette color should remain unchanged.

8. `output_alpha_mode` (COMBO)
   - Options:
     - `binary`
     - `keep`
   - `binary`: output alpha should be either 0 or 1 only, based on threshold.
   - `keep`: preserve original alpha above threshold (if practical).
   - Default `binary`, since my use case is pure sprite transparency.

---

## Outputs

1. `image` (IMAGE)
   - Quantized output image using only the specified palette colors plus transparency.

Optional extra output if easy: 2. `palette_preview` (IMAGE)

- Small swatch preview of the parsed palette.
- Nice to have, not required.

---

## Core behavior

For every pixel:

1. Determine whether pixel is transparent:
   - Use optional mask if available, otherwise infer from image alpha if available.
   - If alpha < `alpha_threshold`, set output pixel to fully transparent.

2. For non-transparent pixels:
   - Compare the pixel color against the supplied palette colors.
   - Find the nearest palette entry using selected distance metric.
   - Replace pixel with that exact palette color.

3. If dithering is enabled:
   - Apply dithering only across non-transparent pixels.
   - Do not dither transparent regions.

Important:

- The node must **never invent new colors** beyond the supplied palette.
- This is the key requirement.

---

## Why this is needed

The current quantization approach is "dumb" for pixel-art cleanup because it tries to spend palette slots fitting many near-whites and near-grays. That is not what I want.

I want to define the palette myself and force the image into it.

Example use case:

- white cat sprite
- mostly white/light gray/blue shadows
- 1 stripe gray
- 1 pink
- 1 green eye
- maybe 1 grass green
- transparency

The automatic quantizer wastes palette slots on multiple whites instead of using my intended palette structure.

---

## Implementation notes

### Parsing palette input

Please write a small robust parser for `palette_text`:

- ignore blank lines
- ignore surrounding whitespace
- support `#RRGGBB`
- support `R,G,B`
- validate range 0–255
- throw a useful error if palette is empty or invalid

### Distance metric

- `rgb`: Euclidean distance in RGB
- `lab`: perceptual distance in CIELAB if feasible
  - if adding LAB dependency is annoying, a lightweight implementation is acceptable
  - otherwise RGB is fine for MVP and LAB can be added cleanly

### Dithering

- Start with `none`
- If `floyd-steinberg` is implemented, keep it simple and deterministic
- Only diffuse error among non-transparent pixels

### Alpha handling

Transparency handling matters a lot.
I want clean sprite-style output.

If Comfy’s `IMAGE` type makes alpha awkward, it is acceptable for MVP to:

- use optional `MASK` as the source of transparency
- produce RGBA-like result as best as Comfy supports

If needed, document any limitations clearly.

---

## Code structure

Please create a proper ComfyUI custom node package with:

- node class
- `NODE_CLASS_MAPPINGS`
- `NODE_DISPLAY_NAME_MAPPINGS`
- minimal README or comments explaining inputs

Prefer readable code over premature optimization.

---

## Non-goals for MVP

Do **not** spend time on:

- adaptive palette extraction
- k-means clustering
- fancy UI beyond basic inputs
- file-writing for true indexed PNG
- resizing/downsampling inside this node

This node is only for:

- taking an image
- remapping it to an explicit fixed palette

If the node works well, I may later want:

- a separate resize/downsample node
- indexed PNG export
- palette input via image swatch
- palette locking by semantic regions

But those are not MVP.

---

## Acceptance criteria

The node is successful if:

1. I can supply a palette of 8 colors manually.
2. The output image uses only those colors (plus transparency).
3. Transparent pixels remain transparent.
4. The output is visually predictable and repeatable.
5. It does not waste colors on auto-generated near-white variants because it never chooses its own palette.

---

## Suggested example palette for testing

Use a test palette like:

```text
#FFFFFF
#DDE3F2
#AAB3C8
#3A3F6D
#9B9EB6
#F6B3C2
#A8C81E
#6DB44A
```
````

This is just an example sprite palette.
The code should work with arbitrary user-supplied palettes.

---

## Nice-to-have after MVP

If implementation goes smoothly, optional enhancements:

- palette preview output image
- optional gamma-corrected RGB matching
- optional `palette_image` input (read swatches from an image strip)
- optional `ignore_background_color` behavior
- optional “lock certain palette colors” or weighted mapping

But do MVP first.

```

---

# My extra notes on the handoff

A few things I’d emphasize if you send this to Codex:

## 1. Keep resize **out** of v1
That should probably be a separate node.

Otherwise the first implementation gets muddy:
- resampling choice
- palette mapping
- alpha handling
- dithering

That’s too many moving parts at once.

So I’d make v1:

> **fixed palette remap only**

Then later:

- `PixelArtResize`
- or `ResizeThenFixedPaletteQuantize`

---

## 2. Mask input is worth it
Because in Comfy, transparency is often awkward.
If the node supports an optional `MASK`, it’ll be much more reliable for sprite work.

That is a very good design choice for Comfy specifically.

---

## 3. “True indexed PNG” is **not necessary** for the first win
You don’t need Codex to solve actual indexed file export yet.

You mainly need:

- output image visually restricted to your palette
- alpha clean
- repeatable mapping

That alone will already be a huge improvement.

Then Aseprite can be the place where you:
- inspect
- clean up
- optionally convert/export as indexed asset

---

## 4. LAB matching is a nice win, but not mandatory
If Codex stalls on color science, tell it:

> get RGB nearest-color working first, cleanly

Then add LAB later.

Because the bigger win is **fixed palette**, not perfect perceptual distance.

---

# My verdict

Yes — this is absolutely worthy of a Codex handoff.

In fact, this is exactly the kind of thing Codex should be good at:
- well-scoped
- custom node
- deterministic behavior
- clear acceptance criteria

If you want, next I can help you make this even more practical by turning that into a **shorter engineer-facing handoff** or a **full repo task list** with:
- file names
- class skeleton
- MVP milestones
- test cases.
```
