# ComfyUI AssetForge

Deterministic utility nodes for ComfyUI asset workflows.

This repo currently contains the original blank-image filter plus the first mask utility node.

The code is organized as a suite:

- `nodes/color_masks.py`: exact color and semantic-map mask extraction.
- `nodes/fixed_palette.py`: deterministic fixed-palette image remapping.
- `nodes/image_filters.py`: deterministic image filtering and batch hygiene.
- `__init__.py`: ComfyUI entrypoint that aggregates node modules.

## Nodes

- `Filter Blank Images`: removes images where every pixel is within a channel tolerance, useful for dropping blank OpenPose/video extraction frames.
- `Mask From Hex Color`: creates a Comfy `MASK` from pixels matching a hex color such as `#3F48CC`.
- `Fixed Palette Quantize`: remaps each non-transparent pixel to the nearest user-supplied palette color. Palette lines can be `#RRGGBB`, `#RRGGBBAA`, `R,G,B`, or `R,G,B,A`. It supports RGB or LAB distance, optional Floyd-Steinberg dithering, and returns a Comfy `MASK` plus an `rgba_image` output for PNG export with transparency.

## Install

Place this repo in `ComfyUI/custom_nodes/` and restart ComfyUI.

## Notes

- Image values are treated as Comfy tensors in the `0.0` to `1.0` range.
- Color tolerances are entered in normal `0` to `255` color units.
- `Mask From Hex Color` matches when every RGB channel is within the tolerance.
- `Fixed Palette Quantize` uses native Comfy `MASK` semantics: `0` means opaque and `1` means transparent. If the input image has an alpha channel, that alpha is used when no mask is connected. Palette entries with alpha below the threshold are accepted as transparent palette entries, but are excluded from visible-color matching.
