from __future__ import annotations

import pytest
import torch

from nodes.fixed_palette import FixedPaletteQuantize, parse_palette
from tests.conftest import (
    assert_visible_palette_subset,
    colors_to_tensor,
    save_mask_png,
    save_tensor_png,
    visible_rgb_set,
)


PALETTE_TEXT = "\n".join(
    [
        "#FFFFFF",
        "#000000",
        "#80A0C0",
        "#6DB44A",
        "#F6B3C2",
        "255,0,0,0",
        "0,255,0",
    ]
)

VISIBLE_PALETTE = {
    (255, 255, 255),
    (0, 0, 0),
    (128, 160, 192),
    (109, 180, 74),
    (246, 179, 194),
    (0, 255, 0),
}


def test_quantize_outputs_only_supplied_visible_palette_colors() -> None:
    image, mask = colors_to_tensor(
        [
            [(250, 250, 248, 255), (10, 8, 12, 255), (130, 158, 200, 255), (0, 0, 0, 0)],
            [(110, 180, 70, 255), (245, 180, 192, 255), (0, 240, 0, 255), (0, 0, 0, 0)],
            [(240, 245, 255, 255), (8, 8, 8, 255), (120, 150, 190, 255), (0, 0, 0, 0)],
        ]
    )

    output, output_mask, palette_preview, rgba = FixedPaletteQuantize().quantize(
        image,
        PALETTE_TEXT,
        "rgb",
        0.01,
        "none",
        True,
        "binary",
        mask,
    )

    assert output.shape == image.shape
    assert rgba.shape == (1, 3, 4, 4)
    assert palette_preview.shape == (1, 24, 144, 3)
    assert torch.equal(output_mask, mask)
    assert torch.equal(rgba[..., 3], 1.0 - output_mask)
    assert_visible_palette_subset(output, output_mask, VISIBLE_PALETTE)

    save_tensor_png(image, "fixed_palette_input.png")
    save_mask_png(mask, "fixed_palette_input_mask.png")
    save_tensor_png(output, "fixed_palette_output_rgb.png")
    save_mask_png(output_mask, "fixed_palette_output_mask.png")
    save_tensor_png(rgba, "fixed_palette_output_rgba.png")
    save_tensor_png(palette_preview, "fixed_palette_palette_preview.png")


def test_palette_parser_accepts_hex_rgb_and_alpha_variants() -> None:
    palette = parse_palette(
        "#010203\n#040506ff\n7,8,9\n10,11,12,255\n#DEADBE00\n13,14,15,0",
        0.01,
        torch.device("cpu"),
        torch.float32,
    )

    colors = {
        tuple(int(value) for value in (row * 255.0).round().to(torch.uint8).tolist())
        for row in palette
    }
    assert colors == {
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
        (10, 11, 12),
    }


def test_palette_parser_rejects_empty_visible_palette() -> None:
    with pytest.raises(ValueError, match="no visible colors"):
        parse_palette("#00000000\n255,255,255,0", 0.01, torch.device("cpu"), torch.float32)


def test_rgba_input_can_supply_alpha_without_mask() -> None:
    rgb, mask = colors_to_tensor([[(255, 255, 255, 255), (255, 255, 255, 0)]])
    rgba_input = torch.cat((rgb, (1.0 - mask).unsqueeze(-1)), dim=-1)

    output, output_mask, _preview, rgba = FixedPaletteQuantize().quantize(
        rgba_input,
        "#FFFFFF\n#000000",
        "rgb",
        0.01,
        "none",
        True,
        "binary",
    )

    assert output.shape == (1, 1, 2, 4)
    assert rgba.shape == (1, 1, 2, 4)
    assert output_mask.tolist() == [[[0.0, 1.0]]]
    assert rgba[0, 0, 0, 3].item() == 1.0
    assert rgba[0, 0, 1, 3].item() == 0.0
    assert visible_rgb_set(rgba, output_mask) == {(255, 255, 255)}
