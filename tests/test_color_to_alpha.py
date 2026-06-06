from __future__ import annotations

import torch

from nodes.image_cleanup import ColorToAlpha
from tests.conftest import colors_to_tensor, save_mask_png, save_tensor_png


def test_color_to_alpha_removes_exact_target_color_globally() -> None:
    image, mask = colors_to_tensor(
        [
            [(169, 168, 161, 255), (255, 255, 255, 255), (169, 168, 161, 255)],
            [(0, 0, 0, 255), (169, 168, 161, 255), (255, 0, 0, 255)],
        ]
    )

    output, output_mask, rgba = ColorToAlpha().convert(
        image,
        "#A9A8A1",
        0.0,
        0.01,
        "binary",
        mask,
    )

    assert torch.nonzero(output_mask[0] == 1.0).tolist() == [[0, 0], [0, 2], [1, 1]]
    assert torch.nonzero(output_mask[0] == 0.0).tolist() == [[0, 1], [1, 0], [1, 2]]
    assert torch.equal(rgba[..., 3], 1.0 - output_mask)
    assert output[0, 0, 0].tolist() == [0.0, 0.0, 0.0]
    assert output[0, 0, 1].tolist() == [1.0, 1.0, 1.0]

    save_tensor_png(image, "color_to_alpha_input.png")
    save_mask_png(mask, "color_to_alpha_input_mask.png")
    save_tensor_png(output, "color_to_alpha_output_rgb.png")
    save_mask_png(output_mask, "color_to_alpha_output_mask.png")
    save_tensor_png(rgba, "color_to_alpha_output_rgba.png")


def test_color_to_alpha_tolerance_uses_max_channel_delta() -> None:
    image, mask = colors_to_tensor(
        [
            [(169, 168, 161, 255), (170, 167, 161, 255), (172, 168, 161, 255)],
        ]
    )

    _output, output_mask, rgba = ColorToAlpha().convert(
        image,
        "#A9A8A1",
        1.0,
        0.01,
        "binary",
        mask,
    )

    assert output_mask.tolist() == [[[1.0, 1.0, 0.0]]]
    save_tensor_png(rgba, "color_to_alpha_tolerance_rgba.png")


def test_color_to_alpha_preserves_existing_transparency() -> None:
    image, mask = colors_to_tensor(
        [
            [(169, 168, 161, 0), (255, 255, 255, 255)],
        ]
    )

    _output, output_mask, rgba = ColorToAlpha().convert(
        image,
        "#A9A8A1",
        0.0,
        0.01,
        "binary",
        mask,
    )

    assert output_mask.tolist() == [[[1.0, 0.0]]]
    assert rgba[0, 0, 0, 3].item() == 0.0
    assert rgba[0, 0, 1, 3].item() == 1.0


def test_color_to_alpha_keep_mode_preserves_non_matching_alpha() -> None:
    rgb, mask = colors_to_tensor(
        [
            [(169, 168, 161, 255), (255, 255, 255, 128)],
        ]
    )
    rgba_input = torch.cat((rgb, (1.0 - mask).unsqueeze(-1)), dim=-1)

    _output, output_mask, rgba = ColorToAlpha().convert(
        rgba_input,
        "#A9A8A1",
        0.0,
        0.01,
        "keep",
    )

    assert output_mask[0, 0, 0].item() == 1.0
    assert torch.isclose(rgba[0, 0, 1, 3], torch.tensor(128 / 255))
    assert torch.isclose(output_mask[0, 0, 1], torch.tensor(1.0 - 128 / 255))
