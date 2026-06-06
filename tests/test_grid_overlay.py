from __future__ import annotations

import pytest
import torch

from nodes.grid import GridOverlay, build_grid_mask
from tests.conftest import colors_to_tensor, save_mask_png, save_tensor_png


def test_grid_mask_uses_cell_size_and_offsets() -> None:
    mask = build_grid_mask(5, 7, 3, 2, 1, 1, torch.device("cpu"))

    assert torch.nonzero(mask[0]).flatten().tolist() == [1, 4]
    assert torch.nonzero(mask[:, 0]).flatten().tolist() == [1, 3]
    assert mask[1].all()
    assert mask[3].all()


def test_grid_overlay_draws_preview_without_changing_comfy_mask() -> None:
    image, mask = colors_to_tensor(
        [
            [(255, 255, 255, 255), (180, 180, 180, 255), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(180, 180, 180, 255), (255, 255, 255, 255), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (255, 255, 255, 255), (180, 180, 180, 255)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (180, 180, 180, 255), (255, 255, 255, 255)],
        ]
    )

    output, output_mask, rgba = GridOverlay().overlay(
        image,
        2,
        2,
        0,
        0,
        "#6060FF",
        1.0,
        True,
        mask,
    )

    assert output.shape == image.shape
    assert rgba.shape == (1, 4, 4, 4)
    assert torch.equal(output_mask, mask)
    assert torch.allclose(output[0, 0, 0], torch.tensor([96 / 255, 96 / 255, 1.0]))
    assert torch.allclose(output[0, 1, 1], torch.tensor([1.0, 1.0, 1.0]))
    assert rgba[0, 0, 2, 3].item() == 1.0

    save_tensor_png(image, "grid_overlay_input.png")
    save_mask_png(mask, "grid_overlay_input_mask.png")
    save_tensor_png(output, "grid_overlay_output_rgb.png")
    save_mask_png(output_mask, "grid_overlay_output_mask.png")
    save_tensor_png(rgba, "grid_overlay_output_rgba.png")


def test_grid_overlay_can_skip_transparent_regions() -> None:
    image, mask = colors_to_tensor(
        [
            [(255, 255, 255, 255), (255, 255, 255, 255), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(255, 255, 255, 255), (255, 255, 255, 255), (0, 0, 0, 0), (0, 0, 0, 0)],
        ]
    )

    _output, output_mask, rgba = GridOverlay().overlay(
        image,
        1,
        1,
        0,
        0,
        "#FF0000",
        1.0,
        False,
        mask,
    )

    assert torch.equal(output_mask, mask)
    assert rgba[0, 0, 0, 3].item() == 1.0
    assert rgba[0, 0, 2, 3].item() == 0.0
    assert rgba[0, 0, 2, :3].tolist() == [0.0, 0.0, 0.0]

    save_tensor_png(rgba, "grid_overlay_skip_transparent_rgba.png")


def test_grid_overlay_rejects_invalid_line_color() -> None:
    image, mask = colors_to_tensor([[(255, 255, 255, 255)]])

    with pytest.raises(ValueError, match="Expected a hex color"):
        GridOverlay().overlay(image, 8, 8, 0, 0, "blue", 0.6, True, mask)
