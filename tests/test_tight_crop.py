from __future__ import annotations

import torch

from nodes.image_cleanup import TightCrop
from tests.conftest import colors_to_tensor, save_mask_png, save_tensor_png


def test_tight_crop_finds_visible_bounds() -> None:
    image, mask = colors_to_tensor(
        [
            [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 255, 255), (255, 255, 255, 255), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
        ]
    )

    output, output_mask, rgba = TightCrop().crop(image, 0.01, 0, "binary", mask)

    assert output.shape == (1, 2, 2, 3)
    assert output_mask.shape == (1, 2, 2)
    assert rgba.shape == (1, 2, 2, 4)
    assert torch.equal(output_mask, torch.zeros((1, 2, 2), dtype=torch.float32))
    assert output[0, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert output[0, 1, 1].tolist() == [1.0, 1.0, 1.0]

    save_tensor_png(image, "tight_crop_input.png")
    save_mask_png(mask, "tight_crop_input_mask.png")
    save_tensor_png(output, "tight_crop_output_rgb.png")
    save_mask_png(output_mask, "tight_crop_output_mask.png")
    save_tensor_png(rgba, "tight_crop_output_rgba.png")


def test_tight_crop_padding_expands_bounds_within_canvas() -> None:
    image, mask = colors_to_tensor(
        [
            [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 255, 255), (255, 255, 255, 255), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
        ]
    )

    output, output_mask, rgba = TightCrop().crop(image, 0.01, 1, "binary", mask)

    assert output.shape == (1, 4, 4, 3)
    assert output_mask.shape == (1, 4, 4)
    assert rgba.shape == (1, 4, 4, 4)
    assert torch.nonzero(output_mask[0] == 0.0).tolist() == [[1, 1], [1, 2], [2, 1], [2, 2]]

    save_tensor_png(output, "tight_crop_padding_output_rgb.png")
    save_mask_png(output_mask, "tight_crop_padding_output_mask.png")
    save_tensor_png(rgba, "tight_crop_padding_output_rgba.png")


def test_tight_crop_returns_empty_image_unchanged() -> None:
    image, mask = colors_to_tensor(
        [
            [(10, 20, 30, 0), (40, 50, 60, 0)],
            [(70, 80, 90, 0), (100, 110, 120, 0)],
        ]
    )

    output, output_mask, rgba = TightCrop().crop(image, 0.01, 1, "binary", mask)

    assert output.shape == image.shape
    assert output_mask.shape == mask.shape
    assert rgba.shape == (1, 2, 2, 4)
    assert torch.equal(output, image)
    assert torch.equal(output_mask, mask)
    assert torch.equal(rgba[..., 3], torch.zeros((1, 2, 2), dtype=torch.float32))

    save_tensor_png(rgba, "tight_crop_empty_unchanged_rgba.png")
