from __future__ import annotations

import torch

from nodes.image_cleanup import StrayPixelCleaner
from tests.conftest import colors_to_tensor, save_mask_png, save_tensor_png


def test_isolated_pixel_is_removed_and_connected_pair_survives() -> None:
    image, mask = colors_to_tensor(
        [
            [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (255, 255, 255, 255), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (255, 255, 255, 255), (255, 255, 255, 255)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
        ]
    )

    output, output_mask, rgba = StrayPixelCleaner().clean(
        image,
        "8",
        1,
        0.01,
        "nontransparent_only",
        "binary",
        mask,
    )

    assert torch.nonzero(output_mask[0] == 0.0).tolist() == [[3, 3], [3, 4]]
    assert rgba.shape == (1, 5, 5, 4)
    assert torch.equal(rgba[..., 3], 1.0 - output_mask)

    save_tensor_png(image, "stray_cleaner_input.png")
    save_mask_png(mask, "stray_cleaner_input_mask.png")
    save_tensor_png(output, "stray_cleaner_output_rgb.png")
    save_mask_png(output_mask, "stray_cleaner_output_mask.png")
    save_tensor_png(rgba, "stray_cleaner_output_rgba.png")


def test_connectivity_controls_diagonal_neighbors() -> None:
    image, mask = colors_to_tensor(
        [
            [(255, 255, 255, 255), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (255, 255, 255, 255), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
        ]
    )

    _image8, mask8, rgba8 = StrayPixelCleaner().clean(
        image,
        "8",
        1,
        0.01,
        "nontransparent_only",
        "binary",
        mask,
    )
    _image4, mask4, rgba4 = StrayPixelCleaner().clean(
        image,
        "4",
        1,
        0.01,
        "nontransparent_only",
        "binary",
        mask,
    )

    assert torch.nonzero(mask8[0] == 0.0).tolist() == [[0, 0], [1, 1]]
    assert torch.nonzero(mask4[0] == 0.0).tolist() == []

    save_tensor_png(rgba8, "stray_cleaner_diagonal_connectivity_8.png")
    save_tensor_png(rgba4, "stray_cleaner_diagonal_connectivity_4.png")


def test_iterations_are_stable_for_true_isolated_pixel_rule() -> None:
    image, mask = colors_to_tensor(
        [
            [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (255, 255, 255, 255), (255, 255, 255, 255), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
        ]
    )

    _once_image, once_mask, once_rgba = StrayPixelCleaner().clean(
        image,
        "4",
        1,
        0.01,
        "nontransparent_only",
        "binary",
        mask,
    )
    _twice_image, twice_mask, twice_rgba = StrayPixelCleaner().clean(
        image,
        "4",
        2,
        0.01,
        "nontransparent_only",
        "binary",
        mask,
    )

    assert torch.nonzero(once_mask[0] == 0.0).tolist() == [[1, 1], [1, 2]]
    assert torch.nonzero(twice_mask[0] == 0.0).tolist() == [[1, 1], [1, 2]]
    save_tensor_png(once_rgba, "stray_cleaner_iterations_once.png")
    save_tensor_png(twice_rgba, "stray_cleaner_iterations_twice.png")
