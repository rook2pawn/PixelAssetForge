from __future__ import annotations

import torch

from nodes.grid import MajorityRulesGrid, build_cell_boundaries
from tests.conftest import colors_to_tensor, save_mask_png, save_tensor_png


def test_majority_color_fills_cell_and_rgba_matches_mask() -> None:
    image, mask = colors_to_tensor(
        [
            [(255, 0, 0, 255), (255, 0, 0, 255), (0, 0, 255, 255), (0, 0, 0, 0)],
            [(255, 0, 0, 255), (0, 0, 255, 255), (0, 0, 0, 0), (0, 0, 0, 0)],
        ]
    )

    output, output_mask, rgba = MajorityRulesGrid().canonicalize(
        image,
        2,
        2,
        0,
        0,
        0.01,
        True,
        "prefer_nonalpha",
        mask,
    )

    assert output[0, :, :2].tolist() == torch.ones((2, 2, 3)).mul(torch.tensor([1.0, 0.0, 0.0])).tolist()
    assert torch.equal(output_mask[0, :, :2], torch.zeros((2, 2), dtype=torch.float32))
    assert torch.equal(rgba[..., 3], 1.0 - output_mask)

    save_tensor_png(image, "majority_grid_input.png")
    save_mask_png(mask, "majority_grid_input_mask.png")
    save_tensor_png(output, "majority_grid_output_rgb.png")
    save_mask_png(output_mask, "majority_grid_output_mask.png")
    save_tensor_png(rgba, "majority_grid_output_rgba.png")


def test_alpha_majority_makes_cell_transparent() -> None:
    image, mask = colors_to_tensor(
        [
            [(255, 0, 0, 255), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0)],
        ]
    )

    output, output_mask, rgba = MajorityRulesGrid().canonicalize(
        image,
        2,
        2,
        0,
        0,
        0.01,
        True,
        "prefer_nonalpha",
        mask,
    )

    assert torch.equal(output, torch.zeros_like(output))
    assert torch.equal(output_mask, torch.ones_like(output_mask))
    assert torch.equal(rgba[..., 3], torch.zeros_like(rgba[..., 3]))


def test_visible_color_beats_alpha_on_tie() -> None:
    image, mask = colors_to_tensor(
        [
            [(0, 255, 0, 255), (0, 255, 0, 255)],
            [(0, 0, 0, 0), (0, 0, 0, 0)],
        ]
    )

    output, output_mask, _rgba = MajorityRulesGrid().canonicalize(
        image,
        2,
        2,
        0,
        0,
        0.01,
        True,
        "prefer_nonalpha",
        mask,
    )

    assert torch.equal(output[0], torch.ones((2, 2, 3)).mul(torch.tensor([0.0, 1.0, 0.0])))
    assert torch.equal(output_mask, torch.zeros_like(output_mask))


def test_visible_color_tie_uses_first_seen_reading_order() -> None:
    image, mask = colors_to_tensor(
        [
            [(0, 0, 255, 255), (255, 0, 0, 255)],
            [(255, 0, 0, 255), (0, 0, 255, 255)],
        ]
    )

    output, _output_mask, rgba = MajorityRulesGrid().canonicalize(
        image,
        2,
        2,
        0,
        0,
        0.01,
        True,
        "prefer_nonalpha",
        mask,
    )

    assert torch.equal(output[0], torch.ones((2, 2, 3)).mul(torch.tensor([0.0, 0.0, 1.0])))
    save_tensor_png(rgba, "majority_grid_visible_tie_rgba.png")


def test_include_alpha_false_ignores_alpha_votes() -> None:
    image, mask = colors_to_tensor(
        [
            [(255, 0, 0, 255), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0)],
        ]
    )

    output, output_mask, _rgba = MajorityRulesGrid().canonicalize(
        image,
        2,
        2,
        0,
        0,
        0.01,
        False,
        "prefer_nonalpha",
        mask,
    )

    assert torch.equal(output[0], torch.ones((2, 2, 3)).mul(torch.tensor([1.0, 0.0, 0.0])))
    assert torch.equal(output_mask, torch.zeros_like(output_mask))


def test_offset_changes_cell_membership() -> None:
    image, mask = colors_to_tensor(
        [
            [(255, 0, 0, 255), (0, 0, 255, 255), (0, 0, 255, 255), (255, 0, 0, 255)],
        ]
    )

    no_offset, _no_offset_mask, no_offset_rgba = MajorityRulesGrid().canonicalize(
        image,
        2,
        1,
        0,
        0,
        0.01,
        True,
        "prefer_nonalpha",
        mask,
    )
    offset, _offset_mask, offset_rgba = MajorityRulesGrid().canonicalize(
        image,
        2,
        1,
        1,
        0,
        0.01,
        True,
        "prefer_nonalpha",
        mask,
    )

    assert no_offset[0, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert no_offset[0, 0, 2].tolist() == [0.0, 0.0, 1.0]
    assert offset[0, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert offset[0, 0, 1].tolist() == [0.0, 0.0, 1.0]
    assert offset[0, 0, 3].tolist() == [1.0, 0.0, 0.0]

    save_tensor_png(no_offset_rgba, "majority_grid_no_offset_rgba.png")
    save_tensor_png(offset_rgba, "majority_grid_offset_rgba.png")


def test_partial_edge_cells_are_processed() -> None:
    image, mask = colors_to_tensor(
        [
            [(0, 0, 255, 255), (255, 0, 0, 255), (255, 0, 0, 255)],
        ]
    )

    output, output_mask, _rgba = MajorityRulesGrid().canonicalize(
        image,
        2,
        1,
        0,
        0,
        0.01,
        True,
        "prefer_nonalpha",
        mask,
    )

    assert output[0, 0, 0].tolist() == [0.0, 0.0, 1.0]
    assert output[0, 0, 1].tolist() == [0.0, 0.0, 1.0]
    assert output[0, 0, 2].tolist() == [1.0, 0.0, 0.0]
    assert torch.equal(output_mask, torch.zeros_like(output_mask))


def test_cell_boundaries_match_grid_overlay_line_semantics() -> None:
    assert build_cell_boundaries(10, 4, 3, torch.device("cpu")) == [0, 3, 7, 10]
