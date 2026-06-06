from __future__ import annotations

import pytest

from nodes.palette_tools import PaletteReportFromImage
from tests.conftest import colors_to_tensor, save_mask_png, save_tensor_png


def test_palette_report_sorts_by_frequency_and_outputs_preview() -> None:
    image, mask = colors_to_tensor(
        [
            [(255, 255, 255, 255), (255, 255, 255, 255), (0, 0, 0, 255), (0, 0, 0, 0)],
            [(128, 160, 192, 255), (255, 255, 255, 255), (0, 0, 0, 255), (0, 0, 0, 0)],
        ]
    )

    palette_text, report_text, color_count, preview = PaletteReportFromImage().report(
        image,
        0.01,
        "frequency_desc",
        True,
        256,
        mask,
    )

    assert palette_text.splitlines() == ["#FFFFFF", "#000000", "#80A0C0", "#00000000"]
    assert color_count == 3
    assert "visible colors: 3" in report_text
    assert "transparent pixels: 2" in report_text
    assert "#FFFFFF,3,50.00%" in report_text
    assert preview.shape == (1, 24, 72, 3)

    save_tensor_png(image, "palette_report_input.png")
    save_mask_png(mask, "palette_report_input_mask.png")
    save_tensor_png(preview, "palette_report_preview_frequency.png")


def test_palette_report_first_seen_and_hex_sort_modes() -> None:
    image, mask = colors_to_tensor(
        [
            [(0, 0, 255, 255), (255, 0, 0, 255), (0, 255, 0, 255)],
        ]
    )

    first_seen_text, _report, _count, first_seen_preview = PaletteReportFromImage().report(
        image,
        0.01,
        "first_seen",
        False,
        256,
        mask,
    )
    hex_text, _report, _count, hex_preview = PaletteReportFromImage().report(
        image,
        0.01,
        "hex",
        False,
        256,
        mask,
    )

    assert first_seen_text.splitlines() == ["#0000FF", "#FF0000", "#00FF00"]
    assert hex_text.splitlines() == ["#0000FF", "#00FF00", "#FF0000"]

    save_tensor_png(first_seen_preview, "palette_report_preview_first_seen.png")
    save_tensor_png(hex_preview, "palette_report_preview_hex.png")


def test_palette_report_can_omit_alpha_entry() -> None:
    image, mask = colors_to_tensor([[(255, 255, 255, 255), (0, 0, 0, 0)]])

    palette_text, _report, color_count, _preview = PaletteReportFromImage().report(
        image,
        0.01,
        "frequency_desc",
        False,
        256,
        mask,
    )

    assert palette_text == "#FFFFFF"
    assert color_count == 1


def test_palette_report_empty_visible_image_returns_empty_palette_and_blank_preview() -> None:
    image, mask = colors_to_tensor([[(10, 20, 30, 0), (40, 50, 60, 0)]])

    palette_text, report_text, color_count, preview = PaletteReportFromImage().report(
        image,
        0.01,
        "frequency_desc",
        True,
        256,
        mask,
    )

    assert palette_text == "#00000000"
    assert color_count == 0
    assert "visible colors: 0" in report_text
    assert preview.shape == (1, 24, 24, 3)
    save_tensor_png(preview, "palette_report_preview_empty.png")


def test_palette_report_rejects_more_than_max_colors() -> None:
    image, mask = colors_to_tensor([[(0, 0, 0, 255), (1, 0, 0, 255), (2, 0, 0, 255)]])

    with pytest.raises(ValueError, match="more than max_colors=2"):
        PaletteReportFromImage().report(image, 0.01, "frequency_desc", True, 2, mask)
