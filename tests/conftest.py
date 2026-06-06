from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = PROJECT_ROOT / "tests" / "previews"


@pytest.fixture(autouse=True)
def ensure_preview_dir() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def colors_to_tensor(rows: list[list[tuple[int, int, int, int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    height = len(rows)
    width = len(rows[0])
    image = torch.zeros((1, height, width, 3), dtype=torch.float32)
    mask = torch.ones((1, height, width), dtype=torch.float32)

    for y, row in enumerate(rows):
        if len(row) != width:
            raise ValueError("All rows must have the same width.")
        for x, (red, green, blue, alpha) in enumerate(row):
            image[0, y, x] = torch.tensor([red, green, blue], dtype=torch.float32) / 255.0
            mask[0, y, x] = 1.0 - alpha / 255.0

    return image, mask


def save_tensor_png(tensor: torch.Tensor, name: str) -> Path:
    path = PREVIEW_DIR / name
    image = tensor[0].detach().cpu().clamp(0.0, 1.0)
    array = (image * 255.0).round().to(torch.uint8).numpy()
    mode = "RGBA" if array.shape[-1] == 4 else "RGB"
    pil_image = Image.fromarray(array, mode=mode)
    scale = preview_scale(pil_image.width, pil_image.height)
    pil_image.resize((pil_image.width * scale, pil_image.height * scale), Image.Resampling.NEAREST).save(path)
    return path


def save_mask_png(mask: torch.Tensor, name: str) -> Path:
    path = PREVIEW_DIR / name
    image = mask[0].detach().cpu().clamp(0.0, 1.0)
    array = (image * 255.0).round().to(torch.uint8).numpy()
    pil_image = Image.fromarray(array, mode="L")
    scale = preview_scale(pil_image.width, pil_image.height)
    pil_image.resize((pil_image.width * scale, pil_image.height * scale), Image.Resampling.NEAREST).save(path)
    return path


def preview_scale(width: int, height: int) -> int:
    return 32 if max(width, height) <= 16 else 4


def visible_rgb_set(image: torch.Tensor, mask: torch.Tensor | None = None) -> set[tuple[int, int, int]]:
    rgb = image[0, ..., :3].detach().cpu().clamp(0.0, 1.0)
    if mask is None:
        visible = torch.ones(rgb.shape[:-1], dtype=torch.bool)
    else:
        visible = mask[0].detach().cpu() < 0.5

    colors: set[tuple[int, int, int]] = set()
    for pixel in rgb[visible]:
        colors.add(tuple(int(value) for value in (pixel * 255.0).round().to(torch.uint8).tolist()))
    return colors


def assert_visible_palette_subset(
    image: torch.Tensor,
    mask: torch.Tensor,
    palette: set[tuple[int, int, int]],
) -> None:
    colors = visible_rgb_set(image, mask)
    extra_colors = colors - palette
    assert not extra_colors, f"Output contained colors outside palette: {sorted(extra_colors)}"
