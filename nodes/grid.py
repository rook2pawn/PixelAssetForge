import re

import torch

from .fixed_palette import get_source_alpha


HEX_COLOR_PATTERN = re.compile(r"^#?([0-9a-fA-F]{6})$")


class GridOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cell_width": ("INT", {"default": 8, "min": 1, "max": 512, "step": 1}),
                "cell_height": ("INT", {"default": 8, "min": 1, "max": 512, "step": 1}),
                "offset_x": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "line_color": ("STRING", {"default": "#6060FF"}),
                "line_opacity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "draw_over_transparent": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "rgba_image")
    FUNCTION = "overlay"
    CATEGORY = "AssetForge/Grid"

    def overlay(
        self,
        image: torch.Tensor,
        cell_width: int,
        cell_height: int,
        offset_x: int,
        offset_y: int,
        line_color: str,
        line_opacity: float,
        draw_over_transparent: bool,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = image[..., :3].clamp(0.0, 1.0)
        alpha = get_source_alpha(image, mask)
        comfy_mask = (1.0 - alpha).clamp(0.0, 1.0)
        grid = build_grid_mask(
            image.shape[1],
            image.shape[2],
            cell_width,
            cell_height,
            offset_x,
            offset_y,
            image.device,
        )

        if not draw_over_transparent:
            grid = grid & (alpha > 0.0).any(dim=0)

        color = parse_hex_color(line_color, image.device, image.dtype)
        blend_amount = grid.to(image.dtype).unsqueeze(0).unsqueeze(-1) * line_opacity
        overlaid_rgb = rgb * (1.0 - blend_amount) + color.view(1, 1, 1, 3) * blend_amount

        if image.shape[-1] >= 4:
            output_image = torch.cat((overlaid_rgb, alpha.unsqueeze(-1)), dim=-1)
        else:
            output_image = overlaid_rgb

        overlay_alpha = alpha
        if draw_over_transparent:
            overlay_alpha = torch.maximum(alpha, grid.to(image.dtype).unsqueeze(0) * line_opacity)
        rgba_image = torch.cat((overlaid_rgb, overlay_alpha.unsqueeze(-1)), dim=-1)
        return (output_image, comfy_mask, rgba_image)


def build_grid_mask(
    height: int,
    width: int,
    cell_width: int,
    cell_height: int,
    offset_x: int,
    offset_y: int,
    device: torch.device,
) -> torch.Tensor:
    x_positions = torch.arange(width, device=device)
    y_positions = torch.arange(height, device=device)
    normalized_x = (x_positions - offset_x) % cell_width
    normalized_y = (y_positions - offset_y) % cell_height
    vertical_lines = normalized_x == 0
    horizontal_lines = normalized_y == 0
    return horizontal_lines[:, None] | vertical_lines[None, :]


def parse_hex_color(hex_color: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = hex_color.strip()
    match = HEX_COLOR_PATTERN.match(value)
    if not match:
        raise ValueError(f"Expected a hex color like #6060FF, got {hex_color!r}.")

    digits = match.group(1)
    channels = [int(digits[index:index + 2], 16) / 255.0 for index in range(0, 6, 2)]
    return torch.tensor(channels, device=device, dtype=dtype)


NODE_CLASS_MAPPINGS = {
    "AssetForgeGridOverlay": GridOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AssetForgeGridOverlay": "Grid Overlay",
}
