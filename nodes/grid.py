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
    CATEGORY = "PixelAssetForge/Grid"

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


class MajorityRulesGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cell_width": ("INT", {"default": 8, "min": 1, "max": 512, "step": 1}),
                "cell_height": ("INT", {"default": 8, "min": 1, "max": 512, "step": 1}),
                "offset_x": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "alpha_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "include_alpha": ("BOOLEAN", {"default": True}),
                "tie_break_mode": (["prefer_nonalpha"], {"default": "prefer_nonalpha"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "rgba_image")
    FUNCTION = "canonicalize"
    CATEGORY = "PixelAssetForge/Grid"

    def canonicalize(
        self,
        image: torch.Tensor,
        cell_width: int,
        cell_height: int,
        offset_x: int,
        offset_y: int,
        alpha_threshold: float,
        include_alpha: bool,
        tie_break_mode: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del tie_break_mode

        rgb = image[..., :3].clamp(0.0, 1.0)
        source_alpha = get_source_alpha(image, mask)
        visible = source_alpha >= alpha_threshold
        output_rgb = torch.zeros_like(rgb)
        output_alpha = torch.zeros_like(source_alpha)

        x_bounds = build_cell_boundaries(image.shape[2], cell_width, offset_x, image.device)
        y_bounds = build_cell_boundaries(image.shape[1], cell_height, offset_y, image.device)

        for batch_index in range(image.shape[0]):
            for top, bottom in pairwise_bounds(y_bounds):
                for left, right in pairwise_bounds(x_bounds):
                    cell_rgb = rgb[batch_index, top:bottom, left:right, :]
                    cell_visible = visible[batch_index, top:bottom, left:right]
                    winner = choose_cell_winner(cell_rgb, cell_visible, include_alpha)

                    if winner is None:
                        continue

                    output_rgb[batch_index, top:bottom, left:right, :] = winner
                    output_alpha[batch_index, top:bottom, left:right] = 1.0

        if image.shape[-1] >= 4:
            output_image = torch.cat((output_rgb, output_alpha.unsqueeze(-1)), dim=-1)
        else:
            output_image = output_rgb

        comfy_mask = (1.0 - output_alpha).clamp(0.0, 1.0)
        rgba_image = torch.cat((output_rgb, output_alpha.unsqueeze(-1)), dim=-1)
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


def build_cell_boundaries(size: int, cell_size: int, offset: int, device: torch.device) -> list[int]:
    positions = torch.arange(size, device=device)
    grid_lines = positions[(positions - offset) % cell_size == 0].detach().cpu().tolist()
    boundaries = sorted({0, size, *(int(position) for position in grid_lines)})
    return boundaries


def pairwise_bounds(boundaries: list[int]) -> list[tuple[int, int]]:
    return [
        (boundaries[index], boundaries[index + 1])
        for index in range(len(boundaries) - 1)
        if boundaries[index] < boundaries[index + 1]
    ]


def choose_cell_winner(
    cell_rgb: torch.Tensor,
    cell_visible: torch.Tensor,
    include_alpha: bool,
) -> torch.Tensor | None:
    flat_rgb = cell_rgb.reshape(-1, 3)
    flat_visible = cell_visible.reshape(-1)
    alpha_count = int((~flat_visible).sum().item()) if include_alpha else -1
    visible_rgb = flat_rgb[flat_visible]

    if visible_rgb.shape[0] == 0:
        return None

    colors, counts = torch.unique(visible_rgb, dim=0, return_counts=True)
    max_visible_count = int(counts.max().item())

    if include_alpha and alpha_count > max_visible_count:
        return None

    candidate_colors = colors[counts == max_visible_count]
    return first_seen_candidate(flat_rgb, flat_visible, candidate_colors)


def first_seen_candidate(
    flat_rgb: torch.Tensor,
    flat_visible: torch.Tensor,
    candidate_colors: torch.Tensor,
) -> torch.Tensor:
    for pixel, is_visible in zip(flat_rgb, flat_visible):
        if not bool(is_visible):
            continue
        if bool((pixel.unsqueeze(0) == candidate_colors).all(dim=1).any()):
            return pixel

    return candidate_colors[0]


def parse_hex_color(hex_color: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = hex_color.strip()
    match = HEX_COLOR_PATTERN.match(value)
    if not match:
        raise ValueError(f"Expected a hex color like #6060FF, got {hex_color!r}.")

    digits = match.group(1)
    channels = [int(digits[index:index + 2], 16) / 255.0 for index in range(0, 6, 2)]
    return torch.tensor(channels, device=device, dtype=dtype)


NODE_CLASS_MAPPINGS = {
    "PixelAssetForgeGridOverlay": GridOverlay,
    "PixelAssetForgeMajorityRulesGrid": MajorityRulesGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelAssetForgeGridOverlay": "Grid Overlay",
    "PixelAssetForgeMajorityRulesGrid": "Majority Rules Grid",
}
