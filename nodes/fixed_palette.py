import re

import torch


HEX_COLOR_PATTERN = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
RGB_COLOR_PATTERN = re.compile(r"^\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*(\d{1,3}))?\s*$")


class FixedPaletteQuantize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "palette_text": (
                    "STRING",
                    {
                        "default": (
                            "#FFFFFF\n"
                            "#DDE3F2\n"
                            "#AAB3C8\n"
                            "#3A3F6D\n"
                            "#9B9EB6\n"
                            "#F6B3C2\n"
                            "#A8C81E\n"
                            "#6DB44A"
                        ),
                        "multiline": True,
                    },
                ),
                "distance_metric": (["lab", "rgb"], {"default": "lab"}),
                "alpha_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dither": (["none", "floyd-steinberg"], {"default": "none"}),
                "preserve_exact_matches": ("BOOLEAN", {"default": True}),
                "output_alpha_mode": (["binary", "keep"], {"default": "binary"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "mask", "palette_preview", "rgba_image")
    FUNCTION = "quantize"
    CATEGORY = "PixelAssetForge/Palette"

    def quantize(
        self,
        image: torch.Tensor,
        palette_text: str,
        distance_metric: str,
        alpha_threshold: float,
        dither: str,
        preserve_exact_matches: bool,
        output_alpha_mode: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        palette = parse_palette(palette_text, alpha_threshold, image.device, image.dtype)
        rgb = image[..., :3].clamp(0.0, 1.0)
        source_alpha = get_source_alpha(image, mask)
        opaque = source_alpha >= alpha_threshold

        if dither == "floyd-steinberg":
            quantized_rgb = floyd_steinberg_quantize(
                rgb,
                palette,
                opaque,
                distance_metric,
                preserve_exact_matches,
            )
        else:
            quantized_rgb = nearest_palette_colors(
                rgb,
                palette,
                distance_metric,
                preserve_exact_matches,
            )

        transparent_rgb = torch.zeros_like(quantized_rgb)
        quantized_rgb = torch.where(opaque.unsqueeze(-1), quantized_rgb, transparent_rgb)

        if output_alpha_mode == "keep":
            alpha = torch.where(opaque, source_alpha, torch.zeros_like(source_alpha))
        else:
            alpha = opaque.to(image.dtype)

        if image.shape[-1] >= 4:
            output_image = torch.cat((quantized_rgb, alpha.unsqueeze(-1)), dim=-1)
        else:
            output_image = quantized_rgb

        rgba_image = torch.cat((quantized_rgb, alpha.unsqueeze(-1)), dim=-1)
        comfy_mask = (1.0 - alpha).clamp(0.0, 1.0)
        return (output_image, comfy_mask, build_palette_preview(palette), rgba_image)


def parse_palette(
    palette_text: str,
    alpha_threshold: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    colors: list[list[float]] = []

    for line_number, raw_line in enumerate(palette_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        hex_match = HEX_COLOR_PATTERN.match(line)
        if hex_match:
            digits = hex_match.group(1)
            alpha = int(digits[6:8], 16) / 255.0 if len(digits) == 8 else 1.0
            if alpha >= alpha_threshold:
                colors.append([int(digits[index:index + 2], 16) / 255.0 for index in range(0, 6, 2)])
            continue

        rgb_match = RGB_COLOR_PATTERN.match(line)
        if rgb_match:
            channels = [int(channel) for channel in rgb_match.groups() if channel is not None]
            if any(channel < 0 or channel > 255 for channel in channels):
                raise ValueError(f"Palette line {line_number} has an RGB value outside 0-255: {raw_line!r}.")
            alpha = channels[3] / 255.0 if len(channels) == 4 else 1.0
            if alpha >= alpha_threshold:
                colors.append([channel / 255.0 for channel in channels[:3]])
            continue

        raise ValueError(f"Palette line {line_number} must be #RRGGBB, #RRGGBBAA, R,G,B, or R,G,B,A, got {raw_line!r}.")

    if not colors:
        raise ValueError("Palette has no visible colors. Add at least one color with alpha above the threshold.")

    return torch.tensor(colors, device=device, dtype=dtype)


def get_source_alpha(image: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        comfy_mask = mask.to(device=image.device, dtype=image.dtype)
        while comfy_mask.ndim < 3:
            comfy_mask = comfy_mask.unsqueeze(0)
        if comfy_mask.ndim == 4:
            comfy_mask = comfy_mask[..., 0]
        if comfy_mask.shape[0] == 1 and image.shape[0] > 1:
            comfy_mask = comfy_mask.expand(image.shape[0], -1, -1)
        return (1.0 - comfy_mask.clamp(0.0, 1.0)).clamp(0.0, 1.0)

    if image.shape[-1] >= 4:
        return image[..., 3].clamp(0.0, 1.0)

    return torch.ones(image.shape[:-1], device=image.device, dtype=image.dtype)


def nearest_palette_colors(
    rgb: torch.Tensor,
    palette: torch.Tensor,
    distance_metric: str,
    preserve_exact_matches: bool,
) -> torch.Tensor:
    distances = color_distances(rgb, palette, distance_metric)
    nearest_indices = distances.argmin(dim=-1)
    quantized = palette[nearest_indices]

    if preserve_exact_matches:
        exact_match = (rgb.unsqueeze(-2) == palette.view(*([1] * (rgb.ndim - 1)), -1, 3)).all(dim=-1).any(dim=-1)
        quantized = torch.where(exact_match.unsqueeze(-1), rgb, quantized)

    return quantized


def color_distances(rgb: torch.Tensor, palette: torch.Tensor, distance_metric: str) -> torch.Tensor:
    if distance_metric == "lab":
        lab = rgb_to_lab(rgb)
        palette_lab = rgb_to_lab(palette)
        delta = lab.unsqueeze(-2) - palette_lab.view(*([1] * (lab.ndim - 1)), -1, 3)
    else:
        delta = rgb.unsqueeze(-2) - palette.view(*([1] * (rgb.ndim - 1)), -1, 3)

    return (delta * delta).sum(dim=-1)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    original_dtype = rgb.dtype
    value = rgb.to(torch.float32).clamp(0.0, 1.0)
    linear = torch.where(value > 0.04045, torch.pow((value + 0.055) / 1.055, 2.4), value / 12.92)

    r = linear[..., 0]
    g = linear[..., 1]
    b = linear[..., 2]

    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    fx = lab_f(x)
    fy = lab_f(y)
    fz = lab_f(z)

    lab = torch.stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)), dim=-1)
    return lab.to(original_dtype)


def lab_f(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value > 0.008856, torch.pow(value, 1.0 / 3.0), 7.787 * value + 16.0 / 116.0)


def floyd_steinberg_quantize(
    rgb: torch.Tensor,
    palette: torch.Tensor,
    opaque: torch.Tensor,
    distance_metric: str,
    preserve_exact_matches: bool,
) -> torch.Tensor:
    work = rgb.clone()
    output = torch.zeros_like(rgb)
    batch_size, height, width, _channels = rgb.shape

    for batch_index in range(batch_size):
        for y in range(height):
            for x in range(width):
                if not bool(opaque[batch_index, y, x]):
                    continue

                old_pixel = work[batch_index, y, x].clamp(0.0, 1.0)
                new_pixel = nearest_palette_colors(
                    old_pixel.view(1, 1, 1, 3),
                    palette,
                    distance_metric,
                    preserve_exact_matches,
                ).view(3)
                output[batch_index, y, x] = new_pixel
                error = old_pixel - new_pixel

                diffuse_error(work, opaque, batch_index, y, x + 1, error, 7.0 / 16.0)
                diffuse_error(work, opaque, batch_index, y + 1, x - 1, error, 3.0 / 16.0)
                diffuse_error(work, opaque, batch_index, y + 1, x, error, 5.0 / 16.0)
                diffuse_error(work, opaque, batch_index, y + 1, x + 1, error, 1.0 / 16.0)

    return output


def diffuse_error(
    image: torch.Tensor,
    opaque: torch.Tensor,
    batch_index: int,
    y: int,
    x: int,
    error: torch.Tensor,
    weight: float,
) -> None:
    if y < 0 or y >= image.shape[1] or x < 0 or x >= image.shape[2]:
        return
    if not bool(opaque[batch_index, y, x]):
        return
    image[batch_index, y, x] = (image[batch_index, y, x] + error * weight).clamp(0.0, 1.0)


def build_palette_preview(palette: torch.Tensor) -> torch.Tensor:
    swatch_size = 24
    colors = palette.shape[0]
    preview = palette.new_zeros((1, swatch_size, swatch_size * colors, 3))

    for index in range(colors):
        start = index * swatch_size
        preview[:, :, start:start + swatch_size, :] = palette[index]

    return preview


NODE_CLASS_MAPPINGS = {
    "PixelAssetForgeFixedPaletteQuantize": FixedPaletteQuantize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelAssetForgeFixedPaletteQuantize": "Fixed Palette Quantize",
}
