import torch

from .fixed_palette import get_source_alpha


class PaletteReportFromImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sort_mode": (["frequency_desc", "first_seen", "hex"], {"default": "frequency_desc"}),
                "include_alpha_entry": ("BOOLEAN", {"default": True}),
                "max_colors": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "IMAGE")
    RETURN_NAMES = ("palette_text", "report_text", "color_count", "palette_preview")
    FUNCTION = "report"
    CATEGORY = "PixelAssetForge/Palette"

    def report(
        self,
        image: torch.Tensor,
        alpha_threshold: float,
        sort_mode: str,
        include_alpha_entry: bool,
        max_colors: int,
        mask: torch.Tensor | None = None,
    ) -> tuple[str, str, int, torch.Tensor]:
        entries, transparent_count = collect_palette_entries(image, alpha_threshold, sort_mode, max_colors, mask)
        palette_lines = [entry.hex_color for entry in entries]
        if include_alpha_entry and transparent_count > 0:
            palette_lines.append("#00000000")

        report_lines = [
            f"visible colors: {len(entries)}",
            f"transparent pixels: {transparent_count}",
            "",
            "color,count,percent",
        ]
        visible_total = sum(entry.count for entry in entries)
        for entry in entries:
            percent = 0.0 if visible_total == 0 else entry.count * 100.0 / visible_total
            report_lines.append(f"{entry.hex_color},{entry.count},{percent:.2f}%")

        return (
            "\n".join(palette_lines),
            "\n".join(report_lines),
            len(entries),
            build_palette_preview_from_entries(entries, image.device, image.dtype),
        )


class PaletteEntry:
    def __init__(self, rgb: tuple[int, int, int], count: int, first_seen: int):
        self.rgb = rgb
        self.count = count
        self.first_seen = first_seen

    @property
    def hex_color(self) -> str:
        return f"#{self.rgb[0]:02X}{self.rgb[1]:02X}{self.rgb[2]:02X}"


def collect_palette_entries(
    image: torch.Tensor,
    alpha_threshold: float,
    sort_mode: str,
    max_colors: int,
    mask: torch.Tensor | None = None,
) -> tuple[list[PaletteEntry], int]:
    rgb8 = (image[..., :3].detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    alpha = get_source_alpha(image, mask).detach().cpu()
    visible = alpha >= alpha_threshold
    transparent_count = int((~visible).sum().item())

    counts: dict[tuple[int, int, int], int] = {}
    first_seen: dict[tuple[int, int, int], int] = {}
    flat_rgb = rgb8.reshape(-1, 3)
    flat_visible = visible.reshape(-1)

    for index, (pixel, is_visible) in enumerate(zip(flat_rgb, flat_visible)):
        if not bool(is_visible):
            continue

        color = tuple(int(channel) for channel in pixel.tolist())
        counts[color] = counts.get(color, 0) + 1
        if color not in first_seen:
            first_seen[color] = index

        if len(counts) > max_colors:
            raise ValueError(f"Image contains more than max_colors={max_colors} visible colors.")

    entries = [PaletteEntry(color, counts[color], first_seen[color]) for color in counts]
    if sort_mode == "first_seen":
        entries.sort(key=lambda entry: entry.first_seen)
    elif sort_mode == "hex":
        entries.sort(key=lambda entry: entry.hex_color)
    else:
        entries.sort(key=lambda entry: (-entry.count, entry.first_seen))

    return entries, transparent_count


def build_palette_preview_from_entries(
    entries: list[PaletteEntry],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    swatch_size = 24
    colors = max(len(entries), 1)
    preview = torch.zeros((1, swatch_size, swatch_size * colors, 3), device=device, dtype=dtype)

    for index, entry in enumerate(entries):
        start = index * swatch_size
        preview[:, :, start:start + swatch_size, :] = torch.tensor(entry.rgb, device=device, dtype=dtype) / 255.0

    return preview


NODE_CLASS_MAPPINGS = {
    "PixelAssetForgePaletteReportFromImage": PaletteReportFromImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelAssetForgePaletteReportFromImage": "Palette Report From Image",
}
