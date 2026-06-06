import re

import torch


HEX_COLOR_PATTERN = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")


class MaskFromHexColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hex_color": ("STRING", {"default": "#000000"}),
                "tolerance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask"
    CATEGORY = "AssetForge/Color Masks"

    def create_mask(self, image: torch.Tensor, hex_color: str, tolerance: float) -> tuple[torch.Tensor]:
        target = self._parse_hex_color(hex_color, image.device, image.dtype)
        rgb = image[..., :3]
        max_channel_delta = torch.abs(rgb - target).amax(dim=-1)
        mask = (max_channel_delta <= tolerance / 255.0).to(image.dtype)
        return (mask,)

    @staticmethod
    def _parse_hex_color(hex_color: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = hex_color.strip()
        match = HEX_COLOR_PATTERN.match(value)
        if not match:
            raise ValueError(f"Expected a hex color like #3F48CC or #fff, got {hex_color!r}.")

        digits = match.group(1)
        if len(digits) == 3:
            digits = "".join(channel * 2 for channel in digits)

        channels = [int(digits[index:index + 2], 16) / 255.0 for index in range(0, 6, 2)]
        return torch.tensor(channels, device=device, dtype=dtype)


NODE_CLASS_MAPPINGS = {
    "AssetForgeMaskFromHexColor": MaskFromHexColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AssetForgeMaskFromHexColor": "Mask From Hex Color",
}
