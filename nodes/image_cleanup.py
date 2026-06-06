import torch
import torch.nn.functional as F

from .fixed_palette import get_source_alpha


class StrayPixelCleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "connectivity": (["8", "4"], {"default": "8"}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "alpha_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_mode": (["nontransparent_only"], {"default": "nontransparent_only"}),
                "output_alpha_mode": (["binary", "keep"], {"default": "binary"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "rgba_image")
    FUNCTION = "clean"
    CATEGORY = "AssetForge/Cleanup"

    def clean(
        self,
        image: torch.Tensor,
        connectivity: str,
        iterations: int,
        alpha_threshold: float,
        target_mode: str,
        output_alpha_mode: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del target_mode

        rgb = image[..., :3].clamp(0.0, 1.0)
        source_alpha = get_source_alpha(image, mask)
        opaque = source_alpha >= alpha_threshold
        kernel = build_neighbor_kernel(connectivity, image.device, image.dtype)

        for _index in range(iterations):
            neighbor_count = count_opaque_neighbors(opaque, kernel)
            isolated = opaque & (neighbor_count == 0)
            if not bool(isolated.any()):
                break
            opaque = opaque & ~isolated

        if output_alpha_mode == "keep":
            alpha = torch.where(opaque, source_alpha, torch.zeros_like(source_alpha))
        else:
            alpha = opaque.to(image.dtype)

        cleaned_rgb = torch.where(opaque.unsqueeze(-1), rgb, torch.zeros_like(rgb))
        if image.shape[-1] >= 4:
            output_image = torch.cat((cleaned_rgb, alpha.unsqueeze(-1)), dim=-1)
        else:
            output_image = cleaned_rgb

        rgba_image = torch.cat((cleaned_rgb, alpha.unsqueeze(-1)), dim=-1)
        comfy_mask = (1.0 - alpha).clamp(0.0, 1.0)
        return (output_image, comfy_mask, rgba_image)


class TightCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "output_alpha_mode": (["binary", "keep"], {"default": "binary"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "rgba_image")
    FUNCTION = "crop"
    CATEGORY = "AssetForge/Cleanup"

    def crop(
        self,
        image: torch.Tensor,
        alpha_threshold: float,
        padding: int,
        output_alpha_mode: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = image[..., :3].clamp(0.0, 1.0)
        source_alpha = get_source_alpha(image, mask)
        visible = source_alpha >= alpha_threshold

        if not bool(visible.any()):
            alpha = build_output_alpha(source_alpha, visible, output_alpha_mode, image.dtype)
            output_image = append_alpha_if_needed(rgb, alpha, image.shape[-1])
            rgba_image = torch.cat((rgb, alpha.unsqueeze(-1)), dim=-1)
            comfy_mask = (1.0 - alpha).clamp(0.0, 1.0)
            return (output_image, comfy_mask, rgba_image)

        rows = torch.where(visible.any(dim=(0, 2)))[0]
        columns = torch.where(visible.any(dim=(0, 1)))[0]
        height = image.shape[1]
        width = image.shape[2]

        top = max(int(rows[0].item()) - padding, 0)
        bottom = min(int(rows[-1].item()) + padding + 1, height)
        left = max(int(columns[0].item()) - padding, 0)
        right = min(int(columns[-1].item()) + padding + 1, width)

        cropped_rgb = rgb[:, top:bottom, left:right, :]
        cropped_source_alpha = source_alpha[:, top:bottom, left:right]
        cropped_visible = visible[:, top:bottom, left:right]
        alpha = build_output_alpha(cropped_source_alpha, cropped_visible, output_alpha_mode, image.dtype)

        output_image = append_alpha_if_needed(cropped_rgb, alpha, image.shape[-1])
        rgba_image = torch.cat((cropped_rgb, alpha.unsqueeze(-1)), dim=-1)
        comfy_mask = (1.0 - alpha).clamp(0.0, 1.0)
        return (output_image, comfy_mask, rgba_image)


def build_output_alpha(
    source_alpha: torch.Tensor,
    visible: torch.Tensor,
    output_alpha_mode: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if output_alpha_mode == "keep":
        return torch.where(visible, source_alpha, torch.zeros_like(source_alpha))
    return visible.to(dtype)


def append_alpha_if_needed(rgb: torch.Tensor, alpha: torch.Tensor, channel_count: int) -> torch.Tensor:
    if channel_count >= 4:
        return torch.cat((rgb, alpha.unsqueeze(-1)), dim=-1)
    return rgb


def build_neighbor_kernel(connectivity: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if connectivity == "4":
        values = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    else:
        values = [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]

    return torch.tensor(values, device=device, dtype=dtype).view(1, 1, 3, 3)


def count_opaque_neighbors(opaque: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    opaque_float = opaque.to(dtype=kernel.dtype).unsqueeze(1)
    neighbor_count = F.conv2d(opaque_float, kernel, padding=1)
    return neighbor_count.squeeze(1)


NODE_CLASS_MAPPINGS = {
    "AssetForgeStrayPixelCleaner": StrayPixelCleaner,
    "AssetForgeTightCrop": TightCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AssetForgeStrayPixelCleaner": "Stray Pixel Cleaner",
    "AssetForgeTightCrop": "Tight Crop",
}
