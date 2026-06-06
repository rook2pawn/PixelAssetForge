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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AssetForgeStrayPixelCleaner": "Stray Pixel Cleaner",
}
