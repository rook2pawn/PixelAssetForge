import torch


class FilterBlankImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tolerance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 255.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "filter_images"
    CATEGORY = "PixelAssetForge/Image Filters"

    def filter_images(self, images: torch.Tensor, tolerance: float) -> tuple[torch.Tensor]:
        pixel_range = images.amax(dim=(1, 2)) - images.amin(dim=(1, 2))
        is_blank = (pixel_range < tolerance / 255.0).all(dim=-1)
        filtered_images = images[~is_blank]
        return (filtered_images,)


NODE_CLASS_MAPPINGS = {
    "PixelAssetForgeFilterBlankImages": FilterBlankImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelAssetForgeFilterBlankImages": "Filter Blank Images",
}
