from .nodes.color_masks import (
    NODE_CLASS_MAPPINGS as _color_mask_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _color_mask_names,
)
from .nodes.image_filters import (
    NODE_CLASS_MAPPINGS as _image_filter_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _image_filter_names,
)
from .nodes.fixed_palette import (
    NODE_CLASS_MAPPINGS as _fixed_palette_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _fixed_palette_names,
)
from .nodes.image_cleanup import (
    NODE_CLASS_MAPPINGS as _image_cleanup_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _image_cleanup_names,
)
from .nodes.grid import (
    NODE_CLASS_MAPPINGS as _grid_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _grid_names,
)


NODE_CLASS_MAPPINGS = {
    **_color_mask_classes,
    **_image_filter_classes,
    **_fixed_palette_classes,
    **_image_cleanup_classes,
    **_grid_classes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_color_mask_names,
    **_image_filter_names,
    **_fixed_palette_names,
    **_image_cleanup_names,
    **_grid_names,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
