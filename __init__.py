"""Agate for ComfyUI: text-to-image with Agate, LogoLabs' 260M model (MIT)."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Browser-side branding: web/js/agate_brand.js draws the Agate mark and
# wordmark on every Agate node. Removing the directory (and this line) changes
# nothing about how the nodes run.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
