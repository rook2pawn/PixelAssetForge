from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest


@pytest.mark.live_comfy
def test_live_comfy_registers_pixel_assetforge_nodes() -> None:
    if os.environ.get("ASSETFORGE_LIVE_COMFY") != "1":
        pytest.skip("Set ASSETFORGE_LIVE_COMFY=1 to check the running ComfyUI server.")

    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=3) as response:
            object_info = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError) as error:
        pytest.skip(f"ComfyUI API is unavailable: {error}")

    expected_nodes = {
        "PixelAssetForgeFixedPaletteQuantize",
        "PixelAssetForgeStrayPixelCleaner",
        "PixelAssetForgeTightCrop",
        "PixelAssetForgeColorToAlpha",
        "PixelAssetForgeGridOverlay",
        "PixelAssetForgeMajorityRulesGrid",
        "PixelAssetForgePaletteReportFromImage",
        "PixelAssetForgeMaskFromHexColor",
        "PixelAssetForgeFilterBlankImages",
    }
    missing_nodes = expected_nodes - set(object_info)
    assert not missing_nodes, f"Missing live Comfy nodes: {sorted(missing_nodes)}"
