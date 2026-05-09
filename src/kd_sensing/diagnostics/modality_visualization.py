"""Compatibility facade for the Gradio viewer manifest workflow."""

from __future__ import annotations

from kd_sensing.diagnostics.visualization.core import *
from kd_sensing.diagnostics.viewer_manifest import export_viewer_manifest


def visualize_modalities(cfg: dict) -> dict:
    """Backward-compatible entry point that now exports a Gradio viewer manifest."""

    return export_viewer_manifest(cfg)
