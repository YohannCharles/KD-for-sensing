from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kd_sensing.data.layouts import multimodal_nf_layout
from kd_sensing.utils.paths import resolve_path

@dataclass(frozen=True)
class MultimodalNFPaths:
    data_root: Path
    raw_root: Path
    codebook_root: Path
    cache_dir: Path
    output_dir: Path


def resolve_multimodal_nf_paths(
    *,
    data_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    codebook_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> MultimodalNFPaths:
    layout = multimodal_nf_layout()
    root = resolve_path(data_root or layout.root)
    raw = resolve_path(raw_root) if raw_root is not None else root / "raw"
    codebooks = resolve_path(codebook_root) if codebook_root is not None else root / "codebooks"
    cache = resolve_path(cache_dir) if cache_dir is not None else root / "cache"
    output = resolve_path(output_dir) if output_dir is not None else resolve_path(layout.audit_output_root)
    return MultimodalNFPaths(
        data_root=root,
        raw_root=raw,
        codebook_root=codebooks,
        cache_dir=cache,
        output_dir=output,
    )

__all__ = ["MultimodalNFPaths", "resolve_multimodal_nf_paths"]
