from __future__ import annotations

from typing import Any

from kd_sensing.preprocessing.multimodal_nf_common import (
    DEFAULT_DENSE_CODEBOOK_SHAPE,
    DEFAULT_SMALL_CODEBOOK_SHAPE,
    audit_multimodal_nf_files,
    build_multimodal_nf_index,
    flatten_beam_triplet,
    parse_codebook_metadata,
    resolve_multimodal_nf_paths,
    unflatten_beam_class,
)
from kd_sensing.registries import PREPROCESSORS


@PREPROCESSORS.register("multimodal_nf_audit")
class MultimodalNFAuditPreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return audit_multimodal_nf_files(**self.kwargs)


@PREPROCESSORS.register("multimodal_nf_index")
class MultimodalNFIndexPreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return build_multimodal_nf_index(**self.kwargs)


__all__ = [
    "DEFAULT_DENSE_CODEBOOK_SHAPE",
    "DEFAULT_SMALL_CODEBOOK_SHAPE",
    "MultimodalNFAuditPreprocessor",
    "MultimodalNFIndexPreprocessor",
    "audit_multimodal_nf_files",
    "build_multimodal_nf_index",
    "flatten_beam_triplet",
    "parse_codebook_metadata",
    "resolve_multimodal_nf_paths",
    "unflatten_beam_class",
]
