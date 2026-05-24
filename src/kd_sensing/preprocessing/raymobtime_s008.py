from __future__ import annotations

from typing import Any

from kd_sensing.preprocessing.raymobtime_s008_beam_labels import build_s008_labels, normalize_beam_labels
from kd_sensing.preprocessing.raymobtime_s008_cache import build_s008_cache
from kd_sensing.preprocessing.raymobtime_s008_common import RAY_FEATURE_NAMES, RaymobtimePaths, resolve_raymobtime_paths
from kd_sensing.preprocessing.raymobtime_s008_index import build_s008_index
from kd_sensing.preprocessing.raymobtime_s008_paths import audit_s008_files
from kd_sensing.preprocessing.raymobtime_s008_ray_features import extract_s008_ray_features
from kd_sensing.registries import PREPROCESSORS


@PREPROCESSORS.register("raymobtime_s008_audit")
class RaymobtimeS008AuditPreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return audit_s008_files(**self.kwargs)


@PREPROCESSORS.register("raymobtime_s008_index")
class RaymobtimeS008IndexPreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return build_s008_index(**self.kwargs)


@PREPROCESSORS.register("raymobtime_s008_ray_features")
class RaymobtimeS008RayFeaturePreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return extract_s008_ray_features(**self.kwargs)


@PREPROCESSORS.register("raymobtime_s008_cache")
class RaymobtimeS008CachePreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return build_s008_cache(**self.kwargs)


__all__ = [
    "RAY_FEATURE_NAMES",
    "RaymobtimePaths",
    "audit_s008_files",
    "build_s008_cache",
    "build_s008_index",
    "build_s008_labels",
    "extract_s008_ray_features",
    "normalize_beam_labels",
    "resolve_raymobtime_paths",
]
