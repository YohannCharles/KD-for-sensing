"""Small differentiable physics helpers for MMW beam baselines."""

from kd_sensing.models.physics.array_response import ula_array_response
from kd_sensing.models.physics.beam_scoring import beam_logits_from_channel
from kd_sensing.models.physics.channel_synthesizer import synthesize_ula_channel
from kd_sensing.models.physics.complex_utils import (
    abs_square,
    complex_mse,
    complex_to_ri,
    normalize_angle,
    ri_to_complex,
)

__all__ = [
    "abs_square",
    "beam_logits_from_channel",
    "complex_mse",
    "complex_to_ri",
    "normalize_angle",
    "ri_to_complex",
    "synthesize_ula_channel",
    "ula_array_response",
]
