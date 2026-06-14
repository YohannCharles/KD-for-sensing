from kd_sensing.data.difficulty.schema import (
    DifficultyContext,
    DifficultyOperatorConfig,
    DifficultyOperatorOutcome,
    DifficultyProfile,
    DifficultyResult,
    DifficultyWarning,
    normalize_config_difficulty,
    normalize_difficulty_profiles,
    profiles_from_resolved_config,
    select_profiles_for_context,
    stable_digest,
    stable_int_seed,
)

_PIPELINE_EXPORTS = {
    "apply_configured_difficulty",
    "apply_difficulty_pipeline",
    "assert_target_preserved",
    "runtime_difficulty_metadata",
}


def __getattr__(name: str):
    if name in _PIPELINE_EXPORTS:
        from kd_sensing.data.difficulty import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)


__all__ = [
    "DifficultyContext",
    "DifficultyOperatorConfig",
    "DifficultyOperatorOutcome",
    "DifficultyProfile",
    "DifficultyResult",
    "DifficultyWarning",
    "apply_configured_difficulty",
    "apply_difficulty_pipeline",
    "assert_target_preserved",
    "normalize_config_difficulty",
    "normalize_difficulty_profiles",
    "profiles_from_resolved_config",
    "runtime_difficulty_metadata",
    "select_profiles_for_context",
    "stable_digest",
    "stable_int_seed",
]
