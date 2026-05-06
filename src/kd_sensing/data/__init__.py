__all__ = [
    "create_samples",
    "SequenceSamples",
    "DeepSense6GDataset",
    "Scenario9Dataset",
    "Scenario32Dataset",
    "SyntheticSequenceDataset",
]


def __getattr__(name: str):
    if name in {"create_samples", "SequenceSamples"}:
        from . import samples

        return getattr(samples, name)
    if name in {"DeepSense6GDataset", "Scenario9Dataset", "Scenario32Dataset", "SyntheticSequenceDataset"}:
        from . import datasets

        return getattr(datasets, name)
    raise AttributeError(f"module 'kd_sensing.data' has no attribute {name!r}")
