__all__ = [
    "DeepSense6GDataset",
    "Scenario9Dataset",
    "Scenario31Dataset",
    "Scenario32Dataset",
    "SyntheticSequenceDataset",
]


def __getattr__(name: str):
    if name in {"DeepSense6GDataset", "Scenario9Dataset", "Scenario31Dataset", "Scenario32Dataset"}:
        from . import scenario9

        return getattr(scenario9, name)
    if name == "SyntheticSequenceDataset":
        from . import synthetic

        return synthetic.SyntheticSequenceDataset
    raise AttributeError(f"module 'kd_sensing.data.datasets' has no attribute {name!r}")
