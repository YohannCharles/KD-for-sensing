__all__ = [
    "DeepSense6GDataset",
    "MMWDataset",
    "SyntheticSequenceDataset",
]


def __getattr__(name: str):
    if name == "DeepSense6GDataset":
        from . import deepsense6g

        return deepsense6g.DeepSense6GDataset
    if name == "MMWDataset":
        from . import mmw

        return mmw.MMWDataset
    if name == "SyntheticSequenceDataset":
        from . import synthetic

        return synthetic.SyntheticSequenceDataset
    raise AttributeError(f"module 'kd_sensing.data.datasets' has no attribute {name!r}")
