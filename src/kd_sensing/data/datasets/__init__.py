__all__ = [
    "DeepSense6GDataset",
    "MMWDataset",
    "RaymobtimeS008SnapshotDataset",
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
    if name == "RaymobtimeS008SnapshotDataset":
        from . import raymobtime_s008

        return raymobtime_s008.RaymobtimeS008SnapshotDataset
    raise AttributeError(f"module 'kd_sensing.data.datasets' has no attribute {name!r}")
