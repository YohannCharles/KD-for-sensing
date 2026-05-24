from __future__ import annotations

MULTIMODAL_NF_DATASET_TYPE = "multimodal_nf"
MULTIMODAL_NF_HDF5_KEYS = {
    "csi": ("H", "channel", "Channel"),
    "gps": ("Pos", "position", "Position"),
    "beam_idx": ("BeamIdx", "beam_idx", "beam_index"),
    "beam_power": ("BeamPower", "beam_power", "power"),
    "los": ("Has_LoS", "HasLOS", "LoS"),
    "nf": ("Is_NF", "IsNF", "near_field"),
    "image": ("image", "Image", "RGB", "rgb"),
    "lidar": ("lidar", "LiDAR", "points", "point_cloud"),
    "city": ("City", "city", "city_id"),
    "trajectory": ("Trajectory", "TrajIdx", "trajectory_id", "traj_id"),
    "frame": ("Frame", "FrameIdx", "frame_id"),
    "metadata": ("Metadata", "metadata"),
    "traj_nlos": ("Traj_Is_NLoS", "traj_is_nlos"),
    "mode": ("Mode_Idx", "mode_idx"),
}
REQUIRED_MULTIMODAL_NF_FIELDS = ("csi", "gps", "beam_idx", "beam_power", "los", "nf")
DEFAULT_DENSE_CODEBOOK_SHAPE = (90, 45, 16)
DEFAULT_SMALL_CODEBOOK_SHAPE = (20, 20, 10)
DEFAULT_FLATTEN_ORDER = "azimuth_elevation_range"

__all__ = [
    "DEFAULT_DENSE_CODEBOOK_SHAPE",
    "DEFAULT_FLATTEN_ORDER",
    "DEFAULT_SMALL_CODEBOOK_SHAPE",
    "MULTIMODAL_NF_DATASET_TYPE",
    "MULTIMODAL_NF_HDF5_KEYS",
    "REQUIRED_MULTIMODAL_NF_FIELDS",
]
