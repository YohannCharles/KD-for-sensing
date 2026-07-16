# 项目表面积 Inventory

当前 source surface：

| 区域 | 保留职责 |
| --- | --- |
| `configs/mmw/` | T2、S1、AMBER-Full、RMBP-MM tracked recipe |
| `configs/deepsense6g/` | Scene31–34 四模态 future-beam T2 recipe |
| `src/kd_sensing/models/` | U-Mask、AMBER-Full、RMBP-MM 与四模态 encoder |
| `src/kd_sensing/data/` | MMW prepared sequence、DeepSense6G 标准 CSV、image/radar/gps/lidar transform、temporal missing |
| `src/kd_sensing/engine/` | train/evaluate、双数据集 dataloader、checkpoint、T2 extension |
| `scripts/` | all-weather、T2 screening、BPA/CMA 与 MMW evidence helpers |
| `pyproject.toml` | train/evaluate/preprocess 三个 public CLI |

非上述路径只能作为通用运行时依赖；若不被 retained workflow 引用，应删除而不是保留 compatibility guard。退役族的用途和 Git/OpenSpec 追溯入口见 [retired_routes.md](retired_routes.md)。
