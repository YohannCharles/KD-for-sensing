# 项目表面积 Inventory

当前 source surface：

| 区域 | 保留职责 |
| --- | --- |
| `configs/mmw/` | shared base、T2、S1、AMBER-Full、RMBP-MM |
| `configs/deepsense6g/` | shared base、Scene31--34 四模态 T2 |
| `src/kd_sensing/models/` | U-Mask、BCACL U2、AMBER-Full、RMBP-MM 与四模态 encoder |
| `src/kd_sensing/losses/` | T2/BPA/CMA/superset、BCACL U2 与 CMSBL M1--M3 |
| `src/kd_sensing/data/` | MMW prepared sequence、DeepSense6G 标准 CSV、四模态 transform 与 temporal missing |
| `src/kd_sensing/engine/` | train/evaluate、双数据集 dataloader、checkpoint 与 training extension |
| `scripts/` | MMW all-weather、BPA/CMA、必要 summary 与 compile verification |
| `pyproject.toml` | train/evaluate/preprocess 三个 public CLI |

无法追溯到上述闭包的代码、YAML、script、test 或 compatibility guard 必须删除。历史路线只通过 Git、OpenSpec archive、`retired_routes.md` 和未改动的本地 `outputs/` 追溯。
