# MMW 实验矩阵

| 方法 | tracked recipe | 角色 |
| --- | --- | --- |
| T2 | `configs/mmw/t2.yaml` | 主方法 |
| S1 | `configs/mmw/s1.yaml` | 关闭 temporal superset consistency 的对照 |
| AMBER-Full | `configs/mmw/amber_full.yaml` | baseline |
| RMBP-MM | `configs/mmw/rmbp_mm.yaml` | baseline |

所有行使用 MMW 15-domain、四模态、40 epoch 和 fixed-last checkpoint。all-weather matrix 与 BPA/CMA helper 的输出必须留在 `outputs/`，不能反向成为 canonical config。CMSBL V0--V4 只属于 inner/development，不是第五个正式比较方法。

DeepSense6G 使用独立的 `configs/deepsense6g/t2.yaml`，仅覆盖 Scene31–34 四模态 future-beam 路径；它不属于本 MMW 比较矩阵，也不复用 MMW evidence helper。
