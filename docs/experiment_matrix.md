# MMW 实验矩阵

| 方法 | tracked recipe | 角色 |
| --- | --- | --- |
| T2 | `configs/mmw/t2.yaml` | 主方法 |
| S1 | `configs/mmw/s1.yaml` | 关闭 temporal superset consistency 的对照 |
| AMBER-Full | `configs/mmw/amber_full.yaml` | baseline |
| RMBP-MM | `configs/mmw/rmbp_mm.yaml` | baseline |

所有行使用 MMW 15-domain、四模态、40 epoch 和 fixed-last checkpoint。all-weather matrix、T2 screening、BPA/CMA 的 helper 是本地脚本；它们的输出必须留在 `outputs/`，不能反向成为 canonical config。
