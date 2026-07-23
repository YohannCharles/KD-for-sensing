# Config Surface

| 路线 | 配置 | 用途 |
| --- | --- | --- |
| Clean MMW U0 | `configs/mmw/u0.yaml` | 当前 MMW 主方法 |
| Clean MMW baselines | `configs/mmw/amber_full.yaml`, `configs/mmw/rmbp_mm.yaml` | 与 U0 使用同一 audited protocol 的本地比较 |
| DeepSense6G T2 | `configs/deepsense6g/t2.yaml` | Scene31--34 四模态 future-beam 路线 |
| MMW preparation | `configs/preprocess/mmw_radar_maps_all_weather.yaml` | 本地雷达图准备 |

MMW recipe 只定义模型和通用训练参数。实际数据域必须由 `scripts/launch_mmw_all_weather_matrix.py` 注入经审计的 clean protocol；不要添加旧 YAML alias 或从 `outputs/` 读取历史 config。
