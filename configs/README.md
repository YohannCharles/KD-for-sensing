# Config Surface

当前源码配置保留 MMW T2/baseline 与受限 DeepSense6G T2 数据路径。

| Family | Paths | Purpose |
| --- | --- | --- |
| T2 / S1 / AMBER-Full / RMBP-MM | `configs/mmw/{_base,t2,s1,amber_full,rmbp_mm}.yaml` | 训练、评估和 all-weather matrix 的 tracked 输入 |
| DeepSense6G T2 | `configs/deepsense6g/{_base,t2}.yaml` | Scene31–34 四模态 future-beam 主线；默认 Scene31 |
| MMW preparation | `configs/preprocess/mmw_radar_maps_all_weather.yaml` | Town03 all-weather 雷达图准备 |

历史 C2、DeepSense6G CSI/mmWave/soft-label/cache 分支、physics、GPS-only、pretraining 和 H5/P1 配置已退役。历史用途见 `docs/retired_routes.md` 与 OpenSpec archive；不要恢复兼容 YAML、alias 或 virtual config。
