# KD for Sensing

本仓库只保留 MMW 四模态波束预测主线：T2、S1、AMBER-Full 与 RMBP-MM。T2 是唯一主方法；S1 是 temporal consistency 关闭的对照；AMBER-Full 和 RMBP-MM 是本地 baseline。

## 入口

所有项目命令使用 `kd_mm_beam` 环境：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/mmw/t2.yaml
conda run -n kd_mm_beam kd-sensing-evaluate --help
conda run -n kd_mm_beam kd-sensing-preprocess --help
```

MMW 主实验与受控消融通过 `scripts/launch_mmw_all_weather_matrix.py`、`scripts/eval_mmw_all_weather_matrix.py`、`scripts/launch_mmw_t2_hyperparameter_screening.py` 和 BPA/CMA runner 运行。它们只读取 `configs/mmw/` 下的 tracked recipe，不读取 `outputs/` 中的历史配置。

## 范围

- 保留四模态 `image/radar/gps/lidar`、MMW prepared sequence dataset、T2 temporal masked-mean router、BPA/CMA、same-model superset consistency 和四方法评估。
- 训练输出、数据、日志、cache 与 checkpoint 均为本地产物，不提交。
- 已退役的蒸馏、CSI/mmWave、physics、DeepSense/Scene31/C2、预训练、GPS-only 和旧诊断路线见 [retired_routes.md](docs/retired_routes.md)。

## 验证

```bash
make verify-quick
make verify-cli-config
make verify-compile
openspec validate --all --strict
conda run -n kd_mm_beam pytest -q
```
