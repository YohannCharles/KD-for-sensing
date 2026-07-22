# KD for Sensing

本仓库保留 MMW 与 DeepSense6G 两个四模态波束预测数据集。MMW 提供 T2、S1、AMBER-Full 与 RMBP-MM 的固定比较协议；DeepSense6G 仅提供 Scene31--34 的 T2 数据路径。当前研究扩展是 T2 上的 BCACL U2/CMSBL，且只改变训练 objective，不改变推理结构。

## 入口

所有项目命令使用 `kd_mm_beam` 环境：

```bash
# MMW 会先校验 15 个 condition/scene/split，并由 launcher 写入带训练画像的 generated config。
conda run -n kd_mm_beam python scripts/launch_mmw_all_weather_matrix.py \
  --output-root outputs/mmw_t2_seed1 --methods T2 --seeds 1 --gpus 0 --preflight-only
conda run -n kd_mm_beam python scripts/launch_mmw_all_weather_matrix.py \
  --output-root outputs/mmw_t2_seed1 --methods T2 --seeds 1 --gpus 0

# DeepSense6G 仍使用其 tracked T2 recipe；本地 CSV 和资源必须已准备完成。
conda run -n kd_mm_beam kd-sensing-train --config configs/deepsense6g/t2.yaml
conda run -n kd_mm_beam kd-sensing-evaluate --help
conda run -n kd_mm_beam kd-sensing-preprocess --help
```

`configs/mmw/t2.yaml` 是 architecture recipe，不包含 MMW 的 condition、scene、split 或训练画像，不能单独作为 MMW 训练命令。MMW 主实验与受控消融通过 all-weather launcher/evaluator 和 BPA/CMA helper 运行；它们只读取 `configs/mmw/` 下的 tracked recipe，不读取 `outputs/` 中的历史配置。

## 范围

- 保留共享四模态 `image/radar/gps/lidar`、T2 temporal masked-mean router、BPA/CMA、same-model superset consistency、BCACL U2 与 CMSBL M1--M3。
- MMW 使用 prepared sequence 和四方法评估；DeepSense6G 仅支持 Scene31–34 标准 CSV、future-beam 64 类硬标签与 T2 recipe，不提供专属 CLI、缓存或 baseline 矩阵。
- 训练输出、数据、日志、cache 与 checkpoint 均为本地产物，不提交。
- 已退役的 PCER/PGCD/动态 Router、PR-SQDF、missing residual、feature/prototype fusion、fallback、BT-SCL 及更早路线见 [retired_routes.md](docs/retired_routes.md)。

## 验证

```bash
make verify-quick
make verify-cli-config
make verify-compile
openspec validate --all --strict
conda run -n kd_mm_beam pytest -q
```
