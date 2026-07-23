# KD for Sensing

当前源码只维护两个四模态波束预测工作流：Clean MMW 的 U0 主线及其 AMBER-Full、RMBP-MM baseline；DeepSense6G Scene31--34 的独立 T2 路线也保留。

MMW 训练只能通过经审计的 `inner_train` / `inner_validation` protocol 启动。outer test、confirmation train 和任何 train/validation 重叠都会在创建数据 loader 前被拒绝。

## MMW 工作流

所有项目命令使用 `kd_mm_beam` 环境。先从本地 split manifest 生成 protocol 和审计报告：

```bash
conda run -n kd_mm_beam python scripts/audit_clean_inner_protocol.py \
  --source-manifest /path/to/inner_split_manifest.json \
  --protocol-output outputs/mmw_clean_u0/protocol.yaml \
  --audit-json outputs/mmw_clean_u0/audit.json \
  --audit-md outputs/mmw_clean_u0/audit.md

conda run -n kd_mm_beam python scripts/launch_mmw_all_weather_matrix.py \
  --protocol outputs/mmw_clean_u0/protocol.yaml \
  --audit-report outputs/mmw_clean_u0/audit.json \
  --output-root outputs/mmw_clean_u0 \
  --methods U0 --seeds 1 --gpus 0

conda run -n kd_mm_beam python scripts/eval_mmw_all_weather_matrix.py \
  --root outputs/mmw_clean_u0 --methods U0 --seeds 1
```

`configs/mmw/u0.yaml`、`amber_full.yaml` 和 `rmbp_mm.yaml` 是 tracked 模型 recipe；它们不携带本地 MMW split，不能绕过 protocol 直接训练。

## DeepSense6G

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/deepsense6g/t2.yaml
conda run -n kd_mm_beam kd-sensing-evaluate --help
conda run -n kd_mm_beam kd-sensing-preprocess --help
```

DeepSense6G 保持自己的 Scene31--34、四模态、64 类 future-beam split 契约，不会被 MMW protocol 重解释。

`dataset/`、`outputs/`、`outputs/cache/`、`cache/`、日志和 checkpoint 都是本地产物，本次收敛不会读取、移动或删除它们。

## 验证

```bash
make verify-quick
make verify-cli-config
make verify-compile
conda run -n kd_mm_beam pytest -q
```
