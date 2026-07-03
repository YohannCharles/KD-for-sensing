## 1. Adaptive Sampler 与 Loss

- [x] 1.1 在现有 U-MaskBeamJEPA 缺失模态 mask 采样路径中实现 opt-in `adaptive_pattern` sampler，包含 EMA state、warmup、score mode、probability clipping、fallback warning 和 epoch CSV log。
- [x] 1.2 在 loss registry 中新增 `beam_neighborhood_ce`，实现 hard CE 与 circular Gaussian soft CE 的 `mix_ce` 混合、ignore index、dtype/device 兼容和启动诊断。
- [x] 1.3 在 loss registry 中新增 `label_smoothing_ce` baseline，并保持未 opt-in 配置的 CE/focal 行为不变。
- [x] 1.4 添加 focused tests 覆盖 adaptive sampler probability/EMA/log fallback 和 beam-neighborhood/label-smoothing loss sanity。

## 2. Scene31 BC 配置矩阵

- [x] 2.1 扩展 `scripts/generate_scene31_next_round.py`，新增 B P0/P1、C P0/P1、BC P0 和 label smoothing run specs。
- [x] 2.2 更新 `configs/scene31/next_round/experiment_manifest.csv` 与 JSON，保持 generated YAML 不纳入源码。
- [x] 2.3 更新 Scene31 generator tests，校验 run name token 与 alpha/temperature/sigma/mix/epoch/seed/sampler/loss 字段一致且不启用 condBTAPA/weakKD。

## 3. Launcher 与 Summary

- [x] 3.1 新增 `scripts/run_scene31_bc_next.sh`，支持 `--group b_p0|c_p0|bc_p0|all_p0|all|baselines`、`--gpu`、`--overwrite`、`--train-only`、`--eval-only`、日志、跳过 complete 和失败继续。
- [x] 3.2 launcher 将 `amr_net_supervised` 与 `amber_full_architecture` 纳入 baseline group，并在 `all_p0`/`all` 中包含这两个需要训练的 baseline。
- [x] 3.3 新增或扩展 BC summary，按 seed 归并 method，保留 core metrics 与可用 top3/top5/within_3/mae，输出 delta vs proto 与 delta vs uniform winner。
- [x] 3.4 添加 launcher/summary focused tests 或可导入 sanity checks，避免真实训练、真实 dataset 和输出产物进入源码。

## 4. 验证与收尾

- [x] 4.1 运行 `openspec validate add-scene31-adaptive-sampler-beamsoft-loss --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_scene31_next_round.py -q`。
- [x] 4.3 运行新增或受影响 focused tests，例如 loss/sampler tests；必要时追加 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q`。
- [x] 4.4 检查 `git status --short --untracked-files=all`，确认没有纳入 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event 或 `__pycache__`。
