## Why

arXiv:2603.29796 提出的 JEPA-MSAC 将多模态感知辅助通信建模为“自监督预测 latent state + 冻结 backbone + 轻量 PHY 任务头”的两阶段 workflow，和本仓库现有 Image+GPS JEPA、BGAM 与多模态 baseline 主线高度相关。当前仓库还缺少对该论文的可审计复现入口，无法用统一的 DeepSense6G 场景、指标、产物边界和文档账本验证其 localization、beam prediction 与 RSSI prediction 结论。

## What Changes

- 新增 JEPA-MSAC paper/workflow reproduction，而不是普通 `modular_sequence` baseline：Stage 1 做多模态 temporal block-masked JEPA 预训练，Stage 2 冻结 backbone 并训练 localization、beam 和 RSSI task heads。
- 增加论文对齐的数据协议：DeepSense6G Scenario 32、13 帧滑窗、`T_hist=8`、`T_pred=5`、64-beam codebook、70/30 随机 train/test split、Image/Radar/LiDAR/GPS/RF 历史输入与未来 target 准备。
- 增加 JEPA-MSAC 模型组件：多模态 tokenizer、factorized time/modality/intra-frame position embedding、temporal block mask sampler、EMA target encoder、full-sequence predictor、predictive latent pooling 和 localization-guided cascading heads。
- 增加 workflow CLI/config：提供 smoke 配置、paper-aligned 配置、Stage 1/Stage 2 运行入口、resume/checkpoint metadata、dry-run/report-only 路径和本地产物输出边界。
- 增加论文指标与报告：RRankMe、RLDA、ADE、FDE、Top-1、Top-3、L1-RSRP diff、RSSI RMSE/MAE、horizon-wise tables、ablation manifest 和 claim status 账本。
- 不提交真实 DeepSense6G 数据、训练 cache、checkpoint、日志或结果图；所有运行产物继续写入 ignored 的 `outputs/`、`logs/` 或显式本地路径。
- 不引入 breaking change；现有 JEPA Image+GPS、BGAM、CSI、viewer 和通用训练入口语义保持不变。

## Capabilities

### New Capabilities
- `jepa-msac-reproduction`: 定义 JEPA-MSAC 论文复现的两阶段训练、数据协议、模型组件、指标、CLI/config、产物和文档/claim 边界。

### Modified Capabilities
- 无。

## Impact

- 代码：`src/kd_sensing/models/`、`src/kd_sensing/engine/`、`src/kd_sensing/data/`、`src/kd_sensing/baselines/jepa_msac/`、`src/kd_sensing/cli/`、`src/kd_sensing/losses/`、`src/kd_sensing/evaluation/`。
- 配置：新增 `configs/pretraining/`、`configs/baselines/` 或 `configs/fusion/experiments/` 下的 JEPA-MSAC smoke 与 paper-aligned 配置，路径命名需避开退役 KD/Hist/Top8/residual 入口。
- 文档：同步 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`、`docs/experiment_matrix.md`、README 简短索引和 `docs/project_surface_inventory.md` lifecycle/allowlist。
- 测试：新增 registry build、synthetic forward、mask sampler、loss、dataset sample assembly、CLI help、workflow smoke、metric/report 和文档同步 guard。
- 依赖：优先复用当前 PyTorch/torchvision/numpy/scikit-learn 栈；若 EfficientNet 或点云/radar 预处理需要新依赖，必须先评估是否可用现有 torchvision/backbone 和本仓库 transform helper 实现。
