## Why

用户提供的 AMR-Net 论文拆解需要落成当前仓库可实施的模型架构方案。仓库已有 `AMR-Net_gps_image` 历史 runner 退役边界，因此需要用新的 OpenSpec 契约明确如何实现 AMR-Net，而不是恢复旧入口或复制训练流程。

## What Changes

- 新增 AMR-Net 模型能力，覆盖 image、LiDAR、GPS 三模态 encoder、概率嵌入、per-modality beam classifier、FEP/PRE 训练损失和 CUAF 推理融合。
- 将 AMR-Net 归类为 `whole-model exception`：其多分支概率采样、训练期 per-modality loss 和推理期 logit/probability fusion 不能只靠现有 `modular_sequence` core/head 简洁表达。
- 复用现有 `engine.batch`、`engine.runtime`、`ModelOutput` 适配、dataset/config/metric 边界；不新增专用训练循环、根目录脚本或旧 `amr_net_gps_image` 兼容入口。
- 新增最小配置、metadata、架构摘要和 focused tests，保证 registry build、synthetic forward、loss、CUAF diagnostics 和普通 baseline 隔离可审计。

## Capabilities

### New Capabilities

- `amr-net-architecture`: 定义 AMR-Net 概率嵌入、多模态不确定性感知融合、训练损失、配置入口、metadata 和测试契约。

### Modified Capabilities

- 无。

## Impact

- 代码：`src/kd_sensing/models/`、`src/kd_sensing/losses/` 或 objective/loss helper、`src/kd_sensing/registries.py` 的默认模型注册导入、模型架构摘要 helper。
- 配置：新增当前 AMR-Net 配置或 overlay，使用 `model.primary.type` 指向新的 current 注册名，避免旧 `amr_net_gps_image` token。
- 测试：新增 AMR-Net focused tests，并按需扩展架构边界、配置加载和模型摘要覆盖。
- 文档：更新模型目录、实验协议或实验矩阵中与 current baseline 可见性有关的最小条目；不提交训练输出、checkpoint、cache 或真实数据。
