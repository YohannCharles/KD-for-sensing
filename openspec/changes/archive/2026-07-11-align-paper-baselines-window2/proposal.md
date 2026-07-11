## Why

`paper/` 目录中的 AMBER 和 AMR-Net baseline 已有本地架构复现骨架，但文档、配置和部分组件仍与用户指定的 `seq_len=2`、`num_pred=1`、仅 `lidar/image/gps/radar` 模态边界不完全一致。

本 change 将这两篇论文 baseline 收敛为可审计的本地复现基线：严格遵守本轮输入/预测窗口和模态限制，同时保留无法声明官方数值复现的 claim 边界。RMBP-MM/WCL 不在本 change 的完成范围内；H5/P1 使用的 `rmbp_mm` component 由其实际 consumer 保护，但不据此声明 WCL 复现完成。

## What Changes

- 将 AMBER full 默认协议固定为 `seq_len=2`、`num_pred=1`，四模态仅 `image/radar/gps/lidar`，并补齐论文对齐的 image ResNet34 spatial-token encoder、AMBER modality indicator / L2 regularization metadata。
- 保持 AMR-Net 为 `image/lidar/gps` 论文三模态 whole-model exception，但明确它仍落在允许模态集合内，并修正文档协议为 `seq_len=2`、`num_pred=1`。
- 更新 configs、tests、OpenSpec delta 和研究协议文档，确保 AMBER/AMR 本地复现不升级为 official reproduction。

## Capabilities

### New Capabilities
- 无。

### Modified Capabilities
- `amber-full-architecture-reproduction`: 调整 AMBER full 的论文对齐 encoder、窗口、模态和 L2 regularization 要求。
- `amr-net-architecture`: 明确 AMR-Net 在用户指定窗口和允许模态集合下的本地复现边界。

## Impact

- 影响模型组件：`src/kd_sensing/models/` 中 AMBER image spatial-token encoder 与 AMBER core。
- 影响配置：AMBER full 与 AMR-Net YAML。
- 影响测试：AMBER full、AMR-Net focused tests，以及必要的 registry/config focused checks。
- 不新增旧 CLI、根目录训练脚本、外部下载依赖、真实数据产物、checkpoint 或输出日志到源码。
