## ADDED Requirements

### Requirement: Fusion G2D 五模态配置入口
项目 MUST 提供五模态 G2D fusion 配置入口。配置 MUST 使用固定模态顺序 `image`、`radar`、`gps`、`lidar`、`mmwave`，MUST 设置 `experiment.task: fusion`，MUST 设置 `distillation.type: g2d`，并 MUST 保持 `model.num_pred: 3`。

#### Scenario: 加载 G2D-lite 配置
- **WHEN** 用户加载 `configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml`
- **THEN** 配置 MUST 启用五个 fusion student 输入模态
- **AND** 配置 MUST 设置 `distillation.type: g2d`
- **AND** 配置 MUST 设置 G2D mode 为 `lite`

#### Scenario: 加载 G2D-global 配置
- **WHEN** 用户加载 `configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml`
- **THEN** 配置 MUST 启用五个 fusion student 输入模态
- **AND** 配置 MUST 设置 `distillation.type: g2d`
- **AND** 配置 MUST 设置 G2D mode 为 `global`
- **AND** 配置 MUST 启用 SMP

#### Scenario: 加载 G2D-horizon 配置
- **WHEN** 用户加载 `configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml`
- **THEN** 配置 MUST 启用五个 fusion student 输入模态
- **AND** 配置 MUST 设置 `distillation.type: g2d`
- **AND** 配置 MUST 设置 G2D mode 为 `horizon_diagnostic`

### Requirement: Fusion student exposes modality features for G2D
fusion student MUST 在不改变主 logits 契约的前提下，为 G2D feature KD 暴露 per-modality branch features。该输出 MUST 能通过 `adapt_model_output()` 的 diagnostics 传递给 G2D distiller。

#### Scenario: legacy fusion_student 输出 modality features
- **WHEN** `fusion_student` 前向完成且启用了多个模态
- **THEN** 输出 diagnostics MUST 包含按模态命名的 branch feature
- **AND** 每个 branch feature MUST 保持 batch 和 sequence 维度与 logits 对齐
- **AND** 主 logits MUST 继续能被解析为 `[B,T,C]`

#### Scenario: CRAF 或 MARF 输出 token features
- **WHEN** G2D student 使用 CRAF、MARF 或 token transformer 风格 fusion 模型
- **THEN** G2D feature extractor MUST 能从 `token_features` 和 `modalities` diagnostics 中按模态拆分 feature
- **AND** 拆分后的 feature MUST 能参与 feature KD

