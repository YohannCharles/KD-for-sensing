## MODIFIED Requirements

### Requirement: AMR-Net 输入与 encoder 契约
AMR-Net MUST 支持 image、LiDAR 和 GPS 三模态序列输入，并保持当前 scene31 本地实验配置边界。模型 MUST 接收仓库 batch 形状：`image_batch [B,T,C,H,W]`、`lidar_batch [B,T,216,2]` 或配置声明的等价 LiDAR tensor、`gps_batch [B,T,2]`，其中 `T >= 1`。论文对齐本地配置 MUST 默认使用 `T=2`、image channel 为 1 的 `224x224` 输入、LiDAR `216x2` 和 GPS `2` 维输入；shape 不匹配时 MUST 抛出包含模态名和实际 shape 的清晰错误。AMR-Net MAY 在进入 snapshot encoder 前将时间维聚合为单个 snapshot 表征。

#### Scenario: paper-aligned 默认序列输入
- **WHEN** AMR-Net 收到 `image_batch [B,2,1,224,224]`、`lidar_batch [B,2,216,2]` 和 `gps_batch [B,2,2]`
- **THEN** 模型 MUST 为三个模态生成 latent feature
- **AND** image feature 维度 MUST 可配置为 128
- **AND** LiDAR 和 GPS feature 维度 MUST 可配置为 512

#### Scenario: 非默认时间长度可运行
- **WHEN** AMR-Net 收到任一 `T >= 1` 的启用模态输入
- **THEN** 模型 MUST 生成 finite fused logits
- **AND** logits 的预测窗口维 MUST 继续由 `num_pred` 控制
- **AND** CUAF availability mask MUST 能处理输入时间维 mask 并映射到预测窗口

### Requirement: FEP 和 PRE 训练损失
系统 MUST 提供 AMR-Net 训练损失 helper，基于 `ModelOutput` diagnostics 计算每模态 FEP 和 PRE。FEP MUST 包含 per-modality cross entropy 和 Gaussian KL；PRE MUST 基于每个模态 posterior 的 `K` 次 Monte Carlo latent sampling 计算 supervised contrastive loss。该 helper MUST 不要求模型 forward 接收标签，并 MUST 在无正样本 batch 上避免 NaN。论文对齐配置 MUST 支持 AMR-only composite objective，避免额外 fused beam/focal 主损失污染 AMR-Net 论文训练目标。

#### Scenario: 计算 AMR loss
- **WHEN** loss helper 收到 AMR-Net `ModelOutput` 和 hard beam labels
- **THEN** 系统 MUST 计算每个模态的 CE 与 KL
- **AND** 系统 MUST 根据 `pre_samples` 从 `mu/logvar` 重新采样 K 个 latent 用于 PRE
- **AND** 系统 MUST 按配置 `alpha`、`beta`、`temperature` 和 `pre_samples` 组合总 loss
- **AND** diagnostics MUST 包含 per-modality loss、KL、PRE 和 skipped anchor 统计

#### Scenario: PRE 无正样本降级
- **WHEN** batch 中没有可用于 supervised contrastive 的同类正样本
- **THEN** PRE loss MUST 返回有限的 0 值或跳过项
- **AND** diagnostics MUST 记录 PRE skipped count

#### Scenario: AMR-only objective 不叠加 fused focal
- **WHEN** 配置声明 `loss.amr.paper_objective_only: true`
- **THEN** prediction loss MUST 使用 AMR composite loss 作为 beam objective 的总损失
- **AND** 系统 MUST NOT 额外叠加 fused logits 的 focal/cross-entropy 主损失

### Requirement: CUAF 推理融合
AMR-Net MUST 实现论文版 CUAF 融合，根据每个模态的 softmax probability 计算 entropy score、average pairwise KL divergence consistency score 和 top-T margin score。三个 score MUST 先分别在模态维度做 softmax 归一化，再聚合并归一化为最终模态权重。融合 MUST 输出 finite fused probability/logits 和可诊断权重。

#### Scenario: CUAF 生成论文版权重
- **WHEN** 三个模态 logits 均可用
- **THEN** CUAF MUST 输出每个模态的 normalized weight
- **AND** 每个样本的权重和 MUST 近似为 1
- **AND** diagnostics MUST 包含 entropy score、pairwise KL score、top-T margin score 和 final weights

#### Scenario: 缺失模态不参与融合
- **WHEN** 配置或 batch mask 标记某个模态不可用
- **THEN** CUAF MUST 将该模态权重降为 0 或按配置 fallback 处理
- **AND** fused logits MUST 仍为 finite tensor

### Requirement: AMR-Net 配置和文档可见性
系统 MUST 提供最小 current 配置或 overlay，使用户能通过现有训练入口选择 AMR-Net。配置 MUST 使用当前 `model.primary.type: amr_net`，保持 scene31 本地实验场景，并标记该能力是 paper-aligned local architecture baseline，不是旧 source-audit runner 或 official reproduction。

#### Scenario: 配置加载 AMR-Net
- **WHEN** 用户加载新增 AMR-Net 配置
- **THEN** 配置 MUST 使用当前 `model.primary.type: amr_net`
- **AND** 配置 MUST 不包含退役 token `amr_net_gps_image` 或旧 runner 名称
- **AND** 配置 MUST 使用 scene31 本地数据边界

#### Scenario: 文档区分旧路线
- **WHEN** 用户阅读模型目录或实验矩阵中的 AMR-Net 条目
- **THEN** 文档 MUST 标记新 AMR-Net 为 current architecture baseline
- **AND** 文档 MUST 保留旧 `AMR-Net_gps_image` source-audit runner 已退役的边界说明
