# amr-net-architecture Specification

## Purpose
TBD - created by archiving change add-amr-net-architecture. Update Purpose after archive.
## Requirements
### Requirement: AMR-Net 模型注册与边界
系统 MUST 提供 current `amr_net` 模型注册名，作为 OpenSpec 记录的 whole-model exception。该模型 MUST 复用现有 batch/runtime、配置加载、评估和 `ModelOutput` 适配路径，不得恢复旧 `amr_net_gps_image` runner、旧 console script、根目录训练脚本或专用训练循环。

#### Scenario: 构建 AMR-Net
- **WHEN** 构建流程导入默认模型组件并请求 `model.primary.type: amr_net`
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建 AMR-Net 模型
- **AND** 模型 MUST 声明 `supports_modality_kwargs=True`
- **AND** 模型 MUST 能由共享 runtime 传入 `image_batch`、`lidar_batch` 和 `gps_batch`

#### Scenario: 旧 AMR runner 不恢复
- **WHEN** 用户请求旧 `amr_net_gps_image` 配置 token、runner 名称或 `kd-sensing-run-amr-net-gps-image`
- **THEN** 系统 MUST 继续拒绝该入口或名称
- **AND** 错误或文档 MUST 指向新的 `amr_net` current baseline，而不是兼容转发到旧 runner

### Requirement: AMR-Net 输入与 encoder 契约
AMR-Net MUST 支持 image、LiDAR 和 GPS 三模态 snapshot 输入。模型 MUST 接收仓库 batch 形状：`image_batch [B,T,C,H,W]`、`lidar_batch [B,T,N,F]` 或配置声明的等价 LiDAR snapshot tensor、`gps_batch [B,T,F]`。初版 AMR-Net MUST 要求 `T=1`，并在 shape 不匹配时抛出包含模态名和实际 shape 的清晰错误。

#### Scenario: paper-aligned snapshot 输入
- **WHEN** AMR-Net 收到 `image_batch [B,1,1,224,224]`、`lidar_batch [B,1,216,2]` 和 `gps_batch [B,1,2]`
- **THEN** 模型 MUST 为三个模态生成 latent feature
- **AND** image feature 维度 MUST 可配置为 128
- **AND** LiDAR 和 GPS feature 维度 MUST 可配置为 512

#### Scenario: 时间维被拒绝
- **WHEN** AMR-Net 收到任一启用模态的时间维 `T != 1`
- **THEN** 模型 MUST 拒绝 forward
- **AND** 错误信息 MUST 指出 AMR-Net 初版需要 snapshot `T=1`

### Requirement: 概率嵌入与重参数化
AMR-Net MUST 为每个启用模态从 deterministic feature 生成 `mu` 和 `logvar`，并通过 reparameterization 得到 latent sample `z`。模型 MUST 对 `logvar` 做数值稳定处理，训练和评估输出 MUST 保留每个模态的 `mu`、`logvar` 和用于分类的 latent diagnostics。

#### Scenario: 输出概率参数
- **WHEN** AMR-Net forward 成功
- **THEN** 输出 diagnostics MUST 包含每个模态的 `mu` 和 `logvar`
- **AND** `mu`、`logvar` 和 `z` 的 batch 维 MUST 与输入 batch 一致

#### Scenario: eval 采样可稳定
- **WHEN** 模型处于 eval 模式且配置使用 deterministic inference
- **THEN** 分类 latent MUST 使用 `mu` 或等价确定性路径
- **AND** repeated forward MUST 不因随机采样改变输出，除非配置显式启用 stochastic inference

### Requirement: Per-modality 分类与 ModelOutput 兼容
AMR-Net MUST 为每个模态生成 beam classification logits，并通过 CUAF 生成主 `logits`。模型输出 MUST 是 `adapt_model_output` 可消费的 dict，其中 `logits` 为 `[B,num_classes]` 或现有 beam loss 可接受的等价形状，其它 AMR 字段进入 diagnostics。

#### Scenario: forward 输出可适配
- **WHEN** `adapt_model_output` 消费 AMR-Net forward 输出
- **THEN** 返回的 `ModelOutput.logits` MUST 是 fused beam logits
- **AND** `ModelOutput.diagnostics` MUST 保留 per-modality logits、概率嵌入参数和 CUAF diagnostics

#### Scenario: 类别数匹配配置
- **WHEN** 配置设置 `num_classes: 64`
- **THEN** 每个模态 logits 和 fused logits 的最后一维 MUST 为 64

### Requirement: FEP 和 PRE 训练损失
系统 MUST 提供 AMR-Net 训练损失 helper，基于 `ModelOutput` diagnostics 计算 per-modality cross entropy、Gaussian KL、FEP 和可选 PRE supervised contrastive loss。该 helper MUST 不要求模型 forward 接收标签，并 MUST 在无正样本 batch 上避免 NaN。

#### Scenario: 计算 AMR loss
- **WHEN** loss helper 收到 AMR-Net `ModelOutput` 和 hard beam labels
- **THEN** 系统 MUST 计算每个模态的 CE 与 KL
- **AND** 系统 MUST 按配置 `alpha`、`beta`、`temperature` 和 `pre_samples` 组合总 loss
- **AND** diagnostics MUST 包含 per-modality loss、KL、PRE 和 skipped anchor 统计

#### Scenario: PRE 无正样本降级
- **WHEN** batch 中没有可用于 supervised contrastive 的同类正样本
- **THEN** PRE loss MUST 返回有限的 0 值或跳过项
- **AND** diagnostics MUST 记录 PRE skipped count

### Requirement: CUAF 推理融合
AMR-Net MUST 实现 CUAF 融合，根据每个模态的 softmax probability 计算 entropy score、cross-modal KL consistency score 和 top-k margin score，并归一化得到模态权重。融合 MUST 输出 finite fused probability/logits 和可诊断权重。

#### Scenario: CUAF 生成权重
- **WHEN** 三个模态 logits 均可用
- **THEN** CUAF MUST 输出每个模态的 normalized weight
- **AND** 每个样本的权重和 MUST 近似为 1
- **AND** diagnostics MUST 包含 entropy、KL consistency、margin 和 final weights

#### Scenario: 缺失模态不参与融合
- **WHEN** 配置或 batch mask 标记某个模态不可用
- **THEN** CUAF MUST 将该模态权重降为 0 或按配置 fallback 处理
- **AND** fused logits MUST 仍为 finite tensor

### Requirement: AMR-Net metadata 和架构摘要
AMR-Net MUST 提供 `training_strategy_metadata()`，至少记录模型注册名、架构类别、启用模态、encoder dims、latent dim、KL/PRE/CUAF 配置、是否消费 reliability metadata、checkpoint/freeze 策略和 `paper_approximation`。模型架构摘要 MUST 能统计 AMR-Net total/trainable params，并保留 metadata。

#### Scenario: metadata 可审计
- **WHEN** 构建 AMR-Net 模型
- **THEN** `training_strategy_metadata()` MUST 返回 `architecture_category: whole_model_exception`
- **AND** metadata MUST 记录 `modalities`、`latent_dim`、`cuaf_enabled` 和 `consumes_reliability_metadata`

#### Scenario: 架构摘要覆盖 AMR-Net
- **WHEN** 模型架构摘要处理 AMR-Net 实例
- **THEN** summary MUST 包含 registry id、total params、trainable params 和 AMR metadata
- **AND** summary MUST 不把 AMR-Net 误归类为 `modular_sequence`

### Requirement: AMR-Net 配置和文档可见性
系统 MUST 提供最小 current 配置或 overlay，使用户能通过现有训练入口选择 AMR-Net。文档 MUST 说明该能力是 paper-inspired architecture baseline，不是旧 source-audit runner 或 official reproduction。

#### Scenario: 配置加载 AMR-Net
- **WHEN** 用户加载新增 AMR-Net 配置
- **THEN** 配置 MUST 使用当前 `model.primary.type: amr_net`
- **AND** 配置 MUST 不包含退役 token `amr_net_gps_image` 或旧 runner 名称

#### Scenario: 文档区分旧路线
- **WHEN** 用户阅读模型目录或实验矩阵中的 AMR-Net 条目
- **THEN** 文档 MUST 标记新 AMR-Net 为 current architecture baseline
- **AND** 文档 MUST 保留旧 `AMR-Net_gps_image` source-audit runner 已退役的边界说明

### Requirement: AMR-Net focused tests
实现 MUST 增加 focused tests，覆盖 registry build、synthetic forward、`adapt_model_output`、AMR loss、CUAF finite 输出、metadata、架构摘要、配置加载和旧入口隔离。测试 MUST 使用 synthetic tensors，不得读取真实 `dataset/` 或写入训练产物。

#### Scenario: synthetic smoke 覆盖
- **WHEN** 运行 AMR-Net focused tests
- **THEN** tests MUST 使用 synthetic image、LiDAR、GPS 和 labels 完成 forward/loss smoke
- **AND** tests MUST 不读取真实数据集、不写入 `outputs/`、`logs/`、cache 或 checkpoint

#### Scenario: 普通 baseline 不被污染
- **WHEN** 运行现有 `modular_sequence` 或 image/GPS baseline focused tests
- **THEN** 新 AMR-Net loss、diagnostics 和 metadata 字段 MUST 不成为普通 baseline 的必需输入
- **AND** 现有 baseline MUST 继续通过 `adapt_model_output` 和默认 beam loss

