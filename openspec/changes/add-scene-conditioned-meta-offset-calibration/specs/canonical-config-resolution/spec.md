## ADDED Requirements

### Requirement: Scenario meta-offset recipe 与矩阵生成
系统 MUST 为 scene-conditioned meta-offset calibration 提供可审查的 base recipe、smoke/example 配置和 override matrix generator。Base/lowmem/真实数据 recipe MUST 默认选择 `overlap_k16_s8_stage1` 作为 canonical visual/JEPA 基底。Generator MUST 生成完整 resolved config 或配置清单到 ignored runtime output boundary，源码中 MUST 不新增与 recipe 等价的大量重复实体 YAML。

#### Scenario: smoke 配置可加载
- **WHEN** 用户加载 scene meta-offset smoke/base 配置
- **THEN** 配置加载 MUST 产生包含 synthetic dataset、scene meta-offset model、`overlap_k16_s8_stage1` canonical base variant、loss、training、evaluation 和 output boundary 的完整 final config
- **AND** 该配置 MUST 不要求真实 dataset 路径、checkpoint 或外部 detector 结果存在

#### Scenario: 默认 base variant 可审计
- **WHEN** 用户没有显式覆盖 `model.primary.canonical_base.variant_id`
- **THEN** resolved config MUST 设置 `variant_id: overlap_k16_s8_stage1`
- **AND** resolved config MUST 设置 visual encoder 为 `overlap_patch`、`kernel_size: 16`、`stride: 8`、`max_tokens: 729`
- **AND** patch16 mean、GPS-biased、ResNet+GPS 或 GPS-only MUST 只能通过显式 control/ablation override 选择

#### Scenario: 矩阵生成记录 provenance
- **WHEN** 用户运行 matrix generator
- **THEN** 每个生成条目 MUST 记录 base recipe、canonical base variant、overrides、experiment family、seed、split protocol、enabled offset heads、meta method 和输出目录
- **AND** generator 默认输出 MUST 位于 ignored `outputs/`、`logs/` 或用户显式指定的本地产物目录

#### Scenario: 不生成退役配置
- **WHEN** 用户请求生成 KD、HiST/Hist、Top8 standalone、GPS residual、camera residual、geometry-residual label、Raymobtime 或 Multimodal-NF 路线配置
- **THEN** generator/config loader MUST 拒绝该请求
- **AND** 错误信息 MUST 指向当前 scene meta-offset 或现有 current workflow 入口
