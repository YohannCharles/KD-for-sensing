# dataset-runtime-contracts Specification

## Purpose
定义跨数据集 runtime contract：用轻量 dataset descriptor 描述数据集家族与本地产物边界，用 sample index、modality adapter 和 target provider 组合 flat sample，并在训练/评估 runtime metadata 中同时记录 dataset family、storage kind、split、enabled modalities、input profiles 和当前 objective 实际消费的 target schema。
## Requirements
### Requirement: Dataset descriptor 注册契约
系统 MUST 提供 dataset descriptor 或等价机制，用于描述当前保留数据集家族的 dataset type、默认目录、存储类型、split 语义、支持模态、支持 target schema 和本地产物边界。descriptor 查询 MUST 保持轻量，不得导入 pandas、h5py、torch dataset、模型或训练模块。已退役的 `multimodal_nf` 和 `raymobtime_s008` descriptor MUST 不再注册。

#### Scenario: 查询退役 Multimodal-NF descriptor
- **WHEN** 代码查询 `multimodal_nf` dataset descriptor
- **THEN** 系统 MUST 报告该 dataset type 不存在或已退役
- **AND** 查询过程 MUST 不读取真实 HDF5 数据、不打开 codebook 文件、不导入训练循环

#### Scenario: 查询退役 Raymobtime descriptor
- **WHEN** 代码查询 `raymobtime_s008` dataset descriptor
- **THEN** 系统 MUST 报告该 dataset type 不存在或已退役
- **AND** 查询过程 MUST 不读取 `dataset/Raymobtime/s008`、不导入 Raymobtime dataset、不导入模型或训练循环

#### Scenario: 查询保留数据集 descriptor
- **WHEN** 代码查询 DeepSense6G 或 MMW 的 descriptor
- **THEN** 系统 MUST 返回对应存储类型和默认路径
- **AND** 这些保留 dataset type、配置和公开输出字段 MUST 保持兼容

### Requirement: Sample index 统一契约
系统 MUST 提供 sample index 契约，把当前保留数据集使用的 CSV、NPZ cache 或 manifest 转换为轻量样本 rows。sample row MUST 至少能表达 `sample_id`、split、数据集家族、scene/condition、trajectory、frame、资源引用、target 引用和 metadata。sample index 初始化 MUST 不物化 image、LiDAR、CSI/channel 等大数组。系统 MUST 不再要求支持 Multimodal-NF HDF5 frame index。

#### Scenario: CSV sequence index 兼容
- **WHEN** DeepSense6G 仍使用现有 CSV sequence 样本构建
- **THEN** 系统 MUST 允许通过适配层暴露 sample index row
- **AND** 该适配 MUST 不改变 `input_beam`、`target_beam`、模态样本字段或 metadata 的既有语义

#### Scenario: Multimodal-NF HDF5 frame index 不再支持
- **WHEN** 用户查找或调用 Multimodal-NF HDF5 frame index builder
- **THEN** 系统 MUST 报告该 builder 不存在或 dataset type 已退役
- **AND** 系统 MUST 不读取 Multimodal-NF HDF5 文件

### Requirement: Modality adapter profile 契约
系统 MUST 支持当前保留数据集按模态和 profile 注册 modality adapter。adapter MUST 声明输入字段、所需资源引用、输出 sample key、shape/dtype 语义、cache/normalization 能力和错误信息。dataset 取样 MUST 只调用启用模态对应的 adapter。系统 MUST 不再提供 Multimodal-NF 专属 adapter/profile。

#### Scenario: 只加载启用模态
- **WHEN** 用户构建当前保留的多模态 dataset 且只启用部分模态
- **THEN** dataset MUST 不读取未启用模态的数据源
- **AND** 返回样本 MUST 只包含启用模态字段、目标字段和 metadata

#### Scenario: Multimodal-NF adapter 删除
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf`
- **THEN** 系统 MUST 不再调用 Multimodal-NF image、LiDAR、GPS 或 CSI adapter
- **AND** dataset 构建 MUST 失败

### Requirement: Target provider 契约
系统 MUST 支持当前保留 objectives 或 target schema 的 target provider。target provider MUST 负责生成主 label、辅助 target、valid mask 和 target metadata，并 MUST 允许 train split 产出的统计或 metadata 复用于 val/test split。系统 MUST 不再提供 Multimodal-NF 近场三维 codebook target provider。

#### Scenario: Artifact 复用
- **WHEN** train dataset 已解析当前保留 objective 所需的 normalizer artifact 或 metadata
- **THEN** data factory MUST 能将需要复用的 artifact 传给 val/test dataset
- **AND** 复用过程 MUST 不要求重新扫描全量大数据文件

#### Scenario: 近场 beam provider 删除
- **WHEN** batch 或配置请求 Multimodal-NF Top-5 三维 beam target
- **THEN** 系统 MUST 不再提供该 target provider
- **AND** 错误信息 MUST 指出 Multimodal-NF 已退役

### Requirement: RuntimeDataset flat sample 契约
系统 MUST 提供薄 runtime dataset 或等价组合方式，通过 sample index、modality adapters 和 target provider 构建 flat dict sample。flat sample keys MUST 与中心化模态契约和 prediction objective 契约一致。

#### Scenario: Flat sample 输出
- **WHEN** 用户从任一支持数据集取样
- **THEN** 返回值 MUST 是 DataLoader 可默认 collate 的 flat dict
- **AND** 输入模态字段 MUST 使用中心化模态契约定义的 sample key
- **AND** target 字段 MUST 使用当前 objective 或 target schema 定义的字段名

#### Scenario: Runtime metadata
- **WHEN** 训练或评估构建 dataloaders
- **THEN** run metadata MUST 记录 dataset type、descriptor family、storage kind、split metadata、enabled modalities、input profiles 和 target schema

### Requirement: Dataset runtime capability purpose 明确
`dataset-runtime-contracts` spec MUST 使用真实目的说明描述 dataset descriptor、sample index、modality adapter、target provider 和 runtime metadata 契约。该 spec MUST 不长期保留 archived TBD Purpose 文案。

#### Scenario: dataset runtime purpose 不再是 TBD
- **WHEN** 开发者阅读 `openspec/specs/dataset-runtime-contracts/spec.md`
- **THEN** Purpose MUST 描述 dataset runtime contract 的当前职责
- **AND** Purpose MUST NOT 包含 `TBD - created by archiving`

### Requirement: Runtime metadata 区分 dataset family 与 target schema
Dataset runtime metadata MUST 同时记录当前保留 dataset family 信息和当前 objective target schema。dataset family MUST 表达数据来源、storage kind、split 和 profiles；target schema MUST 表达当前 run 实际训练或评估的主 target 和辅助 target。系统 MUST 不再写出或要求 Multimodal-NF runtime metadata，也 MUST 不再写出或要求 Raymobtime s008 runtime metadata。

#### Scenario: 保留数据集 metadata 双层记录
- **WHEN** 训练或评估构建当前保留数据集 dataloaders
- **THEN** runtime metadata MUST 记录 dataset type、storage kind、split strategy、enabled modalities 和 input profiles
- **AND** runtime metadata MUST 记录当前 objective 对应的 target schema
- **AND** 二者 MUST 不互相覆盖

#### Scenario: 退役 Raymobtime metadata 不再写出
- **WHEN** 用户加载旧 Raymobtime s008 配置或旧 Raymobtime checkpoint metadata
- **THEN** 当前训练/评估 runtime MUST 不写出新的 `raymobtime_s008_current_snapshot` metadata
- **AND** 系统 MUST 报告 Raymobtime s008 已退役或要求用户使用当前保留 workflow

#### Scenario: Multimodal-NF metadata 不再写出
- **WHEN** 用户加载 Multimodal-NF 配置
- **THEN** 系统 MUST 不写出 Multimodal-NF near-field target schema
- **AND** 系统 MUST 报告该 dataset type 已退役

### Requirement: Path auxiliary target flat sample 契约
RuntimeDataset 或等价 dataset MUST 支持在 flat sample 中表达 path-level auxiliary targets。该契约 MUST 将 `path_params`、`path_descriptor`、`path_semantic_label` 和 `path_valid` 标记为 target/diagnostic 字段，而不是 input modality 字段。

#### Scenario: flat sample 包含 path auxiliary targets
- **WHEN** 当前 dataset family 支持 path-level propagation parameters 且配置启用 path semantics
- **THEN** flat sample MAY 包含 `path_params`、`path_descriptor`、`path_semantic_label` 和 `path_valid`
- **AND** sample metadata MUST 保留 sample_id、dataset family、split、domain、town/scenario/weather 或等价 domain fields

#### Scenario: enabled modalities 不包含 path 或 CSI
- **WHEN** dataloader 根据配置解析 enabled modalities
- **THEN** channel、CSI、path_params 和 path_descriptor MUST NOT 被加入模型输入模态列表
- **AND** runtime metadata MUST 将这些字段记录为 auxiliary target、diagnostic 或 unavailable，而不是 sensing input

### Requirement: Unlabeled target sensitive field guard
训练 runtime MUST 提供 batch 级 sensitive field guard 或等价机制，用于阻止 label_budget 为 0 或 unlabeled target batch 的训练 loss 访问真实 target supervision 字段。

#### Scenario: label_budget 为 0 时访问敏感字段失败
- **WHEN** target adaptation 的 `label_budget=0` 且 loss 代码尝试读取 beam、beam_power、CSI/channel、path_params、path_descriptor、path_semantic_label 或 radio_semantic_label 作为训练监督
- **THEN** 系统 MUST raise error
- **AND** error message MUST 包含 split、field name、label budget 和可执行修复提示

#### Scenario: adapt log 记录防泄漏标志
- **WHEN** target adaptation run 完成或失败
- **THEN** `adapt_log.json` 或等价 metadata MUST 记录 `used_target_beam_for_training`、`used_target_beam_power_for_training`、`used_target_csi_for_training`、`used_target_path_params_for_training`、`used_target_path_label_for_training` 和 `used_target_radio_label_for_training`
- **AND** `label_budget=0` 的成功 run 中这些字段 MUST 为 false

#### Scenario: labeled target subset 可显式使用 path supervision
- **WHEN** `label_budget>0` 且 batch 来自 labeled target subset
- **THEN** training runtime MAY 允许 supervised beam loss
- **AND** 只有 `allow_labeled_target_path_supervision=true` 时，runtime MAY 允许 path_semantic_label 或 path_descriptor supervision
- **AND** unlabeled target subset MUST 继续触发 sensitive field guard

### Requirement: Target sensitive auxiliary supervision policy
训练 runtime MUST 对 target split 中的 sensitive supervision 字段实施显式 policy。`beam`、`beam_power`、CSI/channel、`path_params`、`path_descriptor`、`path_semantic_label` 和 `radio_semantic_label` MUST 按 split、label budget、labeled subset 状态和显式 opt-in 配置决定是否可被训练 loss 使用。

#### Scenario: unlabeled target 禁止 sensitive supervision
- **WHEN** target adaptation batch 来自 unlabeled target subset 或 `label_budget=0`
- **THEN** 训练 loss 访问真实 target `beam`、`beam_power`、CSI/channel、path 或 radio semantic 字段作为监督 MUST 失败
- **AND** error message MUST 包含 split、field name、label budget、labeled subset 状态和可执行修复提示

#### Scenario: labeled target auxiliary supervision 需要 opt-in
- **WHEN** `label_budget>0` 且 batch 来自 labeled target subset
- **THEN** 系统 MUST 允许 supervised beam loss 使用 labeled beam target
- **AND** path auxiliary supervision MUST 只有在显式启用 `allow_labeled_target_path_supervision` 或等价配置时才能使用
- **AND** radio auxiliary supervision MUST 只有在显式启用 `allow_labeled_target_radio_supervision` 或等价配置时才能使用
- **AND** 未启用 opt-in 时访问对应字段作为训练监督 MUST 失败

#### Scenario: sensitive usage metadata 可追踪
- **WHEN** target adaptation run 完成或失败
- **THEN** run metadata MUST 记录 sensitive field policy、label budget、labeled subset 状态和每类 target sensitive 字段是否被训练使用
- **AND** metadata MUST 至少覆盖 `used_target_beam_for_training`、`used_target_beam_power_for_training`、`used_target_csi_for_training`、`used_target_path_params_for_training`、`used_target_path_label_for_training` 和 `used_target_radio_label_for_training`
- **AND** 这些字段 MUST 可被下游 summary 和 quick conclusion 消费

### Requirement: Target-shot split runtime metadata
Dataset runtime metadata MUST record target-shot split state when a run or diagnostic consumes a target-shot split artifact. Metadata MUST include source domains, target domains, target_label_fraction, target_labeled sample count, target_unlabeled sample count, target_test sample count, split artifact path, seed and strict eligibility summary.

#### Scenario: runtime 记录 target-shot split
- **WHEN** 训练、适配、评估或诊断构建 dataloader 并传入 target-shot split artifact
- **THEN** runtime metadata MUST 记录 source/target domain、target_label_fraction、各 split 样本数和 artifact 路径
- **AND** metadata MUST 记录 split strict eligibility 或 leakage diagnostics 摘要

### Requirement: Geometry-residual target schema metadata
Dataset runtime metadata MUST distinguish absolute beam target schema from geometry-residual target schema. When geometry-residual labels are enabled, metadata MUST record num_beams, beam_geo source, residual convention, max_residual, overflow strategy and num_geo_sectors.

#### Scenario: runtime 记录 geometry-residual schema
- **WHEN** dataset 使用 `label_space.type: geometry_residual`
- **THEN** runtime metadata MUST 记录当前 target schema 为 geometry-residual
- **AND** metadata MUST 包含 residual convention、max_residual 和 geometry availability summary

### Requirement: Labeled 与 unlabeled target subset guard
训练 runtime MUST 区分 `target_labeled` 与 `target_unlabeled` subset。`target_unlabeled` batch 的 beam、residual、beam_power、CSI/channel、path 和 radio supervision 字段 MUST 受到 sensitive field guard 保护；`target_labeled` batch MAY 使用 beam/residual supervision，但仍 MUST 遵守 path/radio opt-in policy。

#### Scenario: unlabeled residual supervision 被拒绝
- **WHEN** adaptation loss 在 `target_unlabeled` batch 上读取 `beam_residual` 或 `residual_class` 作为监督
- **THEN** runtime guard MUST raise error
- **AND** error message MUST 包含 subset、field name 和 split artifact path

#### Scenario: labeled target residual supervision 被记录
- **WHEN** adaptation loss 在 `target_labeled` batch 上读取 residual supervision
- **THEN** run metadata MUST 记录 `used_target_residual_for_training=true`
- **AND** metadata MUST 表明监督来源仅限 target_labeled subset

### Requirement: Runtime metadata 记录 difficulty profile
Dataset/dataloader runtime metadata MUST 记录当前 run 实际启用的 difficulty profiles。metadata MUST 包含 profile id、stage、split、operator types、affected modalities、resolved severities、seed、digest、fallback 和 warnings summary。未启用 difficulty 时，metadata MUST 明确记录 clean/default 状态或省略为兼容旧行为。

#### Scenario: train dataloader 记录 difficulty profile
- **WHEN** 训练配置为 train split 启用 GPS mild async profile
- **THEN** dataloader 或 run metadata MUST 记录该 profile 的 resolved stage/split、operator、seed 和 digest
- **AND** validation/test split 若未启用 profile，MUST 不被标记为同一扰动条件

#### Scenario: 未启用 difficulty 保持 clean
- **WHEN** 配置没有声明 difficulty profile
- **THEN** dataset/dataloader 构建 MUST 保持现有 clean 输入行为
- **AND** runtime metadata MUST 不要求新增非空 difficulty 字段才能被下游消费

### Requirement: Difficulty transform 不改变 target contract
Runtime dataset、dataloader 或 batch transform 应用 difficulty profile 时，MUST 保持 target provider 输出的主 label、辅助 target、valid mask、sample id、split 和 dataset family metadata 不变。GPS delay、stride、dropout 或 image degradation MUST 只影响输入模态字段及其输入相关 mask/metadata。

#### Scenario: GPS delay 不移动 target
- **WHEN** batch 应用 GPS delay、low-rate 或 async difficulty
- **THEN** `target_beam`、`target_beam_distribution`、`beam_power`、auxiliary target 和 sample id MUST 与 clean batch 一致
- **AND** runtime metadata MUST 记录该 difficulty 作用于 GPS 输入而非 target schema

#### Scenario: image degradation 不改变 input profiles
- **WHEN** batch 应用 image fog/rain、night、occlusion 或 motion blur
- **THEN** resolved image input profile MUST 仍是原配置对应 profile
- **AND** metadata MUST 将 degradation 记录为 difficulty condition，而不是新的 dataset profile 或新模态

### Requirement: Difficulty 作用阶段边界
系统 MUST 支持按 train、validation、test、evaluation 和 benchmark stage/split 选择 difficulty profile。stage/split 选择 MUST 在 dataloader metadata 和 run metadata 中可审计，MUST 防止训练 profile 隐式泄漏到 evaluation-only benchmark，或 evaluation sweep 隐式改变训练数据。

#### Scenario: evaluation profile 不影响训练 dataloader
- **WHEN** 配置只声明 evaluation difficulty sweep
- **THEN** train dataloader MUST 使用 clean 输入
- **AND** evaluation 或 benchmark pass MUST 按 sweep profile 应用 difficulty

#### Scenario: train profile 不影响 benchmark manifest
- **WHEN** 训练 run 使用 mild async profile，但 benchmark manifest 未引用该 profile
- **THEN** benchmark MUST 不自动继承训练 difficulty profile
- **AND** benchmark metadata MUST 只记录 manifest 显式声明的 difficulty conditions

### Requirement: DeepSense6G dataset contract helper 拆分
DeepSense6G runtime dataset SHALL 将配置 normalization、column validation、target source normalization、GPS feature contract 和 cache path resolution 等低风险契约逻辑拆分到轻量 helper 模块。`DeepSense6GDataset` MUST 继续保持现有 flat sample、target schema、metadata 和资源读取语义兼容。

#### Scenario: helper 不读取真实资源
- **WHEN** 测试或代码调用 DeepSense6G contract helper 解析 GPS feature mode、beam target source、required columns 或 cache path
- **THEN** helper MUST 不读取 image、LiDAR、CSI、GPS 文件或 beam label 文件
- **AND** helper MUST 不导入训练循环、模型 registry 或 heavy runtime module

#### Scenario: dataset 输出保持兼容
- **WHEN** 使用相同 synthetic dataframe、配置和 mock resource paths 构建 DeepSense6G dataset
- **THEN** helper 拆分前后的 sample keys、target fields、sample id、split metadata 和 enabled modality behavior MUST 保持兼容
- **AND** target labels MUST 不因 helper 拆分而改变

### Requirement: DeepSense6G GPS 和 target source contract 可测试
DeepSense6G GPS feature mode、scene calibration、GPS angle offset、GPS BEV XY source 和 beam target source MUST 有集中 helper 和 focused tests。错误信息 MUST 指向具体字段和支持值。

#### Scenario: unsupported GPS feature mode 被拒绝
- **WHEN** 配置声明未知 `gps_feature_mode`
- **THEN** helper MUST raise 清晰错误
- **AND** 错误信息 MUST 列出支持的 GPS feature mode

#### Scenario: current target source 保持 Table III 语义
- **WHEN** BeamBench-fair 或 Table III 风格配置声明 `beam_target_source=current`
- **THEN** helper MUST 保持 current beam target 语义
- **AND** `num_pred`、`seq_len` 和 target path 选择规则 MUST 与现有实现兼容

