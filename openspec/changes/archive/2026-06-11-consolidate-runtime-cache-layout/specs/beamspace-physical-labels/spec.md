## MODIFIED Requirements

### Requirement: Physical label cache and diagnostics
系统 SHALL 缓存构造后的物理标签，并在首次构造时输出 scene-level 统计，便于复现实验和排查物理标签质量。未显式配置 cache_dir 时，物理标签 cache MUST 默认位于 `outputs/cache/physical_labels`。

#### Scenario: 从缓存读取 BSP
- **WHEN** `outputs/cache/physical_labels/<dataset_name>/<scene_name>/beamspace_power_<num_classes>.npz` 已存在且 metadata 匹配当前配置
- **THEN** dataset MUST 优先读取缓存
- **AND** 不应重复解析所有原始 beam power 或 path 文件

#### Scenario: 构造后保存缓存
- **WHEN** 物理标签缓存不存在且样本标签成功构造
- **THEN** 系统 MUST 保存 `.npz` 缓存和必要 metadata
- **AND** metadata MUST 包含 dataset、scene、num_classes、temperature、smoothing_sigma、source 类型和配置摘要

#### Scenario: 首次构造输出统计
- **WHEN** scene-level BSP 缓存首次生成
- **THEN** 系统 MUST 记录 availability、source 分布、entropy、peak probability 和 fallback 原因统计
- **AND** 统计 MUST 可写入 dataset metadata 或训练 runtime metadata
