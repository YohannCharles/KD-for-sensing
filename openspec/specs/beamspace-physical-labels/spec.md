# beamspace-physical-labels Specification

## Purpose
定义 beamspace physical label 的数据契约、构造规则、缓存诊断和 target-domain 泄漏边界。该 spec 约束 v7 shared physical/private residual 等物理先验模型如何读取 beam power、path payload 或等价物理信号，并确保这些标签只在允许的训练阶段参与监督，在 target adaptation 和 target_test 中保持可审计的用途边界。

## Requirements
### Requirement: Beamspace physical label batch contract
系统 SHALL 在启用物理标签时为 beam prediction batch 提供 `beamspace_power_label`，该字段 MUST 与 `target_beam` 的 future horizon 对齐，并表示每个 horizon 上所有 beam class 的归一化 beamspace power distribution。

#### Scenario: batch 包含 beamspace power distribution
- **WHEN** dataset 配置启用 `physical_label.enabled=true` 且 `num_pred=3`、`num_classes=64`
- **THEN** 样本 MUST 包含 shape 为 `[3, 64]` 的 `beamspace_power_label`
- **AND** 每个有效 horizon 的分布和 MUST 在数值容差内等于 1
- **AND** 样本 MUST 继续包含 hard label 字段 `target_beam`

#### Scenario: 标签不可构造时显式失败或标记不可用
- **WHEN** 启用 `physical_label.required=true` 且样本无法从 beam power 或 path 数据构造 `beamspace_power_label`
- **THEN** dataset MUST 抛出包含 sample id、scene 和不可用原因的错误
- **AND** 系统 MUST NOT 静默返回全零或伪造的物理标签

#### Scenario: 非 required 模式记录不可用原因
- **WHEN** 启用 `physical_label.enabled=true` 但 `required=false` 且某个 horizon 无法构造物理标签
- **THEN** batch MUST 包含 `beamspace_power_available=false` 或等价 mask
- **AND** metadata MUST 记录 `beamspace_power_unavailable_reason`

### Requirement: Beam power vector normalization
系统 SHALL 优先从 beam gain、beam power 或 RSS vector 构造 `beamspace_power_label`。输入 vector MUST 先转换为非负线性功率，再按 class 维归一化。

#### Scenario: 线性 beam power 构造 BSP
- **WHEN** 样本 future beam path 指向有效的 `num_classes` 维有限 beam power vector
- **THEN** 系统 MUST clamp 到非负或 `eps` 以上并归一化为概率分布
- **AND** `beamspace_power_source` MUST 标记为 `beam_power_vector`

#### Scenario: dB beam power 构造 BSP
- **WHEN** 配置声明 beam power vector 使用 dB 单位
- **THEN** 系统 MUST 先执行 `10 ** (power_db / 10)` 转换为线性功率
- **AND** 再构造 `beamspace_power_label`

#### Scenario: temperature 平滑
- **WHEN** `physical_label.temperature > 1`
- **THEN** 系统 MUST 对归一化前或归一化后的非负分布应用 temperature 平滑
- **AND** 输出分布 MUST 重新归一化

### Requirement: Path-based beamspace label fallback
系统 SHALL 在 beam power vector 不可用时支持从 path 数据构造 beamspace power approximation。path parser MUST 支持配置字段映射，并记录文件可用 keys 和实际使用字段。

#### Scenario: 从 path AoD 和 complex gain 构造 BSP
- **WHEN** path payload 包含 AoD 或 departure angle 以及 complex gain 或 amplitude 字段
- **THEN** 系统 MUST 依据 path power 和 BS beam bin/codebook 投影累计 beam power
- **AND** 输出 MUST 归一化为 `beamspace_power_label`

#### Scenario: 无真实 codebook 时使用 AoD bin fallback
- **WHEN** 配置未提供真实 BS codebook 或 array response
- **THEN** 系统 MAY 将 AoD 均匀量化到 `num_classes` 个 beam bin
- **AND** 系统 MUST 支持 Gaussian smoothing 以避免 hard bin 过尖锐

#### Scenario: path parser 记录 keys
- **WHEN** path payload 被读取用于构造物理标签
- **THEN** diagnostics MUST 记录可用 keys、字段映射结果和不可用原因
- **AND** 字段缺失时 MUST 提示可通过 config 指定 key name

### Requirement: Physical label cache and diagnostics
系统 SHALL 缓存构造后的物理标签，并在首次构造时输出 scene-level 统计，便于复现实验和排查物理标签质量。

#### Scenario: 从缓存读取 BSP
- **WHEN** `cache/physical_labels/<dataset_name>/<scene_name>/beamspace_power_<num_classes>.npz` 已存在且 metadata 匹配当前配置
- **THEN** dataset MUST 优先读取缓存
- **AND** 不应重复解析所有原始 beam power 或 path 文件

#### Scenario: 构造后保存缓存
- **WHEN** 物理标签缓存不存在且样本标签成功构造
- **THEN** 系统 MUST 保存 `.npz` 缓存和必要 metadata
- **AND** metadata MUST 包含 dataset、scene、num_classes、temperature、smoothing_sigma、source 类型和配置摘要

#### Scenario: 首次构造输出统计
- **WHEN** scene-level BSP 缓存首次生成
- **THEN** 系统 MUST 记录样本数、beam 数、label entropy mean/std
- **AND** 系统 MUST 记录 `argmax(beamspace_power_label) == hard beam` 的 top1 agreement 诊断

### Requirement: Target-domain physical label leakage boundary
系统 SHALL 区分 source pretraining、target labeled adaptation、target unlabeled adaptation 和 target_test 中物理标签的允许用途。target adaptation 默认 MUST NOT 使用 target-side beam power/RSS/path oracle 作为训练监督。

#### Scenario: source pretraining 使用 BSP supervision
- **WHEN** source-domain batch 包含有效 `beamspace_power_label`
- **THEN** v7 source training MAY 使用该字段计算 shared beamspace KL 和 physical head KL
- **AND** diagnostics MUST 标记物理标签来源

#### Scenario: target adaptation 默认不使用 target power oracle
- **WHEN** target adaptation batch 包含 target-side beam power、path payload 或 `beamspace_power_label`
- **THEN** 默认 adaptation loss MUST NOT 对这些字段反传
- **AND** leakage diagnostics MUST 标记未使用 target physical oracle 作为训练监督

#### Scenario: target_test 仅用于评估诊断
- **WHEN** target_test batch 包含 beam power 或 `beamspace_power_label`
- **THEN** evaluation MAY 计算 power metrics 或 physical KL 诊断
- **AND** 这些字段 MUST NOT 影响模型参数更新
