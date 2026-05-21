## ADDED Requirements

### Requirement: Raymobtime snapshot 模态契约
中心化模态契约 MUST 支持 Raymobtime s008 所需的 `coord` 和 `ray` 模态。新增模态 MUST 不改变既有 `image`、`radar`、`gps`、`lidar`、`mmwave` 和 `csi` 的相对顺序、默认字段和旧配置行为。

#### Scenario: 枚举 Raymobtime 新模态
- **WHEN** 开发者查询受支持模态
- **THEN** 系统 MUST 返回既有模态以及 `coord` 和 `ray`
- **AND** 既有六个模态的相对顺序 MUST 保持不变
- **AND** `coord` 和 `ray` MUST 只在用户显式配置或 Raymobtime 配置中启用

#### Scenario: 查询 coord 模态元数据
- **WHEN** 开发者查询 `coord` 模态契约
- **THEN** 系统 MUST 返回样本字段 `coord`
- **AND** 系统 MUST 返回 fusion 输入字段 `coord_batch`
- **AND** 系统 MUST 返回 dataset flag `use_coord`
- **AND** 系统 MUST 返回当前 snapshot coordinate 输入契约和默认 `coord_input_size`

#### Scenario: 查询 ray 模态元数据
- **WHEN** 开发者查询 `ray` 模态契约
- **THEN** 系统 MUST 返回样本字段 `ray`
- **AND** 系统 MUST 返回 fusion 输入字段 `ray_batch`
- **AND** 系统 MUST 返回 dataset flag `use_ray`
- **AND** 系统 MUST 返回 path-level ray-tracing feature 输入契约和默认 `ray_input_size`

### Requirement: Raymobtime 模态列表标准化
系统 MUST 通过中心化模态契约标准化包含 `coord` 和 `ray` 的模态列表。标准化 MUST 继续拒绝未知模态、空列表和重复模态，并 MUST 按固定顺序返回结果。

#### Scenario: 标准化 Raymobtime 全模态
- **WHEN** 用户配置 Raymobtime 模态列表 `["ray", "coord", "lidar", "image"]`
- **THEN** 系统 MUST 返回同一组模态的规范顺序
- **AND** dataset、batch 准备、模型构建和分析输出 MUST 使用同一个标准化结果

#### Scenario: 拒绝重复 Raymobtime 模态
- **WHEN** 用户配置 `modalities: ["coord", "coord", "image"]`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指出模态列表包含重复项

#### Scenario: 拒绝未知 Raymobtime 模态
- **WHEN** 用户配置 `modalities: ["coord", "ray_path"]`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 包含未知模态和可用模态列表

### Requirement: coord/ray batch 输入准备
训练、验证、评估和诊断路径 MUST 能根据模态契约把 `coord` 和 `ray` batch 字段准备为模型输入。Raymobtime snapshot 输入 MUST 保留单步维度，并在模型进入前校验维度和 dtype。

#### Scenario: 准备 coord batch
- **WHEN** 模型启用 `coord` 模态且 batch 包含 `coord`
- **THEN** runtime MUST 构造 `coord_batch`
- **AND** `coord_batch` MUST 具有 `[B, 1, F_coord]` 语义
- **AND** 缺失 `coord` 字段时 MUST 报出包含当前模态名的清晰错误

#### Scenario: 准备 ray batch
- **WHEN** 模型启用 `ray` 模态且 batch 包含 `ray`
- **THEN** runtime MUST 构造 `ray_batch`
- **AND** `ray_batch` MUST 具有 `[B, 1, F_ray]` 语义
- **AND** 缺失 `ray` 字段时 MUST 报出包含当前模态名的清晰错误

#### Scenario: 旧配置不启用 coord/ray
- **WHEN** 用户加载现有 DeepSense6G、MMW 或 CSI 配置
- **THEN** 数据构建流程 MUST 不设置 `use_coord` 或 `use_ray`
- **AND** batch 准备 MUST 不要求 `coord` 或 `ray` 字段
