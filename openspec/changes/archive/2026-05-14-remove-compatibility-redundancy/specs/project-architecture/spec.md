## ADDED Requirements

### Requirement: 兼容冗余入口已删除
项目 MUST 删除已经迁移到 canonical 模块的兼容入口。源码、测试、文档和推荐命令 MUST 不再依赖 `the builder facade module`、`the transform facade module`、`the transform aggregate module`、场景专用 dataset 兼容模块或旧可视化兼容入口。

#### Scenario: 兼容 facade 不再作为公开入口
- **WHEN** 开发者在源码、测试、README 或扩展指南中搜索已删除的兼容 facade
- **THEN** 不得出现 `the builder facade module`、`the transform facade module` 或 `the transform aggregate module` 的运行时引用
- **AND** 对应功能 MUST 通过职责明确的窄模块导入

#### Scenario: 旧入口引用检查
- **WHEN** 开发者运行架构边界检查
- **THEN** 检查 MUST 拒绝新增 `scene-specific dataset class alias`、`the scene-9 dataset-type spelling`、legacy fusion 配置路径或兼容 facade 引用
- **AND** 检查 MUST 在不读取真实数据和不加载 checkpoint 的情况下完成

## MODIFIED Requirements

### Requirement: 横切 builder 职责拆分
训练引擎 MUST 将配置到运行对象的构建逻辑按职责拆分。dataset/dataloader 构建、启用模态推导、cache policy、归一化 artifact、run metadata、optimizer/scheduler/device 构建 MUST 有明确模块边界。已拆分的 builder 功能 MUST 通过对应窄模块使用，项目 MUST 不再保留 `the builder facade module` 兼容 facade。

#### Scenario: 修改 cache policy 不触碰 optimizer 构建
- **WHEN** 开发者调整 image 或 LiDAR cache policy 解析逻辑
- **THEN** 变更 MUST 限定在 cache policy 相关模块及其测试
- **AND** 不需要修改 optimizer、scheduler 或 device 构建逻辑

#### Scenario: 修改 normalization artifact 不触碰 dataset 模态解析
- **WHEN** 开发者调整 GPS、LiDAR 或 mmWave 归一化 artifact 的保存和加载格式
- **THEN** 变更 MUST 限定在归一化 artifact 相关模块及其测试
- **AND** 不需要修改启用模态推导逻辑

#### Scenario: 旧 builders import 被拒绝
- **WHEN** 现有代码尝试从 `the builder facade module` 导入公开构建函数
- **THEN** 导入 MUST 失败或触发清晰迁移错误
- **AND** 错误信息 MUST 指向 `engine.data_factory`、`engine.optim`、`engine.cache_policy`、`engine.normalization_artifacts` 或其它对应窄模块

### Requirement: 模态数据转换职责拆分
数据转换模块 MUST 按 image、radar、lidar、gps、mmwave 和通用 IO/cache/normalization 职责组织。新增或修改某个模态的数据读取、特征构造或 cache key 时，变更 MUST 不要求编辑其它模态的转换实现。项目 MUST 不再保留 `the transform facade module` 或 `the transform aggregate module` 作为兼容聚合入口。

#### Scenario: 修改 GPS 特征不触碰 LiDAR 转换
- **WHEN** 开发者修改 GPS feature sequence 构造
- **THEN** 变更 MUST 限定在 GPS 转换相关模块和测试
- **AND** 不需要修改 LiDAR BEV、image 或 mmWave feature 转换实现

#### Scenario: 旧 transforms import 被拒绝
- **WHEN** 现有代码从 `the transform facade module` 或 `the transform aggregate module` 导入转换函数或 scaler
- **THEN** 导入 MUST 失败或触发清晰迁移错误
- **AND** 错误信息 MUST 指向 `kd_sensing.data.transform_ops.<modality>` 或通用 transform 子模块

### Requirement: builder 实现不得集中在私有聚合模块
训练引擎 MUST 将 builder 实现放在对应职责模块中。`the private builder aggregate` 和 `the builder facade module` MUST 不再作为实现聚合或兼容转发层存在；新实现和测试 MUST 以 `cache_policy`、`modality_resolution`、`data_factory`、`normalization_artifacts`、`run_metadata` 和 `optim` 等窄模块为主。

#### Scenario: cache policy 实现在 cache 模块
- **WHEN** 开发者查看或修改 cache policy 解析逻辑
- **THEN** 主要实现 MUST 位于 `kd_sensing.engine.cache_policy`
- **AND** 不需要编辑 optimizer、run metadata 或 dataset 构建实现

#### Scenario: optimizer 和 device 构建实现在 optim 模块
- **WHEN** 开发者查看或修改 optimizer、scheduler、device 或 distiller 参数组构建逻辑
- **THEN** 主要实现 MUST 位于 `kd_sensing.engine.optim`
- **AND** 不需要编辑 dataset/dataloader 构建实现

#### Scenario: builders facade 已删除
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证 `the builder facade module` 和 `the private builder aggregate` 不再被内部代码引用
- **AND** 测试 MUST 验证构建流程仍能通过窄模块完成

### Requirement: 模态转换实现不得集中在 legacy 聚合模块
数据转换模块 MUST 将仍在使用的 image RGB、GPS、LiDAR、mmWave、radar、IO、cache 和 normalization 实现放入对应模块。`the transform aggregate module` MUST 不再存在或不再作为运行时入口导出任何符号。

#### Scenario: 修改 image 实现不触碰 LiDAR 实现
- **WHEN** 开发者修改 RGB image 加载或标准化逻辑
- **THEN** 主要变更 MUST 限定在 image 转换相关模块和测试
- **AND** 不需要编辑 LiDAR、GPS、mmWave 或 radar 转换实现

#### Scenario: 修改 GPS scaler 不触碰 image 实现
- **WHEN** 开发者修改 GPS feature 或 scaler 加载保存逻辑
- **THEN** 主要变更 MUST 限定在 GPS 或通用 normalization 模块
- **AND** 不需要编辑 image、LiDAR BEV 或 radar map 转换实现

#### Scenario: legacy 聚合模块引用被拒绝
- **WHEN** 开发者运行内部引用扫描
- **THEN** 扫描 MUST 拒绝 `the transform aggregate module`
- **AND** 扫描 MUST 指向对应的窄 transform 模块作为迁移路径
