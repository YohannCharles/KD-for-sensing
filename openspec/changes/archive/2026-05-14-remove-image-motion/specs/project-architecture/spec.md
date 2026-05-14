## MODIFIED Requirements

### Requirement: 模态数据转换职责拆分
数据转换模块 MUST 按 image、radar、lidar、gps、mmwave 和通用 IO/cache/normalization 职责组织。新增或修改某个仍受支持模态的数据读取、特征构造或 cache key 时，变更 MUST 不要求编辑其它模态的转换实现。image 转换职责 MUST 只覆盖 RGB/ImageNet image 加载与标准化，不再包含 image motion mask 或 image motion cache。

#### Scenario: 修改 GPS 特征不触碰 LiDAR 转换
- **WHEN** 开发者修改 GPS feature sequence 构造
- **THEN** 变更 MUST 限定在 GPS 转换相关模块和测试
- **AND** 不需要修改 LiDAR BEV、RGB image 或 mmWave feature 转换实现

#### Scenario: 兼容旧 transforms 导入
- **WHEN** 现有代码从 `kd_sensing.data.transforms` 导入仍受支持的转换函数或 scaler
- **THEN** 导入 MUST 继续成功
- **AND** 函数行为 MUST 与拆分后受支持实现保持一致
- **AND** 已删除的 image motion 公开符号 MUST 不再通过兼容 facade 暴露

### Requirement: 模态转换实现不得集中在 legacy 聚合模块
数据转换模块 MUST 将仍在使用的 image RGB、GPS、LiDAR、mmWave、radar、IO、cache 和 normalization 实现放入对应模块。`kd_sensing.data.transform_ops._legacy` MUST 不再导出或保留已删除的 image motion mask、image motion cache key 或 image motion metadata 符号。

#### Scenario: 删除 image motion 实现不触碰 LiDAR 实现
- **WHEN** 开发者删除 image motion mask 或 image motion cache key
- **THEN** 主要变更 MUST 限定在 image、preprocessing、config、diagnostics 和 tests 中的相关引用
- **AND** 不需要编辑 LiDAR、GPS、mmWave 或 radar 转换实现

#### Scenario: 修改 GPS scaler 不触碰 image 实现
- **WHEN** 开发者修改 GPS feature 或 scaler 加载保存逻辑
- **THEN** 主要变更 MUST 限定在 GPS 或通用 normalization 模块
- **AND** 不需要编辑 RGB image、LiDAR BEV 或 radar map 转换实现

#### Scenario: 旧 transforms facade 不暴露已删除符号
- **WHEN** 现有代码从 `kd_sensing.data.transforms` 导入 `load_motion_masks`、`build_motion_mask_pair` 或 `image_motion_cache_path`
- **THEN** 导入 MUST 失败
- **AND** 错误信息或 ImportError MUST 让开发者能识别这些 image motion 符号已删除

### Requirement: 源码与实验产物边界
项目 MUST 明确源码、配置、文档、OpenSpec artifacts 与本地数据、训练日志、缓存和输出产物的边界。本地运行产物 MUST 保持在 `.gitignore` 覆盖范围内，文档 MUST 指明哪些目录是可复现输入、哪些目录是可删除生成物。删除 image motion 源码与 cache 支持时，系统 MUST 不删除历史 `outputs/` 实验产物。

#### Scenario: 本地产物不进入版本控制
- **WHEN** 用户运行训练、评估、预处理或诊断命令
- **THEN** 生成的 logs、outputs、cache、checkpoint 和 Python bytecode 产物 MUST 位于忽略规则覆盖的路径或文件模式内
- **AND** 项目文档 MUST 不要求提交这些本地产物

#### Scenario: 文档说明产物边界
- **WHEN** 开发者阅读 README 或扩展指南
- **THEN** 文档 MUST 说明 `dataset/`、`All_models/`、`outputs/`、`logs/` 和 cache 目录的角色
- **AND** 文档 MUST 指明哪些目录通常不应纳入源码变更
- **AND** 文档 MUST 明确本次删除 image motion 不会清理历史 `outputs/`
